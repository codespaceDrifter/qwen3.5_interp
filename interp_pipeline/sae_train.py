# train one SAE type (residue / post_attn_norm / post_linear_attn_norm / post_mlp_norm)
# for all 32 layers of Qwen3.5-4B.
#
# Memory strategy:
#   - the base model and all SAEs stay on GPU.
#   - activations are copied to CPU RAM in the hooks so they don't pile up in VRAM.
#   - only one layer's optimizer lives on GPU at a time; its state dict is kept on CPU
#     between steps. this lets a 32 GB card fit the 32 SAEs + model + one batch.
#
# usage from repo root:
#   python3 -m interp_pipeline.sae_train

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from qwen3_5_4b_implementation.config import Qwen3_5Config
from qwen3_5_4b_implementation.loader import load_weights
from qwen3_5_4b_implementation.model import Qwen3_5ForCausalLM
from interp_pipeline.config import (
    BATCH_SIZE,
    BAND_EPS,
    CAPTURE_TYPE,
    CHECKPOINT_EVERY,
    CHUNK_SIZE,
    DEVICE,
    EXPANSION_FACTOR,
    LAYER_END,
    LAYER_START,
    LEARNING_RATE,
    LOG_EVERY,
    SAE_DTYPE,
    SAE_WEIGHTS_DIR,
    SPARSITY_COEFF_INIT,
    SWAP_OPTIMIZERS,
    TARGET_ACTIVE,
    TOKENIZED_DIR,
    TOKEN_DTYPE,
)
from probes.SAE import SAE, train_SAE


# =========================================================================================
# helpers
# =========================================================================================

def make_optimizer(sae: SAE, lr: float):
    """Build an 8-bit AdamW if available, otherwise full AdamW."""
    try:
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(sae.parameters(), lr=lr)
    except ImportError:
        return torch.optim.AdamW(sae.parameters(), lr=lr)


def state_dict_to_cpu(state_dict):
    """Move every tensor in an optimizer state dict to CPU for cheap storage."""
    result = {}
    for k, v in state_dict.items():
        if isinstance(v, dict):
            result[k] = state_dict_to_cpu(v)
        elif isinstance(v, torch.Tensor):
            result[k] = v.cpu()
        else:
            result[k] = v
    return result


def print_vram(label: str = ""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        prefix = f"VRAM [{label}] " if label else "VRAM "
        print(f"{prefix}allocated: {allocated:.2f} GB | reserved: {reserved:.2f} GB | total: {total:.2f} GB")


# =========================================================================================
# activation capture hooks
# =========================================================================================

def register_capture_hooks(model: Qwen3_5ForCausalLM, capture_type: str):
    """Register forward hooks that copy the chosen activations to CPU RAM."""
    captured = {}

    def make_hook(layer_idx: int):
        def hook(module, input, output):
            # modules may return tuples; take first element
            tensor = output[0] if isinstance(output, tuple) else output
            # offload immediately so VRAM doesn't accumulate 32 layer outputs
            captured[layer_idx] = tensor.detach().cpu()
        return hook

    text_model = model.model.language_model
    for i, layer in enumerate(text_model.layers):
        if capture_type == "residue":
            # full layer output: after attention/MLP residual adds
            layer.register_forward_hook(make_hook(i))
        elif capture_type == "post_attn_norm":
            # full-attention subblock output, before residual add
            if hasattr(layer, "self_attn"):
                layer.self_attn.register_forward_hook(make_hook(i))
        elif capture_type == "post_linear_attn_norm":
            # linear-attention subblock output, before residual add
            if hasattr(layer, "linear_attn"):
                layer.linear_attn.register_forward_hook(make_hook(i))
        elif capture_type == "post_mlp_norm":
            # MLP subblock output, before residual add
            layer.mlp.register_forward_hook(make_hook(i))
        else:
            raise ValueError(f"unknown capture_type: {capture_type}")

    return captured


# =========================================================================================
# main
# =========================================================================================

def main(
    device: str = DEVICE,
    dtype: torch.dtype = SAE_DTYPE,
):
    SAE_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = torch.device("cpu")

    # ---- load model ----
    print("loading model...")
    config = Qwen3_5Config()
    model = Qwen3_5ForCausalLM(config)
    load_weights(model, "weights", device=device, dtype=dtype, strict=True)
    model.to(device)
    model.eval()
    print("model loaded")

    # ---- register hooks ----
    captured = register_capture_hooks(model, CAPTURE_TYPE)

    # ---- build SAEs (all on GPU if available) ----
    text_cfg = config.text_config
    hidden_size = text_cfg.hidden_size
    num_layers = text_cfg.num_hidden_layers
    layer_start = max(0, LAYER_START)
    layer_end = min(num_layers - 1, LAYER_END)
    num_train_layers = layer_end - layer_start + 1
    print(f"building {num_train_layers} SAEs for {CAPTURE_TYPE} (layers {layer_start}-{layer_end})...")
    saes = nn.ModuleList([
        SAE(hidden_size, EXPANSION_FACTOR, band_eps=BAND_EPS)
        for _ in range(num_train_layers)
    ]).to(device).to(dtype)
    print_vram("after building SAEs")

    sparsity_coeffs = [SPARSITY_COEFF_INIT for _ in saes]

    if SWAP_OPTIMIZERS:
        # 32 GB card mode: optimizer state dicts live on CPU; only one optimizer on GPU at a time
        optimizer_states = [None for _ in saes]
        optimizers = None
        print("optimizer mode: swap one at a time (low VRAM)")
    else:
        # 96 GB card mode: keep all optimizers on GPU
        optimizers = [make_optimizer(sae, LEARNING_RATE) for sae in saes]
        optimizer_states = None
        print("optimizer mode: all optimizers on GPU (high VRAM)")
    print_vram("after building optimizers")

    # ---- load token bin ----
    bin_path = TOKENIZED_DIR / "tokenized_interpmix.bin"
    if not bin_path.exists():
        raise FileNotFoundError(
            f"tokenized data not found: {bin_path}\n"
            "Run: python3 -m interp_pipeline.tokenize_interp_mix"
        )

    all_tokens = np.fromfile(bin_path, dtype=TOKEN_DTYPE)
    num_chunks = len(all_tokens) // CHUNK_SIZE
    if num_chunks == 0:
        raise ValueError(f"no full {CHUNK_SIZE}-token chunks in {bin_path}")
    print(f"tokens: {len(all_tokens):,}, chunks: {num_chunks:,}")

    # ---- training loop ----
    print("training...")
    start_time = time.time()
    log = []

    for step in range(num_chunks // BATCH_SIZE):
        # build batch
        start = step * BATCH_SIZE * CHUNK_SIZE
        end = start + BATCH_SIZE * CHUNK_SIZE
        if end > len(all_tokens):
            break

        batch_ids = torch.from_numpy(all_tokens[start:end]).view(BATCH_SIZE, CHUNK_SIZE).to(device)

        # forward pass, capture activations to CPU (no KV cache needed for training)
        with torch.no_grad():
            _ = model(input_ids=batch_ids, use_cache=False)

        # train each layer's SAE one at a time
        for i, sae in enumerate(saes):
            layer_idx = layer_start + i
            if layer_idx not in captured:
                continue

            acts = captured[layer_idx].to(device).to(dtype)  # (B, L, D)
            acts = acts.view(-1, acts.shape[-1])             # (B*L, D)

            if SWAP_OPTIMIZERS:
                opt = make_optimizer(sae, LEARNING_RATE)
                if optimizer_states[i] is not None:
                    opt.load_state_dict(optimizer_states[i])
            else:
                opt = optimizers[i]

            sparsity_coeffs[i], loss, l0, recon = train_SAE(
                sae, acts, opt, TARGET_ACTIVE, sparsity_coeffs[i]
            )

            if SWAP_OPTIMIZERS:
                # store optimizer state on CPU and free the GPU optimizer object
                optimizer_states[i] = state_dict_to_cpu(opt.state_dict())
                del opt
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            del acts

            if step % LOG_EVERY == 0 and step > 0:
                log.append({
                    "step": step,
                    "layer": layer_idx,
                    "loss": loss,
                    "l0": l0,
                    "recon": recon,
                    "sparsity_coeff": sparsity_coeffs[i],
                })

        # checkpoint
        if step % CHECKPOINT_EVERY == 0 and step > 0:
            ckpt_dir = SAE_WEIGHTS_DIR / CAPTURE_TYPE / f"step_{step}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            for i, sae in enumerate(saes):
                torch.save(sae.state_dict(), ckpt_dir / f"layer_{layer_start + i}.pt")
            print(f"saved checkpoint at step {step}")

        if step % LOG_EVERY == 0 and step > 0:
            elapsed = time.time() - start_time
            tok_per_sec = (step * BATCH_SIZE * CHUNK_SIZE) / elapsed
            print(f"step {step} | tok/s: {tok_per_sec:,.0f} | elapsed: {elapsed/3600:.2f}h")
            print_vram("log")

        # clear CPU activation cache before the next forward
        captured.clear()

    # ---- final save ----
    final_dir = SAE_WEIGHTS_DIR / CAPTURE_TYPE / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    for i, sae in enumerate(saes):
        torch.save(sae.state_dict(), final_dir / f"layer_{layer_start + i}.pt")

    with open(final_dir / "log.json", "w") as f:
        json.dump(log, f)

    total_time = time.time() - start_time
    print(f"done in {total_time/3600:.2f}h")
    print_vram("final")


def _dtype_from_string(s: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(s, torch.bfloat16)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train layer-wise SAEs on Qwen3.5-4B activations. "
        "Edit interp_pipeline/config.py to change hyperparameters."
    )
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--dtype", default="bfloat16", help="bfloat16|float16|float32")
    args = parser.parse_args()

    main(
        device=args.device,
        dtype=_dtype_from_string(args.dtype),
    )
