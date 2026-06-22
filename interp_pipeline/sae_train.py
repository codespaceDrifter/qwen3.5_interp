# train one SAE type (residue / post_attn_norm / post_linear_attn_norm / post_mlp_norm)
# for all 32 layers of Qwen3.5-4B.
#
# Memory strategy:
#   - the base model and all SAEs stay on GPU.
#   - activations are copied to CPU RAM in the hooks so they don't pile up in VRAM.
#   - optimizers use bitsandbytes PagedAdamW8bit so their state pages to CPU RAM
#     automatically per layer. no manual swapping.
#
# usage from repo root:
#   python3 -m interp_pipeline.sae_train

import argparse
import datetime
import json
import shutil
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
    CLIP_GRAD_NORM,
    CLIP_GRAD_VALUE,
    DEVICE,
    EXPANSION_FACTOR,
    LEARNING_RATE,
    LOG_EVERY,
    PAGE_OPTIMIZERS,
    SAE_DTYPE,
    SAE_WEIGHTS_DIR,
    SPARSITY_COEFF_INIT,
    TARGET_ACTIVE,
    TOKENIZED_DIR,
    TOKEN_DTYPE,
)
from probes.SAE import SAE, train_SAE


# =========================================================================================
# helpers
# =========================================================================================

def make_optimizer(sae: SAE, lr: float):
    """Build PagedAdamW8bit if available and configured, otherwise 8-bit/full AdamW."""
    try:
        import bitsandbytes as bnb
        if PAGE_OPTIMIZERS:
            return bnb.optim.PagedAdamW8bit(sae.parameters(), lr=lr)
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


def latest_checkpoint_dir(capture_type: str) -> Path:
    return SAE_WEIGHTS_DIR / capture_type / "latest"


def prev_checkpoint_dir(capture_type: str) -> Path:
    return SAE_WEIGHTS_DIR / capture_type / "prev"


def checkpoint_dir_order() -> list:
    return ["latest", "prev"]


def _state_dict_has_invalid(state_dict) -> bool:
    """Return True if any tensor in the state dict contains NaN or Inf."""
    for v in state_dict.values():
        if isinstance(v, dict):
            if _state_dict_has_invalid(v):
                return True
        elif isinstance(v, torch.Tensor):
            if torch.isnan(v).any() or torch.isinf(v).any():
                return True
    return False


def _saes_have_invalid(saes: nn.ModuleList) -> bool:
    """Return True if any SAE parameter contains NaN or Inf."""
    for sae in saes:
        if _state_dict_has_invalid(sae.state_dict()):
            return True
    return False


def _load_one_checkpoint(ckpt_dir: Path, saes: nn.ModuleList, optimizers, device, dtype):
    """Load a single checkpoint directory. Returns parsed state or raises on failure/NaN."""
    # SAE weights
    for i, sae in enumerate(saes):
        path = ckpt_dir / f"layer_{i}.pt"
        if not path.exists():
            raise FileNotFoundError(f"missing layer weight: {path}")
        sae.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        sae.to(device).to(dtype)

    if _saes_have_invalid(saes):
        raise ValueError("SAE weights contain NaN or Inf")

    state = torch.load(ckpt_dir / "training_state.pt", map_location="cpu", weights_only=False)
    saved_step = state["step"]
    sparsity_coeffs = state["sparsity_coeffs"]
    log = state["log"]

    optimizer_states = torch.load(
        ckpt_dir / "optimizer_states.pt", map_location="cpu", weights_only=False
    )

    # restore optimizer states into existing optimizer objects
    for opt, opt_state in zip(optimizers, optimizer_states):
        opt.load_state_dict(opt_state)

    return saved_step, sparsity_coeffs, log


def save_checkpoint(
    capture_type: str,
    step: int,
    saes: nn.ModuleList,
    optimizers,
    sparsity_coeffs: list,
    log: list,
):
    """Atomically save a resumable checkpoint, keeping the previous one as `prev`."""
    ckpt_dir = latest_checkpoint_dir(capture_type)
    prev_dir = prev_checkpoint_dir(capture_type)
    tmp_dir = ckpt_dir.with_suffix(".tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # SAE weights
    for i, sae in enumerate(saes):
        torch.save(sae.state_dict(), tmp_dir / f"layer_{i}.pt")

    # optimizer states (move to CPU before saving to keep checkpoint small)
    states_to_save = [state_dict_to_cpu(opt.state_dict()) for opt in optimizers]
    torch.save(states_to_save, tmp_dir / "optimizer_states.pt")

    # everything else needed to resume exactly
    torch.save({
        "step": step,
        "sparsity_coeffs": sparsity_coeffs,
        "log": log,
    }, tmp_dir / "training_state.pt")

    # rotate: latest -> prev, then tmp -> latest
    if prev_dir.exists():
        shutil.rmtree(prev_dir)
    if ckpt_dir.exists():
        ckpt_dir.rename(prev_dir)
    tmp_dir.rename(ckpt_dir)


def load_checkpoint(capture_type: str, saes: nn.ModuleList, optimizers, device, dtype):
    """Load the latest valid checkpoint, falling back to prev if latest is corrupt."""
    roots = {
        "latest": latest_checkpoint_dir(capture_type),
        "prev": prev_checkpoint_dir(capture_type),
    }

    for name in checkpoint_dir_order():
        ckpt_dir = roots[name]
        if not ckpt_dir.exists():
            continue
        try:
            print(f"resuming from checkpoint: {ckpt_dir}")
            saved_step, sparsity_coeffs, log = _load_one_checkpoint(
                ckpt_dir, saes, optimizers, device, dtype
            )
            print(f"  resumed at step {saved_step}")
            print_vram("after resume load")
            return saved_step, sparsity_coeffs, log
        except Exception as e:
            print(f"  failed to load {name}: {e}")

    print("no valid checkpoint found, starting from scratch")
    return None


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

    # ---- build SAEs (random init; overwritten by checkpoint if resuming) ----
    config = Qwen3_5Config()
    text_cfg = config.text_config
    hidden_size = text_cfg.hidden_size
    num_layers = text_cfg.num_hidden_layers
    print(f"building {num_layers} SAEs for {CAPTURE_TYPE}...")
    saes = nn.ModuleList([
        SAE(hidden_size, EXPANSION_FACTOR, band_eps=BAND_EPS)
        for _ in range(num_layers)
    ]).to(device).to(dtype)
    print_vram("after building SAEs")

    # ---- build optimizers (bind to SAE params; state paged to CPU if PagedAdamW8bit) ----
    print("building optimizers...")
    optimizers = [make_optimizer(sae, LEARNING_RATE) for sae in saes]
    print(f"optimizer class: {type(optimizers[0]).__name__}")
    print_vram("after building optimizers")

    sparsity_coeffs = [SPARSITY_COEFF_INIT for _ in saes]
    log = []
    train_log = []
    start_step = 0

    # ---- try to resume before loading the heavy model ----
    loaded = load_checkpoint(CAPTURE_TYPE, saes, optimizers, device, dtype)
    if loaded is not None:
        saved_step, sparsity_coeffs, log = loaded
        start_step = saved_step + 1

    # ---- load model ----
    print("loading model...")
    model = Qwen3_5ForCausalLM(config)
    load_weights(model, "weights", device=device, dtype=dtype, strict=True)
    model.to(device)
    model.eval()
    print("model loaded")
    print_vram("after model load")

    # ---- register hooks ----
    captured = register_capture_hooks(model, CAPTURE_TYPE)

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

    for step in range(start_step, num_chunks // BATCH_SIZE):
        # build batch
        batch_start = step * BATCH_SIZE * CHUNK_SIZE
        batch_end = batch_start + BATCH_SIZE * CHUNK_SIZE
        if batch_end > len(all_tokens):
            break

        batch_ids = torch.from_numpy(all_tokens[batch_start:batch_end]).view(BATCH_SIZE, CHUNK_SIZE).to(device)

        # forward pass, capture activations to CPU (no KV cache needed for training)
        with torch.no_grad():
            _ = model(input_ids=batch_ids, use_cache=False)

        # train each layer's SAE one at a time
        for i, sae in enumerate(saes):
            if i not in captured:
                continue

            acts = captured[i].to(device).to(dtype)  # (B, L, D)
            acts = acts.view(-1, acts.shape[-1])     # (B*L, D)

            sparsity_coeffs[i], loss, l0, recon = train_SAE(
                sae,
                acts,
                optimizers[i],
                TARGET_ACTIVE,
                sparsity_coeffs[i],
                clip_grad_norm=CLIP_GRAD_NORM,
                clip_grad_value=CLIP_GRAD_VALUE,
            )

            del acts

            if step % LOG_EVERY == 0 and step > 0:
                log.append({
                    "step": step,
                    "layer": i,
                    "loss": loss,
                    "l0": l0,
                    "recon": recon,
                    "sparsity_coeff": sparsity_coeffs[i],
                })

        # checkpoint (overwrites `latest`)
        if step % CHECKPOINT_EVERY == 0 and step > 0:
            save_checkpoint(
                CAPTURE_TYPE,
                step,
                saes,
                optimizers,
                sparsity_coeffs,
                log,
            )
            print(f"saved checkpoint at step {step}")

        if step % LOG_EVERY == 0 and step > 0:
            elapsed = time.time() - start_time
            tok_per_sec = (step * BATCH_SIZE * CHUNK_SIZE) / elapsed
            print(
                f"step {step} | tok/s: {tok_per_sec:,.0f} | elapsed: {elapsed/3600:.2f}h | "
                f"recon: {recon:.6f} | sparsity: {l0:.6f}"
            )
            print_vram("log")

            train_log.append({
                "time": datetime.datetime.now().isoformat(),
                "batch": step,
                "tok_per_sec": tok_per_sec,
                "recon": recon,
                "sparsity": l0,
            })
            train_log_path = SAE_WEIGHTS_DIR / CAPTURE_TYPE / "train_log.json"
            with open(train_log_path, "w") as f:
                json.dump(train_log, f, indent=2)

        # clear CPU activation cache before the next forward
        captured.clear()

    # ---- final save ----
    final_dir = SAE_WEIGHTS_DIR / CAPTURE_TYPE / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    for i, sae in enumerate(saes):
        torch.save(sae.state_dict(), final_dir / f"layer_{i}.pt")

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
