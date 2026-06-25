# train one or more SAE capture types for all 32 layers of Qwen3.5-4B.
#
# Memory strategy:
#   - the base model and all SAEs stay on GPU.
#   - activations are copied to CPU RAM in the hooks so they don't pile up in VRAM.
#   - optimizers use bitsandbytes PagedAdamW8bit so their state pages to CPU RAM
#     automatically per layer. no manual swapping.
#
# Supported CAPTURE_TYPE values (set in interp_pipeline/config.py):
#   "residue"      -> full decoder layer output
#   "attn_out"     -> full self-attention subblock output
#   "linear_attn_out" -> linear attention / Gated DeltaNet subblock output
#   "mlp_out"      -> MLP subblock output
#   "attention"    -> trains attn_out + linear_attn_out in a single run
#
# usage from repo root:
#   python3 -m interp_pipeline.sae_train

import datetime
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from qwen3_5_4b_implementation.config import Qwen3_5Config
from qwen3_5_4b_implementation.loader import load_weights
from qwen3_5_4b_implementation.model import Qwen3_5ForCausalLM
from interp_pipeline.config import (
    AUXK_COEFF,
    BATCH_SIZE,
    CAPTURE_TYPE,
    CHECKPOINT_EVERY,
    CHUNK_SIZE,
    CLIP_GRAD_NORM,
    CLIP_GRAD_VALUE,
    DEAD_THRESHOLD,
    DEVICE,
    EXPANSION_FACTOR,
    LEARNING_RATE,
    LOG_EVERY,
    LR_WARMUP_STEPS,
    MAX_TRAIN_CHUNKS,
    PAGE_OPTIMIZERS,
    RESAMPLE_COOLDOWN,
    RESAMPLE_EVERY,
    SAE_DTYPE,
    SAE_TYPE,
    SAE_WEIGHTS_DIR,
    SPARSITY_COEFF_INIT,
    TARGET_ACTIVE,
    TOKENIZED_DIR,
    TOKEN_DTYPE,
)
from probes.AdvTopKSAE import AdvTopKSAE, train_AdvTopKSAE
from probes.GatedSAE import GatedSAE, train_GatedSAE
from probes.TopKSAE import TopKSAE, train_TopKSAE


# =========================================================================================
# helpers
# =========================================================================================

def make_optimizer(sae: nn.Module, lr: float, betas=(0.9, 0.999)):
    """Build PagedAdamW8bit if available and configured, otherwise 8-bit/full AdamW."""
    try:
        import bitsandbytes as bnb
        if PAGE_OPTIMIZERS:
            return bnb.optim.PagedAdamW8bit(sae.parameters(), lr=lr, betas=betas)
        return bnb.optim.AdamW8bit(sae.parameters(), lr=lr, betas=betas)
    except ImportError:
        return torch.optim.AdamW(sae.parameters(), lr=lr, betas=betas)


def plot_train_log(train_log: list, save_path: Path):
    """Plot recon_pct from train_log (matplotlib)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if len(train_log) < 2:
        return

    batches = [entry["batch"] for entry in train_log]
    recon_pcts = [entry["recon_pct"] for entry in train_log]

    plt.figure(figsize=(10, 6))
    plt.plot(batches, recon_pcts, marker="o", markersize=3)
    plt.xlabel("batch")
    plt.ylabel("recon_pct (%)")
    plt.title("SAE training")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_layer_grid_html(group, save_path: Path):
    """Interactive HTML grid: one recon_pct curve per layer."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return
    if len(group.log) < 2:
        return

    layers = sorted({e["layer"] for e in group.log})
    rows, cols = 8, 4
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[f"layer {l}" for l in layers],
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.04,
        horizontal_spacing=0.04,
    )
    for idx, layer in enumerate(layers):
        entries = [e for e in group.log if e["layer"] == layer]
        xs = [e["step"] for e in entries]
        ys = [e["recon_pct"] for e in entries]
        r = idx // cols + 1
        c = idx % cols + 1
        fig.add_trace(
            go.Scatter(x=xs, y=ys, mode="lines", showlegend=False, line=dict(width=1)),
            row=r, col=c,
        )
    fig.update_layout(
        title=f"{group.name} per-layer recon_pct (%)",
        height=1600,
        width=1400,
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(save_path, include_plotlyjs="cdn")


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
# capture specification
# =========================================================================================

def make_hook(captured: dict, key):
    """Factory for a hook that stores the first tensor output under `key`."""
    def hook(module, input, output):
        tensor = output[0] if isinstance(output, tuple) else output
        captured[key] = tensor.detach().cpu()
    return hook


def register_residue_hooks(model: Qwen3_5ForCausalLM, captured: dict):
    text_model = model.model.language_model
    for i, layer in enumerate(text_model.layers):
        layer.register_forward_hook(make_hook(captured, ("residue", i)))


def register_attn_hooks(model: Qwen3_5ForCausalLM, captured: dict):
    text_model = model.model.language_model
    for i, layer in enumerate(text_model.layers):
        if hasattr(layer, "self_attn"):
            layer.self_attn.register_forward_hook(make_hook(captured, ("attn_out", i)))


def register_linear_attn_hooks(model: Qwen3_5ForCausalLM, captured: dict):
    text_model = model.model.language_model
    for i, layer in enumerate(text_model.layers):
        if hasattr(layer, "linear_attn"):
            layer.linear_attn.register_forward_hook(make_hook(captured, ("linear_attn_out", i)))


def register_mlp_hooks(model: Qwen3_5ForCausalLM, captured: dict):
    text_model = model.model.language_model
    for i, layer in enumerate(text_model.layers):
        layer.mlp.register_forward_hook(make_hook(captured, ("mlp_out", i)))


def get_capture_specs(capture_type: str) -> list:
    """Return the list of capture specs to train for a given CAPTURE_TYPE.

    Each spec is (name, hook_registration_function). CAPTURE_TYPE="attention"
    returns two specs so both attention types train in one run.
    """
    specs = {
        "residue": [("residue", register_residue_hooks)],
        "attn_out": [("attn_out", register_attn_hooks)],
        "linear_attn_out": [("linear_attn_out", register_linear_attn_hooks)],
        "mlp_out": [("mlp_out", register_mlp_hooks)],
        "attention": [
            ("attn_out", register_attn_hooks),
            ("linear_attn_out", register_linear_attn_hooks),
        ],
    }
    if capture_type not in specs:
        raise ValueError(f"unknown CAPTURE_TYPE: {capture_type}")
    return specs[capture_type]


# =========================================================================================
# checkpointing
# =========================================================================================

def latest_checkpoint_dir(capture_name: str) -> Path:
    return SAE_WEIGHTS_DIR / capture_name / "latest"


def prev_checkpoint_dir(capture_name: str) -> Path:
    return SAE_WEIGHTS_DIR / capture_name / "prev"


def _state_dict_has_invalid(state_dict) -> bool:
    for v in state_dict.values():
        if isinstance(v, dict):
            if _state_dict_has_invalid(v):
                return True
        elif isinstance(v, torch.Tensor):
            if torch.isnan(v).any() or torch.isinf(v).any():
                return True
    return False


def _saes_have_invalid(saes: nn.ModuleList) -> bool:
    for sae in saes:
        if _state_dict_has_invalid(sae.state_dict()):
            return True
    return False


def _load_one_checkpoint(ckpt_dir: Path, saes: nn.ModuleList, optimizers, device, dtype):
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
    for opt, opt_state in zip(optimizers, optimizer_states):
        opt.load_state_dict(opt_state)

    return saved_step, sparsity_coeffs, log


def load_checkpoint(capture_name: str, saes: nn.ModuleList, optimizers, device, dtype):
    roots = {
        "latest": latest_checkpoint_dir(capture_name),
        "prev": prev_checkpoint_dir(capture_name),
    }
    for name in ["latest", "prev"]:
        ckpt_dir = roots[name]
        if not ckpt_dir.exists():
            continue
        try:
            print(f"  [{capture_name}] resuming from {name}")
            saved_step, sparsity_coeffs, log = _load_one_checkpoint(
                ckpt_dir, saes, optimizers, device, dtype
            )
            print(f"    resumed at step {saved_step}")
            return saved_step, sparsity_coeffs, log
        except Exception as e:
            print(f"    failed to load {name}: {e}")
    print(f"  [{capture_name}] no valid checkpoint, starting from scratch")
    return None


def save_checkpoint(
    capture_name: str,
    step: int,
    saes: nn.ModuleList,
    optimizers,
    sparsity_coeffs: list,
    log: list,
):
    ckpt_dir = latest_checkpoint_dir(capture_name)
    prev_dir = prev_checkpoint_dir(capture_name)
    tmp_dir = ckpt_dir.with_suffix(".tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for i, sae in enumerate(saes):
        torch.save(sae.state_dict(), tmp_dir / f"layer_{i}.pt")

    states_to_save = [state_dict_to_cpu(opt.state_dict()) for opt in optimizers]
    torch.save(states_to_save, tmp_dir / "optimizer_states.pt")

    torch.save({
        "step": step,
        "sparsity_coeffs": sparsity_coeffs,
        "log": log,
    }, tmp_dir / "training_state.pt")

    # rotate: old latest becomes prev, then temp becomes latest.
    # this gives us one backup checkpoint in case latest is corrupt.
    if prev_dir.exists():
        shutil.rmtree(prev_dir)
    if ckpt_dir.exists():
        ckpt_dir.rename(prev_dir)
    tmp_dir.rename(ckpt_dir)


# =========================================================================================
# capture group (one per output type)
# =========================================================================================

# A CaptureGroup holds all state for one SAE output type (e.g. "attn_out").
# When CAPTURE_TYPE="attention" we run two groups (attn_out + linear_attn_out)
# in a single training pass, sharing the model forward.
@dataclass
class CaptureGroup:
    name: str
    saes: nn.ModuleList
    optimizers: list
    sparsity_coeffs: list
    log: list
    train_log: list
    start_step: int
    log_layer_idx: int = 0


def build_capture_group(name: str, num_layers: int, hidden_size: int, device, dtype):
    # pick SAE architecture and optimizer settings from config
    if SAE_TYPE == "adv_topk":
        sae_cls = AdvTopKSAE
        sae_kwargs = {
            "auxk_coeff": AUXK_COEFF,
            "dead_threshold": DEAD_THRESHOLD,
            "cooldown_steps": RESAMPLE_COOLDOWN,
        }
        betas = (0.0, 0.999)
    elif SAE_TYPE == "topk":
        sae_cls = TopKSAE
        sae_kwargs = {}
        betas = (0.9, 0.999)
    else:
        sae_cls = GatedSAE
        sae_kwargs = {}
        betas = (0.9, 0.999)

    # one SAE per layer; all 32 live on GPU
    saes = nn.ModuleList([
        sae_cls(hidden_size, EXPANSION_FACTOR, **sae_kwargs)
        for _ in range(num_layers)
    ]).to(device).to(dtype)
    # one optimizer per SAE; PagedAdamW8bit keeps most state in CPU RAM
    optimizers = [make_optimizer(sae, LEARNING_RATE, betas=betas) for sae in saes]
    sparsity_coeffs = [SPARSITY_COEFF_INIT for _ in saes]
    return CaptureGroup(
        name=name,
        saes=saes,
        optimizers=optimizers,
        sparsity_coeffs=sparsity_coeffs,
        log=[],
        train_log=[],
        start_step=0,
        log_layer_idx=0,
    )


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

    config = Qwen3_5Config()
    text_cfg = config.text_config
    hidden_size = text_cfg.hidden_size
    num_layers = text_cfg.num_hidden_layers

    specs = get_capture_specs(CAPTURE_TYPE)
    print(f"capture type: {CAPTURE_TYPE}")
    print(f"  training: {', '.join(name for name, _ in specs)}")

    # ---- build capture groups ----
    groups = []
    for name, _ in specs:
        print(f"\nbuilding {num_layers} SAEs for {name}...")
        group = build_capture_group(name, num_layers, hidden_size, device, dtype)
        print(f"optimizer class: {type(group.optimizers[0]).__name__}")
        print_vram(f"after building {name}")
        groups.append(group)

    # ---- resume before loading the heavy model ----
    # each capture group loads its own checkpoint independently.
    # for combined attention both groups should have the same saved step.
    print("\nresuming checkpoints...")
    for group in groups:
        loaded = load_checkpoint(group.name, group.saes, group.optimizers, device, dtype)
        if loaded is not None:
            saved_step, group.sparsity_coeffs, group.log = loaded
            group.start_step = saved_step + 1

    # combined mode: both groups saved together, but if they differ use the
    # smaller step so no group misses batches.
    start_step = min((g.start_step for g in groups), default=0)
    if len(set(g.start_step for g in groups)) > 1:
        print(f"warning: capture groups have different resume steps; using min {start_step}")

    # ---- load model ----
    print("\nloading model...")
    model = Qwen3_5ForCausalLM(config)
    load_weights(model, "weights", device=device, dtype=dtype, strict=True)
    model.to(device)
    model.eval()
    print("model loaded")
    print_vram("after model load")

    # ---- register hooks for all capture groups ----
    captured = {}
    for name, register_fn in specs:
        register_fn(model, captured)

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
    print(f"\ntokens: {len(all_tokens):,}, chunks: {num_chunks:,}")

    max_steps = num_chunks // BATCH_SIZE
    if MAX_TRAIN_CHUNKS is not None:
        max_steps = min(max_steps, MAX_TRAIN_CHUNKS // BATCH_SIZE)
    print(f"training up to {max_steps:,} steps (~{max_steps * BATCH_SIZE * CHUNK_SIZE / 1e9:.2f}B tokens)...")

    # ---- training loop ----
    print("training...")
    start_time = time.time()

    def log_metrics(step: int, groups: list):
        """Log one layer per capture group, cycling through that group's actual layers."""
        parts = [f"step {step}"]
        for group in groups:
            entries = [e for e in group.log if e["step"] == step]
            if not entries:
                continue
            layers = sorted({e["layer"] for e in entries})
            layer_idx = layers[group.log_layer_idx % len(layers)]
            entry = next(e for e in entries if e["layer"] == layer_idx)
            group.log_layer_idx += 1
            parts.append(
                f"{group.name}: layer={entry['layer']} "
                f"recon={entry['recon']:.2e} "
                f"recon_pct={entry['recon_pct']:.2f}% "
                f"active={entry['active_count']:.1f}"
            )
            group.train_log.append({
                "time": datetime.datetime.now().isoformat(),
                "batch": step,
                "layer": entry["layer"],
                "recon": entry["recon"],
                "recon_pct": entry["recon_pct"],
                "active_count": entry["active_count"],
            })
            train_log_path = SAE_WEIGHTS_DIR / group.name / "train_log.json"
            train_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(train_log_path, "w") as f:
                json.dump(group.train_log, f, indent=2)
            plot_train_log(group.train_log, train_log_path.with_name("loss.png"))
            plot_layer_grid_html(group, train_log_path.with_name("layers.html"))
        print(" | ".join(parts))

    for step in range(start_step, max_steps):
        # global LR warmup
        if LR_WARMUP_STEPS > 0:
            lr_scale = min(1.0, step / LR_WARMUP_STEPS)
        else:
            lr_scale = 1.0
        for group in groups:
            for opt in group.optimizers:
                for pg in opt.param_groups:
                    pg["lr"] = LEARNING_RATE * lr_scale

        batch_start = step * BATCH_SIZE * CHUNK_SIZE
        batch_end = batch_start + BATCH_SIZE * CHUNK_SIZE
        if batch_end > len(all_tokens):
            break

        batch_ids = torch.from_numpy(all_tokens[batch_start:batch_end]).view(BATCH_SIZE, CHUNK_SIZE).to(device)

        with torch.no_grad():
            _ = model(input_ids=batch_ids, use_cache=False)

        # train each captured activation. `captured` keys are (capture_name, layer_idx).
        # in single-type mode only one capture_name is present; in "attention" mode
        # both attn_out and linear_attn_out keys appear.
        for key in list(captured.keys()):
            capture_name, layer_idx = key
            acts = captured[key].to(device).to(dtype)
            acts = acts.view(-1, acts.shape[-1])

            group = next(g for g in groups if g.name == capture_name)
            sae = group.saes[layer_idx]
            opt = group.optimizers[layer_idx]

            if SAE_TYPE == "topk":
                loss, recon, recon_pct, active_count = train_TopKSAE(
                    sae,
                    acts,
                    opt,
                    k=TARGET_ACTIVE,
                    clip_grad_norm=CLIP_GRAD_NORM,
                    clip_grad_value=CLIP_GRAD_VALUE,
                )
                l0 = 0.0
                unweighted_l0 = 0.0
                group.sparsity_coeffs[layer_idx] = 0.0
            elif SAE_TYPE == "adv_topk":
                loss, recon, recon_pct, active_count = train_AdvTopKSAE(
                    sae,
                    acts,
                    opt,
                    k=TARGET_ACTIVE,
                    step=step,
                    resample_every=RESAMPLE_EVERY,
                    auxk_coeff=AUXK_COEFF,
                    clip_grad_norm=CLIP_GRAD_NORM,
                    clip_grad_value=CLIP_GRAD_VALUE,
                )
                l0 = 0.0
                unweighted_l0 = 0.0
                group.sparsity_coeffs[layer_idx] = 0.0
            else:
                group.sparsity_coeffs[layer_idx], loss, l0, recon, unweighted_l0, recon_pct, active_count = train_GatedSAE(
                    sae,
                    acts,
                    opt,
                    TARGET_ACTIVE,
                    group.sparsity_coeffs[layer_idx],
                    clip_grad_norm=CLIP_GRAD_NORM,
                    clip_grad_value=CLIP_GRAD_VALUE,
                )

            del acts

            if step % LOG_EVERY == 0 and step > 0:
                group.log.append({
                    "step": step,
                    "layer": layer_idx,
                    "loss": loss,
                    "l0": l0,
                    "unweighted_l0": unweighted_l0,
                    "recon": recon,
                    "recon_pct": recon_pct,
                    "active_count": active_count,
                    "sparsity_coeff": group.sparsity_coeffs[layer_idx],
                })

        # checkpoint all groups
        if step % CHECKPOINT_EVERY == 0 and step > 0:
            for group in groups:
                save_checkpoint(
                    group.name,
                    step,
                    group.saes,
                    group.optimizers,
                    group.sparsity_coeffs,
                    group.log,
                )
                print(f"  saved {group.name} checkpoint at step {step}")

        if step % LOG_EVERY == 0 and step > 0:
            log_metrics(step, groups)

        captured.clear()

    # ---- final save ----
    for group in groups:
        final_dir = SAE_WEIGHTS_DIR / group.name / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        for i, sae in enumerate(group.saes):
            torch.save(sae.state_dict(), final_dir / f"layer_{i}.pt")

        with open(final_dir / "log.json", "w") as f:
            json.dump(group.log, f)

    total_time = time.time() - start_time
    print(f"done in {total_time/3600:.2f}h")
    print_vram("final")


if __name__ == "__main__":
    main(device=DEVICE, dtype=SAE_DTYPE)
