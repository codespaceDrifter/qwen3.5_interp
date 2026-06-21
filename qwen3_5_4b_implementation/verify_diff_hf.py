# Side-by-side comparison: our Qwen3.5-4B vs HuggingFace transformers.
# Loads each model separately to keep GPU memory bounded, records decoder-layer
# residual streams and a short greedy generation, then compares from CPU tensors.
#
# usage from repo root:  python3 -m qwen3_5_4b_implementation.verify_diff_hf
# Requires the downloaded weights and the `transformers` package to run the HF
# half; the module itself is import-safe so it can be inspected without it.
#
# run with cpu if not enough vram
# python3 -m qwen3_5_4b_implementation.verify_diff_hf --device cpu --dtype float32

import argparse
import gc
import json
from pathlib import Path

import torch

try:
    from safetensors import safe_open

    _HAS_SAFETENSORS = True
except Exception:  # pragma: no cover
    _HAS_SAFETENSORS = False

try:
    from transformers import AutoConfig, AutoModelForCausalLM

    _HAS_TRANSFORMERS = True
except Exception:  # pragma: no cover
    _HAS_TRANSFORMERS = False

try:
    from tokenizers import Tokenizer

    _HAS_TOKENIZERS = True
except Exception:  # pragma: no cover
    _HAS_TOKENIZERS = False

from qwen3_5_4b_implementation.config import Qwen3_5Config
from qwen3_5_4b_implementation.model import Qwen3_5ForCausalLM, Qwen3_5Cache
from qwen3_5_4b_implementation.loader import (
    load_weights as my_load_weights,
    _resolve_checkpoint_paths,
)


WEIGHTS_PATH = "weights"
TOKENIZER_PATH = "weights/tokenizer.json"
CONFIG_PATH = "weights/config.json"
PROMPT = "hi qwen"
N_GEN = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16


_VISION_PREFIXES = (
    "model.visual.",
    "visual.",
    "vision_tower.",
    "model.vision_tower.",
)


def _is_vision_key(k: str) -> bool:
    return k.startswith(_VISION_PREFIXES)


def _remap_for_hf(k: str) -> str | None:
    """Map our multimodal-style checkpoint keys to HF's text-only layout."""
    if _is_vision_key(k):
        return None
    # HF expects text keys under `model.*`, not `model.language_model.*`.
    if k.startswith("model.language_model."):
        return "model." + k[len("model.language_model.") :]
    # Embeddings are tied; a separate lm_head.weight would break the tie.
    if k.startswith("lm_head."):
        return None
    return k


def _load_into_hf(model: "AutoModelForCausalLM") -> None:  # type: ignore[name-defined]
    sd: dict[str, torch.Tensor] = {}
    for shard_path in _resolve_checkpoint_paths(WEIGHTS_PATH):
        print(f"  loading shard {shard_path.name}...")
        with safe_open(str(shard_path), framework="pt", device=str(DEVICE)) as f:
            for k in f.keys():
                nk = _remap_for_hf(k)
                if nk is None:
                    continue
                sd[nk] = f.get_tensor(k).to(DTYPE)
    model.load_state_dict(sd, strict=False, assign=True)
    # assign=True breaks the embed_tokens/lm_head tie; restore it.
    model.tie_weights()


class _TokenizerWrapper:
    def __init__(self, path: str):
        self.tok = Tokenizer.from_file(str(path))

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text).ids

    def decode(self, ids: list[int] | torch.Tensor, skip_special: bool = False) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self.tok.decode(ids, skip_special_tokens=skip_special)

    def token_to_id(self, token: str) -> int | None:
        return self.tok.token_to_id(token)


def _make_input_ids() -> tuple[torch.Tensor, "_TokenizerWrapper"]:
    tk = _TokenizerWrapper(TOKENIZER_PATH)
    ids = tk.encode(PROMPT)
    return torch.tensor([ids], device=DEVICE, dtype=torch.long), tk


def _build_config() -> Qwen3_5Config:
    cfg = Qwen3_5Config()
    if Path(CONFIG_PATH).exists():
        with open(CONFIG_PATH) as f:
            data = json.load(f)
        text_cfg = data.get("text_config", data)
        if isinstance(text_cfg, dict):
            for k, v in text_cfg.items():
                if hasattr(cfg.text_config, k):
                    setattr(cfg.text_config, k, v)
        cfg.text_config.__post_init__()
    return cfg


def run_mine(input_ids: torch.Tensor, tk: "_TokenizerWrapper"):
    print("\n========== MY MODEL ==========")
    print("building skeleton...")
    my_cfg = _build_config()
    model = Qwen3_5ForCausalLM(my_cfg).eval()
    my_load_weights(model, WEIGHTS_PATH, device=DEVICE, dtype=DTYPE, strict=True)
    model.to(DEVICE)

    # Record residual stream after every decoder layer.
    residuals: list[torch.Tensor] = []
    handles = []
    for layer in model.model.language_model.layers:
        h = layer.register_forward_hook(
            lambda mod, inp, out: residuals.append(out.detach().float().cpu())
        )
        handles.append(h)

    cache = Qwen3_5Cache(my_cfg.text_config.num_hidden_layers)
    with torch.no_grad():
        logits = model(input_ids=input_ids, use_cache=True, past_key_values=cache)["logits"]
    for h in handles:
        h.remove()

    # Greedy generate N_GEN tokens.
    gen_ids: list[int] = []
    next_id = logits[:, -1, :].argmax(-1)
    gen_ids.append(int(next_id.item()))
    for _ in range(N_GEN - 1):
        cur = next_id.unsqueeze(0)
        with torch.no_grad():
            logits = model(input_ids=cur, use_cache=True, past_key_values=cache)["logits"]
        next_id = logits[:, -1, :].argmax(-1)
        gen_ids.append(int(next_id.item()))

    print(f"generated tokens: {gen_ids}")
    print(f"decoded:          {tk.decode(gen_ids, skip_special=False)!r}")

    del model, cache, logits, next_id
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return residuals, gen_ids


def run_hf(input_ids: torch.Tensor, tk: "_TokenizerWrapper"):
    print("\n========== HF MODEL ==========")
    if not _HAS_TRANSFORMERS:
        raise ImportError(
            "the `transformers` package is required to run the HF comparison"
        )

    print("building skeleton...")
    hf_full_cfg = AutoConfig.from_pretrained("weights", trust_remote_code=True)
    hf_text_cfg = getattr(hf_full_cfg, "text_config", hf_full_cfg)
    model = AutoModelForCausalLM.from_config(
        hf_text_cfg, trust_remote_code=True, attn_implementation="eager"
    ).eval()
    print("loading weights...")
    _load_into_hf(model)
    model.to(device=DEVICE, dtype=DTYPE)

    residuals: list[torch.Tensor] = []
    handles = []
    for layer in model.model.layers:
        h = layer.register_forward_hook(
            lambda mod, inp, out: residuals.append(
                (out[0] if isinstance(out, tuple) else out).detach().float().cpu()
            )
        )
        handles.append(h)

    with torch.no_grad():
        _ = model(input_ids=input_ids)
    for h in handles:
        h.remove()

    gen_out = model.generate(
        input_ids=input_ids,
        max_new_tokens=N_GEN,
        do_sample=False,
        use_cache=True,
    )
    gen_ids = gen_out[0, input_ids.shape[1] :].tolist()
    print(f"generated tokens: {gen_ids}")
    print(f"decoded:          {tk.decode(gen_ids, skip_special=False)!r}")

    del model, gen_out
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return residuals, gen_ids


def _weights_exist() -> bool:
    p = Path(WEIGHTS_PATH)
    if p.exists():
        return True
    directory = p.parent
    return (directory / "model.safetensors.index.json").exists() or any(
        directory.glob("*.safetensors")
    )


def main():
    if not _weights_exist():
        print(
            f"weights not found at {WEIGHTS_PATH!r}. "
            "Run: python3 -m qwen3_5_4b_implementation.download"
        )
        return
    if not _HAS_SAFETENSORS:
        print("the `safetensors` package is required to load checkpoint weights.")
        return
    if not _HAS_TOKENIZERS:
        print("the `tokenizers` package is required to tokenize the prompt.")
        return

    input_ids, tk = _make_input_ids()
    print(f"prompt: {PROMPT!r}")
    print(f"input_ids ({input_ids.shape}): {input_ids.tolist()}")

    my_resids, my_tokens = run_mine(input_ids, tk)

    if not _HAS_TRANSFORMERS:
        print("\n[skip] `transformers` not installed; cannot run HF comparison.")
        print(f"MY generated: {my_tokens}")
        return

    hf_resids, hf_tokens = run_hf(input_ids, tk)

    print("\n========== COMPARISON ==========")
    print(f"\nMY  generated: {my_tokens}")
    print(f"HF  generated: {hf_tokens}")
    print(f"tokens match:  {my_tokens == hf_tokens}")

    print(
        f"\nresidual stream per-layer (prompt prefill, T={input_ids.shape[1]}):"
    )
    print(
        f"  {'layer':<6} {'mean diff':<14} {'max diff':<14} "
        f"{'mine norm':<14} {'hf norm':<14}"
    )
    assert len(my_resids) == len(hf_resids)
    for i, (m, h) in enumerate(zip(my_resids, hf_resids)):
        diff = (m - h).abs()
        print(
            f"  {i:<6} {diff.mean().item():<14.4e} "
            f"{diff.max().item():<14.4e} {m.norm().item():<14.4e} "
            f"{h.norm().item():<14.4e}"
        )


def _dtype_from_string(s: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(s, torch.bfloat16)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify the from-scratch Qwen3.5-4B against HuggingFace transformers."
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="device to run on (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="dtype for inference (default: bfloat16; float32 recommended for CPU)",
    )
    args = parser.parse_args()

    DEVICE = torch.device(args.device)
    DTYPE = _dtype_from_string(args.dtype)
    main()
