# Load weights/model.safetensors into Qwen3_5ForCausalLM.
#
# The Qwen3.5-4B text checkpoint stores text decoder keys as
#     model.language_model.embed_tokens.weight
#     model.language_model.layers.0.self_attn.q_proj.weight
#     ...
# which matches our nn.Module key layout exactly.  This loader also tolerates an
# optional leading "model." prefix and skips vision keys / a separate lm_head
# when word embeddings are tied.
#
# Supports both a single .safetensors file and a sharded checkpoint directory
# containing model.safetensors.index.json.

import json
from pathlib import Path

import torch

try:
    from safetensors import safe_open

    _HAS_SAFETENSORS = True
except Exception:  # pragma: no cover
    _HAS_SAFETENSORS = False

from qwen3_5_4b_implementation.model import Qwen3_5ForCausalLM


_VISION_PREFIXES = (
    "model.visual.",
    "visual.",
    "vision_tower.",
    "model.vision_tower.",
)


def _is_vision_key(key: str) -> bool:
    return key.startswith(_VISION_PREFIXES)


def _resolve_checkpoint_paths(safetensors_path: str | Path) -> list[Path]:
    """Return the list of .safetensors shard files to load.

    Accepts either a single file, a directory with an index, or a non-existent
    single-file path whose parent directory contains a sharded checkpoint.
    """
    path = Path(safetensors_path)

    if path.is_file() and path.suffix == ".safetensors":
        return [path]

    directory = path if path.is_dir() else path.parent
    index_file = directory / "model.safetensors.index.json"

    if index_file.exists():
        with open(index_file) as f:
            weight_map = json.load(f).get("weight_map", {})
        shard_names = sorted(set(weight_map.values()))
        shards = [directory / name for name in shard_names]
        missing = [s for s in shards if not s.exists()]
        if missing:
            raise FileNotFoundError(
                f"shard files referenced by {index_file} are missing: {missing}"
            )
        return shards

    if directory.exists():
        shards = sorted(directory.glob("*.safetensors"))
        if shards:
            return shards

    raise FileNotFoundError(
        f"no safetensors checkpoint found at {safetensors_path!r}"
    )


def _remap_key(hf_key: str, model_keys: set[str], tie_word_embeddings: bool) -> str | None:
    # Drop vision branch.
    if _is_vision_key(hf_key):
        return None

    # With tied embeddings the lm_head is shared with embed_tokens; a separate
    # lm_head.weight in the checkpoint would overwrite the tie, so ignore it.
    if tie_word_embeddings and hf_key.startswith("lm_head."):
        return None

    # Prefer exact match, then a key with one leading "model." stripped,
    # then the multimodal-style "model.language_model.*" prefix added.
    candidates = [hf_key]
    if hf_key.startswith("model."):
        candidates.append(hf_key[len("model.") :])
        if not hf_key.startswith("model.language_model."):
            candidates.append("model.language_model." + hf_key[len("model.") :])

    for candidate in candidates:
        if candidate in model_keys:
            return candidate

    # No recognised mapping — let load_state_dict report it as unexpected.
    return hf_key


def load_weights(
    model: Qwen3_5ForCausalLM,
    safetensors_path: str,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
    strict: bool = True,
):
    """Load a single safetensors checkpoint into ``model`` in-place."""
    if not _HAS_SAFETENSORS:
        raise ImportError(
            "the `safetensors` package is required to load checkpoint weights"
        )

    model_keys = set(model.state_dict().keys())
    tie_word_embeddings = getattr(model.config, "tie_word_embeddings", False)

    state_dict: dict[str, torch.Tensor] = {}
    our_to_hf: dict[str, str] = {}

    shard_paths = _resolve_checkpoint_paths(safetensors_path)
    print(f"loading checkpoint from {len(shard_paths)} shard(s)...")
    for shard_path in shard_paths:
        print(f"  opening {shard_path}...")
        with safe_open(str(shard_path), framework="pt", device=str(device)) as f:
            hf_keys = list(f.keys())
            print(f"    {len(hf_keys)} tensors")
            for hf_key in hf_keys:
                our_key = _remap_key(hf_key, model_keys, tie_word_embeddings)
                if our_key is None:
                    continue
                tensor = f.get_tensor(hf_key)
                if dtype is not None:
                    tensor = tensor.to(dtype)
                state_dict[our_key] = tensor
                our_to_hf[our_key] = hf_key

    missing = model_keys - set(state_dict.keys())
    unexpected = set(state_dict.keys()) - model_keys

    if missing:
        print(f"\n[missing in checkpoint, kept at init: {len(missing)} keys]")
        for k in sorted(missing)[:20]:
            print(f"  {k}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")

    if unexpected:
        print(f"\n[in checkpoint but not in model: {len(unexpected)} keys]")
        for k in sorted(unexpected)[:20]:
            print(f"  {k}")
        if len(unexpected) > 20:
            print(f"  ... and {len(unexpected) - 20} more")

    if strict and missing:
        # The tied lm_head is intentionally not loaded from a separate checkpoint key.
        strict_missing = set(missing)
        if tie_word_embeddings:
            strict_missing.discard("lm_head.weight")
        if strict_missing:
            raise RuntimeError(
                f"strict load failed — {len(strict_missing)} model params have no checkpoint weight:\n"
                + "\n".join(sorted(strict_missing)[:30])
            )

    # assign=True keeps the checkpoint dtype/device without an extra copy.
    # It breaks tied embeddings, so restore the tie afterwards.
    model.load_state_dict(state_dict, strict=False, assign=True)
    if tie_word_embeddings and hasattr(model, "tie_weights"):
        model.tie_weights()
    print(f"\nloaded {len(state_dict)} tensors into model.")
    return model
