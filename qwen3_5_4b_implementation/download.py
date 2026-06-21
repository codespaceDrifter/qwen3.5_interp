# Pull Qwen3.5-4B weights + tokenizer files from HuggingFace.
# usage from repo root:  python3 -m qwen3_5_4b_implementation.download

import argparse
from pathlib import Path

try:
    from huggingface_hub import snapshot_download

    _HAS_HF_HUB = True
except Exception:  # pragma: no cover
    _HAS_HF_HUB = False


MODEL_ID = "Qwen/Qwen3.5-4B"
WEIGHTS_DIR = Path("weights")

# Only files the from-scratch text implementation needs.  Sharded models are
# handled automatically via model.safetensors.index.json.
ALLOW_PATTERNS = [
    "*.safetensors",
    "*.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "config.json",
    "chat_template.jinja",
]


def main():
    parser = argparse.ArgumentParser(
        description="Download Qwen3.5-4B weights and tokenizer files from HuggingFace."
    )
    parser.add_argument(
        "--repo-id", default=MODEL_ID, help="HuggingFace model identifier"
    )
    parser.add_argument(
        "--weights-dir", default=str(WEIGHTS_DIR), help="local output directory"
    )
    args = parser.parse_args()

    if not _HAS_HF_HUB:
        raise ImportError(
            "the `huggingface_hub` package is required to download weights"
        )

    weights_dir = Path(args.weights_dir)
    weights_dir.mkdir(exist_ok=True)
    print(f"[get ] {args.repo_id} -> {weights_dir}")
    snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(weights_dir),
        allow_patterns=ALLOW_PATTERNS,
    )
    print("done.")


if __name__ == "__main__":
    main()
