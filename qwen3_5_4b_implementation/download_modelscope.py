import argparse
from pathlib import Path

try:
    from modelscope import snapshot_download
    _HAS_MODELSCOPE = True
except Exception:
    _HAS_MODELSCOPE = False

MODEL_ID = "qwen/Qwen3.5-4B"
WEIGHTS_DIR = Path("weights")


def main():
    parser = argparse.ArgumentParser(
        description="Download Qwen3.5-4B weights from ModelScope (China-friendly mirror of HF)."
    )
    parser.add_argument(
        "--repo-id", default=MODEL_ID, help="ModelScope model identifier"
    )
    parser.add_argument(
        "--weights-dir", default=str(WEIGHTS_DIR), help="local output directory"
    )
    args = parser.parse_args()

    if not _HAS_MODELSCOPE:
        raise ImportError(
            "the `modelscope` package is required.\n"
            "install it with: pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple"
        )

    weights_dir = Path(args.weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)

    print(f"[get ] {args.repo_id} -> {weights_dir}")
    print("[info] This downloads the exact same safetensors shards + tokenizer files as HF.")

    snapshot_download(
        args.repo_id,
        cache_dir=str(weights_dir),
        local_dir=str(weights_dir)
    )
    print("done.")


if __name__ == "__main__":
    main()
