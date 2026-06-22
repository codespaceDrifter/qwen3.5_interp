# Shuffle an already-tokenized .bin file at the chunk level.
# Reads CHUNK_SIZE-token chunks, randomly permutes them, and writes back.
# Does NOT break chunks.
#
# Usage from repo root:
#   python3 -m interp_pipeline.shuffle_tokenized
#
# Optional: set seed for reproducibility
#   python3 -m interp_pipeline.shuffle_tokenized --seed 123

import argparse
from pathlib import Path

import numpy as np

from interp_pipeline.config import CHUNK_SIZE, TOKENIZED_DIR, TOKEN_DTYPE


def shuffle_bin(in_path: Path, out_path: Path | None = None, seed: int | None = None):
    """Shuffle chunks in a tokenized .bin file.

    If out_path is None, overwrites in_path in place via a temporary file.
    """
    in_path = Path(in_path)
    if out_path is None:
        out_path = in_path.with_suffix(".shuffled.tmp")
    else:
        out_path = Path(out_path)

    file_bytes = in_path.stat().st_size
    bytes_per_chunk = CHUNK_SIZE * np.dtype(TOKEN_DTYPE).itemsize
    num_chunks = file_bytes // bytes_per_chunk

    if num_chunks == 0:
        raise ValueError(f"no complete chunks in {in_path}")

    print(f"shuffling {num_chunks:,} chunks from {in_path}")
    print(f"  file size: {file_bytes / 1e9:.2f} GB")
    print(f"  seed: {seed}")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_chunks)

    # memory-efficient: random read via memmap, sequential write to output
    src = np.memmap(in_path, dtype=TOKEN_DTYPE, mode="r").reshape(num_chunks, CHUNK_SIZE)
    dst = np.memmap(out_path, dtype=TOKEN_DTYPE, mode="w+", shape=(num_chunks, CHUNK_SIZE))

    batch_size = 4096  # write in batches to reduce Python loop overhead
    written = 0
    for start in range(0, num_chunks, batch_size):
        end = min(start + batch_size, num_chunks)
        batch_perm = perm[start:end]
        dst[start:end] = src[batch_perm]
        written += (end - start)
        if written % (batch_size * 10) == 0 or written == num_chunks:
            print(f"  {written:,} / {num_chunks:,} chunks written")

    dst.flush()
    del dst
    del src

    if out_path != in_path:
        # atomic-ish replace
        tmp_backup = in_path.with_suffix(".bak")
        in_path.rename(tmp_backup)
        out_path.rename(in_path)
        tmp_backup.unlink()
        print(f"overwrote {in_path}")

    print("done")


def main():
    parser = argparse.ArgumentParser(description="Shuffle a tokenized .bin file by chunks.")
    parser.add_argument("--input", type=Path, default=TOKENIZED_DIR / "tokenized_interpmix.bin")
    parser.add_argument("--output", type=Path, default=None, help="optional output path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    shuffle_bin(args.input, args.output, args.seed)


if __name__ == "__main__":
    main()
