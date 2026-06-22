# Quick inspect: pick a random starting chunk and print 10 consecutive chunks
# to visually verify the .bin file is shuffled.
#
# Usage:
#   python3 -m interp_pipeline.inspect_tokenized

import argparse
import random
from pathlib import Path

import numpy as np

from interp_pipeline.config import CHUNK_SIZE, DEFAULT_TOKENIZER_PATH, TOKENIZED_DIR, TOKEN_DTYPE
from interp_pipeline.tokenize_interp_mix import Qwen3_5Tokenizer


def inspect(bin_path: Path, start_chunk: int | None = None):
    tok = Qwen3_5Tokenizer(DEFAULT_TOKENIZER_PATH)
    flat = np.fromfile(bin_path, dtype=TOKEN_DTYPE)
    num_chunks = len(flat) // CHUNK_SIZE

    if start_chunk is None:
        start_chunk = random.randint(0, max(0, num_chunks - 10))

    print(f"total chunks: {num_chunks:,}")
    print(f"showing chunks {start_chunk} to {start_chunk + 9}\n")

    for offset in range(10):
        idx = start_chunk + offset
        if idx >= num_chunks:
            break
        chunk = flat[idx * CHUNK_SIZE:(idx + 1) * CHUNK_SIZE]
        text = tok.tok.decode(chunk.tolist())
        print(f"--- chunk {idx} ---")
        print(text[:500])
        print(f"[length: {len(text)} chars, first 10 tokens: {chunk[:10].tolist()}]\n")


def main():
    parser = argparse.ArgumentParser(description="Inspect consecutive chunks of tokenized data.")
    parser.add_argument("--bin", type=Path, default=TOKENIZED_DIR / "tokenized_interpmix.bin")
    parser.add_argument("--start-chunk", type=int, default=None, help="starting chunk index")
    args = parser.parse_args()

    inspect(args.bin, args.start_chunk)


if __name__ == "__main__":
    main()
