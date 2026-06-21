# Sample and decode random chunks from the tokenized interpmix bin.
# Usage from repo root:
#   python3 -m interp_pipeline.inspect_tokenized

import numpy as np
from tokenizers import Tokenizer

from interp_pipeline.config import CHUNK_SIZE, DEFAULT_TOKENIZER_PATH, TOKENIZED_DIR


def main():
    tok = Tokenizer.from_file(str(DEFAULT_TOKENIZER_PATH))
    bin_path = TOKENIZED_DIR / "tokenized_interpmix.bin"

    tokens = np.fromfile(bin_path, dtype="int32")
    total_chunks = len(tokens) // CHUNK_SIZE

    print(f"total tokens: {len(tokens):,}")
    print(f"total chunks: {total_chunks:,}\n")

    num_samples = 5
    for i in range(num_samples):
        idx = np.random.randint(0, total_chunks) * CHUNK_SIZE
        chunk = tokens[idx : idx + CHUNK_SIZE]
        text = tok.decode(chunk.tolist())
        print(f"=== sample {i + 1} (chunk {idx // CHUNK_SIZE}) ===")
        print(text[:1200])
        print("...\n")


if __name__ == "__main__":
    main()
