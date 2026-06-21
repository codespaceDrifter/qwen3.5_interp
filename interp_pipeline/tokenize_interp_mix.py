# Tokenize the dvtasteps interpretability mix with the Qwen3.5 tokenizer.
# Reads DATA/interpMix/*.arrow from the external SSD and writes one flat
# int32 .bin file of 1024-token chunks.
#
# Single-pass: tokenizes each row once, writes full chunks as soon as they are
# ready, and keeps only a small rolling remainder buffer in memory.
#
# Usage from repo root:
#   python3 -m interp_pipeline.tokenize_interp_mix

import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
from tokenizers import Tokenizer

from interp_pipeline.config import (
    CHUNK_SIZE,
    DEFAULT_TOKENIZER_PATH,
    INTERP_MIX_DIR,
    TOKENIZED_DIR,
    TOKEN_DTYPE,
)


class Qwen3_5Tokenizer:
    """Thin wrapper over the Qwen3.5 tokenizer.json (no transformers dependency)."""

    def __init__(self, tokenizer_json_path: str | Path):
        self.tok = Tokenizer.from_file(str(tokenizer_json_path))

    def encode(self, text: str) -> list[int]:
        # encode raw text as-is; newlines and whitespace are preserved
        return self.tok.encode(text, add_special_tokens=False).ids


def load_arrow_texts(path: Path):
    """Yield text strings from an Arrow file (IPC file format), batch by batch."""
    with pa.ipc.open_file(str(path)) as reader:
        if "text" not in reader.schema.names:
            raise ValueError(
                f"no 'text' column in {path}; columns: {reader.schema.names}"
            )

        for i in range(reader.num_record_batches):
            batch = reader.get_batch(i)
            col = batch.column("text")
            for j in range(batch.num_rows):
                yield col[j].as_py()


def main():
    # set this env var if you want to reduce CPU thread usage / heat:
    #   export TOKENIZERS_PARALLELISM=false
    # by default the tokenizers Rust backend uses all cores.

    if not INTERP_MIX_DIR.exists():
        raise FileNotFoundError(f"interp mix dir not found: {INTERP_MIX_DIR}")

    tokenizer_path = Path(DEFAULT_TOKENIZER_PATH)
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"tokenizer not found at {tokenizer_path}; run from repo root"
        )

    TOKENIZED_DIR.mkdir(parents=True, exist_ok=True)

    tok = Qwen3_5Tokenizer(tokenizer_path)
    print(f"tokenizer: {tokenizer_path}")
    print(f"vocab size: {tok.tok.get_vocab_size()}")
    print(f"chunk size: {CHUNK_SIZE}")
    print(f"dtype: {TOKEN_DTYPE}")
    print()

    arrow_files = sorted(INTERP_MIX_DIR.glob("*.arrow"))
    if not arrow_files:
        raise FileNotFoundError(f"no .arrow files found in {INTERP_MIX_DIR}")

    print(f"found {len(arrow_files)} arrow files:")
    for f in arrow_files:
        print(f"  {f.name}")
    print()

    out_bin = TOKENIZED_DIR / "tokenized_interpmix.bin"
    print(f"writing {out_bin}")

    buffer: list[int] = []
    total_docs = 0
    total_tokens = 0
    full_chunks = 0
    manifest_sources = []

    with open(out_bin, "wb") as f:
        for arrow_path in arrow_files:
            print(f"tokenizing {arrow_path.name}...")
            doc_count = 0
            token_count = 0

            for text in load_arrow_texts(arrow_path):
                if not isinstance(text, str):
                    continue
                ids = tok.encode(text)
                buffer.extend(ids)
                token_count += len(ids)
                doc_count += 1
                total_docs += 1
                total_tokens += len(ids)

                # flush every full chunk
                while len(buffer) >= CHUNK_SIZE:
                    chunk = np.array(buffer[:CHUNK_SIZE], dtype=TOKEN_DTYPE)
                    chunk.tofile(f)
                    buffer = buffer[CHUNK_SIZE:]
                    full_chunks += 1

                    if full_chunks % 10000 == 0:
                        print(f"  {full_chunks:,} chunks written")

            manifest_sources.append({
                "file": arrow_path.name,
                "docs": doc_count,
                "tokens": token_count,
            })
            print(f"  {doc_count:,} docs, {token_count:,} tokens")

    dropped_tokens = len(buffer)

    print(f"\n=== DONE ===")
    print(f"  total docs: {total_docs:,}")
    print(f"  total tokens: {total_tokens:,}")
    print(f"  full chunks: {full_chunks:,}")
    print(f"  dropped remainder: {dropped_tokens:,} tokens")

    manifest = {
        "tokenizer": str(tokenizer_path),
        "chunk_size": CHUNK_SIZE,
        "dtype": TOKEN_DTYPE,
        "total_docs": total_docs,
        "total_tokens": total_tokens,
        "full_chunks": full_chunks,
        "dropped_tokens": dropped_tokens,
        "sources": manifest_sources,
    }

    manifest_path = TOKENIZED_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    size_gb = out_bin.stat().st_size / 1e9
    print(f"bin: {out_bin} ({size_gb:.2f} GB)")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
