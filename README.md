# qwen3.5 interp

Master repo for mechanistic-interpretability work on **Qwen3.5-4B**.

The model is implemented from scratch in `qwen3_5_4b_implementation/` (no runtime dependency on `transformers`, `causal_conv1d`, or `flash_linear_attention`), then we run interp pipelines on top of it. This mirrors the structure of the old gemma4 repo but targets a dense hybrid-attention architecture instead of the Per-Layer Embedding mess.


## repo layout

```
qwen3_5_4b_implementation/   from-scratch model, loader, download, run, chat, HF verifier
probes/                      SAE implementation (copied as-is)
interp_pipeline/             tokenize data, inspect tokens, train SAEs
weights/                     downloaded Qwen3.5-4B checkpoints (gitignored)
```

No `embedding_visual/` — skipped entirely per request.


## getting the model

```bash
python3 -m qwen3_5_4b_implementation.download
```

This pulls `Qwen/Qwen3.5-4B` into `weights/`:
- `model.safetensors`
- `tokenizer.json`
- `tokenizer_config.json`
- `config.json`


## quick smoke tests

```bash
# verify the from-scratch implementation matches HF on a shared prompt
python3 -m qwen3_5_4b_implementation.verify_diff_hf

# interactive chat
python3 -m qwen3_5_4b_implementation.chat
```

All commands assume you run them from the repo root (`/home/drift/Desktop/qwen3.5_interp`).


## interp pipeline file order

1. `python3 -m interp_pipeline.tokenize_interp_mix` — tokenize the dvtasteps interpretability mix.
2. `python3 -m interp_pipeline.inspect_tokenized` — sanity-check random chunks.
3. `python3 -m interp_pipeline.sae_train` — train SAEs on the capture point you set.

See `interp_pipeline.md` for the hardware/memory design and `qwen_architecture.md` for where to hook.


## methods we will implement

### SAEs
Train sparse autoencoders on the residual stream and on every sublayer contribution to it (full-attention output, linear-attention output, MLP output). Hypersearch sparsity and expansion, then label features.

### WCCs
Same locations as the SAEs but with the WCC algorithm from MINDOLOGY.

### Attention labeling
QK and OV feature-pair interpretation.

### MLP neurons
Interpret individual hidden-dim neurons by their encoding context and their decoding projection onto the word embedding basis.

### MLP feature tables
Residue→MLP-out feature tables using SAEs or WCCs.


## tech stack

- PyTorch for inference and weight loading
- `safetensors`, `huggingface_hub`, `tokenizers`
- `pyarrow` for reading the `.arrow` interp mix
- FastAPI + Uvicorn if/when we add a web backend
- Pure HTML/CSS/JS frontend if/when we add a web UI
