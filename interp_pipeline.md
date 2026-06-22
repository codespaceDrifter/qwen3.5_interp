# random gpu mon note  
while true; do nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv,noheader; sleep 1; done



# file order


Run these from the repo root in order:

1. `python3 -m interp_pipeline.tokenize_interp_mix`
2. `python3 -m interp_pipeline.inspect_tokenized`
3. `python3 -m interp_pipeline.sae_train`
4. `sae_label` (to be added)



# interp pipeline

Designing software on a hardware level means thinking about where data lives and how it moves.

Rough tiers:
- SSD ↔ RAM
- RAM ↔ VRAM
- VRAM ↔ vregister
- vregister ↔ vregister (actual compute)
- RAM ↔ register
- register ↔ register (CPU compute)

Caches are kernel-managed by PyTorch, so we ignore them in planning.

1 CPU cycle ≈ 0.25 ns (~4 GHz). 1 GPU cycle ≈ 0.5 ns (~2 GHz, 5090 SM clock).

**speed (cycles per operation)**

| operation | cycles |
|---|---|
| register ↔ register (CPU add/mul) | ~4 CPU |
| RAM ↔ register (read one value) | ~200 CPU |
| vregister ↔ vregister (GPU FMA) | ~4 GPU |
| VRAM ↔ vregister (read one value) | ~400 GPU |
| RAM ↔ VRAM (start a transfer) | ~40,000 CPU, then 30 GB/s |
| SSD ↔ RAM (start a read) | ~200,000 CPU, then 7 GB/s |

**space**

| tier | size |
|---|---|
| CPU register | ~1 KB per core |
| RAM | 32–128 GB |
| SSD | TBs (14 TB planned) |
| GPU vregister | ~40 MB total on 5090 |
| VRAM | 32 GB on 5090 |

Parallelism on the GPU is bound by cores; estimate via FLOPs. For decision making, just fit as many batches as possible capped by VRAM and RAM.



# 1: data

We use the same dvtasteps interpretability mix as the gemma4 repo, just re-tokenized with the Qwen3.5 tokenizer.

Target coverage: ~4B tokens.

- 1024-token chunks
- ~16 GB of int32 `.bin` once tokenized
- Source text ~20 GB

The exact mix is randomly sampled from these datasets (sizes from dvtasteps):

- 2 GB all the news
- 2 GB reddit mix
- 1 GB bookcorpus
- 5 GB English Wikipedia
- 3 GB OpenWebText
- 2 GB OpenWebMath
- 1 GB StarCoder Python
- 1 GB StarCoder C++
- 1 GB StarCoder JavaScript
- 500 MB StarCoder HTML
- 500 MB StarCoder CSS

After sampling and tokenizing, expected:
- ~5B tokens across ~30M docs

Paths are configured in `interp_pipeline/config.py`:
- input: `/media/drift/Extreme SSD/DATA/interpMix/*.arrow`
- output: `/media/drift/Extreme SSD/qwen3.5_interp_data/tokenized_interpmix/`



# 2: activations

Storing all activations to SSD would be enormous, so we stream them.

We want activations at every residual-contribution point:
- `residue` — full layer output
- `post_attn_norm` — full-attention subblock output
- `post_linear_attn_norm` — linear-attention (Gated DeltaNet) subblock output
- `post_mlp_norm` — MLP subblock output

We run the model over the data once and use hooks in our implementation to stream activations. The SAE training script offloads captured activations from VRAM to RAM immediately in the hooks, so VRAM does not accumulate 32 layer outputs. Only one layer's activation is moved back to GPU at a time for training.



# 3: probes

We train one epoch over the whole dataset to avoid overfitting.

Qwen3.5-4B text model shape:
- 32 hidden layers
- residual stream dim `D = 2560`
- MLP intermediate dim = 9216
- ~4B total parameters

A 16× expansion SAE uses a feature dim of `16 × 2560 = 40,960`.

**Batch-size calculation**

We want the largest batch that saturates memory/compute bandwidth without OOM. With the streaming setup in `sae_train.py`:

- Model (bf16): ~4B params → ~8 GB
- 32 SAEs (bf16 params): 32 × 210M params ≈ 13.4 GB
- One SAE's gradients (bf16): ~0.4 GB
- One SAE's 8-bit optimizer state: ~0.4 GB
- One batch activation on GPU: `B × 1024 × 2560 × 2 bytes` = `B × 5.24 MB`
- Forward intermediates: ~2 GB headroom

Total ≈ `8 + 13.4 + 0.4 + 0.4 + 2 + B × 0.0052` GB.

For a 32 GB card, solving for B:

```
24.2 + B × 0.0052 < 32
B < ~1500
```

So `B = 128` is conservative and leaves ~6 GB of slack for fragmentation. If you have less VRAM or no `bitsandbytes`, lower `B` to 64 or 32.

**Why not train all SAE optimizers at once?**

Keeping 32 full optimizers on GPU would require ~3× the SAE parameter memory just for optimizer states/gradients, pushing the footprint past 32 GB. The script therefore keeps all SAEs on GPU but swaps one optimizer at a time: after training layer `i`, its optimizer state dict is moved to CPU RAM and the GPU optimizer object is freed. This lets us do a single forward pass per batch while fitting on one 32 GB card.

**Training config (defaults in `sae_train.py`)**

```python
EXPANSION_FACTOR = 16
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
TARGET_ACTIVE = 64
SPARSITY_COEFF_INIT = 1e-3
BAND_EPS = 0.001
```

Run one capture type at a time (`residue`, `post_attn_norm`, `post_linear_attn_norm`, `post_mlp_norm`) by editing `CAPTURE_TYPE`.

Train on both the sublayer contributions and the residue, then inspect whether the residue is well approximated by the sum of its sublayer SAEs on specific examples.



# 4: labeling

To be added: automated feature labeling (likely via API calls) and a labeling UI/workflow.
