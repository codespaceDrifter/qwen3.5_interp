# config for qwen3.5_interp interpretability pipeline
# mirrors dvtasteps/scripts/config.py so we read from the same external SSD when it is present,
# otherwise falls back to a local `data/` directory inside the repo.

from pathlib import Path

import torch


# Try the external SSD first; fall back to a local `data/` directory inside the repo.
# `Path("data")` resolves to `<repo_root>/data/` when the SSD is not plugged in.
_EXTERNAL_SSD = Path("/media/drift/Extreme SSD")
if _EXTERNAL_SSD.exists():
    DATA_ROOT = Path(f"{_EXTERNAL_SSD}/qwen3.5_interp_data")
    INTERP_MIX_DIR = Path(f"{_EXTERNAL_SSD}/DATA/interpMix")
else:
    DATA_ROOT = Path("data")
    INTERP_MIX_DIR = DATA_ROOT / "interpMix"

# where tokenized outputs go
TOKENIZED_DIR = DATA_ROOT / "tokenized_interpmix"

# where trained SAE weights and checkpoints go.
SAE_WEIGHTS_DIR = DATA_ROOT / "sae_weights"

# qwen3.5 tokenizer bundled in this repo
DEFAULT_TOKENIZER_PATH = Path("weights/tokenizer.json")

CHUNK_SIZE = 1024
TOKEN_DTYPE = "int32"

# =========================================================================================
# SAE training config
# =========================================================================================

# which subblock output / residual stream point to train on
# these are all the tensors that get ADDED INTO or CARRIED BY the residual stream:
#   "residue"              = full decoder layer output (after both residual adds)
#   "post_attn_norm"       = full-attention subblock output, before the first residual add
#   "post_linear_attn_norm"= linear-attention (Gated DeltaNet) subblock output, before residual add
#   "post_mlp_norm"        = MLP subblock output, before the second residual add
CAPTURE_TYPE = "post_attn_norm"

EXPANSION_FACTOR = 16
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
TARGET_ACTIVE = 64
SPARSITY_COEFF_INIT = 1e-3
BAND_EPS = 0.001

CHECKPOINT_EVERY = 10000   # chunks
LOG_EVERY = 100            # chunks

# On GPUs with enough VRAM (e.g. RTX Pro 6000 96 GB), keep all 32 optimizers on GPU.
# On 32 GB cards, set this True to swap only one optimizer to GPU at a time.
SWAP_OPTIMIZERS = True

# device/dtype for the base model and SAEs
DEVICE = "cuda"
SAE_DTYPE = torch.bfloat16
