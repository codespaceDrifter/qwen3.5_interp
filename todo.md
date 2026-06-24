# SAE Training TODO

## Current Code Layout
- `probes/GatedSAE.py` — your custom gated SAE (slab + cone gates + adaptive sparsity coeff).
- `probes/TopKSAE.py` — basic top-K SAE: `encoder_weight/bias`, `decoder_weight/bias`, ReLU, top-K.
- `interp_pipeline/config.py` — `SAE_TYPE = "gated" | "topk"`, `CAPTURE_TYPE`, `TARGET_ACTIVE` (=K), LR, etc.
- `interp_pipeline/sae_train.py` — training loop, logging, checkpointing, HTML/PNG plots.

## TopK SAE Improvements to Try

1. **AuxK loss (main dead-feature fix)**
   - Add an auxiliary reconstruction loss on the top-K *currently dead* features, reconstructing the residual `activation - pred`.
   - Typical weight: `1/32` of main recon loss.
   - Need to track which features are dead over a window.

2. **LR warmup**
   - Ramp LR from `0` to `LEARNING_RATE` over first ~1000 steps.
   - Helps avoid early feature death.

3. **Adam β1 = 0**
   - Change optimizer `betas=(0.0, 0.999)` instead of default `(0.9, 0.999)`.
   - Reduces momentum, helps dead features revive.

4. **Feature resampling**
   - Every N steps, identify dead features. Exponential Moving Average of firing frequency.  
   - Keep a small ring buffer of recent activations (or high-error activations).
   - Reset dead feature encoder row to a high-error activation, decoder column to its normalized direction.
   - Reset that feature’s optimizer state.

5. **Geometric median decoder_bias init**
   - Write a one-off script to sample ~100k activations for a capture type.
   - Compute geometric median (Weiszfeld’s algorithm) and use it as initial `decoder_bias`.
   - Can approximate with mean if geometric median is too annoying.

6. **Remove encoder_bias experiment**
   - Top-K doesn’t obviously need per-feature thresholds.
   - Removing it fixes the threshold at 0 and makes full collapse harder.

7. **Try larger K**
   - `TARGET_ACTIVE = 128` or `256`.
   - Residue stream is the hardest; 64 may just be too few.

## GatedSAE Improvements (if revisiting)

0. sparsity loss only affect gate like gemmascope

1. **Two-sided sparsity loss**
   - Change `clamp(error, min=0)` to `error.abs()` so dead features also get pushed open.

2. **Softer gates / larger band_eps**
   - Current `band_eps=0.001` makes gradients vanish away from thresholds.

3. **Initialize slab lower bound negative**
   - Start with all features open, let training sparsify.

4. **Feature resampling**
   - Same idea as TopK: reset dead feature directions.

## Experiments to Run

- [ ] TopK on `residue` with `k=64` (current run)
- [ ] TopK on `residue` with `k=128` or `256`
- [ ] TopK on `attn_out` / `linear_attn_out` / `mlp_out` with `k=64`
- [ ] Gated on `attention` with fixed two-sided sparsity loss

## Reference Notes

- Anthropic TopK SAE: top-K activation + AuxK loss + occasional resampling + LR warmup + β1=0 + geometric-median decoder bias init.
- Gemma Scope: JumpReLU/Gated SAEs + resampling.
- Ghost gradients are older and have been reported as suboptimal in recent work.
