# Qwen3.5-4B architecture — what's different from a vanilla transformer

Assumes you know a standard decoder-only transformer (token embed → N × [attn + MLP] → norm → LM head). This walks the modules in `qwen3_5_4b_implementation/model.py` in data-flow order and flags the Qwen3.5-specific pieces.

Qwen3.5-4B dimensions:

| | value |
|---|---|
| num_hidden_layers | 32 |
| hidden_size | 2560 |
| intermediate_size | 9216 |
| vocab_size | 248320 |
| num_attention_heads | 16 |
| num_key_value_heads | 4 (GQA group = 4) |
| head_dim | 256 |
| partial_rotary_factor | 0.25 (RoPE on 64 of 256 dims) |
| rope_theta | 10,000,000 |
| layer_types | 8 blocks of `[3 linear + 1 full]` attention |
| linear_num_key_heads | 16 |
| linear_num_value_heads | 32 |
| linear_key_head_dim | 128 |
| linear_value_head_dim | 128 |
| rms_norm_eps | 1e-6 |
| tie_word_embeddings | True |
| max_position_embeddings | 262144 |


---

## Overall data flow

```
              input_ids (B, T)
                   │
                   ▼
            ┌──────────────────────┐
            │   nn.Embedding       │
            │   embed_tokens       │  (tied to lm_head)
            │                      │
            └──────────┬───────────┘
                       │ (B, T, 2560)
                       ▼
          residual ──► [Qwen3_5DecoderLayer 0]
                           │
                           ▼
                      [Qwen3_5DecoderLayer 1]
                           │
                          ...
                           │
                      [Qwen3_5DecoderLayer 31]
                           │
                           ▼
                 ┌─────────────────────┐
                 │  Qwen3_5RMSNorm     │
                 │  final norm         │
                 └──────────┬──────────┘
                            ▼
                 ┌─────────────────────┐
                 │  nn.Linear          │
                 │  lm_head            │  (tied to embed_tokens)
                 └──────────┬──────────┘
                            ▼
                 logits (B, T, 248320)
```

No PLE, no layer scalars, no vision/audio in this text-only implementation.

---

## One decoder layer (zoomed in)

```
         residual_in (B, T, 2560)
                │
        ┌───────▼─────────────────┐
        │  Qwen3_5RMSNorm         │
        │  input_layernorm        │
        └───────┬─────────────────┘
                ▼
        ┌───────────────────────────────┐
        │  Qwen3_5Attention             │  (layers 3, 7, 11, ...)
        │  full_attn                    │
        │  OR                           │
        │  Qwen3_5GatedDeltaNet         │  (all other layers)
        │  linear_attn                  │
        │  └─ Qwen3_5RMSNormGated       │
        └───────┬───────────────────────┘
                │
        residual┤
                ▼
        ┌───────▼─────────────────┐
        │  Qwen3_5RMSNorm         │
        │  post_attention_layernorm│
        └───────┬─────────────────┘
                ▼
        ┌─────────────────────────┐
        │  Qwen3_5MLP             │
        │  mlp                    │
        │  SwiGLU, no bias        │
        └───────┬─────────────────┘
                │
        residual┤
                ▼
         residual_out (B, T, 2560)
```

Two residual adds per layer: one after the attention branch, one after the MLP branch.

---

## 1. `Qwen3_5RMSNorm`

Standard RMSNorm with a **learnable weight centered around 1**:

```python
output = x_rms * (1 + weight)
```

`weight` is initialized to zeros, so the norm starts as the identity scale. This is the Llama/Qwen style, not the Gemma-4 raw-weight style.

A gated variant, `Qwen3_5RMSNormGated`, lives inside the linear-attention block and multiplies by `silu(gate)` after normalizing.

---

## 2. `Qwen3_5TextRotaryEmbedding` — MRoPE for text

Qwen3.5 uses **MRoPE** (Multi-modal RoPE). In the multimodal model the three spatial grids are T, H, W and get interleaved. For text-only inference the three grids are identical, so the interleaving is a no-op in terms of information but is kept for checkpoint fidelity.

Key parameter: `partial_rotary_factor = 0.25`. Only the first `256 * 0.25 = 64` dims of each head are rotated; the remaining `192` dims are position-invariant (NoPE channels).

```python
rotary_dim = int(head_dim * partial_rotary_factor)  # 64
rope_theta = 10_000_000
```

`mrope_section = [11, 11, 10]` splits the 32 frequency bins across the three grids for the interleaving code.

---

## 3. `Qwen3_5Attention` — full-attention layers

Every 4th layer (indices 3, 7, 11, 15, 19, 23, 27, 31) is a full softmax-attention layer.

Unusual details:

### 3a. Query-gate split
`q_proj` projects hidden states to **twice** the query dimension. The output is split into actual queries and a multiplicative gate:

```python
query, gate = chunk(q_proj(x))      # query: (B, T, num_heads, head_dim)
                                    # gate:  (B, T, num_heads * head_dim)
output = o_proj(attention_output * sigmoid(gate))
```

The gate is a per-head soft switch on the attention output.

### 3b. Q/K norms
Queries and keys are individually RMSNorm'd (`q_norm`, `k_norm`) before the dot product.

### 3c. Standard scaling
Unlike Gemma-4, this attention uses normal `scale = 1 / sqrt(head_dim)`.

### 3d. GQA
`num_attention_heads = 16`, `num_key_value_heads = 4`, so each K/V head is broadcast to 4 query heads.

### 3e. Partial RoPE
Only the first 64 dims of the 256-dim heads get RoPE; the rest pass through unchanged.

---

## 4. `Qwen3_5GatedDeltaNet` — linear-attention layers

The other 24 layers are linear-attention layers using the **Gated DeltaNet** formulation.

High-level flow:

```
x
│
├─► in_proj_qkv ──► causal Conv1d + SiLU ──► split q, k, v
│
├─► in_proj_b ──► sigmoid ──► beta
│
└─► in_proj_a + dt_bias + A_log ──► g (negative decay)

q, k, v, beta, g ──► gated delta rule ──► gated RMSNorm(z) ──► out_proj
```

- `in_proj_qkv` maps `2560 → key_dim*2 + value_dim = 16*128*2 + 32*128 = 8192`.
- A depthwise causal 1D conv (kernel size 4) is applied over the sequence.
- `beta` and `g` implement the delta-rule update on a recurrent state.
- The output is normalized with `Qwen3_5RMSNormGated`, where `z` from `in_proj_z` gates the normalized result.

This is the only place in the model that looks "Mamba-like"; it replaces softmax attention with a stateful linear recurrent layer.

---

## 5. `Qwen3_5MLP` — SwiGLU

Standard gated MLP:

```python
down_proj(silu(gate_proj(x)) * up_proj(x))
```

- `gate_proj`: `2560 → 9216`
- `up_proj`:   `2560 → 9216`
- `down_proj`: `9216 → 2560`

No bias. This is the same SwiGLU pattern as Llama/Qwen2.5.

---

## 6. `Qwen3_5TextModel` / `Qwen3_5ForCausalLM`

- 32 `Qwen3_5DecoderLayer`s stacked.
- Final `RMSNorm`.
- `lm_head` is a `Linear(2560, 248320)` whose weight is tied to `embed_tokens.weight`.
- The top-level `model.language_model` wrapper keeps the state_dict key layout identical to the official HF checkpoint (`model.language_model.embed_tokens.weight`, etc.).

---

## What's NOT here (vs. full Qwen3.5)

- **Vision tower** — text-only implementation.
- **Audio tower** — not implemented.
- **Multimodal MRoPE inputs** — text grids only.

---

## Cheat sheet — where to put your interp hooks

| Site | What you'd get | How |
|---|---|---|
| Residual stream | Trunk between layers | `register_forward_hook` on each `Qwen3_5DecoderLayer` |
| Full-attention output | Attention's contribution to the residual | hook after `o_proj` inside `Qwen3_5Attention.forward`, or around `self_attn(...)` in `Qwen3_5DecoderLayer.forward` |
| Linear-attention output | Gated DeltaNet's contribution to the residual | hook on `layer.linear_attn` |
| MLP output | MLP's contribution to the residual | hook on `layer.mlp` |
| Q / K / V (post-RoPE) | Full-attention head dynamics | hook inside `Qwen3_5Attention.forward` |
| Pre/post MLP | MLP input/output | hook before/after `mlp(...)` in decoder layer |
| Final pre-LM-head hidden | Residual after final norm | output of `model.language_model.norm` |

The SAE training script already captures `residue`, `post_attn_norm`, `post_linear_attn_norm`, and `post_mlp_norm` via hooks.
