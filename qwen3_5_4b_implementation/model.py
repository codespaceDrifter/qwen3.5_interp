"""
From-scratch PyTorch implementation of the Qwen3.5-4B text model.

This module implements only the text-only inference path.  It intentionally
re-implements the torch fallback kernels for the linear-attention (Gated
DeltaNet) layers so that it has no runtime dependency on `transformers`,
`causal_conv1d`, or `flash_linear_attention`.

Weight key naming is chosen to match the official HuggingFace Qwen3.5
safetensors layout for the text-only / language-model branch:

    model.language_model.embed_tokens.weight
    model.language_model.layers.{i}.self_attn.q_proj.weight
    model.language_model.layers.{i}.mlp.gate_proj.weight
    model.language_model.norm.weight
    lm_head.weight
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Qwen3_5Config, Qwen3_5TextConfig


def _get_text_config(config):
    """Allow modules to accept either a Qwen3_5Config or Qwen3_5TextConfig."""
    if hasattr(config, "text_config") and config.text_config is not None:
        return config.text_config
    return config


# ---------------------------------------------------------------------------
# Normalization layers
# ---------------------------------------------------------------------------

class Qwen3_5RMSNorm(nn.Module):
    """RMSNorm used throughout the Qwen3.5 text model.

    Qwen3.5 centers the norm around 1, i.e. ``output = x_norm * (1 + weight)``,
    and initializes ``weight`` to zeros.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class Qwen3_5RMSNormGated(nn.Module):
    """Gated RMSNorm used inside the Gated DeltaNet linear-attention block."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)


# ---------------------------------------------------------------------------
# Rotary position embeddings (MRoPE-like, text-only)
# ---------------------------------------------------------------------------

class Qwen3_5TextRotaryEmbedding(nn.Module):
    """MRoPE-style rotary embeddings for the text branch.

    The HF implementation computes cos/sin for three spatial grids (T, H, W)
    and then interleaves them.  For text-only inference all three grids are
    identical, but the interleaving code is kept for fidelity.
    """

    def __init__(self, config: Qwen3_5TextConfig, device=None):
        super().__init__()
        self.config = config
        dim = int(config.head_dim * config.partial_rotary_factor)
        inv_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # Number of frequency bins taken from each grid dimension.
        self.mrope_section = [11, 11, 10]

    @staticmethod
    def _apply_interleaved_mrope(freqs: torch.Tensor, mrope_section) -> torch.Tensor:
        """Interleave 3D rotary frequencies into a single 2D frequency tensor.

        freqs: (3, batch_size, seq_len, head_dim // 2)
        Returns: (batch_size, seq_len, head_dim // 2)
        """
        freqs_t = freqs[0].clone()
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    def forward(
        self, x: torch.Tensor, position_ids: torch.LongTensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # position_ids may be 2D (bs, seq) or 3D (3, bs, seq).  Expand to 3D.
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        # inv_freq: (dim // 2,) -> (3, bs, dim // 2, 1)
        inv_freq_expanded = (
            self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        )
        # position_ids: (3, bs, seq) -> (3, bs, 1, seq)
        position_ids_expanded = position_ids[:, :, None, :].float()

        # (3, bs, 1, seq) @ (3, bs, dim//2, 1).transpose -> (3, bs, seq, dim//2)
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
        freqs = self._apply_interleaved_mrope(freqs, self.mrope_section)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query/key states.

    q, k: (bs, num_heads, seq_len, head_dim) when unsqueeze_dim=1.
    cos, sin: (bs, seq_len, rotary_dim).
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class Qwen3_5MLP(nn.Module):
    """SwiGLU MLP."""

    def __init__(self, config: Qwen3_5TextConfig, intermediate_size: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Torch-only fallback kernels for Gated DeltaNet
# ---------------------------------------------------------------------------

def torch_causal_conv1d_update(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
) -> torch.Tensor:
    """Single-token causal Conv1d update used during decode.

    hidden_states: (batch, hidden_size, seq_len)  -- usually seq_len == 1
    conv_state:    (batch, hidden_size, state_len)
    weight:        (hidden_size, kernel_size)
    """
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]

    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(
        hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size
    )
    out = F.silu(out[:, :, -seq_len:])
    out = out.to(hidden_states.dtype)
    return out


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """L2 normalization matching the FLA reference implementation."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def torch_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Chunk-parallel Gated Delta Rule (torch fallback).

    Inputs are expected in (batch, seq_len, num_heads, head_dim) layout; they
    are transposed to (batch, num_heads, seq_len, head_dim) internally.
    """
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)

    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size

    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size

    scale = 1.0 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)

    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0,
    )

    # Intra-chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)

    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)

    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

    last_recurrent_state = (
        torch.zeros(
            batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device
        )
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)

    # Cross-chunk recurrent state
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = k_cumdecay[:, :, i] @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2)
            @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None

    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def torch_recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Sequential Gated Delta Rule used for single-token decode."""
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)

    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1.0 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(
        batch_size, num_heads, sequence_length, v_head_dim, dtype=value.dtype, device=value.device
    )
    last_recurrent_state = (
        torch.zeros(
            batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device
        )
        if initial_state is None
        else initial_state.to(value)
    )

    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None

    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


# ---------------------------------------------------------------------------
# Linear attention: Gated DeltaNet
# ---------------------------------------------------------------------------

def apply_mask_to_padding_states(hidden_states: torch.Tensor, attention_mask):
    """Zero out hidden states for padding tokens (used by linear attention)."""
    if (
        attention_mask is not None
        and attention_mask.shape[1] > 1
        and attention_mask.shape[0] > 1
    ):
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)
    return hidden_states


class Qwen3_5GatedDeltaNet(nn.Module):
    """Gated DeltaNet linear-attention layer."""

    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        config = _get_text_config(config)
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.layer_norm_epsilon = config.rms_norm_eps

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        # Time-step / decay parameters
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))

        self.norm = Qwen3_5RMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        self.in_proj_qkv = nn.Linear(
            self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False
        )
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params=None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)

        batch_size, seq_len, _ = hidden_states.shape
        use_precomputed_states = (
            cache_params is not None and cache_params.has_previous_state(self.layer_idx)
        )

        if use_precomputed_states:
            conv_state = cache_params.conv_states[self.layer_idx]
            recurrent_state = cache_params.recurrent_states[self.layer_idx]

        mixed_qkv = self.in_proj_qkv(hidden_states)
        mixed_qkv = mixed_qkv.transpose(1, 2)  # (bs, conv_dim, seq)

        z = self.in_proj_z(hidden_states)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        if use_precomputed_states and seq_len == 1:
            # Single-token decode path
            mixed_qkv = torch_causal_conv1d_update(
                mixed_qkv,
                conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                activation="silu",
            )
        else:
            # Prefill or chunked-decode path
            if use_precomputed_states:
                mixed_qkv = torch.cat([conv_state, mixed_qkv], dim=-1)

            if cache_params is not None:
                # Keep the last kernel_size-1 inputs as the conv context for the next step.
                keep = self.conv_kernel_size - 1
                if mixed_qkv.shape[-1] >= keep:
                    new_conv_state = mixed_qkv[:, :, -keep:].contiguous()
                else:
                    new_conv_state = F.pad(
                        mixed_qkv, (keep - mixed_qkv.shape[-1], 0)
                    )
                cache_params.update_conv_state(new_conv_state, self.layer_idx)

            mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, : mixed_qkv.shape[-1]])

            if use_precomputed_states:
                mixed_qkv = mixed_qkv[:, :, -seq_len:]

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )

        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        # `g` is the negative decay rate used by the gated delta rule.
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        if use_precomputed_states and seq_len == 1:
            core_attn_out, last_recurrent_state = torch_recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state if use_precomputed_states else None,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )

        if cache_params is not None:
            cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        output = self.out_proj(core_attn_out)
        return output


# ---------------------------------------------------------------------------
# Full (softmax) attention
# ---------------------------------------------------------------------------

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads to match the number of query heads (GQA)."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Standard causal softmax attention."""
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class Qwen3_5Attention(nn.Module):
    """Multi-head full attention with query/key norms, RoPE, and gated output."""

    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        config = _get_text_config(config)
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim * 2,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]  # (bs, seq_len)
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)  # (bs, seq_len, num_heads * head_dim)

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, unsqueeze_dim=1
        )

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


# ---------------------------------------------------------------------------
# Decoder layer, text model, and CausalLM head
# ---------------------------------------------------------------------------

class Qwen3_5DecoderLayer(nn.Module):
    """A single transformer decoder layer.

    Switches between linear attention and full attention according to
    ``config.layer_types[layer_idx]``.
    """

    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        config = _get_text_config(config)
        self.hidden_size = config.hidden_size
        self.layer_type = config.layer_types[layer_idx]

        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3_5Attention(config, layer_idx)
        else:
            raise ValueError(f"Unknown layer_type: {self.layer_type}")

        self.mlp = Qwen3_5MLP(config, config.intermediate_size)
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                attention_mask=attention_mask,
            )
        else:
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Qwen3_5Cache:
    """Minimal KV + recurrent cache for text-only generation.

    Holds per-layer key/value tensors for full-attention layers and
    conv/recurrent states for linear-attention layers.
    """

    def __init__(self, num_hidden_layers: int):
        self.num_hidden_layers = num_hidden_layers
        self.key_cache = [None] * num_hidden_layers
        self.value_cache = [None] * num_hidden_layers
        self.conv_states = [None] * num_hidden_layers
        self.recurrent_states = [None] * num_hidden_layers
        self._seen_tokens = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=-2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=-2
            )
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def update_conv_state(self, conv_state: torch.Tensor, layer_idx: int):
        self.conv_states[layer_idx] = conv_state

    def update_recurrent_state(self, recurrent_state: Optional[torch.Tensor], layer_idx: int):
        self.recurrent_states[layer_idx] = recurrent_state

    def get_seq_length(self, layer_idx: int = 0) -> int:
        # Overall sequence length seen by the cache. Linear-attention layers do not
        # store per-token KV tensors, so we maintain an explicit counter.
        if self._seen_tokens:
            return self._seen_tokens
        # Fallback for caches created before _seen_tokens was tracked.
        if self.key_cache[layer_idx] is not None:
            return self.key_cache[layer_idx].shape[-2]
        return 0

    def set_seq_length(self, seq_length: int):
        self._seen_tokens = seq_length

    def has_previous_state(self, layer_idx: Optional[int] = None) -> bool:
        if layer_idx is not None:
            return (
                self.conv_states[layer_idx] is not None
                or self.recurrent_states[layer_idx] is not None
                or self.key_cache[layer_idx] is not None
            )
        return any(s is not None for s in self.key_cache + self.conv_states + self.recurrent_states)


def _prepare_causal_mask(
    seq_len: int,
    past_key_values,
    attention_mask: Optional[torch.Tensor],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Build a causal mask compatible with cached generation."""
    cache_len = past_key_values.get_seq_length() if past_key_values is not None else 0
    total_len = cache_len + seq_len

    mask = torch.zeros(seq_len, total_len, dtype=dtype, device=device)
    if seq_len > 1:
        causal_part = torch.triu(
            torch.ones(seq_len, seq_len, dtype=dtype, device=device), diagonal=1
        )
        causal_part = causal_part.masked_fill(causal_part == 1, float("-inf"))
        mask[:, cache_len:] = causal_part

    if attention_mask is not None:
        # attention_mask: (bs, total_len) with 1 for valid tokens, 0 for padding.
        if attention_mask.shape[-1] < total_len:
            pad_len = total_len - attention_mask.shape[-1]
            attention_mask = F.pad(attention_mask, (pad_len, 0), value=1)
        padding_mask = (1 - attention_mask).to(dtype) * float("-inf")
        mask = mask.unsqueeze(0) + padding_mask[:, None, None, :]
    else:
        mask = mask.unsqueeze(0).unsqueeze(0)

    return mask


class Qwen3_5TextModel(nn.Module):
    """Text-only transformer trunk."""

    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__()
        config = _get_text_config(config)
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3_5DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config)

    def _update_linear_attn_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        past_key_values,
    ) -> Optional[torch.Tensor]:
        """Linear attention uses a simple left-padding mask, or None when safe."""
        if past_key_values is not None and past_key_values.has_previous_state():
            return None
        if attention_mask is not None and torch.all(attention_mask == 1):
            return None
        return attention_mask

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if use_cache and past_key_values is None:
            past_key_values = Qwen3_5Cache(self.config.num_hidden_layers)

        batch_size, seq_len, _ = inputs_embeds.shape

        # Build position ids.  For text-only inference all three MRoPE grids are
        # identical, so we replicate a single 2D tensor to 3D.
        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values else 0
            position_ids = (
                torch.arange(seq_len, device=inputs_embeds.device) + past_seen_tokens
            )
            position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        text_position_ids = position_ids[0]

        causal_mask = _prepare_causal_mask(
            seq_len=seq_len,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        linear_attn_mask = self._update_linear_attn_mask(attention_mask, past_key_values)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for i, decoder_layer in enumerate(self.layers):
            layer_mask = (
                linear_attn_mask if self.config.layer_types[i] == "linear_attention" else causal_mask
            )
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
            )

        hidden_states = self.norm(hidden_states)

        if use_cache and past_key_values is not None:
            past_key_values.set_seq_length(past_seen_tokens + seq_len)

        return {
            "last_hidden_state": hidden_states,
            "past_key_values": past_key_values if use_cache else None,
        }


class Qwen3_5ForCausalLM(nn.Module):
    """Text-only causal language model.

    The ``model.language_model`` wrapper is kept so that the state_dict key
    layout matches the official HF Qwen3.5 text checkpoint:
    ``model.language_model.embed_tokens.weight``, etc.
    """

    def __init__(self, config: Qwen3_5Config):
        super().__init__()
        text_config = _get_text_config(config)
        self.config = text_config

        self.model = nn.Module()
        self.model.language_model = Qwen3_5TextModel(text_config)

        self.vocab_size = text_config.vocab_size
        self.lm_head = nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False)

        if text_config.tie_word_embeddings:
            self.tie_weights()

    def tie_weights(self):
        self.lm_head.weight = self.model.language_model.embed_tokens.weight

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        logits_to_keep: int = 0,
    ):
        outputs = self.model.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
        )

        hidden_states = outputs["last_hidden_state"]
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
            )

        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": outputs["past_key_values"],
        }
