# Qwen3.5-4B configuration
# specs from the official Qwen3.5-4B checkpoint:
#   hidden_size = 2560
#   intermediate_size = 9216
#   num_hidden_layers = 32
#   num_attention_heads = 16
#   num_key_value_heads = 4
#   head_dim = 256
#   vocab_size = 248320
#   tie_word_embeddings = True
# architecture: 8 blocks of [3 linear-attention layers + 1 full-attention layer]

from dataclasses import dataclass


@dataclass
class Qwen3_5TextConfig:
    vocab_size: int = 248320
    hidden_size: int = 2560
    intermediate_size: int = 9216
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 256
    hidden_act: str = "silu"
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    attention_dropout: float = 0.0
    tie_word_embeddings: bool = True
    use_cache: bool = True

    # RoPE
    rope_theta: float = 10000000.0
    partial_rotary_factor: float = 0.25  # rope on 64 of 256 head dims

    # linear attention (Gated DeltaNet)
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32

    # layout: every 4th layer is full attention; the rest are linear attention
    full_attention_interval: int = 4

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    def __post_init__(self):
        # generate layer_types if not overridden
        if not hasattr(self, "layer_types") or self.layer_types is None:
            self.layer_types = [
                "linear_attention" if (i + 1) % self.full_attention_interval else "full_attention"
                for i in range(self.num_hidden_layers)
            ]


@dataclass
class Qwen3_5VisionConfig:
    # not implemented; kept for API compatibility
    pass


@dataclass
class Qwen3_5Config:
    text_config: Qwen3_5TextConfig = None
    image_token_id: int = 248056
    video_token_id: int = 248057
    vision_start_token_id: int = 248053
    vision_end_token_id: int = 248054

    def __post_init__(self):
        if self.text_config is None:
            self.text_config = Qwen3_5TextConfig()
        self.text_config.__post_init__()
