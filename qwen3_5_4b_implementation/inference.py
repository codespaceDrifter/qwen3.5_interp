# Shared inference utilities: load model + tokenizer, generate text.

from pathlib import Path

import torch
import torch.nn.functional as F

from qwen3_5_4b_implementation.config import Qwen3_5Config
from qwen3_5_4b_implementation.model import Qwen3_5ForCausalLM, Qwen3_5Cache
from qwen3_5_4b_implementation.loader import load_weights
from qwen3_5_4b_implementation.tokenizer import QwenTokenizer


def _dtype_from_string(s: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(s, torch.bfloat16)


def load_model(
    weights_path: str = "weights",
    tokenizer_path: str = "weights/tokenizer.json",
    device: str | torch.device = "cuda",
    dtype: torch.dtype | str = torch.bfloat16,
    config: Qwen3_5Config | None = None,
) -> tuple[Qwen3_5ForCausalLM, QwenTokenizer]:
    """Build the model, load safetensors, and load the tokenizer."""
    if isinstance(dtype, str):
        dtype = _dtype_from_string(dtype)
    device = torch.device(device)
    cfg = config or Qwen3_5Config()

    if not Path(weights_path).exists():
        raise FileNotFoundError(
            f"weights not found: {weights_path!r}. "
            "Run: python3 -m qwen3_5_4b_implementation.download"
        )

    print("building model skeleton...")
    model = Qwen3_5ForCausalLM(cfg)
    load_weights(model, weights_path, device=device, dtype=dtype, strict=True)
    model.eval()
    # Move any buffers / re-initialized params (rope, etc.) onto the device.
    model.to(device)

    print("loading tokenizer...")
    tokenizer = QwenTokenizer(tokenizer_path)
    return model, tokenizer


@torch.no_grad()
def generate(
    model: Qwen3_5ForCausalLM,
    tokenizer: QwenTokenizer,
    prompt: str,
    device: str | torch.device = "cuda",
    system: str | None = "You are a helpful assistant.",
    chat: bool = True,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
) -> str:
    """Greedy/temperature-sampled generation for a single prompt."""
    device = torch.device(device)

    if chat:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], system=system
        )

    input_ids = torch.tensor(
        [tokenizer.encode(prompt)], device=device, dtype=torch.long
    )
    cache = Qwen3_5Cache(model.config.num_hidden_layers)

    # Prefill.
    logits = model(input_ids=input_ids, use_cache=True, past_key_values=cache)["logits"]
    next_token = _sample(logits[:, -1, :], temperature)
    generated = [int(next_token.item())]

    # Decode loop.
    eos_id = tokenizer.token_to_id(tokenizer.endoftext)
    eot_id = tokenizer.token_to_id(tokenizer.im_end)
    for _ in range(max_new_tokens - 1):
        tok = generated[-1]
        if eos_id is not None and tok == eos_id:
            break
        if eot_id is not None and tok == eot_id:
            break

        logits = model(
            input_ids=next_token.unsqueeze(0),
            use_cache=True,
            past_key_values=cache,
        )["logits"]
        next_token = _sample(logits[:, -1, :], temperature)
        generated.append(int(next_token.item()))

    stop_ids = {i for i in (eos_id, eot_id) if i is not None}
    out_ids = [t for t in generated if t not in stop_ids]
    return tokenizer.decode(out_ids, skip_special=True)


def _sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0.0:
        return logits.argmax(dim=-1)
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
