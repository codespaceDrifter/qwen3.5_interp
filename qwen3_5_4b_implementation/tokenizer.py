# Tiny wrapper around the HF `tokenizer.json` format for Qwen3.5.

from tokenizers import Tokenizer


class QwenTokenizer:
    def __init__(self, tokenizer_path: str = "weights/tokenizer.json"):
        self.tok = Tokenizer.from_file(str(tokenizer_path))

        self.im_start = "<|im_start|>"
        self.im_end = "<|im_end|>"
        self.endoftext = "<|endoftext|>"
        self.eos_token = self.endoftext

    def token_to_id(self, token: str) -> int | None:
        return self.tok.token_to_id(token)

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text).ids

    def decode(self, ids, skip_special: bool = True) -> str:
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return self.tok.decode(ids, skip_special_tokens=skip_special)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        add_generation_prompt: bool = True,
    ) -> str:
        """Build the Qwen chat prompt string manually."""
        parts: list[str] = []
        if system is not None:
            parts.append(f"{self.im_start}system\n{system}{self.im_end}\n")
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "" )
            parts.append(f"{self.im_start}{role}\n{content}{self.im_end}\n")
        if add_generation_prompt:
            parts.append(f"{self.im_start}assistant\n")
        return "".join(parts)
