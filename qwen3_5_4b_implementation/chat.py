# Interactive terminal chat for Qwen3.5-4B.
# usage:  python3 -m qwen3_5_4b_implementation.chat --device cpu --dtype float32
# type 'exit' or press ctrl-c/ctrl-d to quit.
# python3 -m qwen3_5_4b_implementation.chat --device cpu --dtype float32

import argparse

from qwen3_5_4b_implementation.inference import load_model, generate


def main():
    parser = argparse.ArgumentParser(description="Interactive Qwen3.5-4B chat.")
    parser.add_argument("--weights", default="weights")
    parser.add_argument("--tokenizer", default="weights/tokenizer.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", help="bfloat16|float16|float32")
    parser.add_argument("--system", default="You are a helpful assistant.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    model, tokenizer = load_model(
        weights_path=args.weights,
        tokenizer_path=args.tokenizer,
        device=args.device,
        dtype=args.dtype,
    )

    print("\n--- Qwen3.5-4B chat loop ---")
    print("  type 'exit' to quit\n")

    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line or line.lower() == "exit":
            break

        out = generate(
            model,
            tokenizer,
            prompt=line,
            device=args.device,
            system=args.system,
            chat=True,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        print(f"qwen> {out}\n")


if __name__ == "__main__":
    main()
