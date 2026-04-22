#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dflash.ddtree_engine import generate_ddtree
from dflash.model_mlx import load, load_draft


DEFAULT_MODEL_PATH = "/Users/samuelfajreldines/dev/models/Qwen3.6-35B-A3B-4bit"
DEFAULT_DRAFT_PATH = "/Users/samuelfajreldines/dev/models/Qwen3.6-35B-A3B-DFlash"
DEFAULT_PROMPT = "Explique speculative decoding em portugues em 5 frases curtas."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a local MLX DDTree generation test with Qwen3.6-35B-A3B-4bit."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Local path to the base model.")
    parser.add_argument("--draft-model", default=DEFAULT_DRAFT_PATH, help="Local path to the DFlash draft model.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="User prompt.")
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum new tokens to generate.")
    parser.add_argument("--tree-budget", type=int, default=4, help="DDTree node budget.")
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Disable Qwen thinking mode in the chat template if supported.",
    )
    parser.add_argument(
        "--target-turboquant-bits",
        type=float,
        default=4.0,
        help="Optional TurboQuant bit width for compatible target-model KV cache layers. Use 0 to disable.",
    )
    return parser.parse_args()


def build_prompt(tokenizer, user_prompt: str, enable_thinking: bool) -> str:
    messages = [{"role": "user", "content": user_prompt}]
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    try:
        return tokenizer.apply_chat_template(
            messages,
            enable_thinking=enable_thinking,
            **kwargs,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def validate_path(label: str, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.exists():
        raise SystemExit(f"{label} not found: {path}")
    return path


def main() -> int:
    args = parse_args()
    model_path = validate_path("Base model", args.model)
    draft_path = validate_path("Draft model", args.draft_model)

    print(f"Loading base model from {model_path}", file=sys.stderr)
    model, tokenizer = load(str(model_path))

    print(f"Loading draft model from {draft_path}", file=sys.stderr)
    draft = load_draft(str(draft_path))

    prompt = build_prompt(tokenizer, args.prompt, enable_thinking=not args.disable_thinking)
    print("Starting DDTree generation...", file=sys.stderr)

    result = generate_ddtree(
        target_model=model,
        draft_model=draft,
        tokenizer=tokenizer,
        prompt_tokens=prompt,
        max_new_tokens=args.max_tokens,
        tree_budget=args.tree_budget,
        target_turboquant_bits=(None if args.target_turboquant_bits <= 0 else args.target_turboquant_bits),
    )

    print(result["text"], end="", flush=True)
    print()
    print(
        (
            f"finish_reason={result['finish_reason']} "
            f"engine={result['engine']} "
            f"tree_budget={result['tree_budget']} "
            f"prompt_tps={result['prompt_tps']:.2f} "
            f"generation_tps={result['generation_tps']:.2f} "
            f"generated_tokens={result['generated_tokens']} "
            f"peak_memory_gb={result['peak_memory_gb']:.2f}"
        ),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
