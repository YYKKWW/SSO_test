#!/usr/bin/env python3
"""Download and save a Hugging Face tokenizer for offline cluster jobs."""

from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="Tokenizer model id or local path.")
    parser.add_argument("output_dir", help="Directory to save the tokenizer.")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.source,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer.save_pretrained(output_dir)
    print(f"saved tokenizer: {output_dir}")


if __name__ == "__main__":
    main()
