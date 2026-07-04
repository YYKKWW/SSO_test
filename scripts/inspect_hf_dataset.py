#!/usr/bin/env python3
"""Inspect a Hugging Face dataset repository layout."""

from __future__ import annotations

import argparse
import collections

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", default="allenai/olmo-mix-1124")
    parser.add_argument("--limit", type=int, default=120)
    args = parser.parse_args()

    info = HfApi().dataset_info(args.dataset)
    print(f"dataset: {args.dataset}")
    print(f"sha: {info.sha}")
    print(f"siblings: {len(info.siblings)}")
    if info.cardData:
        print(f"card_data: {info.cardData}")

    counts: dict[str, int] = collections.defaultdict(int)
    for sibling in info.siblings:
        root = sibling.rfilename.split("/", 1)[0]
        counts[root] += 1

    print("top-level entries:")
    for root, count in sorted(counts.items()):
        print(f"  {root}: {count}")

    print("files:")
    for sibling in info.siblings[: args.limit]:
        print(f"  {sibling.rfilename}")


if __name__ == "__main__":
    main()
