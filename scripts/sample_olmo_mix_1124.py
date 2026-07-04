#!/usr/bin/env python3
"""Stream a weighted OLMo mix sample into JSONL shards.

The default weights match the OLMo 2 Mix (November 2024) token counts used in
the SSO paper appendix: DCLM plus arxiv, pes2o, starcoder, algebraic-stack,
open-web-math, and wiki.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests
import zstandard as zstd
from datasets import load_dataset
from huggingface_hub import HfApi, get_token, hf_hub_url
from transformers import AutoTokenizer


DATASET_ID = "allenai/olmo-mix-1124"
TOKENIZER_ID = "allenai/OLMo-2-1124-7B"

COMPONENT_TOKEN_COUNTS = {
    "dclm": 3_700_000_000_000,
    "arxiv": 20_800_000_000,
    "pes2o": 58_600_000_000,
    "starcoder": 83_000_000_000,
    "algebraic-stack": 11_800_000_000,
    "open-web-math": 12_200_000_000,
    "wiki": 3_660_000_000,
}

DEFAULT_COMPONENTS = tuple(COMPONENT_TOKEN_COUNTS)


@dataclass
class SplitStats:
    docs: int = 0
    estimated_tokens: int = 0
    shards: int = 0


@dataclass
class ComponentStats:
    train: SplitStats
    valid: SplitStats


class ShardWriter:
    def __init__(
        self,
        output_dir: Path,
        split: str,
        component: str,
        tokens_per_shard: int,
        force: bool,
    ) -> None:
        self.output_dir = output_dir
        self.split = split
        self.component = component.replace("-", "_")
        self.tokens_per_shard = tokens_per_shard
        self.force = force
        self.index = 0
        self.current_tokens = 0
        self.handle = None
        self.path: Path | None = None
        self.shards_written = 0

    def _next_path(self) -> Path:
        return self.output_dir / f"{self.split}_{self.component}_{self.index:06d}.jsonl"

    def _open(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self._next_path()
        if self.path.exists() and not self.force:
            raise FileExistsError(f"{self.path} exists; pass --force to overwrite")
        self.handle = self.path.open("w", encoding="utf-8")
        self.current_tokens = 0
        self.shards_written += 1

    def write(self, record: dict, token_count: int) -> None:
        if self.handle is None:
            self._open()
        assert self.handle is not None
        assert self.path is not None
        if self.current_tokens >= self.tokens_per_shard:
            self.handle.close()
            self.index += 1
            self._open()
        self.handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.current_tokens += token_count

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None


def parse_components(raw: str) -> list[str]:
    if raw == "all":
        return list(DEFAULT_COMPONENTS)
    components = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [item for item in components if item not in COMPONENT_TOKEN_COUNTS]
    if unknown:
        raise ValueError(f"unknown components: {unknown}")
    return components


def allocate_tokens(total_tokens: int, components: Iterable[str]) -> dict[str, int]:
    components = list(components)
    denominator = sum(COMPONENT_TOKEN_COUNTS[name] for name in components)
    raw = {
        name: total_tokens * COMPONENT_TOKEN_COUNTS[name] / denominator
        for name in components
    }
    quotas = {name: int(math.floor(value)) for name, value in raw.items()}
    remainder = total_tokens - sum(quotas.values())
    for name in sorted(components, key=lambda item: raw[item] - quotas[item], reverse=True):
        if remainder <= 0:
            break
        quotas[name] += 1
        remainder -= 1
    return quotas


def estimate_tokens(tokenizer, text: str, count_eod: bool) -> int:
    token_count = len(tokenizer.encode(text, add_special_tokens=False))
    if count_eod:
        token_count += 1
    return token_count


def get_dataset_sha(dataset_id: str) -> str | None:
    try:
        return HfApi().dataset_info(dataset_id).sha
    except Exception as exc:  # pragma: no cover - metadata is best effort.
        print(f"warning: could not resolve dataset sha: {exc}", file=sys.stderr)
        return None


def list_component_files(dataset_id: str, revision: str | None, component: str) -> list[str]:
    info = HfApi().dataset_info(
        repo_id=dataset_id,
        revision=revision,
    )
    files = [sibling.rfilename for sibling in info.siblings]
    prefix = f"data/{component}/"
    suffixes = (".jsonl", ".json", ".jsonl.gz", ".json.gz", ".jsonl.zstd", ".json.zstd")
    component_files = [
        path for path in files if path.startswith(prefix) and path.endswith(suffixes)
    ]
    if not component_files:
        raise FileNotFoundError(f"no JSON files found for component {component!r}")
    return component_files


def iter_hf_json_records(
    *,
    dataset_id: str,
    filename: str,
    revision: str | None,
    token: str | None,
) -> Iterable[dict]:
    url = hf_hub_url(
        repo_id=dataset_id,
        filename=filename,
        repo_type="dataset",
        revision=revision,
    )
    headers = {"Authorization": f"Bearer {token}"} if token else None
    with requests.get(
        url,
        headers=headers,
        stream=True,
        timeout=(30, 300),
        allow_redirects=True,
    ) as response:
        response.raise_for_status()
        response.raw.decode_content = True
        if filename.endswith(".gz"):
            binary_stream = gzip.GzipFile(fileobj=response.raw)
        elif filename.endswith(".zstd"):
            binary_stream = zstd.ZstdDecompressor().stream_reader(response.raw)
        else:
            binary_stream = response.raw

        with io.TextIOWrapper(binary_stream, encoding="utf-8") as text_stream:
            for line in text_stream:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    print(
                        f"warning: skip malformed JSON in {filename}: {exc}",
                        file=sys.stderr,
                    )


def iter_component_records_hub_files(
    *,
    dataset_id: str,
    component: str,
    revision: str | None,
    seed: int,
    token: str | None,
) -> Iterable[dict]:
    files = list_component_files(dataset_id, revision, component)
    random.Random(seed).shuffle(files)
    print(f"{component}: streaming {len(files)} files from HuggingFace Hub", flush=True)
    for file_index, filename in enumerate(files, start=1):
        print(f"{component}: file {file_index}/{len(files)} {filename}", flush=True)
        try:
            yield from iter_hf_json_records(
                dataset_id=dataset_id,
                filename=filename,
                revision=revision,
                token=token,
            )
        except Exception as exc:
            print(f"warning: skip failed file {filename}: {exc}", file=sys.stderr, flush=True)


def sample_component(
    *,
    dataset_id: str,
    component: str,
    tokenizer,
    output_dir: Path,
    train_quota: int,
    valid_quota: int,
    tokens_per_shard: int,
    shuffle_buffer: int,
    seed: int,
    count_eod: bool,
    force: bool,
    backend: str,
    revision: str | None,
    hf_token: str | None,
) -> ComponentStats:
    if backend == "datasets":
        stream = load_dataset(dataset_id, component, split="train", streaming=True)
        if shuffle_buffer > 0:
            stream = stream.shuffle(buffer_size=shuffle_buffer, seed=seed)
    elif backend == "hub-files":
        stream = iter_component_records_hub_files(
            dataset_id=dataset_id,
            component=component,
            revision=revision,
            seed=seed,
            token=hf_token,
        )
    else:
        raise ValueError(f"unknown backend: {backend}")

    writers = {
        "train": ShardWriter(output_dir, "train", component, tokens_per_shard, force),
        "valid": ShardWriter(output_dir, "valid", component, tokens_per_shard, force),
    }
    stats = ComponentStats(train=SplitStats(), valid=SplitStats())
    split = "train"
    quotas = {"train": train_quota, "valid": valid_quota}

    try:
        for row in stream:
            if split == "train" and stats.train.estimated_tokens >= quotas["train"]:
                split = "valid"
            if split == "valid" and stats.valid.estimated_tokens >= quotas["valid"]:
                break

            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            token_count = estimate_tokens(tokenizer, text, count_eod)
            if token_count <= 0:
                continue

            record = {"text": text, "source": component}
            writers[split].write(record, token_count)
            split_stats = getattr(stats, split)
            split_stats.docs += 1
            split_stats.estimated_tokens += token_count

            if (split_stats.docs % 1000) == 0:
                print(
                    f"{component} {split}: docs={split_stats.docs} "
                    f"estimated_tokens={split_stats.estimated_tokens} "
                    f"quota={quotas[split]}",
                    flush=True,
                )
    finally:
        for writer in writers.values():
            writer.close()

    stats.train.shards = writers["train"].shards_written
    stats.valid.shards = writers["valid"].shards_written
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DATASET_ID)
    parser.add_argument("--tokenizer", default=TOKENIZER_ID)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-tokens", type=int, default=1_000_000_000)
    parser.add_argument("--valid-tokens", type=int, default=10_000_000)
    parser.add_argument("--tokens-per-shard", type=int, default=50_000_000)
    parser.add_argument("--components", default="all")
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count-eod", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--backend", choices=["hub-files", "datasets"], default="hub-files")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    components = parse_components(args.components)
    train_quotas = allocate_tokens(args.target_tokens, components)
    valid_quotas = allocate_tokens(args.valid_tokens, components)

    output_dir = Path(args.output_dir)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not args.force:
        raise FileExistsError(f"{manifest_path} exists; pass --force to overwrite")

    print("component quotas:")
    for component in components:
        weight = COMPONENT_TOKEN_COUNTS[component] / sum(
            COMPONENT_TOKEN_COUNTS[item] for item in components
        )
        print(
            f"  {component}: weight={weight:.8f} "
            f"train={train_quotas[component]} valid={valid_quotas[component]}"
        )

    if args.dry_run:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    dataset_sha = args.revision or get_dataset_sha(args.dataset)
    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or get_token()

    all_stats: dict[str, ComponentStats] = {}
    for index, component in enumerate(components):
        print(f"start component: {component}", flush=True)
        stats = sample_component(
            dataset_id=args.dataset,
            component=component,
            tokenizer=tokenizer,
            output_dir=output_dir,
            train_quota=train_quotas[component],
            valid_quota=valid_quotas[component],
            tokens_per_shard=args.tokens_per_shard,
            shuffle_buffer=args.shuffle_buffer,
            seed=args.seed + index,
            count_eod=args.count_eod,
            force=args.force,
            backend=args.backend,
            revision=dataset_sha,
            hf_token=hf_token,
        )
        all_stats[component] = stats
        print(
            f"done {component}: "
            f"train_tokens={stats.train.estimated_tokens} "
            f"valid_tokens={stats.valid.estimated_tokens}",
            flush=True,
        )

    manifest = {
        "dataset": args.dataset,
        "dataset_sha": dataset_sha,
        "tokenizer": args.tokenizer,
        "target_tokens": args.target_tokens,
        "valid_tokens": args.valid_tokens,
        "tokens_per_shard": args.tokens_per_shard,
        "shuffle_buffer": args.shuffle_buffer,
        "seed": args.seed,
        "count_eod": args.count_eod,
        "backend": args.backend,
        "component_token_counts": COMPONENT_TOKEN_COUNTS,
        "train_quotas": train_quotas,
        "valid_quotas": valid_quotas,
        "actual": {
            component: {
                "train": all_stats[component].train.__dict__,
                "valid": all_stats[component].valid.__dict__,
            }
            for component in components
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
