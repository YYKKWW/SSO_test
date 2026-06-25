#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path


VAL_RE = re.compile(
    r"validation loss(?: at iteration (?P<iter>\d+))?.*?lm loss value: (?P<loss>[0-9.E+-]+)"
)
TRAIN_RE = re.compile(
    r"iteration\s+(?P<iter>\d+)/\s*(?P<total>\d+).*?lm loss: (?P<loss>[0-9.E+-]+)"
)


def parse_log(path: Path):
    last_train_iter = None
    last_train_loss = None
    best_val_iter = None
    best_val_loss = None
    last_val_iter = None
    last_val_loss = None

    if not path.exists():
        return {
            "status": "missing_log",
            "last_train_iter": "",
            "last_train_loss": "",
            "last_val_iter": "",
            "last_val_loss": "",
            "best_val_iter": "",
            "best_val_loss": "",
        }

    text = path.read_text(errors="replace")
    for match in TRAIN_RE.finditer(text):
        last_train_iter = int(match.group("iter"))
        last_train_loss = float(match.group("loss"))
    for match in VAL_RE.finditer(text):
        val_iter = int(match.group("iter") or last_train_iter or 0)
        val_loss = float(match.group("loss"))
        last_val_iter = val_iter
        last_val_loss = val_loss
        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_iter = val_iter

    status = "ok" if best_val_loss is not None else "no_validation_yet"
    return {
        "status": status,
        "last_train_iter": last_train_iter or "",
        "last_train_loss": last_train_loss or "",
        "last_val_iter": last_val_iter or "",
        "last_val_loss": last_val_loss or "",
        "best_val_iter": best_val_iter or "",
        "best_val_loss": best_val_loss or "",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="/workspace/results/optimizer_arena_v2/mup_lr_width_sweep/sweep_manifest.tsv",
    )
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    manifest = Path(args.manifest)
    rows = []
    with manifest.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            row.update(parse_log(Path(row["log_file"])))
            rows.append(row)

    fieldnames = [
        "optimizer",
        "width",
        "lr",
        "job_name",
        "status",
        "last_train_iter",
        "last_train_loss",
        "last_val_iter",
        "last_val_loss",
        "best_val_iter",
        "best_val_loss",
        "log_file",
    ]

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(out)
    else:
        writer = csv.DictWriter(__import__("sys").stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
