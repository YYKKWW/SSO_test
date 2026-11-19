"""Publish compact endpoint evidence from locally archived logs, never weights."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
INPUTS = {
    '4442918_2': 'warm_ambient_radial_3b_20261116/train.log',
    **{f'4427402_{i}': f'direct_top_repair_20261112/h20/results/4427402_{i}/train.log' for i in range(3)},
    '4426353_3': 'kernel_lmo_h20_20261112/final_4426353/exact/train.log',
    '4426353_5': 'kernel_lmo_h20_20261112/final_4426353/warm/train.log',
}


def main():
    rows = []
    for job, relative in INPUTS.items():
        path = ROOT/'docs/results'/relative
        raw = path.read_bytes()
        text = raw.decode()
        vals = [dict(step=int(step), loss=float(loss)) for step, loss in re.findall(
            r'validation loss at iteration (\d+)(?: on validation set)? \| lm loss value: ([\d.Ee+-]+)', text)]
        assert vals, job
        counts = {}
        for kind in ('nan', 'skipped'):
            found = [int(v) for v in re.findall(r'number of '+kind+r' iterations:\s*(\d+)', text)]
            counts['max_reported_'+kind] = max(found) if found else None
        rows.append(dict(job=job, local_log=str(path.relative_to(ROOT)),
            log_sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw),
            validation=vals, final_validation=vals[-1], health=counts))
    report = dict(published_on='2026-11-19',
        scope='Compact extracts, not a new evaluation; bulk source logs remain local.', rows=rows)
    output = ROOT/'docs/results/published_log_summaries_20261119.json'
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    for row in rows:
        print(row['job'], row['final_validation'], row['health'])


if __name__ == '__main__':
    main()
