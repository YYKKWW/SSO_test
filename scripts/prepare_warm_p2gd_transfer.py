"""Freeze a source-only overlay against completed 3B control 4453017_0.

Never copy data/checkpoints or change the completed reference project.
Submission is idempotent; an ambiguous sbatch result requires inspection.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
from unittest.mock import patch

BASE_HASH = "ec80d212d45bf1c7dfcad269e38a515c533874c8eb712fc0e57dd03a363cd186"
FILES = {"scripts/warm_p2gd_transfer.py", "scripts/test_warm_p2gd_transfer.py",
         "scripts/audit_warm_p2gd_transfer.py", "scripts/prepare_warm_p2gd_transfer.py",
         "scripts/run_kernel_lmo_pilot.py", "scripts/pretrain_kernel_lmo_pilot.py",
         "slurm/warm_p2gd_transfer_3b_20261126.sbatch"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, indent=2)+"\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--target", type=Path, required=True)
    p.add_argument("--overlay", type=Path, required=True)
    p.add_argument("--submit", action="store_true")
    args = p.parse_args()
    base, target = args.base.resolve(), args.target.resolve()
    if target == base or target.is_relative_to(base):
        raise RuntimeError("independent project required")
    overlay_hash = sha(args.overlay)
    if sha(base/"source.sha256") != BASE_HASH:
        raise RuntimeError("wrong frozen baseline")
    receipt = target/"results/submission.json"
    if receipt.exists():
        record = json.loads(receipt.read_text())
        if record["overlay_sha256"] != overlay_hash:
            raise RuntimeError("submitted source differs")
        print(json.dumps(record, indent=2))
        return
    marker = target/"results/preparation.json"
    if target.exists() and not marker.exists():
        raise RuntimeError("unmarked target; inspect before retry")
    if not target.exists():
        if shutil.disk_usage(base.parent).free < 4*1024**3:
            raise RuntimeError("less than 4 GiB free")
        with tarfile.open(args.overlay) as tar:
            members = tar.getmembers()
            if len(members) != len(FILES) or {m.name for m in members} != FILES or not all(m.isfile() for m in members):
                raise RuntimeError("unexpected overlay contents")
            entries = {}
            target.mkdir(parents=True)
            for line in (base/"source.sha256").read_text().splitlines():
                digest, name = line.split(maxsplit=1)
                name = name.lstrip("*")
                source = (base/name).resolve()
                if not source.is_relative_to(base) or sha(source) != digest:
                    raise RuntimeError("baseline mismatch: "+name)
                dest = target/name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)
                entries[name] = digest
            for member in members:
                dest = target/member.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as src, dest.open("wb") as out:
                    shutil.copyfileobj(src, out)
                entries[member.name] = sha(dest)
        for name in ("logs", "results"):
            (target/name).mkdir(exist_ok=True)
        (target/"source.sha256").write_text("".join(f"{digest}  {name}\n" for name, digest in sorted(entries.items())))
        save(marker, dict(base=str(base), target=str(target), overlay_sha256=overlay_hash,
                          source_manifest_sha256=sha(target/"source.sha256"),
                          reference_job="4453017_0", changed_files=sorted(FILES)))
    prepared = json.loads(marker.read_text())
    if prepared["overlay_sha256"] != overlay_hash:
        raise RuntimeError("overlay changed since preparation")
    subprocess.run(["sha256sum", "--check", "--quiet", "source.sha256"], cwd=target, check=True)
    old = json.loads((base/"results/4453017_0/launch_config.json").read_text())
    env = dict(os.environ, PROJECT_DIR=str(target), MEGATRON_PATH=str(target/"Megatron-LM"),
               P2GD_TEST_DEVICE="cpu", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", NVIDIA_TF32_OVERRIDE="0")
    for key in ("TRAIN_DATA_PREFIX", "VALID_DATA_PREFIX", "TOKENIZER_MODEL"):
        env[key] = old[key]
        expected = [Path(old[key])] if key == "TOKENIZER_MODEL" else [Path(old[key]+ext) for ext in (".bin", ".idx")]
        if not all(path.exists() for path in expected):
            raise RuntimeError("missing existing data: "+key)
    python = str(Path.home()/"envs/sso_h20/bin/python")
    with (target/"results/cpu_preflight.log").open("w") as log:
        subprocess.run([python, "-B", "-m", "unittest", "discover", "-s", "scripts", "-p", "test_warm*.py", "-v"],
                       cwd=target, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    sys.path.insert(0, str(target/"scripts"))
    configs = []
    with patch.dict(os.environ, env):
        import run_kernel_lmo_pilot as launcher
        for mode in ("cone", "top1", "ambient"):
            def capture(e, folder, smoke):
                folder.mkdir(parents=True, exist_ok=True)
                if not smoke:
                    configs.append(dict(e))
            argv = ["run", "--study", "direct_pk", "--arm", "0", "--budget", "3b", "--solver", "warm_block8",
                    "--ambient-return", "radial", "--warm-p2gd-mode", mode, "--warm-p2gd-ratio", "0.075"]
            with tempfile.TemporaryDirectory() as scratch, \
                 patch.dict(os.environ, dict(PROJECT_DIR=scratch, SLURM_ARRAY_JOB_ID="preview", SLURM_ARRAY_TASK_ID="0")), \
                 patch("sys.argv", argv), patch.object(launcher, "run_training", side_effect=capture), \
                 patch.object(launcher.subprocess, "run"):
                launcher.main()
    ignore = {"PRETRAIN_ENTRYPOINT", "KERNEL_SNAPSHOT_DIR", "smoke", "processed_tokens",
              "geometry_precision", "data_protocol", "new_window_routing"}
    for c in configs:
        changes = {k: [v, c.get(k)] for k, v in old.items() if k not in ignore and c.get(k) != v}
        if changes:
            raise RuntimeError("unexpected baseline changes: "+repr(changes))
        if c["SAVE_CHECKPOINT"] != "0" or c["LOAD_CHECKPOINT"]:
            raise RuntimeError("fresh no-checkpoint training required")
    save(target/"results/config_comparison.json", dict(reference="4453017_0",
        all_shared_recorded_training_fields_equal=True,
        new_modes=[{k:v for k,v in c.items() if k.startswith("KERNEL_WARM_P2GD_")} for c in configs]))
    for path in base.parent.glob("*/results/*/launch_config.json"):
        c = json.loads(path.read_text())
        if c.get("KERNEL_WARM_P2GD_MODE") and c.get("TRAIN_ITER") == "5723" and not path.is_relative_to(target):
            raise RuntimeError("existing warm-transfer 3B run: "+str(path))
    if not args.submit:
        print(json.dumps(prepared, indent=2))
        return
    with (target/"results/submission.intent.json").open("x") as out:
        json.dump(dict(created_utc=datetime.now(timezone.utc).isoformat(), overlay_sha256=overlay_hash), out)
    job = subprocess.check_output(["sbatch", "--parsable", "--export=ALL", "slurm/warm_p2gd_transfer_3b_20261126.sbatch"],
                                  cwd=target, env=env, text=True).strip()
    record = dict(**prepared, job_id=job, submitted_utc=datetime.now(timezone.utc).isoformat(),
                  arms=["cone", "top1", "ambient"], max_concurrent_gpus=2, gpus_per_arm=1,
                  width=256, seed=2345, iterations=5723, processed_tokens=5723*128*4096,
                  lr=.015, gap=2e-4, ratio=.075, capacity=6, block_size=8, block_iterations=10,
                  checkpoints=False, actual_elr_matched=False, cpu_tests_passed=True,
                  cuda_tests_and_four_step_smoke="inside each allocation before fresh main training")
    save(receipt, record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
