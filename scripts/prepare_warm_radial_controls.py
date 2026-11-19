"""Freeze and submit only the two missing controls for warm radial job 4442918_2."""
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

BASE_MANIFEST = "9617f0aa4c6df6219115789ac295248dda30dcbbd449308d47d858f192f479d6"
FILES = {
    "scripts/direct_top_pgd_pk.py", "scripts/run_kernel_lmo_pilot.py",
    "scripts/test_warm_radial_controls.py", "scripts/prepare_warm_radial_controls.py",
    "slurm/warm_radial_controls_3b_20261117.sbatch",
}
PATTERNS = ("test_kernel_lmo_h20.py", "test_direct_top_repair.py", "test_direct_top_pgd_pk.py",
            "test_direct_radial_3b.py", "test_warm_radial_3b.py", "test_warm_radial_controls.py")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def duplicates(projects):
    matches = []
    for path in projects.glob("*/results/*/launch_config.json"):
        try:
            c = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        required = dict(KERNEL_STUDY="direct_pk", KERNEL_SOLVER="warm_block8",
                        KERNEL_AMBIENT_RETURN="radial", TRAIN_ITER="5723", WIDTH="256",
                        SEED="2345", SPEL_PGD_GAP_THRESHOLD_REL="2e-4",
                        SPEL_PGD_CLUSTER_ITERATIONS="10")
        if c.get("KERNEL_METHOD") in ("none", "tp_half") and all(str(c.get(k)) == v for k, v in required.items()):
            matches.append(str(path))
    return matches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    base, target = args.base.resolve(), args.target.resolve()
    if target == base or target.is_relative_to(base):
        raise RuntimeError("independent target required")
    overlay_hash = sha(args.overlay)
    if sha(base/"source.sha256") != BASE_MANIFEST:
        raise RuntimeError("wrong frozen warm reference")
    receipt = target/"results/submission.json"
    if receipt.exists():
        record = json.loads(receipt.read_text())
        if record["overlay_sha256"] != overlay_hash:
            raise RuntimeError("submission already exists with different source")
        print(json.dumps(record, indent=2))
        return
    if duplicates(base.parent):
        raise RuntimeError("matching control already exists: " + repr(duplicates(base.parent)))
    marker = target/"results/preparation.json"
    if target.exists() and not marker.exists():
        raise RuntimeError("target exists without preparation; inspect it before retrying")
    if not target.exists():
        if shutil.disk_usage(base.parent).free < 4*1024**3:
            raise RuntimeError("less than 4 GiB free")
        with tarfile.open(args.overlay) as archive:
            members = archive.getmembers()
            if len(members) != len(FILES) or {m.name for m in members} != FILES or not all(m.isfile() for m in members):
                raise RuntimeError("unexpected overlay contents")
            entries = {}
            target.mkdir(parents=True)
            for line in (base/"source.sha256").read_text().splitlines():
                digest, relative = line.split(maxsplit=1)
                relative = relative.lstrip("*")
                source = (base/relative).resolve()
                if not source.is_relative_to(base) or sha(source) != digest:
                    raise RuntimeError("base source mismatch: " + relative)
                dest = target/relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)
                entries[relative] = digest
            changed = []
            for member in members:
                dest = target/member.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as src, dest.open("wb") as out:
                    shutil.copyfileobj(src, out)
                digest = sha(dest)
                changed.append(dict(path=member.name, old_sha256=entries.get(member.name), new_sha256=digest))
                entries[member.name] = digest
        for name in ("logs", "results"):
            (target/name).mkdir(exist_ok=True)
        (target/"source.sha256").write_text("".join(f"{digest}  {name}\n" for name, digest in sorted(entries.items())))
        marker.write_text(json.dumps(dict(base=str(base), target=str(target), overlay_sha256=overlay_hash,
            base_manifest_sha256=BASE_MANIFEST, source_manifest_sha256=sha(target/"source.sha256"),
            source_files=len(entries), changed_files=changed, reference_job="4442918_2",
            created_utc=datetime.now(timezone.utc).isoformat()), indent=2)+"\n")
    prepared = json.loads(marker.read_text())
    if prepared["overlay_sha256"] != overlay_hash:
        raise RuntimeError("overlay changed since preparation")
    subprocess.run(["sha256sum", "--check", "--quiet", "source.sha256"], cwd=target, check=True)
    reference = json.loads((base/"results/4442918_2/launch_config.json").read_text())
    env = dict(os.environ, PROJECT_DIR=str(target), MEGATRON_PATH=str(target/"Megatron-LM"),
               P2GD_TEST_DEVICE="cpu", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", NVIDIA_TF32_OVERRIDE="0")
    for key in ("TRAIN_DATA_PREFIX", "VALID_DATA_PREFIX", "TOKENIZER_MODEL"):
        env[key] = reference[key]
    python = str(Path.home()/"envs/sso_h20/bin/python")
    for pattern in PATTERNS:
        with (target/"results"/(pattern+".cpu.log")).open("w") as log:
            subprocess.run([python, "-B", "-m", "unittest", "discover", "-s", "scripts", "-p", pattern, "-v"],
                           cwd=target, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    sys.path.insert(0, str(target/"scripts"))
    comparisons = []
    with patch.dict(os.environ, env):
        import run_kernel_lmo_pilot as launcher
        for arm, method in ((0, "none"), (1, "tp_half")):
            captured = []
            def capture(e, folder, smoke):
                folder.mkdir(parents=True, exist_ok=True)
                if not smoke:
                    captured.append(dict(e))
            argv = ["run_kernel_lmo_pilot.py", "--study", "direct_pk", "--arm", "0",
                    "--budget", "3b", "--ambient-return", "radial", "--solver", "warm_block8"]
            if arm == 1:
                argv.append("--near-tp-half")
            with tempfile.TemporaryDirectory() as scratch, \
                 patch.dict(os.environ, dict(PROJECT_DIR=scratch, SLURM_ARRAY_JOB_ID="preview", SLURM_ARRAY_TASK_ID=str(arm))), \
                 patch("sys.argv", argv), patch.object(launcher, "run_training", side_effect=capture), \
                 patch.object(launcher.subprocess, "run"):
                launcher.main()
            c = captured[0]
            ignored = {"PRETRAIN_ENTRYPOINT", "KERNEL_SNAPSHOT_DIR", "smoke", "processed_tokens",
                       "geometry_precision", "data_protocol", "new_window_routing"}
            changes = {k:dict(reference=v, control=c.get(k)) for k, v in reference.items()
                       if k not in ignored and c.get(k) != v}
            if set(changes) != {"KERNEL_METHOD", "KERNEL_PGD_TRIAL_RATIO"} or c["KERNEL_METHOD"] != method:
                raise RuntimeError("unexpected configuration differences: " + repr(changes))
            if c["SAVE_CHECKPOINT"] != "0" or c["LOAD_CHECKPOINT"] != "":
                raise RuntimeError("fresh training with checkpoints disabled required")
            comparisons.append(dict(arm=arm, method=method, changes=changes,
                tp_near_trial_ratio=c.get("KERNEL_TP_NEAR_RATIO"), other_recorded_training_fields_equal=True))
    (target/"results/config_comparison.json").write_text(json.dumps(comparisons, indent=2)+"\n")
    if not args.submit:
        print(json.dumps(prepared, indent=2))
        return
    if duplicates(base.parent):
        raise RuntimeError("matching control appeared before submission")
    # Exclusive intent prevents an ambiguous sbatch response from causing a duplicate retry.
    with (target/"results/submission.intent.json").open("x") as out:
        json.dump(dict(overlay_sha256=overlay_hash, created_utc=datetime.now(timezone.utc).isoformat()), out)
    job = subprocess.check_output(["sbatch", "--parsable", "--export=ALL",
        "slurm/warm_radial_controls_3b_20261117.sbatch"], cwd=target, env=env, text=True).strip()
    record = dict(**prepared, job_id=job, arm_indices=[0, 1], gpus_per_arm=1, max_concurrent_gpus=2,
        submitted_utc=datetime.now(timezone.utc).isoformat(), seed=2345, width=256,
        train_iterations=5723, processed_tokens=5723*128*4096, solver="warm_block8", iterations=10,
        gap=2e-4, return_rule="radial_both_branches", checkpoints=False, cpu_preflight_passed=True,
        cuda_preflight="inside allocation before smoke and fresh main training", config_comparison=comparisons,
        data={k:env[k] for k in ("TRAIN_DATA_PREFIX", "VALID_DATA_PREFIX", "TOKENIZER_MODEL")})
    receipt.write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
