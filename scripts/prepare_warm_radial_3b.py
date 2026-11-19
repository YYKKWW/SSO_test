"""Freeze one warm-K10 radial follow-up from the completed exact 3B source."""
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

BASE_MANIFEST="b6d9a49bfb5e4c4f750e62427b3f9c75ed63d27f95800c4a63bd002810bd7ed2"
OVERLAY_FILES={
    "scripts/direct_top_repair.py", "scripts/direct_top_pgd_pk.py",
    "scripts/run_kernel_lmo_pilot.py", "scripts/pretrain_kernel_lmo_pilot.py",
    "scripts/audit_warm_radial.py", "scripts/test_warm_radial_3b.py",
    "scripts/prepare_warm_radial_3b.py", "slurm/warm_ambient_radial_3b_20261116.sbatch",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def duplicate_runs(projects):
    matches=[]
    for path in projects.glob("*/results/*/launch_config.json"):
        try:
            c=json.loads(path.read_text())
        except (ValueError,OSError):
            continue
        required=dict(KERNEL_STUDY="direct_pk",KERNEL_SOLVER="warm_block8",
                      KERNEL_METHOD="pgd_half",KERNEL_AMBIENT_RETURN="radial",
                      TRAIN_ITER="5723",WIDTH="256",SEED="2345",
                      SPEL_PGD_GAP_THRESHOLD_REL="2e-4",
                      SPEL_PGD_CLUSTER_ITERATIONS="10")
        if all(str(c.get(k))==v for k,v in required.items()):
            matches.append(str(path))
    return matches


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base",type=Path,required=True)
    parser.add_argument("--target",type=Path,required=True)
    parser.add_argument("--overlay",type=Path,required=True)
    parser.add_argument("--submit",action="store_true")
    args=parser.parse_args()
    base,target=args.base.resolve(),args.target.resolve()
    if target==base or target.is_relative_to(base):
        raise RuntimeError("use an independent target, never the exact source")
    overlay_hash=sha(args.overlay)
    if sha(base/"source.sha256")!=BASE_MANIFEST:
        raise RuntimeError("wrong exact reference manifest")
    receipt=target/"results/submission.json"
    if receipt.exists():
        old=json.loads(receipt.read_text())
        if old["overlay_sha256"]!=overlay_hash:
            raise RuntimeError("existing submission has different source")
        print(json.dumps(old,indent=2));return
    duplicates=duplicate_runs(base.parent)
    if duplicates:
        raise RuntimeError("matching run already exists: "+repr(duplicates))
    marker=target/"results/preparation.json"
    if target.exists() and not marker.exists():
        raise RuntimeError("target exists without a preparation record")
    if not target.exists():
        if shutil.disk_usage(base.parent).free < 3*1024**3:
            raise RuntimeError("less than 3 GiB free; do not stage another run")
        target.mkdir(parents=True)
        entries={}
        for line in (base/"source.sha256").read_text().splitlines():
            digest,relative=line.split(maxsplit=1)
            relative=relative.lstrip("*")
            source=(base/relative).resolve()
            if not source.is_relative_to(base) or sha(source)!=digest:
                raise RuntimeError("base source mismatch: "+relative)
            dest=target/relative
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,dest)
            entries[relative]=digest
        changed=[]
        with tarfile.open(args.overlay) as archive:
            members=archive.getmembers()
            if len(members)!=len(OVERLAY_FILES) or {m.name for m in members}!=OVERLAY_FILES:
                raise RuntimeError("unexpected overlay contents")
            for member in members:
                if not member.isfile():
                    raise RuntimeError("overlay must contain regular files only")
                dest=target/member.name
                dest.parent.mkdir(parents=True,exist_ok=True)
                with archive.extractfile(member) as src,dest.open("wb") as out:
                    shutil.copyfileobj(src,out)
                digest=sha(dest)
                if entries.get(member.name)!=digest:
                    changed.append(dict(path=member.name,old_sha256=entries.get(member.name),new_sha256=digest))
                entries[member.name]=digest
        for name in ("logs","results"):
            (target/name).mkdir(exist_ok=True)
        (target/"source.sha256").write_text("".join(f"{digest}  {name}\n" for name,digest in sorted(entries.items())))
        preparation=dict(base=str(base),target=str(target),overlay_sha256=overlay_hash,
            base_manifest_sha256=BASE_MANIFEST,source_manifest_sha256=sha(target/"source.sha256"),
            source_files=len(entries),changed_files=changed,
            created_utc=datetime.now(timezone.utc).isoformat(),reference_job="4434107_2")
        marker.write_text(json.dumps(preparation,indent=2)+"\n")
    preparation=json.loads(marker.read_text())
    if preparation["overlay_sha256"]!=overlay_hash:
        raise RuntimeError("overlay changed since preparation")
    subprocess.run(["sha256sum","--check","--quiet","source.sha256"],cwd=target,check=True)
    reference=json.loads((base/"results/4434107_2/launch_config.json").read_text())
    env=dict(os.environ,PROJECT_DIR=str(target),MEGATRON_PATH=str(target/"Megatron-LM"),
             P2GD_TEST_DEVICE="cpu",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",
             NVIDIA_TF32_OVERRIDE="0")
    for key in ("TRAIN_DATA_PREFIX","VALID_DATA_PREFIX","TOKENIZER_MODEL"):
        env[key]=reference[key]
    python=str(Path.home()/"envs/sso_h20/bin/python")
    patterns=("test_kernel_lmo_h20.py","test_direct_top_repair.py","test_direct_top_pgd_pk.py",
              "test_direct_radial_3b.py","test_warm_radial_3b.py")
    for pattern in patterns:
        subprocess.run([python,"-B","-m","unittest","discover","-s","scripts","-p",pattern,"-v"],
                       cwd=target,env=env,check=True)
    # Capture the real launcher's environment; suppress training and offline audits.
    sys.path.insert(0,str(target/"scripts"))
    captured=[]
    def capture(e,folder,smoke):
        folder.mkdir(parents=True,exist_ok=True)
        if not smoke:captured.append(dict(e))
    with tempfile.TemporaryDirectory() as scratch,patch.dict(os.environ,env):
        import run_kernel_lmo_pilot as launcher
        with patch.dict(os.environ,dict(PROJECT_DIR=scratch,SLURM_ARRAY_JOB_ID="preview",SLURM_ARRAY_TASK_ID="2")), \
             patch("sys.argv",["run_kernel_lmo_pilot.py","--study","direct_pk","--arm","2",
                               "--budget","3b","--ambient-return","radial","--solver","warm_block8"]), \
             patch.object(launcher,"run_training",side_effect=capture), \
             patch.object(launcher.subprocess,"run"):
            launcher.main()
    ignored={"PRETRAIN_ENTRYPOINT","KERNEL_SNAPSHOT_DIR","smoke","processed_tokens",
             "geometry_precision","data_protocol","new_window_routing"}
    changes={k:dict(exact=v,warm=captured[0].get(k)) for k,v in reference.items()
             if k not in ignored and captured[0].get(k)!=v}
    if set(changes)!={"KERNEL_SOLVER","SPEL_PGD_CLUSTER_SPECTRUM_MODE"}:
        raise RuntimeError("unexpected training-config changes: "+repr(changes))
    (target/"results/config_comparison.json").write_text(json.dumps(dict(
        reference_job="4434107_2",changes=changes,ignored_output_metadata=sorted(ignored),
        other_recorded_training_fields_equal=True),indent=2)+"\n")
    if not args.submit:
        print(json.dumps(preparation,indent=2));return
    duplicates=duplicate_runs(base.parent)
    if duplicates:
        raise RuntimeError("matching run appeared before submission: "+repr(duplicates))
    intent=target/"results/submission.intent.json"
    with intent.open("x") as out:
        json.dump(dict(overlay_sha256=overlay_hash,created_utc=datetime.now(timezone.utc).isoformat()),out)
    script="slurm/warm_ambient_radial_3b_20261116.sbatch"
    job=subprocess.check_output(["sbatch","--parsable","--export=ALL",script],
                                cwd=target,env=env,text=True).strip()
    record=dict(**preparation,job_id=job,arm_indices=[2],gpus_per_arm=1,max_concurrent_gpus=1,
        submitted_utc=datetime.now(timezone.utc).isoformat(),seed=2345,width=256,
        train_iterations=5723,processed_tokens=5723*128*4096,solver="warm_block8",iterations=10,
        gap=2e-4,rho=.5,return_rule="radial_both_branches",checkpoints=False,
        cpu_preflight_passed=True,cuda_preflight="inside allocation before smoke and fresh main run",
        training_config_changes=changes,data={k:env[k] for k in ("TRAIN_DATA_PREFIX","VALID_DATA_PREFIX","TOKENIZER_MODEL")})
    receipt.write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps(record,indent=2))


if __name__=="__main__":
    main()
