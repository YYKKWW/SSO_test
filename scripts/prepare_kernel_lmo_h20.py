"""Server-side frozen-copy preparation and idempotent Slurm submission."""
import argparse
from datetime import datetime, timezone
import hashlib,json,os,shutil,subprocess,tarfile
from pathlib import Path


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_arms(value):
    try:
        arms = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("use comma-separated arm indices") from error
    if not arms or tuple(sorted(set(arms))) != arms or any(i < 0 or i > 7 for i in arms):
        raise argparse.ArgumentTypeError("arms must be unique increasing indices from 0 to 7")
    return arms


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base",type=Path,required=True)
    parser.add_argument("--target",type=Path,required=True)
    parser.add_argument("--overlay",type=Path,required=True)
    parser.add_argument("--submit",action="store_true")
    parser.add_argument("--arms",type=parse_arms,default=tuple(range(8)))
    parser.add_argument("--max-concurrent",type=int,choices=(1,2),default=2)
    parser.add_argument("--study",choices=("kernel","direct_top","direct_pk"),default="kernel")
    parser.add_argument("--budget",choices=("100m","3b"),default="100m")
    parser.add_argument("--ambient-return",choices=("sphere_projection","radial"),default="sphere_projection")
    args=parser.parse_args();target=args.target;overlay_hash=sha(args.overlay)
    if args.study != "direct_pk" and (args.budget != "100m" or args.ambient_return != "sphere_projection"):
        parser.error("budget/ambient-return overrides require direct_pk")
    if args.study=="direct_top" and any(i>2 for i in args.arms):
        parser.error("direct_top requires explicit --arms within 0,1,2")
    if args.study=="direct_pk" and any(i>5 for i in args.arms):
        parser.error("direct_pk requires explicit --arms within 0 to 5")
    receipt=target/"results/submission.json"
    if receipt.exists():
        old=json.loads(receipt.read_text())
        if old["overlay_sha256"]!=overlay_hash: raise RuntimeError("existing submission has different source")
        if tuple(old.get("arm_indices",range(8)))!=args.arms or old["max_concurrent_gpus"]!=args.max_concurrent:
            raise RuntimeError("existing submission has different arm selection")
        if old.get("study","kernel")!=args.study: raise RuntimeError("existing submission has different study")
        if old.get("budget","100m")!=args.budget or old.get("ambient_return","sphere_projection")!=args.ambient_return:
            raise RuntimeError("existing submission has different budget or return")
        print(json.dumps(old,indent=2));return
    marker=target/"results/preparation.json"
    if target.exists() and not marker.exists(): raise RuntimeError("target already exists without preparation record")
    if not target.exists():
        target.mkdir(parents=True)
        entries=[]
        for line in (args.base/"source.sha256").read_text().splitlines():
            digest,relative=line.split(maxsplit=1);relative=relative.lstrip("*")
            source=(args.base/relative).resolve()
            if not source.is_relative_to(args.base.resolve()): raise ValueError(relative)
            if sha(source)!=digest: raise RuntimeError("base manifest mismatch: "+relative)
            dest=target/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
            entries.append(relative)
        with tarfile.open(args.overlay) as archive:
            for member in archive.getmembers():
                if not member.isfile(): continue
                dest=(target/member.name).resolve()
                if not dest.is_relative_to(target.resolve()) or member.name in entries:
                    raise RuntimeError("unsafe or overwriting overlay: "+member.name)
                dest.parent.mkdir(parents=True,exist_ok=True)
                with archive.extractfile(member) as src,dest.open("wb") as out: shutil.copyfileobj(src,out)
                entries.append(member.name)
        for directory in ("logs","results"): (target/directory).mkdir(exist_ok=True)
        manifest="".join(f"{sha(target/path)}  {path}\n" for path in sorted(entries))
        (target/"source.sha256").write_text(manifest)
        preparation=dict(base=str(args.base),base_manifest_sha256=sha(args.base/"source.sha256"),
            target=str(target),overlay_sha256=overlay_hash,source_manifest_sha256=sha(target/"source.sha256"),
            source_files=len(entries),created_utc=datetime.now(timezone.utc).isoformat(),
            budget=args.budget,ambient_return=args.ambient_return)
        marker.write_text(json.dumps(preparation,indent=2)+"\n")
    preparation=json.loads(marker.read_text())
    if preparation["overlay_sha256"]!=overlay_hash: raise RuntimeError("overlay changed")
    if preparation.get("budget","100m")!=args.budget or preparation.get("ambient_return","sphere_projection")!=args.ambient_return:
        raise RuntimeError("prepared source has a different budget or return")
    config=json.loads((args.base/"results/4419145_0/launch_config.json").read_text())
    env=dict(os.environ,PROJECT_DIR=str(target),PREVIOUS_CONFIRMATION_DIR=str(args.base),
             MEGATRON_PATH=str(target/"Megatron-LM"),P2GD_TEST_DEVICE="cpu",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",
             KERNEL_TRAINING_BUDGET=args.budget,KERNEL_AMBIENT_RETURN=args.ambient_return)
    for key in ("TRAIN_DATA_PREFIX","VALID_DATA_PREFIX","TOKENIZER_MODEL"):
        env[key]=config[key]
    python=str(Path.home()/"envs/sso_h20/bin/python")
    patterns=["test_kernel_lmo_h20.py"]
    if args.study in ("direct_top","direct_pk"): patterns.append("test_direct_top_repair.py")
    if args.study=="direct_pk": patterns.append("test_direct_top_pgd_pk.py")
    if args.budget=="3b": patterns.append("test_direct_radial_3b.py")
    for pattern in patterns:
        subprocess.run([python,"-B","-m","unittest","discover","-s","scripts","-p",pattern,"-v"],
                       cwd=target,env=env,check=True)
    if not args.submit:
        print(json.dumps(preparation,indent=2));return
    selection=",".join(map(str,args.arms))+f"%{args.max_concurrent}"
    script={"kernel":"slurm/kernel_lmo_pilot_100m_20261112.sbatch",
            "direct_top":"slurm/direct_top_repair_100m_20261112.sbatch",
            "direct_pk":"slurm/direct_top_pgd_pk_100m_20261112.sbatch"}[args.study]
    if args.budget=="3b": script="slurm/direct_ambient_radial_3b_20261113.sbatch"
    job=subprocess.check_output(["sbatch","--parsable","--export=ALL",f"--array={selection}",script],
                                cwd=target,env=env,text=True).strip()
    submitted=dict(**preparation,job_id=job,study=args.study,submitted_utc=datetime.now(timezone.utc).isoformat(),
                   arms=len(args.arms),arm_indices=list(args.arms),max_concurrent_gpus=args.max_concurrent,
                   gpus_per_arm=1,seed=2345 if args.study=="direct_pk" else 1234,
                   train_iterations=5723 if args.budget=="3b" else 191,
                   processed_tokens=(5723 if args.budget=="3b" else 191)*128*4096,
                   checkpoints=False,cpu_preflight_passed=True,cuda_preflight="runs inside each allocation",
                   data={k:env[k] for k in ("TRAIN_DATA_PREFIX","VALID_DATA_PREFIX","TOKENIZER_MODEL")})
    receipt.write_text(json.dumps(submitted,indent=2)+"\n")
    print(json.dumps(submitted,indent=2))


if __name__=="__main__": main()
