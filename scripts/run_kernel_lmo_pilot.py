"""Launch one frozen pilot arm; smoke and main use separate training processes."""
import argparse,json,os,subprocess,time
from pathlib import Path
from kernel_lmo_h20 import ARMS


def training_schedule(budget):
    if budget == "100m":
        return dict(TRAIN_ITER="191",TRAIN_TOKENS="100000000",LR_WARMUP_ITERS="25",
                    LR_DECAY_ITERS="191",EVAL_INTERVAL="50")
    if budget == "3b":
        return dict(TRAIN_ITER="5723",TRAIN_TOKENS="3000000000",LR_WARMUP_ITERS="250",
                    LR_DECAY_ITERS="5723",EVAL_INTERVAL="250")
    raise ValueError("budget must be 100m or 3b")


def arm_spec(arm, study):
    if study == "direct_pk":
        from direct_top_pgd_pk import ARMS as PK_ARMS
        if not 0 <= arm < len(PK_ARMS): raise ValueError("direct-PK arm must be 0 to 5")
        return PK_ARMS[arm]
    if study == "direct_top":
        from direct_top_repair import MODES
        if not 0 <= arm < len(MODES): raise ValueError("direct-top arm must be 0, 1 or 2")
        return MODES[arm], "exact", "spel_pgd_dist"
    if study != "kernel" or not 0 <= arm < len(ARMS):
        raise ValueError("invalid study/arm")
    return ARMS[arm]


def run_training(env,folder,smoke):
    folder.mkdir(parents=True,exist_ok=True)
    config=dict(env); config.update(RUN_ROOT=str(folder),DATA_CACHE_PATH=str(folder/"data_cache"))
    if smoke: config.update(TRAIN_ITER="4",TRAIN_TOKENS="2097152",EVAL_ITERS="1",LR_WARMUP_ITERS="1",LOG_INTERVAL="1")
    if env["OPTIMIZER"]=="spel_pgd_dist":
        config["KERNEL_SNAPSHOT_DIR"]=str(folder/"snapshots")
    recorded={k:v for k,v in config.items() if k in ("OPTIMIZER","SEED","WIDTH","NUM_LAYERS","HEAD_DIM","SEQ_LENGTH","GLOBAL_BATCH","MICRO_BATCH","TRAIN_ITER","TRAIN_TOKENS","LR","MIN_LR","LR_WARMUP_ITERS","LR_DECAY_ITERS","EVAL_INTERVAL","EVAL_ITERS","TRAIN_DATA_PREFIX","VALID_DATA_PREFIX","TOKENIZER_MODEL","PRETRAIN_ENTRYPOINT") or k.startswith(("KERNEL_","SPEL_PGD_"))}
    recorded.update(smoke=smoke,processed_tokens=int(config["TRAIN_ITER"])*128*4096,
        geometry_precision="fp32_highest" if env["OPTIMIZER"]=="spel_pgd_dist" else "reference",
        data_protocol=f"{env.get('KERNEL_TRAINING_BUDGET','100m')} budget on existing 3B indexed source",
        new_window_routing=env["OPTIMIZER"]=="spel_pgd_dist")
    (folder/"launch_config.json").write_text(json.dumps(recorded,indent=2)+"\n")
    start=time.time()
    with (folder/"train.log").open("w") as log:
        process=subprocess.Popen(["bash",str(Path(env["PROJECT_DIR"])/"slurm/spel_olmo_1b_h20.sbatch")],env=config,
                                 stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in process.stdout:
            print(line,end="",flush=True);log.write(line);log.flush()
        code=process.wait()
    (folder/"training_wall_seconds.txt").write_text(str(round(time.time()-start))+"\n")
    if code: raise subprocess.CalledProcessError(code,process.args)
    (folder/"TRAINING_COMPLETED").touch()


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--arm",type=int,required=True)
    parser.add_argument("--study",choices=("kernel","direct_top","direct_pk"),default="kernel")
    parser.add_argument("--ambient-return",choices=("sphere_projection","radial"),
                        default="sphere_projection")
    parser.add_argument("--budget",choices=("100m","3b"),default="100m")
    parser.add_argument("--solver",choices=("exact","warm_block8"),default=None,
                        help="Opt-in spectral override for direct-PK radial arms only")
    parser.add_argument("--near-tp-half",action="store_true",
                        help="Radial control: halve D0 only inside the near window")
    args=parser.parse_args()
    if args.ambient_return != "sphere_projection" and args.study != "direct_pk":
        parser.error("--ambient-return radial requires --study direct_pk")
    if args.budget != "100m" and args.study != "direct_pk":
        parser.error("--budget 3b requires --study direct_pk")
    method,solver,optimizer=arm_spec(args.arm,args.study)
    if args.solver is not None:
        if args.study != "direct_pk" or args.ambient_return != "radial" or method not in ("none","pgd_half","pgd_full"):
            parser.error("--solver requires an uncorrected direct-PK radial arm")
        solver=args.solver
    if args.near_tp_half:
        if args.study != "direct_pk" or method != "none" or args.ambient_return != "radial":
            parser.error("--near-tp-half requires the direct-PK radial control arm")
        method="tp_half"
    root=Path(os.environ["PROJECT_DIR"])
    env=dict(os.environ)
    env.update(OPTIMIZER=optimizer,SEED="2345" if args.study=="direct_pk" else "1234",WIDTH="256",NUM_LAYERS="28",HEAD_DIM="128",SEQ_LENGTH="4096",
        GLOBAL_BATCH="128",MICRO_BATCH="4",GPUS_PER_NODE="1",TP_SIZE="1",PP_SIZE="1",TRAIN_ITER="191",TRAIN_TOKENS="100000000",
        LR="1.5e-2",MIN_LR="1.5e-3",LR_WARMUP_ITERS="25",LR_DECAY_ITERS="191",EVAL_INTERVAL="50",EVAL_ITERS="20",LOG_INTERVAL="10",
        SAVE_CHECKPOINT="0",WEIGHT_DECAY="0.1",LOAD_CHECKPOINT="",KERNEL_METHOD=method,KERNEL_SOLVER=solver,KERNEL_STUDY=args.study,
        SPEL_PGD_BRANCH_MODE="auto",SPEL_PGD_GAP_THRESHOLD_REL="2e-4",SPEL_PGD_CLUSTER_THRESHOLD_REL="2e-4",
        SPEL_PGD_RADIAL_CLUSTER_MODE="adaptive4",SPEL_PGD_PGD_LR_SCALE="1",SPEL_PGD_CLUSTER_DIRECTION_MODE="momentum_fro",
        SPEL_PGD_CLUSTER_NEAR_PROJECTION="cone",SPEL_PGD_CLUSTER_SPECTRUM_MODE="warm_block8" if solver=="warm_block8" else "exact",
        SPEL_PGD_CLUSTER_ITERATIONS="10",SPEL_PGD_PROJECTION_MODE="tp_radial_pgd_exact",SPEL_PGD_POST_RETURN_SPECTRAL_CACHE="1",
        SPEL_PGD_GAP_ESTIMATOR_MODE="exact_svd_gap_only",SPEL_PGD_GAP_PROBE_INTERVAL="1",SPEL_PGD_GAP_AUDIT_INTERVAL="0",
        SPEL_PGD_MAIN_POWER_DTYPE="fp32",SPEL_PGD_POWER_ITERATION_STEPS="10",SPEL_PGD_SIGMA2_POWER_ITERATION_STEPS="10",
        SPEL_PGD_MSIGN_STEPS="8",SPEL_PGD_TANGENT_PROJECT_AFTER_MSIGN="1",SPEL_PGD_NORMALIZE_MSIGN_SPECTRAL="0",
        SPEL_PGD_MATRIX_WEIGHT_DECAY="0",SPEL_PGD_DIRECTION_NORMALIZATION="none",SPEL_PGD_FALLBACK_DIRECTION="pgd",SPEL_PGD_WARM_START_UV="0",
        SPECTRAL_BALL_POWER_ITERATION_STEPS="10",SPECTRAL_BALL_SOLVER_TOLERANCE_F="2e-4",SPECTRAL_BALL_SOLVER_MAX_ITERATIONS="20",
        MUON_BALL_POWER_ITERATION_STEPS="10",MUON_BALL_MSIGN_STEPS="8",JOB_NAME=f"{args.study}_{method}_{solver}_{args.budget}")
    env.update(training_schedule(args.budget),KERNEL_TRAINING_BUDGET=args.budget)
    if args.budget == "3b":
        env.update(KERNEL_SNAPSHOT_STEPS="2,250,1000,3000,5000,5720",KERNEL_SNAPSHOT_MAX="48")
    env["PRETRAIN_ENTRYPOINT"]=str(root/"scripts/pretrain_kernel_lmo_pilot.py") if optimizer=="spel_pgd_dist" else str(root/"Megatron-LM/pretrain_gpt.py")
    if args.study=="direct_pk":
        from direct_top_pgd_pk import RATIOS
        env["KERNEL_AMBIENT_RETURN"]=args.ambient_return
        env["KERNEL_PGD_TRIAL_RATIO"]=str(RATIOS.get(method,0.))
        env["KERNEL_PGD_LENGTH_REFERENCE"]="current_D0_fro_norm_or_sqrt_rank_if_zero"
        if method == "tp_half":
            env["KERNEL_TP_NEAR_RATIO"]="0.5"
        if args.ambient_return == "radial":
            env["JOB_NAME"] += "_radial"
    run_id=f"{env['SLURM_ARRAY_JOB_ID']}_{env['SLURM_ARRAY_TASK_ID']}"
    if args.arm==0 and args.study=="kernel":
        source=Path(env["PREVIOUS_CONFIRMATION_DIR"])
        subprocess.run(["python","-B",str(root/"scripts/audit_kernel_lmo_snapshots.py"),
            str(source/"results/4419145_0/snapshots"),str(source/"results/4419145_1/snapshots"),
            "--output",str(root/"results/real_snapshot_audit.json"),"--device","cuda"],check=True,env=env)
    run_training(env,root/"results"/(run_id+"_smoke"),True)
    folder=root/"results"/run_id
    run_training(env,folder,False)
    if args.study=="direct_pk" and optimizer=="spel_pgd_dist":
        subprocess.run(["python","-B",str(root/"scripts/audit_direct_pk.py"),
            "--snapshots",str(folder/"snapshots"),"--limit","8","--device","cuda",
            "--ambient-return",args.ambient_return,
            "--output",str(folder/"offline_audit.json")],check=True,env=env)
        if solver == "warm_block8":
            subprocess.run(["python","-B",str(root/"scripts/audit_warm_radial.py"),
                "--snapshots",str(folder/"snapshots"),"--limit","8","--device","cuda",
                "--output",str(folder/"warm_spectrum_audit.json")],check=True,env=env)
    elif args.study=="direct_top":
        subprocess.run(["python","-B",str(root/"scripts/local_direct_top_repair.py"),
            "--snapshots",str(folder/"snapshots"),"--limit","8","--skip-synthetic",
            "--device","cuda","--output",str(folder/"offline_audit.json")],check=True,env=env)
    elif optimizer=="spel_pgd_dist":
        subprocess.run(["python","-B",str(root/"scripts/audit_kernel_lmo_snapshots.py"),str(folder/"snapshots"),
            "--per-directory","8","--output",str(folder/"offline_audit.json"),"--device","cuda"],check=True,env=env)
    (folder/"COMPLETED").touch()


if __name__=="__main__": main()
