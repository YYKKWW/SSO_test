"""Install the research adapter only in this training process and observe it."""
import json
import os
from pathlib import Path
import runpy

import torch
from kernel_lmo_h20 import PilotAdapter, pilot_dispatch


def main():
    out=Path(os.environ["KERNEL_SNAPSHOT_DIR"]); out.mkdir(parents=True,exist_ok=True)
    if os.environ.get("KERNEL_STUDY", "kernel") == "direct_pk":
        from direct_top_pgd_pk import DirectPKAdapter
        adapter=DirectPKAdapter(os.environ["KERNEL_METHOD"],os.environ["KERNEL_SOLVER"],
            ambient_return=os.environ.get("KERNEL_AMBIENT_RETURN","sphere_projection"))
    elif os.environ.get("KERNEL_STUDY", "kernel") == "direct_top":
        from direct_top_repair import DirectTopAdapter
        adapter=DirectTopAdapter(os.environ["KERNEL_METHOD"],os.environ["KERNEL_SOLVER"])
    else:
        adapter=PilotAdapter(os.environ["KERNEL_METHOD"],os.environ["KERNEL_SOLVER"])
    saved=early=0; scheduled=set(); successful=False
    snapshot_steps={int(s) for s in os.environ.get("KERNEL_SNAPSHOT_STEPS","2,50,100,190").split(",")}
    snapshot_max=int(os.environ.get("KERNEL_SNAPSHOT_MAX","32"))
    def observed(**kwargs):
        nonlocal saved,early
        result=adapter(**kwargs)
        state=adapter.last
        step=kwargs.get("optimizer_step",-1)
        key=(step,tuple(kwargs["W"].shape))
        scheduled_sample=step in snapshot_steps and key not in scheduled
        if scheduled_sample: scheduled.add(key)
        near_sample=state is not None and state["record"]["near"] and early<8
        if near_sample: early+=1
        if state is not None and saved<snapshot_max and (scheduled_sample or near_sample):
            data={k:v.detach().cpu() if isinstance(v,torch.Tensor) else v
                  for k,v in state.items() if k!="cache"}
            data.update(step=step,param_name=kwargs.get("param_name"),component_label=kwargs.get("component_label"),
                        selection="scheduled_shape" if scheduled_sample else "early_near")
            torch.save(data,out/f"sample_{saved:03d}.pt"); saved+=1
        if step>0 and step%10==0 and key not in log_keys:
            log_keys.add(key)
            if tuple(kwargs["W"].shape)==first_shape[0]:
                print("KERNEL_LMO_DIAGNOSTICS "+json.dumps(adapter.report()),flush=True)
        if first_shape[0] is None: first_shape[0]=tuple(kwargs["W"].shape)
        return result
    log_keys=set(); first_shape=[None]
    try:
        with pilot_dispatch(observed):
            runpy.run_path(str(Path(os.environ["MEGATRON_PATH"])/"pretrain_gpt.py"),run_name="__main__")
        successful=True
    finally:
        report=adapter.report(); report.update(successful_training=successful,snapshots=saved)
        (out/"pilot_statistics.json").write_text(json.dumps(report,indent=2)+"\n")
        print("KERNEL_LMO_FINAL "+json.dumps(report),flush=True)
        if successful:
            c=report["counts"]
            assert c["trial_solves"]==c["updates"]>0,c
            assert c.get("full_svd",0)+c.get("small_ritz_svd",0)==c["trial_solves"]+c.get("bootstrap_solves",0),c
            if adapter.solver == "warm_block8":
                assert c.get("full_svd",0) == 0,c


if __name__=="__main__": main()
