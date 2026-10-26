"""Training-only observer: save bounded samples, never compute exact spectra.

Full spectra are audited in a separate process AFTER training. The call guard
enforces the implementation contract, not a numerical acceptance criterion.
"""
import json
import os
import runpy
import sys
from pathlib import Path

import torch

ROOT = Path(os.environ["MEGATRON_PATH"])
sys.path.insert(0, str(ROOT))
from emerging_optimizers.orthogonalized_optimizers import spectral_cluster_radial as cluster


def snapshot_options(env):
    """Keep the old pilot defaults; long runs explicitly request late samples."""
    steps = {int(s) for s in env.get("CLUSTER_SNAPSHOT_STEPS", "2,50,150,190").split(",") if s}
    cap = int(env.get("CLUSTER_SNAPSHOT_MAX", "24"))
    early = int(env.get("CLUSTER_SNAPSHOT_EARLY", "8"))
    late = int(env.get("CLUSTER_SNAPSHOT_LATE", "0"))
    after = int(env.get("CLUSTER_SNAPSHOT_LATE_AFTER", "1000"))
    if not steps or min(steps) < 1 or cap < 1 or min(early, late, after) < 0:
        raise ValueError("invalid bounded snapshot schedule")
    return steps, cap, early, late, after


def main():
    out = Path(os.environ["CLUSTER_SNAPSHOT_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    original = cluster.compute_cluster_radial_update
    original_svd, original_vals = torch.linalg.svd, torch.linalg.svdvals
    counts = dict(component_updates=0, small_svd=0, full_svd=0, svdvals=0,
                  saved=0, successful_training=False)
    active = False
    scheduled = set()
    early_clusters = 0
    late_clusters = 0
    sample_steps, sample_cap, early_cap, late_cap, late_after = snapshot_options(os.environ)

    def guarded_svd(a, *args, **kwargs):
        if active:
            if max(a.shape[-2:]) > 8:
                counts["full_svd"] += 1
                raise RuntimeError("unexpected full SVD inside ungated Ritz update")
            counts["small_svd"] += 1
        return original_svd(a, *args, **kwargs)

    def guarded_vals(*args, **kwargs):
        if active:
            counts["svdvals"] += 1
            raise RuntimeError("unexpected svdvals inside ungated Ritz update")
        return original_vals(*args, **kwargs)

    def observed(**kw):
        nonlocal active, early_clusters, late_clusters
        step = kw.get("optimizer_step", -1)
        shape = tuple(kw["W"].shape)
        key = (step, shape)
        selected = step in sample_steps and key not in scheduled
        if selected:
            scheduled.add(key)
        active = True
        try:
            result, info = original(**kw)
        finally:
            active = False
        counts["component_updates"] += 1
        early = info.branch == "pgd" and early_clusters < early_cap
        if early:
            early_clusters += 1
        late = info.branch == "pgd" and not early and step >= late_after and late_clusters < late_cap
        if late:
            late_clusters += 1
        cache = kw.get("current_spectral_cache")
        if (selected or early or late) and cache is not None and counts["saved"] < sample_cap:
            current = kw["W"].detach().float()
            returned = current - kw["effective_lr"] * result
            payload = dict(
                step=step, param_name=kw.get("param_name"), component_label=kw.get("component_label"),
                selection="scheduled_shape" if selected else (
                    "first_cluster_calls" if early else "late_cluster_calls"),
                current=current.cpu(), returned=returned.cpu(), momentum=kw["M"].detach().float().cpu(),
                radius=kw["target_radius"], gap_threshold=kw["gap_threshold_rel"],
                effective_lr=float(kw["effective_lr"]),
                cluster_threshold=kw["cluster_threshold_rel"], coefficient=kw["pgd_lr_scale"],
                cluster_mode=kw["cluster_mode"], branch=info.branch,
                direction_mode=kw.get("cluster_direction_mode", "matched_fro"),
                step_multiplier=info.cluster_step_multiplier,
                current_u=cache.u_basis.cpu(), current_v=cache.v_basis.cpu(),
                current_s=cache.s_basis.cpu(), current_gap=cache.rel_gap,
                returned_s=info.next_spectral_cache.s_basis.cpu(),
                returned_gap=info.next_spectral_cache.rel_gap,
            )
            torch.save(payload, out / f"sample_{counts['saved']:03d}.pt")
            counts["saved"] += 1
        return result, info

    cluster.compute_cluster_radial_update = observed
    torch.linalg.svd, torch.linalg.svdvals = guarded_svd, guarded_vals
    try:
        runpy.run_path(str(ROOT / "pretrain_gpt.py"), run_name="__main__")
        counts["successful_training"] = True
    finally:
        cluster.compute_cluster_radial_update = original
        torch.linalg.svd, torch.linalg.svdvals = original_svd, original_vals
        (out / "spectral_call_counts.json").write_text(json.dumps(counts, indent=2)+"\n")
        print("CLUSTER_RITZ_SPECTRAL_CALLS " + json.dumps(counts), flush=True)


if __name__ == "__main__":
    main()
