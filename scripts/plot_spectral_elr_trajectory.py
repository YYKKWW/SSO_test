"""Plot the six common-AUS trajectories and same-state return audit."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {1.0:"#176ba0", .985:"#dc8729", .95:"#24957a"}


def finish(fig, output, name):
    for ax in fig.axes:
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(alpha=.2)
    for extension in ("png","svg"):
        fig.savefig(output/(name+"."+extension),dpi=180,bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary",type=Path)
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    data = json.loads(args.summary.read_text())
    args.output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.size":10,"font.family":"DejaVu Sans","axes.titlesize":12})
    arms = data["arms"]
    fig,axes = plt.subplots(1,2,figsize=(13.5,4.8),layout="constrained")
    for i,arm in enumerate(arms):
        steps = [r["iteration"]-252 for r in arm["by_step"]]
        label = ("Muon" if i < 3 else "MCSD-TP")+f" x {arm['multiplier']:g}"
        axes[0].plot([0]+steps,[arm["initial_loss"]]+[r["loss_after"] for r in arm["by_step"]],
            color=COLORS[arm["multiplier"]],ls="-" if i < 3 else "--",lw=1.5,label=label)
    for i in range(3):
        delta = [a["loss_after"]-b["loss_after"] for a,b in zip(arms[i+3]["by_step"],arms[i]["by_step"])]
        axes[1].plot(range(1,65),delta,color=COLORS[arms[i]["multiplier"]],lw=1.7,
                     label=f"AUS multiplier {arms[i]['multiplier']:g}")
    axes[0].set(xlabel="Updates after checkpoint 252",ylabel="Held-out loss",title="Same shared AUS targets; 20 fixed evaluation microbatches")
    axes[0].legend(frameon=False,ncol=2,fontsize=8.5)
    axes[1].axhline(0,color="#555555",lw=.8)
    axes[1].set(xlabel="Updates after checkpoint 252",ylabel="MCSD-TP loss minus matched Muon loss",
                title="Positive values favor Muon at the same multiplier")
    axes[1].legend(frameon=False,fontsize=9)
    finish(fig,args.output,"loss_trajectories")

    fig,axes = plt.subplots(1,2,figsize=(12,4.5),layout="constrained")
    keys = ["master_aus_ratio","bf16_aus_ratio","frobenius_ratio","spectral_ratio","output_ratio"]
    labels = ["Master AUS\n252 components","BF16 AUS\n252 components","Frobenius step\n12 components",
              "Spectral step\n12 components","Local output\n252 components"]
    for i in range(3):
        a,b = arms[i+3],arms[i]
        ratios = [a["mean_ratios_to_reference_trajectory"][k]/b["mean_ratios_to_reference_trajectory"][k]-1 for k in keys]
        axes[0].bar([j+(i-1)*.24 for j in range(len(keys))],[100*r for r in ratios],width=.23,
                    color=COLORS[a["multiplier"]],label=f"x {a['multiplier']:g}")
    axes[0].set_xticks(range(len(keys)),labels,fontsize=8)
    axes[0].set(ylabel="Difference between mean reference ratios (%)",title="Matched AUS does not match every geometric measure")
    axes[0].legend(frameon=False)
    names = ["Muon\nx1","Muon\nx0.985","Muon\nx0.95","MCSD-TP\nx1","MCSD-TP\nx0.985","MCSD-TP\nx0.95"]
    for offset,stat,color in ((-.16,"median","#176ba0"),(.16,"p95","#d66c45")):
        axes[1].bar([i+offset for i in range(6)],[100*a["return_residual"][stat] for a in arms],
                    width=.3,label=stat,color=color)
    axes[1].set_xticks(range(6),names,fontsize=8)
    axes[1].set(ylabel="Relative first-order return residual (%)",title="Each arm's own states; 12 components x 64 updates")
    axes[1].legend(frameon=False)
    finish(fig,args.output,"geometry_measures")

    fig,axes = plt.subplots(1,2,figsize=(12,4.4),layout="constrained")
    for i in (0,2,3,5):
        arm = arms[i]
        label = ("Muon" if i < 3 else "MCSD-TP")+f" x {arm['multiplier']:g}"
        for ax,key in zip(axes,("gap_median","perturbation_over_gap_median")):
            ax.plot([r["iteration"]-252 for r in arm["by_step"]],[r[key] for r in arm["by_step"]],
                color=COLORS[arm["multiplier"]],ls="-" if i < 3 else "--",lw=1.7,label=label)
    for ax in axes:
        ax.set(xlabel="Updates after checkpoint 252",yscale="log")
        ax.legend(frameon=False,fontsize=9)
    axes[0].set(ylabel="Median relative top-two gap",title="Twelve sampled components at each step")
    axes[1].set(ylabel="Median perturbation norm / absolute gap",title="Diagnostic, not an error certificate")
    finish(fig,args.output,"gap_conditioning")

    fig,ax = plt.subplots(figsize=(7.5,4.6),layout="constrained")
    for method,label,color in (("muon_common","Muon","#176ba0"),("sso_aus","SSO","#dc8729"),
                               ("mcsd_aus","MCSD","#a7558d"),("mcsd_tp_aus","MCSD-TP","#24957a")):
        selected = [r for r in data["audit"] if r["method"] == method and r["policy"] == "same_returned_aus"]
        selected.sort(key=lambda r:r["fraction"])
        ax.plot([r["fraction"] for r in selected],[100*r["metrics"]["return_residual"]["median"] for r in selected],
                marker="o",lw=1.8,color=color,label=label)
    ax.set(xticks=[.25,.5,1.],xlabel="Coefficient fraction after full-step AUS matching",
        ylabel="Median relative return residual (%)",title="Same-state replay: 24 snapshots from the Muon reference only")
    ax.legend(frameon=False,ncol=2)
    finish(fig,args.output,"same_state_returns")


if __name__ == "__main__":
    main()
