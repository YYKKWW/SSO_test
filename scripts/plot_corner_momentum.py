"""Plot measured snapshot geometry separately from constructed loss tests."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    report=json.loads(args.report.read_text())
    near=[s for s in report["real_states"] if s["cluster_q"]>1]
    plt.rcParams.update({"pdf.fonttype":42,"font.size":10})
    fig,axs=plt.subplots(2,2,figsize=(12.8,8),layout="constrained")
    colors=dict(mcsd="#355d8a",ambient="#c85c48",cone="#188577")
    x=np.arange(len(near))
    labels=[str(s["step"])+"\n"+"x".join(map(str,s["shape"])) for s in near]
    for ax,key,title in ((axs[0,0],"angle_degrees","Real snapshots: global angle to cluster cone"),
                         (axs[0,1],"top_block_angle_degrees","Same snapshots: angle within top symmetric block")):
        ax.plot(x,[s["negative_momentum_cluster_cone"][key] for s in near],"o",color=colors["ambient"],label="Negative Nesterov input -M")
        ax.plot(x,[s["positive_momentum_cluster_cone"][key] for s in near],"x",color=colors["mcsd"],label="Positive Nesterov input M")
        ax.set_xticks(x,labels,fontsize=7.5)
        ax.set_ylabel("Angle (degrees)")
        ax.set_xlabel("Training step / matrix shape (9 selected near states)")
        ax.set_title(title,fontsize=11)
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False,fontsize=8)
    methods=["d0","half_d0","ambient_050","top1_050","cluster_cone_050"]
    labels=["MCSD-TP","0.5 D0","Ambient\n0.5","Top-1\n0.5","Cone\n0.5"]
    selected={r["method"]:r for r in report["real_near_summary"]}
    ax=axs[1,0]
    ax.bar(np.arange(5),[selected[m]["mean_return_rotation_degrees"] for m in methods],
           color=[colors["mcsd"],colors["mcsd"],colors["ambient"],"#b38a31",colors["cone"]],width=.6)
    ax.set_xticks(np.arange(5),labels)
    ax.set_ylabel("Mean return rotation (degrees)")
    ax.set_title("Real snapshots: direction change from radial return",fontsize=11)
    ax=axs[1,1]
    methods=["half_d0","ambient_050","top1_050","cluster_cone_050","ambient_050_project"]
    labels=["0.5 D0","Ambient\nradial","Top-1\nradial","Cone\nradial","Ambient\nproject"]
    for offset,key,label,color in ((-.18,"toy_corner_aligned_summary","M = true gradient",colors["cone"]),
                                   (.18,"toy_corner_noisy_summary","cos(M, gradient) = 0.5",colors["ambient"])):
        s={r["method"]:r for r in report[key]}
        ax.bar(np.arange(5)+offset,[100*s[m]["quadratic_loss_increase_count"]/s[m]["states"] for m in methods],
               width=.34,label=label,color=color)
    ax.set_xticks(np.arange(5),labels)
    ax.set_ylabel("True quadratic loss increased (%)")
    ax.set_title("Synthetic exact corners: 72 paired cases per condition",fontsize=11)
    ax.legend(frameon=False,fontsize=8)
    for ax in axs.flat:
        ax.grid(axis="y",color="#e2e5e8",linewidth=.6)
        ax.set_axisbelow(True)
        ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("Cone geometry and complete returns: snapshot audit versus toy loss",fontsize=14)
    args.output.mkdir(parents=True,exist_ok=True)
    for ext in ("png","pdf"):
        fig.savefig(args.output/f"corner_momentum_audit.{ext}",dpi=200,facecolor="white")
    plt.close(fig)


if __name__=="__main__":
    main()
