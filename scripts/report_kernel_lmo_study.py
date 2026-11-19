"""Generate a compact, versionable record from the local paired study."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    data=json.loads(args.input.read_text())
    compact={k:data[k] for k in ("config","source_sha256","base_sha256","elapsed_seconds","summary","pairs")}
    compact["raw_sha256"]=hashlib.sha256(args.input.read_bytes()).hexdigest()
    compact["raw_local_path"]=str(args.input)
    compact["trajectory_count"]=len(data["rows"])
    compact["max_errors"]={k:max(row["maximum"].get(k,0.) for row in data["rows"])
        for k in ("model_error","reference_guard_violation","radius_error","profile_length_error")}
    compact["audit_counts"]={k:sum(row["counts"].get(k,0) for row in data["rows"])
        for k in ("coupling_audits","coupling_bound_violation","complete_bound_violation")}
    lookup={(r["seed"],tuple(r["shape"]),r["protocol"],r["method"]):r for r in data["rows"]}
    additional=[]
    for protocol in data["config"]["protocols"]:
        for a,b in (("K-LMO-0.5","A2-input-1.5"),("K-LMO-1","A2-input-2")):
            differences=[r["final_loss"]-lookup[(r["seed"],tuple(r["shape"]),protocol,b)]["final_loss"]
                for r in data["rows"] if r["method"]==a and r["protocol"]==protocol]
            additional.append(dict(protocol=protocol,method=a,reference=b,pairs=len(differences),
                mean_delta=sum(differences)/len(differences),wins=sum(d < -1e-10 for d in differences),
                ties=sum(abs(d)<=1e-10 for d in differences),losses=sum(d > 1e-10 for d in differences)))
    compact["additional_pairs"]=additional
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"summary.json").write_text(json.dumps(compact,indent=2,allow_nan=False)+"\n")
    lines=["# Generated Paired Results", "", "Generated from the recorded raw JSON; see the narrative report for caveats.", ""]
    for protocol in data["config"]["protocols"]:
        lines += ["## "+protocol,"", "| Method | Mean f-f* | Near rate | Near uphill | Mean trial/R | Mean returned/R |",
                  "|---|---:|---:|---:|---:|---:|"]
        for r in data["summary"]:
            if r["protocol"]!=protocol: continue
            uphill=f"{100*r['near_uphill_rate']:.2f}%" if r['near_rate'] else "n/a"
            lines.append(f"| {r['method']} | {r['mean_excess']:.9g} | {100*r['near_rate']:.3f}% | {uphill} | {r['mean_trial_length']:.8g} | {r['mean_returned_length']:.8g} |")
        lines += ["", "Mean f-f* uses 36 known-optimum instances; paired comparisons below use all 48.", "",
                  "| Method | Reference | Mean paired delta | Wins | Ties | Losses |", "|---|---|---:|---:|---:|---:|"]
        for r in data["pairs"]+additional:
            if r["protocol"]==protocol:
                lines.append(f"| {r['method']} | {r['reference']} | {r['mean_delta']:+.9g} | {r['wins']} | {r['ties']} | {r['losses']} |")
        lines.append("")
    (args.output/"computed_tables.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(trajectories=compact["trajectory_count"],output=str(args.output),
                         errors=compact["max_errors"],audits=compact["audit_counts"])))


if __name__=="__main__":
    main()
