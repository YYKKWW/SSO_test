"""Render paired local-study tables and curves from the immutable result JSON."""
import argparse
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input",type=Path)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    data=json.loads(args.input.read_text())
    args.out.mkdir(parents=True,exist_ok=True)
    protocols=data['config']['protocols']; methods=data['config']['methods']
    colors={'MCSD-TP':'#315E9E','A1':'#D18826','A2':'#248D79','A4':'#C84857'}
    titles={'certified_det':'Theory schedule, deterministic',
            'certified_ema':'Theory schedule, bounded-noise EMA',
            'practical_nesterov':'Larger-step Nesterov stress test'}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,3,figsize=(14,7),layout='constrained')
    for col,protocol in enumerate(protocols):
        for method in methods:
            rows=[r for r in data['rows'] if r['protocol']==protocol and r['method']==method]
            known=[r for r in rows if r['fstar'] is not None]
            steps=[p['step'] for p in rows[0]['trace']]
            gaps=[statistics.mean(max(1e-15,r['trace'][i]['loss']-r['fstar']) /
                      max(1e-15,r['trace'][0]['loss']-r['fstar']) for r in known) for i in range(len(steps))]
            moves=[statistics.mean(r['trace'][i]['actual_fro'] for r in rows) for i in range(len(steps))]
            axes[0,col].plot(steps,gaps,color=colors[method],label=method,lw=1.8)
            axes[1,col].plot(steps,moves,color=colors[method],lw=1.8)
        axes[0,col].set_title(titles[protocol],fontsize=11)
        for row in range(2):
            axes[row,col].set_yscale('log'); axes[row,col].grid(alpha=.2)
            axes[row,col].set_xlabel('Iteration')
    axes[0,0].set_ylabel('Mean normalized objective gap')
    axes[1,0].set_ylabel('Mean returned step / radius')
    axes[0,2].legend(frameon=False,fontsize=9)
    fig.savefig(args.out/'paired_trajectories.png',dpi=160)
    plt.close(fig)
    lines=['# Local Step-Aware Hybrid Results','',
      'Exact FP64 spectral-sphere reference; no new H20 jobs. '+
      '[Theory audit](../step_aware_hybrid_review_20261111.md).','',
      '## Protocol','',
      f"{len(data['rows'])} trajectories: 20 fixed instance seeds x five shapes x three protocols x four methods, "
      f"each for {data['config']['steps']} steps. The 400-step screening and this extension are not independent replications.",
      '', 'Shapes: 4x4, 16x12, 12x16, 32x24, 24x32. Initial top multiplicity is 1, 2, 4 or full. '+
      'Isotropic inside/outside/plateau targets have known optima; anisotropic quadratics do not. '+
      'This is a stratified synthetic suite, not a full factorial design or an LLM matrix sample.', '',
      'All methods share initialization, noise samples and schedules within each case. '+
      'Near routing uses the complete h-window, not the existing two-threshold production routing. '+
      'No rank cap is imposed. Exact polar LMOs use a zero extension on roundoff-sized null modes.', '',
      '- Theory protocols: h=0.05R(1+t/100)^(-0.3), alpha=0.2*0.05R/(2sqrt(r))*(1+t/100)^(-0.9). '+
      'EMA fresh-sample weight is 0.5(1+t/100)^(-0.5). Noise has bounded Frobenius norm 0.05R.',
      '- Stress protocol: peak alpha=0.0075R, 20-step warmup and cosine factor to 0.1, h=0.0002R; '+
      'B=0.9B+0.1G and M=0.9B+0.1G with updated B. These are synthetic relative steps, not measured LLM displacements.',
      '- The reference algorithms do not normalize after model correction. A2 is unit-momentum based; '+
      'A1/A4 use a spectral LMO, so equal alpha does not imply equal trial lengths.', '',
      '![Paired objectives and returned step sizes](paired_trajectories.png)', '',
      'Curves average per-instance objective gaps normalized at the first recorded update; '+
      'only the 75 known-optimum instances per method/protocol enter the upper row. '+
      'The lower row uses all 100 instances. No confidence band is implied.', '',
      '## Complete Return and Step Size','',
      '| Protocol | Method | Mean f-f* | Return bound applicable | Uphill steps | Prediction reversed | Mean trial / R | Mean returned / R |',
      '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in data['summary']:
        lines.append(f"| {r['protocol']} | {r['method']} | {r['mean_excess']:.6g} | {r['bound_rate']:.2%} | "
                     f"{r['uphill_rate']:.2%} | {r['prediction_reversal_rate']:.2%} | {r['mean_trial_fro']:.6g} | {r['mean_actual_fro']:.6g} |")
    lines += ['', 'Mean f-f* excludes the anisotropic objectives. A prediction reversal means '+
              'alpha<grad f,D><0 but f(X_next)-f(X)>0, with numerical tolerance 1e-11 R^2. '+
              'Applicability of the return bound alone does not imply objective decrease.', '',
              '## Paired Final Objectives','',
              '| Protocol | Left minus right | Mean difference | Wins / ties / losses |',
              '|---|---|---:|---:|']
    for r in data['paired']:
        lines.append(f"| {r['protocol']} | {r['left']} - {r['right']} | {r['mean_delta']:+.6g} | "
                     f"{r['wins']} / {r['pairs']-r['wins']-r['losses']} / {r['losses']} |")
    lines += ['', 'Differences pair the same objective and initial state, including anisotropic cases. '+
              'They are descriptive; heterogeneous problem scales and repeated seed usage preclude a simple significance claim.', '',
              '## Capacity and Protection','',
              '| Protocol | Method | Window q>4 | Audited rebuilt q>4 | q>8 | Max rebuilt q | A4 acceptance |',
              '|---|---|---:|---:|---:|---:|---:|']
    for r in data['summary']:
        if r['method']=='MCSD-TP': continue
        accept='--' if r['guard_accept_rate'] is None else f"{r['guard_accept_rate']:.2%}"
        lines.append(f"| {r['protocol']} | {r['method']} | {r['window_gt4']:.2%} | {r['rebuilt_q_gt4']:.2%} | "
                     f"{r['rebuilt_q_gt8']:.2%} | {r['max_rebuilt_q']} | {accept} |")
    lines += ['', 'Capacity audits occur at initialization, midpoint and final pre-update state: '+
              '300 per method/protocol. They search q>=q_window, recomputing D for each extension until '+
              '2*alpha*||D||_2<=Gamma. This is a sufficient-bound audit, not necessary geometric capacity '+
              'or a change to the trajectory. Window rates count every step; audit rates count only these sampled states.', '',
              '## Same-State Trial-Length Control','',
              'Only A4 states are sampled. Each candidate is compared with the top-1 MCSD-TP direction '+
              'scaled to the candidate\'s Frobenius trial length. The finite-step model direction is never rescaled. '+
              'The objective is evaluated after each radial return; these counterfactuals do not update the trajectory.', '',
              '| Protocol | Near samples | A4 minus matched top-1, mean change | A4 wins / ties / losses |',
              '|---|---:|---:|---:|']
    amp=[]
    for protocol in protocols:
        deltas=[]
        for row in data['rows']:
            if row['protocol']!=protocol or row['method']!='A4':continue
            for audit in row['amplitude_audit']:
                if audit['q']<=1: continue
                a=next(a for a in audit['rows'] if a['method']=='A4')
                if a['matched_top1_change'] is not None:
                    deltas.append(a['actual_change']-a['matched_top1_change'])
        wins=sum(d < -1e-10 for d in deltas); losses=sum(d > 1e-10 for d in deltas)
        entry=dict(protocol=protocol,n=len(deltas),mean_delta=statistics.mean(deltas) if deltas else None,
                   wins=wins,ties=len(deltas)-wins-losses,losses=losses)
        amp.append(entry)
        lines.append(f"| {protocol} | {len(deltas)} | {entry['mean_delta']:+.6g} | {wins} / {entry['ties']} / {losses} |")
    violations=sum(r['upper_violations']+r['return_bound_violations'] for r in data['summary'])
    lines += ['', '## Verification and Limits','',
        f"- Applicable-bound violations: **{violations}**. Largest sampled radius error: "+
        f"**{max(r['max_radius_error'] for r in data['summary']):.3g}**.",
        '- Each step uses one trial-weight SVD and the transformed post-return cache. '+
        'Exact polar LMOs, spectral-norm diagnostics and capacity/amplitude audits use additional decompositions. '+
        'This is not a one-total-SVD engineering benchmark.',
        f"- Total measured CPU time including diagnostics: {data['elapsed_seconds']:.1f} seconds. "
        'No GPU timing claim is made.',
        '- No online line search, sphere-projection fallback or full-loss acceptance test.',
        '- Finite trajectories do not prove stochastic convergence or universal superiority.',
        '', 'Raw results: `docs/results/local_step_aware_20261111/trajectories_2000.json`. '+
        f"Source SHA256: `{data['source_sha256']}`.", '']
    (args.out/'computed_tables.md').write_text('\n'.join(lines))
    (args.out/'amplitude_summary.json').write_text(json.dumps(amp,indent=2)+'\n')
    print(json.dumps(dict(amplitude=amp,violations=violations),indent=2))


if __name__=='__main__':main()
