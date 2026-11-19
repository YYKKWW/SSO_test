"""Render the published 32-update confirmation summary (matplotlib required)."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    'muon_reference': 'MuonSphere reference',
    'sso_aus': 'SSO / AUS',
    'mcsd_tp_aus': 'MCSD-TP / AUS',
    'mcsd_tp_output': 'MCSD-TP / output',
    'mcsd_output': 'MCSD / output',
    'mcsd_tp_aus_scale0985': 'MCSD-TP / AUS x 0.985',
    'mcsd_tp_aus_scale095': 'MCSD-TP / AUS x 0.95',
    'sso_spectral': 'SSO / spectral',
    'mcsd_tp_spectral': 'MCSD-TP / spectral',
    'sso_spectral_cap_aus': 'SSO / spectral + AUS cap',
    'mcsd_tp_spectral_cap_aus': 'MCSD-TP / spectral + AUS cap',
    'sso_spectral_cap_output': 'SSO / spectral + output cap',
    'mcsd_tp_spectral_cap_output': 'MCSD-TP / spectral + output cap',
}
COLORS = ['#343a40', '#d95f02', '#1b9e77', '#377eb8', '#b44e8b', '#8e6c02', '#6a51a3']


def finish(fig, directory, name):
    for ax in fig.axes:
        ax.spines[['top', 'right']].set_visible(False)
    for ext in ('png', 'pdf'):
        fig.savefig(directory/(name+'.'+ext), dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('summary', type=Path)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    data = json.loads(args.summary.read_text())
    arms = data['arms']; by_name = {a['name']: a for a in arms}
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.titlesize': 12, 'axes.labelsize': 11,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.6), layout='constrained',
                             gridspec_kw={'width_ratios': [1, 1.05]})
    for a, color in zip(arms[:7], COLORS):
        axes[0].plot([0]+list(range(1, 33)), [a['initial_loss']]+[s['loss_after'] for s in a['by_step']],
                     label=LABELS[a['name']], color=color, lw=1.8)
    axes[0].set(xlabel='Updates after the common checkpoint', ylabel='Held-out loss',
                title='Same initial state; 20 fixed evaluation microbatches')
    axes[0].grid(alpha=.18)
    axes[0].legend(loc='upper right', fontsize=9, frameon=False)
    order = list(reversed(arms))
    values = [a['final_delta_vs_muon'] for a in order]
    y = np.arange(len(order))
    axes[1].barh(y, values, color=['#c44e52' if v > 0 else '#249283' for v in values], height=.65)
    axes[1].set_yticks(y, [LABELS[a['name']] for a in order], fontsize=9)
    axes[1].axvline(0, color='#666666', lw=.9)
    for pos, val in zip(y, values):
        axes[1].text(val+(.00065 if val >= 0 else -.00065), pos, f'{val:+.5f}',
                     va='center', ha='left' if val >= 0 else 'right', fontsize=9)
    axes[1].set(xlim=(-.038, .031), xlabel='Final loss minus MuonSphere reference',
                title='All 13 arms at update 32; lower is better')
    axes[1].grid(axis='x', alpha=.18); axes[1].set_axisbelow(True)
    fig.suptitle('Width 256: complete-return step-control confirmation', fontsize=14)
    finish(fig, args.output, 'loss_confirmation')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout='constrained')
    for name, color in zip(['muon_reference', 'sso_aus', 'mcsd_output', 'mcsd_tp_output'], COLORS):
        a = by_name[name]
        for ax, key in zip(axes, ['nesterov_input_angle_to_tangent_degrees', 'direction_angle_to_tangent_degrees']):
            ax.plot([s['iteration'] for s in a['by_step']], [s[key] for s in a['by_step']],
                    label=LABELS[name], color=color, lw=1.6)
    for ax, title in zip(axes, ['Nesterov input before matrix sign', 'Direction before radial return']):
        ax.set(xlabel='Training iteration', ylabel='Mean angle to tangent plane (degrees)', title=title)
        ax.grid(alpha=.18); ax.legend(frameon=False, fontsize=9)
    fig.suptitle('12 sampled matrix components, not all model weights', fontsize=12)
    finish(fig, args.output, 'tangent_angles')

    names = ['sso_spectral', 'mcsd_tp_spectral', 'mcsd_tp_output', 'mcsd_tp_aus_scale095',
             'sso_spectral_cap_output', 'mcsd_tp_spectral_cap_output']
    fig, ax = plt.subplots(figsize=(11, 4.6), layout='constrained')
    for i, (key, label, color) in enumerate([
        ('aus_ratio_to_own_muon', 'AUS (12 components)', '#4c78a8'),
        ('actual_elr_spectral_ratio_to_own_muon', 'Spectral ELR (12 components)', '#f28e2b'),
        ('output_heldout_ratio_to_own_muon', 'Local output ELR (252 components)', '#249283')]):
        vals = [np.mean([s[key] for s in by_name[n]['by_step']]) for n in names]
        ax.bar(np.arange(len(names))+(i-1)*.24, np.array(vals)-1, width=.23, label=label, color=color)
    ax.axhline(0, color='#555555', lw=.8)
    ax.set_xticks(np.arange(len(names)), [LABELS[n].replace(' / ', '\n').replace(' + ', '\n+ ') for n in names], fontsize=9)
    ax.set(ylabel='Mean ratio to own-state Muon target minus 1',
           title='Equal spectral ELR does not equalize angular or output movement')
    ax.grid(axis='y', alpha=.18); ax.set_axisbelow(True)
    ax.legend(ncol=3, fontsize=9, frameon=False, loc='upper right')
    finish(fig, args.output, 'actual_step_ratios')


if __name__ == '__main__':
    main()
