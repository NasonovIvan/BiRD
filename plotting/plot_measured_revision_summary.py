#!/usr/bin/env python3
"""Export four one-column PDF plots of the completed primary-partition results.

Uses the same serif typography and semantic colors as plot_cross_model.py.
PNG companions are previews; the manuscript includes the vector PDFs.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Importing the original plot module applies the shared publication rcParams.
from plot_cross_model import C_POSITIVE, C_REVERSE, VALID_ANSWER_THRESHOLD

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'plotting/_results/extended_validation/analysis/statistics.json'
OUT = ROOT / 'article/figures'
MODELS = ['gemma2_9b', 'gemma3_12b', 'llama31_8b', 'llama32_3b',
          'qwen25_7b', 'ministral_3_8b']
NAMES = ['Gemma-2-9B', 'Gemma-3-12B', 'Llama-3.1-8B', 'Llama-3.2-3B',
         'Qwen-2.5-7B', 'Ministral-3-8B']
GAMES = ['dictator', 'ultimatum_proposer', 'ultimatum_responder', 'trust',
         'public_goods', 'prisoner_dilemma']
LABELS = ['Given', 'Offered', 'Acceptance', 'Sent', 'Contribution', 'Cooperation']
COLORS = {'+1': C_POSITIVE, '-1': C_REVERSE}


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {name}.pdf and preview')


def main():
    rows = json.loads(DATA.read_text())['results']
    def record(model, target, scope='aggregate'):
        return next(r for r in rows if r['model'] == model and
                    r['split'] == 'split_101' and r['condition'] == 'full_' + target
                    and r['reference'] == 'baseline' and r['scope'] == scope)

    # Embed TrueType fonts so PDF labels remain text and match the serif plots.
    plt.rcParams.update({'pdf.fonttype': 42, 'mathtext.fontset': 'dejavuserif',
                         'savefig.facecolor': 'white'})
    for metric, filename in [('shift', 'revision_primary_effects'),
                              ('validity', 'revision_primary_validity')]:
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        for target, offset, label in [('+1', -.15, 'Higher-index'),
                                      ('-1', .15, 'Lower-index')]:
            selected = [record(m, target) for m in MODELS]
            y = np.arange(len(MODELS)) + offset
            if metric == 'shift':
                shift = np.array([r['point']['conditional_shift'] for r in selected])
                ci = np.array([r['ci95']['conditional_shift'] for r in selected])
                ax.errorbar(shift, y, xerr=np.stack([shift-ci[:, 0], ci[:, 1]-shift]),
                            fmt='o', color=COLORS[target], capsize=3, markersize=6,
                            label=label)
            else:
                ax.scatter([r['point']['q_treatment'] for r in selected], y,
                           color=COLORS[target], s=40, label=label)
        ax.set_yticks(np.arange(len(MODELS)), NAMES)
        ax.invert_yaxis()
        if metric == 'shift':
            ax.axvline(0, color='gray', lw=.8)
            ax.set_xlim(-.48, .64)
            ax.set_xlabel('Action-index shift from baseline')
        else:
            ax.axvline(VALID_ANSWER_THRESHOLD, color='gray', lw=1, linestyle='--')
            ax.set_xlim(.68, 1.025)
            ax.set_xticks([.7, .8, .9, 1.0])
            ax.set_xlabel('Valid-answer rate $q$')
        ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.01),
                  ncol=2, frameon=False, fontsize=12)
        ax.grid(axis='x', alpha=.3)
        ax.set_axisbelow(True)
        fig.tight_layout()
        save(fig, filename)

    for target, filename in [('+1', 'revision_native_higher'),
                              ('-1', 'revision_native_lower')]:
        fig, ax = plt.subplots(figsize=(6.8, 5.2))
        selected = [[record(m, target, g) for g in GAMES] for m in MODELS]
        values = np.array([[-r['point']['conditional_shift'] for r in row]
                           for row in selected])
        im = ax.imshow(values, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
        for i in range(len(MODELS)):
            for j in range(len(GAMES)):
                value = values[i, j]
                value = 0.0 if abs(value) < .0005 else value
                star = '*' if selected[i][j]['p_holm'] < .05 else ''
                ax.text(j, i, (f'{value:+.2f}' if value else '0.00') + star,
                        ha='center', va='center', fontsize=11,
                        color='white' if abs(value) > .6 else '#152332')
        ax.set_xticks(np.arange(6), LABELS, rotation=45, ha='right')
        ax.set_yticks(np.arange(6), NAMES)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        cb = fig.colorbar(im, ax=ax, orientation='horizontal',
                          pad=.34, fraction=.045, aspect=32)
        cb.set_label('Change in native outcome')
        fig.tight_layout()
        save(fig, filename)


if __name__ == '__main__':
    main()
