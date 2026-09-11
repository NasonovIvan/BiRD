#!/usr/bin/env python3
"""Plot reviewed revision statistics; never read legacy point estimates.
Usage: python plotting/plot_revision.py --statistics runs/revision/analysis/statistics.json --out-dir article/figures/revision
"""
import argparse
import json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--statistics',required=True,type=Path)
    ap.add_argument('--out-dir',required=True,type=Path)
    a=ap.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    document=json.loads(a.statistics.read_text())
    rows=document['results']
    a.out_dir.mkdir(parents=True,exist_ok=True)
    for model in sorted({r['model'] for r in rows}):
        for split in sorted({r['split'] for r in rows if r['model']==model}):
            for reference in ['baseline','ablation']:
                selected=[r for r in rows if r['model']==model and r['split']==split and r['scope']=='aggregate'
                          and (r['reference']=='baseline' if reference=='baseline' else r['reference']!='baseline')]
                fig,ax=plt.subplots(figsize=(8,max(3,len(selected)*.28)))
                labels=[]
                for i,r in enumerate(selected):
                    label=r['condition'] if reference=='baseline' else r['condition']+' minus '+r['reference']
                    labels.append(label)
                    mean=r['point']['conditional_shift'];ci=r['ci95']['conditional_shift']
                    if mean is None: continue
                    color='#0072B2' if r['point']['q_treatment']>=.7 else '#D55E00'
                    ax.plot(mean,i,'o',color=color,markersize=4)
                    if ci is not None: ax.plot(ci,[i,i],color=color,linewidth=1)
                ax.set_yticks(range(len(labels)),labels);ax.axvline(0,color='gray',lw=.8)
                ax.set_xlabel('Equal-game action-index shift (95% scenario-cluster CI)')
                ax.invert_yaxis();ax.spines[['top','right']].set_visible(False)
                fig.tight_layout()
                for ext in ['png','pdf']:
                    fig.savefig(a.out_dir/f'{model}_{split}_{reference}.{ext}',dpi=220,bbox_inches='tight')
                plt.close(fig)
    print('Figures retain split identities; overlapping splits are not pooled.')

if __name__=='__main__': main()
