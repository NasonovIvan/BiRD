#!/usr/bin/env python3
"""Analysis of fixed-condition transfer and narrow capabilities; no inference.
Blinded human ratings are intentionally not inferred from answer parseability.
"""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dirs',nargs='+',required=True)
    ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--resamples',type=int,default=10000)
    ap.add_argument('--descriptive-resamples',type=int,default=1000)
    ap.add_argument('--confirmatory-split',default='split_101')
    a=ap.parse_args()
    from experiments.revision.provenance import provenance
    origin=provenance(a.run_dirs)
    from experiments.revision.stats import infer,holm
    results=[]
    for root in map(Path,a.run_dirs):
        for split in sorted(root.glob('split_*')):
            v=split/'validation'
            for condition in ['full_+1','full_-1']:
                for domain in ['russian','multiturn','capability']:
                    path=v/f'{condition}_{domain}.json';bp=v/f'baseline_{domain}.json'
                    if not path.exists() or not bp.exists(): continue
                    b,t=json.loads(bp.read_text()),json.loads(path.read_text())
                    if domain=='capability':
                        def convert(rows):
                            return [dict(game=r['kind'],scenario_id=r['id'],repeat=0,metric=float(r['correct'])) for r in rows]
                        b,t=convert(b),convert(t)
                    scopes=sorted({r.get('policy',r.get('game')) for r in b}) if domain!='russian' else ['aggregate']
                    for scope in scopes:
                        bb=b if scope=='aggregate' else [r for r in b if r.get('policy',r.get('game'))==scope]
                        tt=t if scope=='aggregate' else [r for r in t if r.get('policy',r.get('game'))==scope]
                        family=('capability' if domain=='capability' else 'transfer') \
                            if split.name==a.confirmatory_split else 'descriptive_stability'
                        n_resamples=a.resamples if family in ('capability','transfer') \
                            else a.descriptive_resamples
                        result=infer(bb,tt,n_resamples)
                        results.append(dict(**origin,model=json.loads((root/'manifest.json').read_text())['arguments']['model'],
                          split=split.name,condition=condition,domain=domain,scope=scope,
                          multiplicity_family=family,resamples=n_resamples,**result))
    family_sizes={}
    for row in results: row['p_holm']=None
    for family in ('transfer','capability'):
        indices=[i for i,r in enumerate(results) if r['multiplicity_family']==family]
        family_sizes[family]=len(indices)
        adjusted=holm([results[i]['p_cluster_swap'] if results[i]['p_cluster_swap'] is not None else 1
                       for i in indices])
        for i,p in zip(indices,adjusted):
            results[i]['p_holm']=p if results[i]['p_cluster_swap'] is not None else None
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(dict(**origin,results=results,total_comparisons=len(results),
      corrected_family_sizes=family_sizes,confirmatory_split=a.confirmatory_split,
      confirmatory_resamples=a.resamples,descriptive_resamples=a.descriptive_resamples,
      note='Capability shifts are accuracy differences. Multiturn clusters are complete episodes. No claim of global capability preservation.'),indent=2,allow_nan=False)+'\n')

if __name__=='__main__': main()
