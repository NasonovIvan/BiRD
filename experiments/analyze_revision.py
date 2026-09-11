#!/usr/bin/env python3
"""Analyze measured revision runs with prespecified multiplicity families.

The first split is confirmatory.  Additional overlapping discovery splits are
reported as stability analyses and are never counted as independent tests.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dirs', nargs='+', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--resamples', type=int, default=10000)
    ap.add_argument('--descriptive-resamples', type=int, default=1000)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--confirmatory-split',default='split_101')
    a=ap.parse_args()
    from experiments.revision.provenance import provenance
    origin=provenance(a.run_dirs)
    from experiments.revision.stats import infer, holm
    from experiments.revision.data import GAMES
    output=[]
    for root in map(Path,a.run_dirs):
        manifest=json.loads((root/'manifest.json').read_text())
        for split in sorted(root.glob('split_*')):
            test=split/'test'
            baseline=json.loads((test/'baseline.json').read_text())
            comparisons=[]
            for target in ('+1','-1'):
                full_path=test/f'full_{target}.json'
                if not full_path.exists(): continue
                full_records=json.loads(full_path.read_text())
                comparisons.append((f'full_{target}','baseline',full_records,
                                    baseline,'full'))
                # Contribution = full minus each reselected ablation on the
                # same held-out set.  An ablation-vs-baseline test does not
                # answer whether the component restriction hurts BiRD.
                for path in sorted(test.glob(f'*_{target}.json')):
                    if path.stem.startswith(('drop_','single_layer_','fixed_','no_','only_')):
                        comparisons.append((f'full_{target}',path.stem,full_records,
                                            json.loads(path.read_text()),'ablation'))
                # Direct specificity contrasts against matched nulls.
                for reference in (f'random_{target}',f'shuffled_{target}',
                                  f'full_{target}_permuted',f'full_{target}_zero_hook'):
                    reference_path=test/(reference+'.json')
                    if reference_path.exists():
                        comparisons.append((f'full_{target}',reference,full_records,
                                            json.loads(reference_path.read_text()),'specificity'))
                for reference in (f'full_{target}_sign_flip',f'full_{target}_split_budget'):
                    reference_path=test/(reference+'.json')
                    if reference_path.exists():
                        comparisons.append((f'full_{target}',reference,full_records,
                                            json.loads(reference_path.read_text()),'diagnostic'))
            for condition,reference,records,base,kind in comparisons:
                scopes=['aggregate',*GAMES] if kind=='full' else ['aggregate']
                for scope in scopes:
                    b=base if scope=='aggregate' else [r for r in base if r['game']==scope]
                    t=records if scope=='aggregate' else [r for r in records if r['game']==scope]
                    is_confirmatory=split.name==a.confirmatory_split
                    if is_confirmatory and kind=='full':
                        family='primary_model_game'
                    elif is_confirmatory and kind=='ablation':
                        family='aggregate_ablation'
                    elif is_confirmatory and kind=='specificity':
                        family='aggregate_specificity'
                    else:
                        family='descriptive'
                    n_resamples=a.resamples if family in (
                        'primary_model_game','aggregate_ablation','aggregate_specificity') \
                        else a.descriptive_resamples
                    result=infer(b,t,n_resamples,a.seed)
                    output.append(dict(model=manifest['arguments']['model'],split=split.name,
                                       condition=condition,reference=reference,scope=scope,
                                       multiplicity_family=family,resamples=n_resamples,**result))
            # Fairness-sensitive responder analysis: compare within offer-fraction bands.
            if split.name != a.confirmatory_split:
                continue
            for condition in ('full_+1','full_-1'):
                path=test/(condition+'.json')
                if not path.exists(): continue
                records=json.loads(path.read_text())
                for low,high in [(0,.3),(.3,.5),(.5,1.01)]:
                    def in_band(r):
                        return r['game']=='ultimatum_responder' and low <= r['meta']['offer']/r['meta']['total'] < high
                    b=[r for r in baseline if in_band(r)]
                    t=[r for r in records if in_band(r)]
                    if not b: continue
                    result=infer(b,t,a.descriptive_resamples,a.seed)
                    output.append(dict(model=manifest['arguments']['model'],split=split.name,
                       condition=condition,reference='baseline',scope=f'UR_offer_{low}_{high}',
                       multiplicity_family='descriptive_sensitivity',
                       resamples=a.descriptive_resamples,**result))
            # Index sensitivity: exclude each game in turn, including Ultimatum Responder.
            for condition in ('full_+1','full_-1'):
                path=test/(condition+'.json')
                if not path.exists(): continue
                records=json.loads(path.read_text())
                for omitted in GAMES:
                    b=[r for r in baseline if r['game']!=omitted]
                    t=[r for r in records if r['game']!=omitted]
                    result=infer(b,t,a.descriptive_resamples,a.seed)
                    output.append(dict(model=manifest['arguments']['model'],split=split.name,
                          condition=condition,reference='baseline',scope='without_'+omitted,
                          multiplicity_family='descriptive_sensitivity',
                          resamples=a.descriptive_resamples,**result))
    if not output: raise RuntimeError('No completed test data found')
    # Undefined tests occupy a slot with p=1, preserving the declared family size.
    native_names={'dictator':'given_share','ultimatum_proposer':'offered_share',
      'ultimatum_responder':'acceptance_rate','trust':'sent_share',
      'public_goods':'contribution_share','prisoner_dilemma':'cooperation_rate'}
    for row in output:
        row.update(origin)
        for game, values in row['per_game'].items():
            values['native_outcome']=native_names[game]
            values['native_share_shift']=-values['conditional_shift'] if values['conditional_shift'] is not None else None
    corrected_families=('primary_model_game','aggregate_ablation','aggregate_specificity')
    family_sizes={}
    for row in output: row['p_holm']=None
    for family in corrected_families:
        indices=[i for i,r in enumerate(output) if r['multiplicity_family']==family]
        family_sizes[family]=len(indices)
        adjusted=holm([output[i]['p_cluster_swap'] if output[i]['p_cluster_swap'] is not None else 1.
                       for i in indices])
        for i,p in zip(indices,adjusted):
            output[i]['p_holm']=p if output[i]['p_cluster_swap'] is not None else None
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    (out/'statistics.json').write_text(json.dumps(dict(**origin,total_comparisons=len(output),
        corrected_family_sizes=family_sizes,confirmatory_split=a.confirmatory_split,
        confirmatory_resamples=a.resamples,descriptive_resamples=a.descriptive_resamples,
        results=output),indent=2,allow_nan=False)+'\n')
    with (out/'statistics.csv').open('w') as f:
        fields=['model','split','condition','reference','scope','multiplicity_family',
                'shift','ci_low','ci_high','q','p','p_holm','bound_low','bound_high']
        fields += list(origin)
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in output:
            ci=r['ci95']['conditional_shift'] or [None,None]
            writer.writerow({**origin,**{k:r[k] for k in fields[:6]},'shift':r['point']['conditional_shift'],
              'ci_low':ci[0],'ci_high':ci[1],'q':r['point']['q_treatment'],'p':r['p_cluster_swap'],
              'p_holm':r['p_holm'],'bound_low':r['point']['worst_case_low'],'bound_high':r['point']['worst_case_high']})
    print(f'Wrote {len(output)} comparisons; corrected families: {family_sizes}.')

if __name__=='__main__': main()
