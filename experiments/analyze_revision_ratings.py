#!/usr/bin/env python3
"""Analyze two independently completed blind-review sheets before adjudication.
Usage: python experiments/analyze_revision_ratings.py --rater-a a.csv --rater-b b.csv --key blind_review_key.json --out ratings.json
"""
import argparse
import csv
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
CRITERIA=('rule_consistency','semantic_coherence','decision_explanation_consistency')


def load_sheet(path):
    # utf-8-sig accepts both ordinary UTF-8 and the BOM used by spreadsheet
    # applications and the generated annotation sheets.
    with path.open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    ids=[r['review_id'] for r in rows]
    if len(ids)!=len(set(ids)):raise ValueError(f'Duplicate review IDs in {path}')
    for r in rows:
        for criterion in CRITERIA:
            if r[criterion] not in ('0','1'):raise ValueError(f'Unfilled/nonbinary rating: {path}, {r["review_id"]}, {criterion}')
    return {r['review_id']:r for r in rows}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--rater-a',type=Path,required=True);ap.add_argument('--rater-b',type=Path,required=True)
    ap.add_argument('--key',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--resamples',type=int,default=10000);args=ap.parse_args()
    from experiments.revision.stats import infer,holm
    a,b=load_sheet(args.rater_a),load_sheet(args.rater_b)
    key_rows=json.loads(args.key.read_text());key={r['review_id']:r for r in key_rows}
    if len(key)!=len(key_rows) or not (a.keys()==b.keys()==key.keys()):
        raise ValueError('Both sheets must contain every blinded item exactly once')
    agreement={};comparisons=[]
    for criterion in CRITERIA:
        n=len(a);po=sum(a[k][criterion]==b[k][criterion] for k in a)/n
        pa=sum(int(a[k][criterion]) for k in a)/n;pb=sum(int(b[k][criterion]) for k in b)/n
        pe=pa*pb+(1-pa)*(1-pb)
        agreement[criterion]=dict(n=n,agreement=po,kappa=(po-pe)/(1-pe) if pe<1 else None,
                                  disagreement_ids=[k for k in a if a[k][criterion]!=b[k][criterion]])
        records={}
        for k,m in key.items():
            for judge,sheet in enumerate((a,b)):
                records.setdefault(m['condition'],[]).append(dict(game=sheet[k]['game'],
                    scenario_id=m['scenario_id'],repeat=judge,metric=float(sheet[k][criterion])))
        for condition,rows in records.items():
            if condition=='baseline':continue
            comparisons.append(dict(criterion=criterion,condition=condition,
                **infer(records['baseline'],rows,args.resamples)))
    adjusted=holm([r['p_cluster_swap'] if r['p_cluster_swap'] is not None else 1 for r in comparisons])
    for row,p in zip(comparisons,adjusted):row['p_holm']=p if row['p_cluster_swap'] is not None else None
    def rates(ids, criterion):
        ra=sum(int(a[k][criterion]) for k in ids)/len(ids)
        rb=sum(int(b[k][criterion]) for k in ids)/len(ids)
        return dict(n_items=len(ids),rater_a=ra,rater_b=rb,pooled=(ra+rb)/2)
    descriptive=[]
    group_specs=[('overall',None,lambda m:True)]
    for model in sorted({m.get('model') for m in key.values() if m.get('model')}):
        group_specs.append(('model',model,lambda m,model=model:m.get('model')==model))
    for game in sorted({row['game'] for row in a.values()}):
        group_specs.append(('game',game,lambda m,game=game:a[m['review_id']]['game']==game))
    for group_type,group,keep in group_specs:
        for condition in sorted({m['condition'] for m in key.values()}):
            ids=[k for k,m in key.items() if m['condition']==condition and keep(dict(m,review_id=k))]
            for criterion in CRITERIA:
                descriptive.append(dict(group_type=group_type,group=group,
                    condition=condition,criterion=criterion,**rates(ids,criterion)))
    any_disagreement=sum(any(a[k][c]!=b[k][c] for c in CRITERIA) for k in a)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(dict(agreement=agreement,
       all_criteria_agreement=dict(n=len(a),items_with_any_disagreement=any_disagreement,
          items_with_complete_agreement=len(a)-any_disagreement,
          complete_agreement=(len(a)-any_disagreement)/len(a)),
       descriptive_rates=descriptive,comparisons=comparisons,
       note='Independent ratings before adjudication. Scenario clusters retain both raters. Positive shifts mean better rated quality. Correction is within this supplied sheet pair; combine families for cross-model confirmatory claims.'),indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
