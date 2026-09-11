#!/usr/bin/env python3
"""Frozen-configuration transfer, capability checks and mechanistic diagnostics.
Runs after run_revision_experiments.py; never retunes on validation outcomes.
"""
import argparse
from contextlib import nullcontext
import csv
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.run_revision_experiments import (checkpoint_provenance, evaluate,
                                                   read, save, specs_for)
from experiments.revision.data import render,digest,GAMES


def strict_answer(game, raw, total):
    text=raw.strip()
    if game=='ultimatum_responder': return text in ('ACCEPT','REJECT')
    if game=='prisoner_dilemma': return text in ('COOPERATE','BETRAY')
    try: return text.replace('.','',1).isdigit() and 0 <= float(text) <= total
    except ValueError: return False


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir',required=True)
    ap.add_argument('--model-name')
    ap.add_argument('--turns',type=int,default=5)
    ap.add_argument('--episodes',type=int,default=30)
    ap.add_argument('--capability-items',type=int,default=100)
    ap.add_argument('--decision-max-new-tokens',type=int,default=12)
    ap.add_argument('--multiturn-max-new-tokens',type=int,default=8)
    ap.add_argument('--capability-max-new-tokens',type=int,default=32)
    ap.add_argument('--explanation-max-new-tokens',type=int,default=96)
    args=ap.parse_args()
    if min(args.turns,args.episodes,args.capability_items,args.decision_max_new_tokens,
           args.multiturn_max_new_tokens,args.capability_max_new_tokens,
           args.explanation_max_new_tokens)<1: ap.error('counts and token limits must be positive')
    import numpy as np
    from emo.config import load_config
    from emo.models import SteeringModel
    from emo.hooks import steer
    from rlhf_break.games import SYSTEM_PROMPT, parse_decision, is_incoherent
    root=Path(args.run_dir); manifest=read(root/'manifest.json'); exp=manifest['experiment_config']
    weights=args.model_name or manifest['arguments'].get('model_name')
    overrides={'model':{'gen_batch_size':manifest['model_config']['model']['gen_batch_size']}}
    if weights: overrides['model']['hf_name']=weights
    model=SteeringModel(load_config(manifest['arguments']['model'],overrides=overrides))
    provenance=checkpoint_provenance(model)
    if provenance != read(root/'checkpoint_provenance.json'):
        raise RuntimeError('Validation loaded a different checkpoint than the core experiment')
    dec=exp.get('decoding',{}); sampling=dict(greedy=False,temperature=dec.get('temperature',1.),
                                            top_p=dec.get('top_p',.95))
    decision_sampling=dict(sampling,max_new_tokens=args.decision_max_new_tokens)
    for split in sorted(root.glob('split_*')):
        out=split/'validation'; out.mkdir(exist_ok=True)
        validation_manifest=dict(arguments=vars(args),core_manifest_hash=digest(manifest))
        if (out/'manifest.json').exists() and read(out/'manifest.json') != validation_manifest:
            raise RuntimeError('Validation settings changed; use a fresh core run directory')
        save(out/'manifest.json',validation_manifest)
        seed=int(split.name.split('_')[1]); rows=read(split/'scenarios.json')['test']
        vectors=dict(np.load(split/'vectors.npz',allow_pickle=False))
        winners=read(split/'winners.json')
        conditions={'baseline':None,**{k:v for k,v in winners.items() if k.startswith('full_') and v}}
        # Human review covers all explanation outputs; condition is hidden in the rating sheet.
        review, key=[] ,[]
        for condition,c in conditions.items():
            specs=specs_for(model,c,vectors)
            def generate(prompts, system=SYSTEM_PROMPT, max_tokens=128):
                with steer(model,specs) if specs else nullcontext():
                    return model.generate(prompts,system=system,**dict(sampling,max_new_tokens=max_tokens),seed=seed+30000)
            russian=[]
            for r in rows:
                m=r['meta']; russian.append(dict(r,prompt=render(r['game'],r['context'],m['total'],m['ratio'],'ru')))
            path=out/(condition+'_russian.json')
            if not path.exists():
                # Same system prompt and Latin decision tokens isolate user-prompt language transfer.
                save(path,evaluate(model,russian,specs,decision_sampling,seed+30000,
                                   manifest['arguments']['test_repeats']))
            capability_path=out/(condition+'_capability.json')
            if not capability_path.exists():
                items=[]
                for i in range(args.capability_items):
                    a,b=13+i*7,9+i*3
                    items.append(dict(id=f'arithmetic_{i}',prompt=f'Compute {a} + {b} * 3. Reply with only the integer.',answer=str(a+b*3),kind='arithmetic'))
                    word=f'code{i:04d}'
                    items.append(dict(id=f'format_{i}',prompt=f'Return exactly this JSON object with no extra text: {{"code":"{word}"}}',
                                      answer={'code':word},kind='instruction'))
                raws=generate([x['prompt'] for x in items],system='Follow the user instruction exactly.',
                              max_tokens=args.capability_max_new_tokens)
                if len(raws)!=len(items): raise RuntimeError('Capability output count mismatch')
                for x,raw in zip(items,raws):
                    try: parsed=json.loads(raw) if x['kind']=='instruction' else raw.strip()
                    except ValueError: parsed=None
                    x.update(raw=raw,correct=parsed==x['answer'])
                save(capability_path,items)
            explanation_path=out/(condition+'_explanations.json')
            if not explanation_path.exists():
                prompts=[r['prompt']+'\nFor this diagnostic only, put your decision on the first line, then give a brief explanation consistent with the stated rules.' for r in rows]
                raws=generate(prompts,max_tokens=args.explanation_max_new_tokens)
                if len(raws)!=len(rows): raise RuntimeError('Explanation output count mismatch')
                explanation=[]
                for r,raw in zip(rows,raws):
                    first=raw.strip().split('\n')[0]
                    explanation.append(dict(r,raw=raw,strict_first_line=strict_answer(r['game'],first,r['meta']['total'])))
                save(explanation_path,explanation)
            for r in read(explanation_path):
                rid=digest([split.name,condition,r['scenario_id']])[:24]
                review.append(dict(review_id=rid,game=r['game'],prompt=r['prompt'],response=r['raw'],
                                   rule_consistency='',semantic_coherence='',decision_explanation_consistency='',notes=''))
                key.append(dict(review_id=rid,condition=condition,scenario_id=r['scenario_id']))
            episode_path=out/(condition+'_multiturn.json')
            if not episode_path.exists():
                episodes=[]
                policies=('always_cooperate','always_betray','tit_for_tat')
                for policy_index,policy in enumerate(policies):
                    histories=[[] for _ in range(args.episodes)]
                    previous=['COOPERATE']*args.episodes
                    failed=[False]*args.episodes
                    for turn in range(args.turns):
                        opponents=[('COOPERATE' if policy=='always_cooperate' else
                                    'BETRAY' if policy=='always_betray' else previous[e])
                                   for e in range(args.episodes)]
                        prompts=[]
                        for episode in range(args.episodes):
                            prompts.append(
                              f'You play {args.turns} rounds of a simultaneous Prisoner\'s Dilemma with the same partner. '
                              f'Both COOPERATE: each gets 3. Both BETRAY: each gets 1. A sole betrayer gets 5; the cooperator gets 0. '
                              f'This is round {turn+1}. Your partner\'s policy is undisclosed.\n'
                              +'\n'.join(histories[episode])+'\nReply exactly COOPERATE or BETRAY.')
                        # Generate one batch per round and policy.  Keeping all
                        # episode slots in the batch, including already failed
                        # trajectories, preserves RNG alignment across frozen
                        # conditions while reducing 450 generate calls to 15.
                        with steer(model,specs) if specs else nullcontext():
                            raws=model.generate(prompts,system=SYSTEM_PROMPT,
                                **dict(sampling,max_new_tokens=args.multiturn_max_new_tokens),
                                seed=seed+40000+policy_index*args.turns+turn)
                        for episode,(raw,opponent) in enumerate(zip(raws,opponents)):
                            was_failed=failed[episode]
                            action=None if was_failed else (
                                raw.strip() if strict_answer('prisoner_dilemma',raw,0) else None)
                            valid=action is not None
                            metric=float(action=='BETRAY') if valid else None
                            payoff=(3 if action==opponent=='COOPERATE' else
                                    1 if action==opponent=='BETRAY' else
                                    5 if action=='BETRAY' else 0) if valid else None
                            episodes.append(dict(game='prisoner_dilemma',
                                scenario_id=f'{policy}:{episode}',repeat=turn,
                                metric=metric,raw='' if was_failed else raw,
                                opponent_action=opponent,policy=policy,payoff=payoff,
                                continuation_invalid=was_failed))
                            if not valid:
                                failed[episode]=True
                            else:
                                histories[episode].append(
                                    f'Round {turn+1}: you chose {action}, partner chose {opponent}, your payoff {payoff}.')
                                previous[episode]=action
                save(episode_path,episodes)
        rng=np.random.default_rng(seed);rng.shuffle(review)
        review_path=out/'blind_review.csv'
        key_path=out/'blind_review_key.json'
        # A resumed validation run must never erase ratings entered by a human.
        # The response set is immutable because the validation manifest is
        # checked above, so existing review material can safely be retained.
        if not review_path.exists():
            with review_path.open('w') as f:
                writer=csv.DictWriter(f,fieldnames=list(review[0]));writer.writeheader();writer.writerows(review)
        if not key_path.exists():
            save(key_path,key)
        # Associations use one averaged baseline label per scenario, separately by game.
        projection_path=out/'projection_diagnostics.json'
        if not projection_path.exists():
            baseline=read(split/'test/baseline.json')
            layers=sorted({c['source_layer'] for c in conditions.values() if c})
            acts=model.collect_residuals([r['prompt'] for r in rows],layers=layers,pool='last',system=SYSTEM_PROMPT)
            diagnostics=[]
            for name,c in conditions.items():
                if not c: continue
                v=vectors[c['vector_key']];x=acts[c['source_layer']].float().numpy()
                projection=x@v
                for game in GAMES:
                    px,ys=[],[]
                    for i,r in enumerate(rows):
                        if r['game']!=game: continue
                        vals=[z['metric'] for z in baseline if z['scenario_id']==r['scenario_id'] and z['metric'] is not None]
                        if vals: px.append(float(projection[i]));ys.append(float(np.mean(vals)))
                    corr=float(np.corrcoef(px,ys)[0,1]) if len(px)>2 and np.std(px)>0 and np.std(ys)>0 else None
                    diagnostics.append(dict(condition=name,game=game,n=len(px),pearson=corr,
                       note='Descriptive prompt-projection association, not identification of a causal semantic feature.'))
            save(projection_path,diagnostics)
    print('Validation collected. Blind human ratings and statistical analysis remain separate.')

if __name__=='__main__': main()
