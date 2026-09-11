#!/usr/bin/env python3
"""Reviewer-1 experiment suite. Run only on the author's inference server.

No network judge; no mutation of legacy results. See article/revision/README.md.
Heavy imports occur after CLI parsing so --help needs only the standard library.
"""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict
import json
from pathlib import Path
import platform
import shutil
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.revision.data import GAMES, digest, make_splits, select, summarize

PROFILES = ['full', 'drop_generic', 'drop_game', 'drop_decision', 'drop_sae',
            'only_generic', 'only_game', 'only_decision', 'only_sae',
            'single_layer', 'fixed_layer', 'fixed_coefficient', 'no_validity',
            'no_q_weight', 'random', 'shuffled']


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def checkpoint_provenance(model):
    """Record the loaded checkpoint without hashing multi-gigabyte weights."""
    config = model.model.config
    parameter_count = sum(parameter.numel() for parameter in model.model.parameters())
    return dict(
        requested_name=model.mcfg.hf_name,
        resolved_name=getattr(config, '_name_or_path', None),
        commit_hash=getattr(config, '_commit_hash', None),
        model_type=getattr(config, 'model_type', None),
        architectures=getattr(config, 'architectures', None),
        parameter_count=parameter_count,
        parameter_dtype=str(next(model.model.parameters()).dtype),
        config=digest(config.to_dict()),
    )


def evaluate(model, rows, specs, decoding, seed, repeats):
    from rlhf_break.baseline import _run_games
    from rlhf_break.games import Scenario
    scenarios = [Scenario(r['game'], r['prompt'], r['meta']) for r in rows]
    recs = _run_games(model, scenarios, specs=specs, n_repeats=repeats,
                      seed=seed, check_incoherent=True, **decoding)
    if len(recs) != len(rows)*repeats:
        raise RuntimeError('Generation count does not match the paired design')
    for i, r in enumerate(recs):
        source = rows[i//repeats]
        r.update(scenario_id=source['scenario_id'], repeat=i%repeats)
        # Guard against permissive legacy parser accepting total+1 / negative indices.
        if r['metric'] is not None and not 0 <= r['metric'] <= 1:
            r.update(metric=None, coherent=False)
    return recs


def unit(x):
    import numpy as np
    x = np.asarray(x, dtype=np.float32)
    norm = float(np.linalg.norm(x))
    if not np.isfinite(x).all() or norm < 1e-8:
        raise ValueError('Non-finite or zero contrast; cannot silently replace with noise')
    return x / norm


def extract(model, rows, layers, dec, args, seed, out):
    import numpy as np
    import torch
    from rlhf_break.games import SYSTEM_PROMPT
    from rlhf_break.safety_direction import (REFUSAL_PROMPTS, COMPLY_PROMPTS,
        GAME_AGGRESSIVE_FRAMING, GAME_FAIR_FRAMING)
    from emo.vectors.sae_train import train_sae
    recs = evaluate(model, rows, None, dec, seed, args.extraction_repeats)
    save(out/'extraction_records.json', recs)
    # Average stochastic labels by scenario BEFORE constructing contrasts.
    labels = []
    for i in range(len(rows)):
        vals = [r['metric'] for r in recs[i*args.extraction_repeats:(i+1)*args.extraction_repeats]
                if r['metric'] is not None]
        labels.append(float(np.mean(vals)) if vals else np.nan)
    acts = model.collect_residuals([r['prompt'] for r in rows], layers=layers,
                                  pool='last', system=SYSTEM_PROMPT)
    high, low = [], []
    for game in GAMES:
        idx = [i for i, r in enumerate(rows) if r['game'] == game and np.isfinite(labels[i])]
        if not idx:
            continue
        median = np.median([labels[i] for i in idx])
        h, l = [i for i in idx if labels[i] > median], [i for i in idx if labels[i] < median]
        if h and l:
            high.append(h)
            low.append(l)
    if not high:
        raise RuntimeError('No within-game label contrast. Increase extraction scenarios/repeats.')
    # Equal weight for each contributing game, avoiding game-composition confounding.
    def contrast(x, hi=high, lo=low):
        return np.mean([x[h].mean(0)-x[l].mean(0) for h, l in zip(hi, lo)], axis=0)
    rng = np.random.default_rng(seed)
    shuffled_hi, shuffled_lo = [], []
    for h, l in zip(high, low):
        perm = rng.permutation(h+l).tolist()
        shuffled_hi.append(perm[:len(h)])
        shuffled_lo.append(perm[len(h):])
    vectors, diagnostics = {}, {}
    for family, pos, neg in [('generic', REFUSAL_PROMPTS, COMPLY_PROMPTS),
                             ('game', GAME_AGGRESSIVE_FRAMING, GAME_FAIR_FRAMING)]:
        contrast_system = 'You are a helpful assistant.' if family == 'generic' else SYSTEM_PROMPT
        a = model.collect_residuals(pos, layers=layers, pool='last', system=contrast_system)
        b = model.collect_residuals(neg, layers=layers, pool='last', system=contrast_system)
        for layer in layers:
            vectors[f'{family}:{layer}'] = unit(a[layer].float().numpy().mean(0)-b[layer].float().numpy().mean(0))
    for layer in layers:
        x = acts[layer].float().numpy()
        vectors[f'decision:{layer}'] = unit(contrast(x))
        vectors[f'shuffled:{layer}'] = unit(contrast(x, shuffled_hi, shuffled_lo))
        for k in range(args.random_directions):
            vectors[f'random{k}:{layer}'] = unit(rng.normal(size=x.shape[1]))
        cfg = SimpleNamespace(seed=seed, sae_params=dict(n_features=args.sae_features,
             topk=args.sae_topk, lr=.001, batch_size=min(256, len(x)),
             n_epochs=args.sae_epochs, aux_loss_weight=.1))
        sae, metrics = train_sae(torch.from_numpy(x), cfg, model.input_device)
        with torch.no_grad():
            feats = sae.encode(torch.from_numpy(x)).numpy()
            top = np.argsort(contrast(feats))[-args.sae_selected:]
            vectors[f'sae:{layer}'] = unit(sae.W_dec.detach().numpy()[top].sum(0))
        diagnostics[str(layer)] = dict(training=metrics, selected_features=top.tolist())
        for family in ('generic', 'game'):
            vectors[f'{family}_combined:{layer}'] = unit(vectors[f'{family}:{layer}']+vectors[f'decision:{layer}'])
    np.savez(out/'vectors.npz', **vectors)
    save(out/'extraction_diagnostics.json', dict(sae=diagnostics,
         contrast_games=len(high), label_policy='scenario means, strict median, equal game weights'))
    return vectors


def candidates(vectors, layers, groups, coeffs, norm_scale, full):
    result = []
    # Keep extraction layer in every identity, including multi-layer interventions.
    # The original sweep omitted it from some multi-layer resume keys.
    for key in sorted(vectors):
        family, source = key.split(':')
        components = (['generic', 'decision'] if family == 'generic_combined' else
                      ['game', 'decision'] if family == 'game_combined' else
                      ['random'] if family.startswith('random') else [family])
        for group in [[int(source)]] + groups:
            for coeff in sorted(set([-float(c) for c in coeffs]+[float(c) for c in coeffs])):
                c = dict(vector_key=key, source_layer=int(source), layers=group, coeff=coeff,
                         norm_scale=norm_scale, full=full, components=components)
                c['id'] = digest(c)[:20]
                result.append(c)
    return result


def specs_for(model, c, vectors):
    import torch
    from emo.hooks import SteerSpec
    if c is None:
        return None
    coeff = c['coeff'] / (1 if c['full'] else len(c['layers']))
    return [SteerSpec(layer=l, vector=torch.tensor(vectors[c['vector_key']],
            device=model.input_device, dtype=model.model.dtype), coeff=coeff,
            norm_scale=c['norm_scale']) for l in c['layers']]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True)
    ap.add_argument('--model-name')
    ap.add_argument('--config')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--batch-size', type=int,
                    help='Override the per-model generation batch size.')
    ap.add_argument('--coeffs', nargs='+', type=float,
                    help='Prespecified positive coefficient magnitudes; overrides config.')
    ap.add_argument('--max-multilayer-groups', type=int,
                    help='Use only the first N prespecified layer groups from config.')
    ap.add_argument('--split-seeds', nargs='+', type=int, default=[101, 202, 303, 404, 505])
    ap.add_argument('--counts', nargs=3, type=int, default=[48, 36, 60], metavar=('EXTRACT','SELECT','TEST'))
    ap.add_argument('--extraction-repeats', type=int, default=3)
    ap.add_argument('--selection-repeats', type=int, default=3)
    ap.add_argument('--test-repeats', type=int, default=5)
    ap.add_argument('--random-directions', type=int, default=5)
    ap.add_argument('--sae-features', type=int, default=16384)
    ap.add_argument('--sae-topk', type=int, default=64)
    ap.add_argument('--sae-selected', type=int, default=16)
    ap.add_argument('--sae-epochs', type=int, default=500)
    ap.add_argument('--selection-max-new-tokens', type=int, default=8,
                    help='Short generation budget used only during candidate selection.')
    ap.add_argument('--test-max-new-tokens', type=int, default=12,
                    help='Generation budget for extraction and held-out test decisions.')
    ap.add_argument('--save-all-selection', action='store_true',
                    help='Store raw JSON for every candidate (very large and slow).')
    ap.add_argument('--selection-checkpoint-every', type=int, default=25,
                    help='Atomically checkpoint compact selection summaries every N candidates.')
    args = ap.parse_args()
    if len(set(args.split_seeds)) != len(args.split_seeds):
        ap.error('split seeds must be unique')
    if min(args.extraction_repeats, args.selection_repeats, args.test_repeats,
           args.random_directions, args.sae_epochs, args.selection_max_new_tokens,
           args.test_max_new_tokens, args.selection_checkpoint_every) < 1:
        ap.error('repeats, random directions, and epochs must be positive')
    if args.batch_size is not None and args.batch_size < 1:
        ap.error('batch size must be positive')
    if args.coeffs is not None and (not args.coeffs or min(args.coeffs) <= 0):
        ap.error('coefficient magnitudes must be positive')
    if args.max_multilayer_groups is not None and args.max_multilayer_groups < 0:
        ap.error('max multilayer groups must be nonnegative')
    if not 1 <= args.sae_topk <= args.sae_features or not 1 <= args.sae_selected <= args.sae_features:
        ap.error('invalid SAE dimensions')
    # Validate disjointness/capacity before loading weights.
    all_parts = {s: make_splits(s, args.counts) for s in args.split_seeds}
    import numpy as np
    import torch
    import transformers
    # SAE training uses float32 matrices.  ``high`` enables TF32 tensor cores
    # on Ampere/Blackwell while retaining float32 accumulation, which is ample
    # precision for the exploratory reconstruction objective and much faster
    # than strict IEEE matmul on RTX 5090.
    torch.set_float32_matmul_precision('high')
    from emo.config import load_config
    from emo.jsonc import load_jsonc
    from emo.models import SteeringModel
    exp_path = Path(args.config) if args.config else ROOT/'config/rlhf_break'/f'{args.model}.json'
    # Config files use the project's JSONC loader.
    exp = load_jsonc(str(exp_path))
    # ``gen_batch_size`` belongs to ModelConfig; a top-level override is
    # silently ignored by ``load_config`` and previously defeated the
    # per-model OOM settings in the experiment configs.
    overrides = {'model': {'gen_batch_size': args.batch_size or exp.get('gen_batch_size', 8)}}
    if args.model_name:
        overrides['model']['hf_name'] = args.model_name
    cfg = load_config(args.model, overrides=overrides)
    out_root = Path(args.out_dir)
    source_files = [Path(__file__), ROOT/'experiments/revision/data.py',
                    ROOT/'emo/hooks.py', ROOT/'rlhf_break/games.py',
                    ROOT/'emo/models.py', ROOT/'rlhf_break/safety_direction.py',
                    ROOT/'emo/vectors/sae_train.py', ROOT/'emo/config.py']
    manifest = dict(arguments=vars(args), experiment_config=exp,
                    model_config=asdict(cfg), protocol_version=1,
                    source_hashes={str(p.relative_to(ROOT)):digest(p.read_text())
                                   for p in source_files},
                    versions=dict(python=platform.python_version(), numpy=np.__version__,
                                  torch=torch.__version__, transformers=transformers.__version__,
                                  cuda=torch.version.cuda,
                                  gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None))
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root/'manifest.json'
    if manifest_path.exists() and read(manifest_path) != manifest:
        raise RuntimeError('Output directory belongs to a different protocol/configuration; use a new directory')
    save(manifest_path, manifest)
    # Preserve the exact executable sources alongside the recorded hashes.
    # This keeps resumed multi-model runs independently auditable even when a
    # later architecture needs a memory-only implementation correction.
    for source in source_files:
        target = out_root/'source_snapshot'/source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    dec_cfg = exp.get('decoding', {})
    dec = dict(greedy=False, temperature=dec_cfg.get('temperature', 1.),
               top_p=dec_cfg.get('top_p', .95))
    extraction_dec = dict(dec, max_new_tokens=args.test_max_new_tokens)
    selection_dec = dict(dec, max_new_tokens=args.selection_max_new_tokens)
    test_dec = dict(dec, max_new_tokens=args.test_max_new_tokens)
    model = SteeringModel(cfg)
    provenance_path = out_root/'checkpoint_provenance.json'
    provenance = checkpoint_provenance(model)
    if provenance_path.exists() and read(provenance_path) != provenance:
        raise RuntimeError('Loaded checkpoint differs from the checkpoint used by this output directory')
    save(provenance_path, provenance)
    layers = sorted({min(cfg.model.num_layers-1, int(round(f*cfg.model.num_layers)))
                     for f in exp['direction_extraction']['layer_fractions']})
    sw = exp['sweep']
    groups = [list(g) for g in sw.get('multi_layer_groups', [])]
    if args.max_multilayer_groups is not None:
        groups = groups[:args.max_multilayer_groups]
    coeffs = sorted(set(args.coeffs or sw['coeffs']))
    if any(len(set(g)) != len(g) or any(l < 0 or l >= cfg.model.num_layers for l in g) for g in groups):
        raise ValueError('Invalid or duplicate multi-layer indices')
    winners_all = []
    for seed, parts in all_parts.items():
        split_started = time.perf_counter()
        out = out_root/f'split_{seed}'
        out.mkdir(exist_ok=True)
        save(out/'scenarios.json', parts)
        if (out/'vectors.npz').exists():
            vectors = dict(np.load(out/'vectors.npz', allow_pickle=False))
        else:
            vectors = extract(model, parts['extraction'], layers, extraction_dec, args, seed, out)
        extraction_finished = time.perf_counter()
        cs = candidates(vectors, layers, groups, coeffs, sw.get('norm_scale', True),
                        sw.get('multi_layer_full_coeff', False))
        save(out/'candidates.json', cs)
        def cached(stage, name, rows, c, repeats, decoding):
            path = out/stage/(name+'.json')
            if path.exists():
                return read(path)
            recs = evaluate(model, rows, specs_for(model,c,vectors), decoding, seed+10000,
                            repeats)
            save(path, recs)
            return recs
        base = summarize(cached('selection','baseline',parts['selection'],None,
                                args.selection_repeats, selection_dec))
        summaries_path = out/'selection_summaries.json'
        sums = read(summaries_path) if summaries_path.exists() else {}
        for i, c in enumerate(cs):
            if c['id'] not in sums:
                recs = evaluate(model, parts['selection'], specs_for(model,c,vectors),
                                selection_dec, seed+10000, args.selection_repeats)
                sums[c['id']] = summarize(recs)
                if args.save_all_selection:
                    save(out/'selection'/(c['id']+'.json'), recs)
                if (i + 1) % args.selection_checkpoint_every == 0:
                    save(summaries_path, sums)
            print(f'{args.model} split={seed} candidate={i+1}/{len(cs)}', flush=True)
        save(summaries_path, sums)
        selection_finished = time.perf_counter()
        winners = {}
        for target in (-1,1):
            for profile in PROFILES:
                key = f'{profile}_{target:+d}'
                c = select(cs, sums, base, target, profile, layers[len(layers)//2],
                           coeffs[len(coeffs)//2], exp.get('coherence_threshold', .7))
                winners[key] = c
        # Freeze every winner before evaluating any test record.
        save(out/'winners.json', winners)
        # Materialize raw selection outputs only for the distinct frozen
        # winners.  This preserves an auditable sample without writing tens of
        # gigabytes of redundant JSON for candidates that are never selected.
        for c in {x['id']: x for x in winners.values() if x}.values():
            cached('selection', c['id'], parts['selection'], c,
                   args.selection_repeats, selection_dec)
        save(out/'selection/storage.json', dict(
            format='compact-summaries-plus-selected-raw-v1',
            candidate_count=len(cs), selected_raw_count=len({x['id'] for x in winners.values() if x}),
            repeats=args.selection_repeats,
            note='All candidate summaries are retained; raw JSON is retained for frozen winners.'))
        test_base_records = cached('test','baseline',parts['test'],None,
                                   args.test_repeats, test_dec)
        test_base = summarize(test_base_records)
        test_config_cache = {}

        def effective_signature(c):
            if c is None or c['coeff'] == 0:
                return 'baseline'
            effective_coeff = c['coeff'] / (1 if c['full'] else len(c['layers']))
            return digest([c['vector_key'], c['layers'], effective_coeff,
                           c['norm_scale']])

        def cached_test(name, c):
            path = out/'test'/(name+'.json')
            if path.exists():
                records = read(path)
                test_config_cache.setdefault(effective_signature(c), records)
                return records
            signature = effective_signature(c)
            if signature == 'baseline':
                records = test_base_records
            elif signature in test_config_cache:
                records = test_config_cache[signature]
            else:
                records = evaluate(model, parts['test'], specs_for(model,c,vectors),
                                   test_dec, seed+10000, args.test_repeats)
                test_config_cache[signature] = records
            save(path, records)
            return records

        for key, c in winners.items():
            if c is None:
                winners_all.append(dict(split=seed, profile=key, config=None, status='no_feasible_selection'))
                continue  # explicit null winner; never replace with a degraded fallback
            recs = cached_test(key,c)
            summary = summarize(recs)
            shift = summary['mean']-test_base['mean'] if summary['mean'] is not None and test_base['mean'] is not None else None
            target = int(key.rsplit('_',1)[1])
            passed = shift is not None and target*shift > 0 and summary['q'] >= exp.get('coherence_threshold',.7)
            discovery_shift = sums[c['id']]['mean']-base['mean']
            winners_all.append(dict(split=seed, profile=key, config=c, status='test_target_met' if passed else 'test_target_failed',
               discovery_shift=discovery_shift, test_shift=shift,
               discovery_test_gap=target*(discovery_shift-shift) if shift is not None else None, test=summary))
            if key.startswith('full_'):
                for label, changes in [('sign_flip',dict(coeff=-c['coeff'])),
                                       ('zero_hook',dict(coeff=0.)),
                                       ('split_budget',dict(full=False))]:
                    cached_test(key+'_'+label,dict(c,**changes))
                # Same vector/layers/size, destroy coordinate alignment (matched norm null).
                rng = np.random.default_rng(seed+20000)
                perm_key = c['id']+':permuted'
                vectors[perm_key] = rng.permutation(vectors[c['vector_key']])
                cached_test(key+'_permuted',dict(c,vector_key=perm_key))
        test_finished = time.perf_counter()
        save(out/'runtime.json', dict(
            extraction_seconds=extraction_finished-split_started,
            selection_seconds=selection_finished-extraction_finished,
            test_seconds=test_finished-selection_finished,
            total_seconds=test_finished-split_started,
            candidate_count=len(cs)))
        print(f'{args.model} split={seed} finished in '
              f'{test_finished-split_started:.1f}s '
              f'(extract={extraction_finished-split_started:.1f}s, '
              f'select={selection_finished-extraction_finished:.1f}s, '
              f'test={test_finished-selection_finished:.1f}s)', flush=True)
        save(out_root/'selection_stability.json', dict(results=winners_all,
             note='Splits overlap with each other; report distributions, never treat splits as independent replications.',
             frequencies={p:dict(Counter(str(x['config']['components']) if x['config'] else 'NO_FEASIBLE_SELECTION' for x in winners_all if x['profile']==p))
                          for p in sorted({x['profile'] for x in winners_all})}))
    print('Complete. No manuscript values were changed. Run analyze_revision.py separately.')


if __name__ == '__main__':
    main()
