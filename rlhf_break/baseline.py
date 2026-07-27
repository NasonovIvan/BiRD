"""Phase 0 + Phase 2: baseline measurement and safety-breaking sweep.

``run_baseline`` — measure the model's default decision-making (no steering).
``run_sweep``     — for each candidate (direction, layer, coeff), steer and
                    measure the shift in aggression metrics + coherence.

Both share the same game battery and the same generation/parsing pipeline; the
only difference is whether a ``SteerSpec`` is applied.

v2 changes (see RLHF_BREAK_IMPROVEMENTS.md):
  - Temperature sampling + n_repeats: instead of 1 greedy answer per prompt,
    generate K answers at temperature T and average the metric. This lets
    steering shift the output *distribution* rather than needing to flip the
    argmax (which greedy makes nearly impossible).
  - Multi-layer specs: inject at several layers simultaneously via
    ``make_multi_layer_specs`` — more effective on instruct models.

v3 changes (based on v2 analysis: coeff=3.0 norm_scale gave only +0.027):
  - ``norm_scale`` parameter: when False, uses absolute-magnitude steering
    (like emotion steering with coeff=8). This is much stronger than
    norm_scale=True at the same coeff.
  - ``multi_layer_full_coeff``: when True, injects the FULL coeff at each
    layer (not split). Total perturbation = coeff × n_layers.
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch
from emo.hooks import SteerSpec, make_multi_layer_specs, steer
from emo.models import SteeringModel

from .games import (
    SYSTEM_PROMPT,
    all_scenarios,
    is_incoherent,
    is_refusal,
    parse_decision,
)
from .io_local import append_jsonl, read_jsonl, write_jsonl


def _run_games(
    model: SteeringModel,
    scenarios,
    specs: list[SteerSpec] | None = None,
    max_new_tokens: int = 32,
    system: str | None = None,
    greedy: bool = True,
    temperature: float = 0.8,
    top_p: float = 0.95,
    n_repeats: int = 1,
    seed: int | None = None,
    check_incoherent: bool = False,
) -> list[dict]:
    """Run the game battery under optional steering; return decision records.

    With ``n_repeats > 1`` and ``greedy=False``, each scenario is generated
    ``n_repeats`` times (with different sampling seeds) and each repeat gets
    its own record. The caller can then average per-scenario to reduce noise.
    """
    if system is None:
        system = SYSTEM_PROMPT
    prompts = [s.prompt for s in scenarios]

    if n_repeats > 1:
        expanded_prompts = []
        expanded_scenarios = []
        for s in scenarios:
            for _ in range(n_repeats):
                expanded_prompts.append(s.prompt)
                expanded_scenarios.append(s)
        prompts = expanded_prompts
        scenarios = expanded_scenarios

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        system=system,
        greedy=greedy,
        temperature=temperature if not greedy else None,
        top_p=top_p if not greedy else None,
        seed=seed,
    )
    if specs:
        with steer(model, specs):
            raws = model.generate(prompts, **gen_kwargs)
    else:
        raws = model.generate(prompts, **gen_kwargs)
    records = []
    for s, raw in zip(scenarios, raws):
        try:
            d = parse_decision(s.game, raw, s.meta)
        except Exception:
            # Parser crash (e.g. unexpected model output under extreme
            # steering). Treat as incoherent rather than crashing the run.
            d = type("D", (), {"coherent": False, "metric": float("nan"),
                              "parsed": None})()
        coherent = d.coherent
        # When enabled (sweep/final-eval), also check for garbled output
        # that the parser may have accidentally parsed (e.g. a number
        # embedded in a degenerate repetition loop). This catches the
        # "steering too strong → incoherent text" failure mode that
        # inflates aggression while destroying coherence.
        if coherent and check_incoherent and is_incoherent(raw, s.game):
            coherent = False
        records.append({
            "game": s.game,
            "prompt": s.prompt,
            "raw": raw,
            "metric": float(d.metric) if coherent and not np.isnan(d.metric) else None,
            "coherent": coherent,
            "refusal": is_refusal(raw),
            "parsed": d.parsed,
            "meta": s.meta,
        })
    return records


def run_baseline(
    model: SteeringModel,
    n_per_game: int = 12,
    seed: int = 0,
    system: str | None = None,
    out_path: str | None = None,
    greedy: bool = True,
    temperature: float = 0.8,
    top_p: float = 0.95,
    n_repeats: int = 1,
    max_new_tokens: int = 32,
    check_incoherent: bool = False,
) -> dict:
    """Phase 0: measure baseline decision-making (no steering)."""
    scenarios = all_scenarios(n_per_game=n_per_game, seed=seed)
    records = _run_games(
        model, scenarios, specs=None, system=system,
        max_new_tokens=max_new_tokens,
        greedy=greedy, temperature=temperature, top_p=top_p,
        n_repeats=n_repeats, seed=seed,
        check_incoherent=check_incoherent,
    )
    summary = _summarize(records, condition="baseline")
    summary["n_per_game"] = n_per_game
    summary["seed"] = seed
    summary["n_repeats"] = n_repeats
    summary["greedy"] = greedy
    summary["temperature"] = temperature if not greedy else None
    if out_path:
        write_jsonl(records, out_path)
        summary["records_path"] = out_path
    return summary


def _run_config_multiseed(
    model: SteeringModel,
    specs: list[SteerSpec],
    seeds: list[int],
    n_per_game: int,
    system: str | None,
    greedy: bool,
    temperature: float,
    top_p: float,
    n_repeats: int,
    condition: str = "",
    max_new_tokens: int = 32,
    check_incoherent: bool = False,
) -> dict:
    """Run one steering config across multiple scenario seeds and pool records.

    Each seed draws a *fresh* scenario sample via ``all_scenarios(seed=...)``,
    so cross-seed variance in the mean metric exposes a seed-dependent
    boundary-saturation direction (one that collapses numeric games to the
    extreme "0" token on one sample and the extreme "100" token on another).
    A genuine aggression axis shifts offers into the mid-range consistently
    across seeds and therefore has a small ``metric_seed_std``.

    The pooled ``_summarize`` output is augmented with:
      - ``sweep_seeds``: the seeds used.
      - ``per_seed_mean_metric``: per-seed mean aggression metric.
      - ``metric_seed_std``: std of the per-seed means (0.0 for a single seed).
    """
    all_records: list[dict] = []
    per_seed_means: list[float | None] = []
    for sd in seeds:
        sc = all_scenarios(n_per_game=n_per_game, seed=sd)
        recs = _run_games(
            model, sc, specs=specs, system=system,
            max_new_tokens=max_new_tokens,
            greedy=greedy, temperature=temperature, top_p=top_p,
            n_repeats=n_repeats, seed=sd,
            check_incoherent=check_incoherent,
        )
        all_records.extend(recs)
        ms = [r["metric"] for r in recs if r["metric"] is not None]
        per_seed_means.append(float(np.mean(ms)) if ms else None)
    summary = _summarize(all_records, condition=condition)
    summary["sweep_seeds"] = list(seeds)
    summary["per_seed_mean_metric"] = per_seed_means
    valid = [m for m in per_seed_means if m is not None]
    summary["metric_seed_std"] = float(np.std(valid)) if len(valid) >= 2 else 0.0
    return summary


def run_sweep(
    model: SteeringModel,
    candidates: list[dict],
    coeffs: list[float],
    n_per_game: int = 8,
    seed: int = 1,
    system: str | None = None,
    out_path: str | None = None,
    greedy: bool = True,
    temperature: float = 0.8,
    top_p: float = 0.95,
    n_repeats: int = 1,
    multi_layer_groups: list[list[int]] | None = None,
    norm_scale: bool = True,
    multi_layer_full_coeff: bool = False,
    sweep_seeds: list[int] | None = None,
    max_new_tokens: int = 32,
    check_incoherent: bool = False,
) -> list[dict]:
    """Phase 2: sweep over (direction, layer, coeff) breaking configs.

    v3: ``norm_scale=False`` uses absolute-magnitude steering (like emotion
    steering with coeff=8). ``multi_layer_full_coeff=True`` injects the FULL
    coeff at each layer (not split), which is much stronger.

    v4: ``sweep_seeds`` runs every config across multiple scenario seeds and
    records the cross-seed std of the mean metric (``metric_seed_std``). This
    surfaces seed-unstable boundary-saturation directions that only look
    aggressive on one particular sample. Defaults to ``[seed]`` (single seed,
    identical to prior behaviour).
    """
    seeds = [int(s) for s in sweep_seeds] if sweep_seeds else [int(seed)]
    done = set()
    if out_path and os.path.exists(out_path):
        for row in read_jsonl(out_path):
            done.add((row.get("layer", -1), row["kind"], row["coeff"]))
    results = []

    # --- Count total configs for progress reporting ---
    n_single = len(candidates) * len(coeffs)
    n_ml = 0
    if multi_layer_groups:
        n_ml = len(candidates) * len(multi_layer_groups) * len(coeffs)
    n_total = n_single + n_ml
    sweep_done = 0
    sweep_t0 = time.time()

    def _sweep_progress(label, summary):
        nonlocal sweep_done
        sweep_done += 1
        elapsed = time.time() - sweep_t0
        rate = sweep_done / max(1e-6, elapsed)
        remaining = (n_total - sweep_done) / max(1e-6, rate)
        _m = summary.get("mean_metric")
        _m_str = f"{_m:.3f}" if _m is not None else "  --"
        print(
            f"  [{sweep_done}/{n_total}] {label:<40} "
            f"agg={_m_str} coh={summary['coherence_rate']:.2f} "
            f"ref={summary['refusal_rate']:.2f}  "
            f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)",
            flush=True,
        )

    # --- Single-layer candidates ---
    for cand in candidates:
        layer = cand["layer"]
        kind = cand["kind"]
        vec = cand["vector"]
        v_tensor = torch.tensor(vec, dtype=model.model.dtype, device=model.input_device)
        for coeff in coeffs:
            if (layer, kind, coeff) in done:
                sweep_done += 1
                continue
            specs = [SteerSpec(layer=layer, vector=v_tensor, coeff=coeff, norm_scale=norm_scale)]
            t0 = time.time()
            try:
                summary = _run_config_multiseed(
                    model, specs, seeds, n_per_game, system,
                    greedy, temperature, top_p, n_repeats,
                    condition=f"{kind}_L{layer}_c{coeff}",
                    max_new_tokens=max_new_tokens,
                    check_incoherent=check_incoherent,
                )
            except Exception as e:
                # Extreme coefficients with norm_scale=False can cause CUDA
                # errors or OOM. Log and skip this config instead of crashing
                # the entire sweep.
                print(
                    f"  [sweep] L{layer} {kind} c={coeff}: ERROR ({e}) — skipping",
                    flush=True,
                )
                summary = {
                    "condition": f"{kind}_L{layer}_c{coeff}",
                    "n_total": 0,
                    "mean_metric": None,
                    "std_metric": None,
                    "coherence_rate": 0.0,
                    "refusal_rate": 1.0,
                    "per_game": {},
                    "layer": layer,
                    "kind": kind,
                    "coeff": coeff,
                    "multi_layer": False,
                    "norm_scale": norm_scale,
                    "elapsed_s": round(time.time() - t0, 1),
                    "error": str(e),
                }
                results.append(summary)
                if out_path:
                    append_jsonl(summary, out_path)
                # Clear CUDA cache in case of OOM so subsequent configs can run.
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                continue
            summary["layer"] = layer
            summary["kind"] = kind
            summary["coeff"] = coeff
            summary["multi_layer"] = False
            summary["norm_scale"] = norm_scale
            summary["elapsed_s"] = round(time.time() - t0, 1)
            results.append(summary)
            if out_path:
                append_jsonl(summary, out_path)
            _sweep_progress(f"{kind}_L{layer}_c{coeff}", summary)

    # --- Multi-layer candidates ---
    if multi_layer_groups:
        for cand in candidates:
            vec = cand["vector"]
            v_tensor = torch.tensor(vec, dtype=model.model.dtype, device=model.input_device)
            for ml_group in multi_layer_groups:
                valid_layers = [l for l in ml_group if l < model.cfg.model.num_layers]
                if len(valid_layers) < 2:
                    continue
                for coeff in coeffs:
                    kind = cand["kind"]
                    ml_key = f"{kind}_ML{'-'.join(str(l) for l in valid_layers)}"
                    if (-1, ml_key, coeff) in done:
                        sweep_done += 1
                        continue
                    if multi_layer_full_coeff:
                        # v3: inject FULL coeff at each layer (not split).
                        specs = [
                            SteerSpec(
                                layer=l, vector=v_tensor.clone(),
                                coeff=coeff, norm_scale=norm_scale,
                            )
                            for l in valid_layers
                        ]
                    else:
                        # v2: split coeff evenly across layers.
                        specs = make_multi_layer_specs(
                            layers=valid_layers,
                            vector=v_tensor,
                            coeff=coeff,
                            norm_scale=norm_scale,
                        )
                    t0 = time.time()
                    try:
                        summary = _run_config_multiseed(
                            model, specs, seeds, n_per_game, system,
                            greedy, temperature, top_p, n_repeats,
                            condition=f"{ml_key}_c{coeff}",
                            max_new_tokens=max_new_tokens,
                            check_incoherent=check_incoherent,
                        )
                    except Exception as e:
                        # Multi-layer full-coeff with high α can overflow the
                        # residual stream → CUDA error. Skip instead of crashing.
                        print(
                            f"  [sweep] ML{valid_layers} {ml_key} c={coeff}: "
                            f"ERROR ({e}) — skipping",
                            flush=True,
                        )
                        summary = {
                            "condition": f"{ml_key}_c{coeff}",
                            "n_total": 0,
                            "mean_metric": None,
                            "std_metric": None,
                            "coherence_rate": 0.0,
                            "refusal_rate": 1.0,
                            "per_game": {},
                            "layer": valid_layers[0],
                            "kind": ml_key,
                            "coeff": coeff,
                            "multi_layer": True,
                            "norm_scale": norm_scale,
                            "ml_layers": valid_layers,
                            "ml_full_coeff": multi_layer_full_coeff,
                            "elapsed_s": round(time.time() - t0, 1),
                            "error": str(e),
                        }
                        results.append(summary)
                        if out_path:
                            append_jsonl(summary, out_path)
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                        continue
                    summary["layer"] = valid_layers[0]
                    summary["kind"] = ml_key
                    summary["coeff"] = coeff
                    summary["multi_layer"] = True
                    summary["norm_scale"] = norm_scale
                    summary["ml_layers"] = valid_layers
                    summary["ml_full_coeff"] = multi_layer_full_coeff
                    summary["elapsed_s"] = round(time.time() - t0, 1)
                    results.append(summary)
                    if out_path:
                        append_jsonl(summary, out_path)
                    _sweep_progress(f"{ml_key}_c{coeff}", summary)
    sweep_elapsed = time.time() - sweep_t0
    print(f"  [sweep] done: {len(results)} configs in {sweep_elapsed:.0f}s "
          f"({len(results) / max(1e-6, sweep_elapsed):.2f} configs/s)", flush=True)
    return results


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


def _summarize(records: list[dict], condition: str = "") -> dict:
    """Aggregate per-game metrics + coherence/refusal rates."""
    metrics = [r["metric"] for r in records if r["metric"] is not None]
    coherent = [r["coherent"] for r in records]
    refusal = [r["refusal"] for r in records]
    by_game = {}
    for r in records:
        g = r["game"]
        by_game.setdefault(g, []).append(r)
    per_game = {}
    for g, rs in by_game.items():
        gm = [r["metric"] for r in rs if r["metric"] is not None]
        per_game[g] = {
            "n": len(rs),
            "mean_metric": float(np.mean(gm)) if gm else None,
            "std_metric": float(np.std(gm)) if gm else None,
            "coherence": sum(r["coherent"] for r in rs) / len(rs),
            "refusal": sum(r["refusal"] for r in rs) / len(rs),
        }
    return {
        "condition": condition,
        "n_total": len(records),
        "mean_metric": float(np.mean(metrics)) if metrics else None,
        "std_metric": float(np.std(metrics)) if metrics else None,
        "coherence_rate": float(np.mean(coherent)) if coherent else 0.0,
        "refusal_rate": float(np.mean(refusal)) if refusal else 0.0,
        "per_game": per_game,
    }
