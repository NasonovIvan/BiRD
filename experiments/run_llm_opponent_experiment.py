#!/usr/bin/env python3
"""Run RLHF-break with LLM-API opponents (not just hardcoded strategies).

This module extends the core experiment to support **LLM-judge-as-opponent**
mode for 2-player games (Ultimatum proposer/responder). Instead of the model
playing against a fixed automated strategy, it plays against a second LLM
accessed via the judge API gateway.

Two opponent modes:
1. ``automated`` — hardcoded fair strategy (default, deterministic, reproducible)
2. ``llm_judge`` — a second LLM plays the opponent role via the API

For single-player games (Dictator, Trolley, Trust, Public Goods, Prisoner's
Dilemma), the opponent mode has no effect — these games have no opponent.

Usage:
    # Run with LLM-judge opponents:
    python run_llm_opponent_experiment.py --model llama31_8b --opponent llm_judge

    # Run with automated opponents (default, for comparison):
    python run_llm_opponent_experiment.py --model llama31_8b --opponent automated

    # Run both and compare:
    python run_llm_opponent_experiment.py --model llama31_8b --opponent both
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

# Add parent to path.
PKG_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PKG_ROOT)

from emo.config import load_config
from emo.hooks import SteerSpec, make_multi_layer_specs, steer
from emo.models import SteeringModel
from rlhf_break import analyze as A
from rlhf_break import baseline as B
from rlhf_break import safety_direction as sd
from rlhf_break.games import (
    GAMES,
    SYSTEM_PROMPT,
    Decision,
    Scenario,
    all_scenarios,
    dictator_scenarios,
    is_incoherent,
    is_refusal,
    parse_decision,
    parse_dictator,
    parse_ultimatum_proposer,
    parse_ultimatum_responder,
    ultimatum_proposer_scenarios,
    ultimatum_responder_scenarios,
)
from rlhf_break.io_local import (
    read_jsonl,
    save_json,
    write_jsonl,
)
from rlhf_break.opponents import (
    FairAutomatedProposer,
    FairAutomatedResponder,
    LLMJudgeOpponent,
)


def _load_exp_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _resolve_layers(fractions: list[float], num_layers: int) -> list[int]:
    return [min(num_layers - 1, max(0, int(f * num_layers))) for f in fractions]


def _print_sep(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


# ---------------------------------------------------------------------------
# LLM-opponent game runner
# ---------------------------------------------------------------------------

def run_games_with_opponent(
    model: SteeringModel,
    opponent_mode: str = "automated",
    n_per_game: int = 8,
    seed: int = 0,
    system: str | None = None,
    specs: list[SteerSpec] | None = None,
    greedy: bool = True,
    temperature: float = 0.8,
    top_p: float = 0.95,
    n_repeats: int = 1,
) -> list[dict]:
    """Run the game battery with a specified opponent mode.

    For single-player games (Dictator, Trolley, Trust, Public Goods,
    Prisoner's Dilemma), the opponent mode is irrelevant — the model
    makes a unilateral decision.

    For 2-player games (Ultimatum proposer/responder), the opponent
    determines the other player's move:
    - ``automated``: hardcoded fair strategy (deterministic)
    - ``llm_judge``: a second LLM via the judge API

    Returns decision records with an ``opponent_mode`` and ``opponent_move``
    field added.
    """
    if system is None:
        system = SYSTEM_PROMPT

    # Build scenarios for single-player games + Ultimatum.
    scenarios = all_scenarios(n_per_game=n_per_game, seed=seed)

    if n_repeats > 1:
        expanded = []
        for s in scenarios:
            for _ in range(n_repeats):
                expanded.append(s)
        scenarios = expanded

    # For Ultimatum responder, the opponent is the proposer who makes an offer.
    # For Ultimatum proposer, the opponent is the responder who accepts/rejects.
    # We need to run these interactively if using LLM judge.

    # Initialize opponents.
    llm_responder = None
    llm_proposer = None
    if opponent_mode == "llm_judge":
        try:
            from emo.measure.judge_client import JudgeClient
            client = JudgeClient()
            if not client.ping():
                print("  ⚠ Judge API not reachable; falling back to automated.")
                opponent_mode = "automated"
            else:
                llm_responder = LLMJudgeOpponent(client, role="responder")
                llm_proposer = LLMJudgeOpponent(client, role="proposer")
                print(f"  LLM judge opponent initialized (model: {client.model})")
        except Exception as e:
            print(f"  ⚠ Cannot init LLM judge opponent: {e}")
            print(f"  Falling back to automated opponents.")
            opponent_mode = "automated"

    auto_responder = FairAutomatedResponder(threshold=0.30)
    auto_proposer = FairAutomatedProposer(offer_fraction=0.5)

    # Generate model outputs.
    prompts = [s.prompt for s in scenarios]
    gen_kwargs = dict(
        max_new_tokens=32, system=system,
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

    # Process each scenario.
    records = []
    for s, raw in zip(scenarios, raws):
        try:
            d = parse_decision(s.game, raw, s.meta)
        except Exception:
            d = type("D", (), {"coherent": False, "metric": float("nan"),
                               "parsed": None})()
        record = {
            "game": s.game,
            "prompt": s.prompt,
            "raw": raw,
            "metric": float(d.metric) if d.coherent and not np.isnan(d.metric) else None,
            "coherent": d.coherent,
            "refusal": is_refusal(raw),
            "parsed": d.parsed,
            "meta": s.meta,
            "opponent_mode": opponent_mode,
        }

        # For 2-player games, get the opponent's move.
        if s.game == "ultimatum_proposer":
            # The model proposed an offer; the opponent (responder) decides.
            offer = d.parsed.get("offer") if d.parsed and isinstance(d.parsed, dict) else None
            if offer is not None:
                context = {"offer": offer, "total": s.meta.get("total", 100)}
                if opponent_mode == "llm_judge" and llm_responder:
                    opp_result = llm_responder.respond(context)
                else:
                    opp_result = auto_responder.respond(context)
                record["opponent_move"] = opp_result
                # If rejected, both get 0 — adjust metric to reflect futility.
                if opp_result.get("accept") is False:
                    record["opponent_rejected"] = True
                    # The model's unfairness still counts, but the outcome is 0 for both.
                    record["outcome"] = "rejected"
                else:
                    record["opponent_rejected"] = False
                    record["outcome"] = "accepted"
            else:
                record["opponent_move"] = None
                record["outcome"] = "unparseable"

        elif s.game == "ultimatum_responder":
            # The model is the responder; the opponent (proposer) made the offer.
            # The offer is already in the scenario meta.
            offer = s.meta.get("offer", 0)
            total = s.meta.get("total", 100)
            context = {"total": total}
            if opponent_mode == "llm_judge" and llm_proposer:
                # In LLM mode, the proposer generates its own offer.
                # But the scenario already has a fixed offer — we use that.
                # (The LLM proposer would be used if we generated scenarios dynamically.)
                opp_result = {"offer": offer, "strategy": "scenario_fixed"}
            else:
                opp_result = auto_proposer.respond(context)
            record["opponent_move"] = opp_result
            record["outcome"] = "responded"

        else:
            # Single-player game — no opponent.
            record["opponent_move"] = None
            record["outcome"] = "unilateral"

        records.append(record)

    return records


def _summarize_with_opponent(records: list[dict], condition: str = "") -> dict:
    """Summarize records including opponent interaction data."""
    coherent = [r for r in records if r.get("coherent")]
    refusals = [r for r in records if r.get("refusal")]
    metrics = [r["metric"] for r in coherent if r.get("metric") is not None]

    summary = {
        "condition": condition,
        "n_total": len(records),
        "n_coherent": len(coherent),
        "n_refusal": len(refusals),
        "coherence_rate": len(coherent) / len(records) if records else 0,
        "refusal_rate": len(refusals) / len(records) if records else 0,
        "mean_metric": float(np.mean(metrics)) if metrics else None,
        "std_metric": float(np.std(metrics)) if metrics else None,
    }

    # Per-game breakdown.
    per_game = {}
    for game in GAMES:
        game_records = [r for r in records if r.get("game") == game]
        game_coherent = [r for r in game_records if r.get("coherent")]
        game_metrics = [r["metric"] for r in game_coherent if r.get("metric") is not None]
        per_game[game] = {
            "n": len(game_records),
            "n_coherent": len(game_coherent),
            "coherence": len(game_coherent) / len(game_records) if game_records else 0,
            "refusal": sum(1 for r in game_records if r.get("refusal")) / len(game_records) if game_records else 0,
            "mean_metric": float(np.mean(game_metrics)) if game_metrics else None,
        }

        # For 2-player games, add opponent interaction stats.
        if game in ("ultimatum_proposer", "ultimatum_responder"):
            opp_moves = [r.get("opponent_move") for r in game_records if r.get("opponent_move")]
            if opp_moves:
                if game == "ultimatum_proposer":
                    n_rejected = sum(1 for r in game_records if r.get("opponent_rejected"))
                    per_game[game]["opponent_reject_rate"] = n_rejected / len(game_records)
                per_game[game]["opponent_strategy"] = opp_moves[0].get("strategy", "unknown") if opp_moves else "unknown"

    summary["per_game"] = per_game
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="RLHF-break with LLM-API opponents (vs hardcoded strategies)"
    )
    ap.add_argument("--model", default="llama31_8b", help="model id")
    ap.add_argument("--model-name", default=None,
                    help="Override weight loading path (local dir or HF repo). "
                         "Use this to point to local weights on the server.")
    ap.add_argument("--config", default=None, help="path to rlhf_break.json")
    ap.add_argument("--opponent", default="both",
                    choices=["automated", "llm_judge", "both"],
                    help="opponent mode for 2-player games")
    ap.add_argument("--out-dir", default=None, help="output directory")
    ap.add_argument("--n-per-game", type=int, default=8)
    args = ap.parse_args()

    # Auto-resolve per-model config (see run_rlhf_break.py for rationale).
    mid = args.model or "llama31_8b"
    if args.config:
        cfg_path = args.config
    else:
        per_model = os.path.join(PKG_ROOT, "config", "rlhf_break", f"{mid}.json")
        if os.path.exists(per_model):
            cfg_path = per_model
            print(f"  [config] auto-resolved per-model config: {per_model}")
        else:
            cfg_path = os.path.join(PKG_ROOT, "config", "rlhf_break.json")
    exp = _load_exp_config(cfg_path)
    model_id = args.model or exp.get("model", "llama31_8b")
    out_dir = args.out_dir or os.path.join(
        PKG_ROOT, exp.get("out_dir", "runs/rlhf_break"), model_id, "opponent_comparison"
    )
    os.makedirs(out_dir, exist_ok=True)


    dec = exp.get("decoding", {})
    greedy = dec.get("greedy", True)
    temperature = dec.get("temperature", 0.8)
    top_p = dec.get("top_p", 0.95)
    n_repeats = dec.get("n_repeats", 1)

    _print_sep("LLM-OPPONENT EXPERIMENT")
    print(f"  model:      {model_id}")
    print(f"  opponent:   {args.opponent}")
    print(f"  output dir: {out_dir}")

    # --model-name: override the weight loading path.
    overrides: dict = {}
    if args.model_name:
        overrides.setdefault("model", {})["hf_name"] = args.model_name
        if os.path.isdir(args.model_name):
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            print(f"  weights:     {args.model_name} (local dir -> HF offline mode)")
        else:
            print(f"  weights:     {args.model_name}")

    # Pass gen_batch_size from the experiment config into ModelConfig so
    # that generate() chunks large prompt lists and avoids CUDA OOM.
    if "gen_batch_size" in exp:
        overrides["gen_batch_size"] = exp["gen_batch_size"]

    # Load model.
    mcfg = load_config(model_id, overrides=overrides)
    print(f"  hf_name:    {mcfg.model.hf_name}")
    print(f"  gen_batch:  {mcfg.model.gen_batch_size}")
    print(f"  loading model...")
    model = SteeringModel(mcfg)
    num_layers = mcfg.model.num_layers
    print(f"  loaded:     {num_layers} layers, hidden={mcfg.model.hidden_size}")

    # Load directions (from previous RLHF-break run).
    parent_dir = os.path.dirname(os.path.dirname(out_dir))
    refusal_path = os.path.join(parent_dir, "refusal_directions.npz")
    aggr_path = os.path.join(parent_dir, "aggression_directions.npz")
    game_refusal_path = os.path.join(parent_dir, "game_refusal_directions.npz")
    ranked_path = os.path.join(parent_dir, "ranked_configs.json")

    refusal_dirs = sd.load_directions(refusal_path) if os.path.exists(refusal_path) else {}
    aggr_dirs = sd.load_directions(aggr_path) if os.path.exists(aggr_path) else {}
    game_refusal_dirs = sd.load_directions(game_refusal_path) if os.path.exists(game_refusal_path) else {}

    # Load top config.
    top_config = None
    if os.path.exists(ranked_path):
        with open(ranked_path) as f:
            ranked = json.load(f)
        if ranked:
            top_config = ranked[0]
            print(f"  Top config: L{top_config['layer']} {top_config['kind']} c={top_config['coeff']}")

    # Build steering specs.
    specs = None
    if top_config:
        de = exp["direction_extraction"]
        layers = _resolve_layers(de["layer_fractions"], num_layers)
        # candidate_breaking_vectors_v2 returns a list; convert to dict for lookup.
        candidates = {
            c["kind"] + f"_L{c['layer']}": c
            for c in sd.candidate_breaking_vectors_v2(
                refusal_dirs, aggr_dirs, game_refusal_dirs,
                None, None, [top_config["layer"]],
            )
        }
        base_kind = top_config["kind"].split("_ML")[0] if "_ML" in top_config["kind"] else top_config["kind"]
        key = base_kind + f"_L{top_config['layer']}"
        if key in candidates:
            cand = candidates[key]
            v = torch.tensor(cand["vector"], dtype=model.model.dtype, device=model.input_device)
            ml_layers = top_config.get("ml_layers")
            norm_scale = top_config.get("norm_scale", True)
            if ml_layers:
                specs = [
                    SteerSpec(layer=l, vector=v.clone(), coeff=top_config["coeff"], norm_scale=norm_scale)
                    for l in ml_layers
                ]
            else:
                specs = [SteerSpec(layer=top_config["layer"], vector=v, coeff=top_config["coeff"], norm_scale=norm_scale)]
            print(f"  Steering specs ready: {len(specs)} layer(s)")
        else:
            print(f"  ⚠ Could not find steering vector {key}; running baseline only")

    # Run experiments.
    modes = ["automated", "llm_judge"] if args.opponent == "both" else [args.opponent]
    all_results = {}

    for mode in modes:
        _print_sep(f"OPPONENT MODE: {mode.upper()}")

        # Baseline (no steering).
        print(f"\n  --- Baseline ({mode}) ---")
        baseline_records = run_games_with_opponent(
            model, opponent_mode=mode,
            n_per_game=args.n_per_game, seed=42,
            system=SYSTEM_PROMPT, specs=None,
            greedy=greedy, temperature=temperature, top_p=top_p,
            n_repeats=n_repeats,
        )
        baseline_path = os.path.join(out_dir, f"baseline_{mode}.jsonl")
        write_jsonl(baseline_records, baseline_path)
        baseline_summary = _summarize_with_opponent(baseline_records, f"baseline_{mode}")
        save_json(baseline_summary, os.path.join(out_dir, f"baseline_{mode}_summary.json"))
        print(f"  Baseline mean: {baseline_summary['mean_metric']:.3f} "
              f"coh: {baseline_summary['coherence_rate']:.2f}")

        # Steered.
        if specs:
            print(f"\n  --- Steered ({mode}) ---")
            steered_records = run_games_with_opponent(
                model, opponent_mode=mode,
                n_per_game=args.n_per_game, seed=42,
                system=SYSTEM_PROMPT, specs=specs,
                greedy=greedy, temperature=temperature, top_p=top_p,
                n_repeats=n_repeats,
            )
            steered_path = os.path.join(out_dir, f"steered_{mode}.jsonl")
            write_jsonl(steered_records, steered_path)
            steered_summary = _summarize_with_opponent(steered_records, f"steered_{mode}")
            save_json(steered_summary, os.path.join(out_dir, f"steered_{mode}_summary.json"))
            print(f"  Steered mean:  {steered_summary['mean_metric']:.3f} "
                  f"coh: {steered_summary['coherence_rate']:.2f}")

            # Compute lift.
            if baseline_summary["mean_metric"] is not None and steered_summary["mean_metric"] is not None:
                lift = steered_summary["mean_metric"] - baseline_summary["mean_metric"]
                print(f"  Lift: {lift:+.3f}")

            all_results[mode] = {
                "baseline": baseline_summary,
                "steered": steered_summary,
                "lift": lift if specs else None,
            }
        else:
            all_results[mode] = {"baseline": baseline_summary}

    # Comparison table.
    if len(modes) == 2:
        _print_sep("COMPARISON: AUTOMATED vs LLM_JUDGE")
        print(f"  {'Metric':<25} {'Automated':>12} {'LLM Judge':>12} {'Diff':>8}")
        print(f"  {'-' * 25} {'-' * 12} {'-' * 12} {'-' * 8}")
        for mode in modes:
            s = all_results[mode].get("baseline", {})
            print(f"  Baseline mean           {s.get('mean_metric', 0):>12.3f} ", end="")
        print()
        for mode in modes:
            s = all_results[mode].get("steered", {})
            if s:
                print(f"  Steered mean            {s.get('mean_metric', 0):>12.3f} ", end="")
        print()

    # Save combined results.
    combined_path = os.path.join(out_dir, "opponent_comparison_results.json")
    save_json(all_results, combined_path)

    _print_sep("DONE")
    print(f"  Artifacts in: {out_dir}")
    print(f"\n  Summary:")
    for mode, res in all_results.items():
        b = res.get("baseline", {})
        s = res.get("steered", {})
        bm = b.get("mean_metric", 0) or 0
        sm = s.get("mean_metric", 0) or 0 if s else 0
        print(f"    {mode:<15}  baseline={bm:.3f}  steered={sm:.3f}  lift={sm - bm:+.3f}")


if __name__ == "__main__":
    main()
