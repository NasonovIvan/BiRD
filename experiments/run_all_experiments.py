#!/usr/bin/env python3
"""Unified runner: runs all RLHF-break experiments in one command.

Executes the full experiment suite for a single model:
  1. Core RLHF-break experiment (baseline → directions → sweep → final eval → publication)
  2. Advanced experiments (steering detection, auto-alpha, justification mode)
  3. LLM-opponent experiment (automated vs LLM-judge opponents)

All results are saved locally on the machine where experiments run.

Usage:
    # Run everything for a model (weights from HuggingFace):
    python run_all_experiments.py --model llama31_8b

    # Run with local weights (from server path):
    python run_all_experiments.py --model llama31_8b --model-name /path/to/weights

    # Run only specific stages:
    python run_all_experiments.py --model llama31_8b --stages core,advanced

    # Run with custom output dir:
    python run_all_experiments.py --model llama31_8b --out-dir ./runs/my_run

    # Skip LLM-opponent (no judge API available):
    python run_all_experiments.py --model llama31_8b --stages core,advanced

Environment variables for judge API (needed for LLM-opponent experiment):
    JUDGE_API_URL=https://<host>/v1/chat/completions
    JUDGE_API_TOKEN=<token>
    JUDGE_MODEL=<model_name>
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Resolve paths.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(SCRIPT_DIR)  # emo_exp/
EXP_DIR = os.path.join(PKG_ROOT, "experiments")


def _print_sep(title: str) -> None:
    print(f"\n{'#' * 70}")
    print(f"#  {title}")
    print(f"{'#' * 70}")


def _run_subprocess(cmd: list[str], stage_name: str) -> bool:
    """Run a subprocess and stream output. Returns True on success."""
    _print_sep(f"RUNNING: {stage_name}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Working dir: {os.getcwd()}")
    start = time.time()
    try:
        result = subprocess.run(cmd, cwd=PKG_ROOT, check=False)
        elapsed = time.time() - start
        if result.returncode == 0:
            print(f"\n  ✓ {stage_name} completed in {elapsed:.1f}s")
            return True
        else:
            print(f"\n  ✗ {stage_name} FAILED (exit code {result.returncode}) after {elapsed:.1f}s")
            return False
    except Exception as e:
        print(f"\n  ✗ {stage_name} CRASHED: {e}")
        return False


def main():
    ap = argparse.ArgumentParser(
        description="Unified runner: all RLHF-break experiments in one command"
    )
    ap.add_argument("--model", required=True,
                    help="Model ID from config/models/ (e.g. llama31_8b, qwen3_4b)")
    ap.add_argument("--model-name", default=None,
                    help="Override weight loading path (local dir or HF repo). "
                         "Use this to point to local weights on the server, e.g. "
                         "--model-name /data/models/llama-3.1-8b-instruct")
    ap.add_argument("--config", default=None,
                    help="Path to rlhf_break.json config. If omitted, auto-resolves "
                         "config/rlhf_break/<model>.json (per-model), falling back to "
                         "config/rlhf_break.json")
    ap.add_argument("--out-dir", default=None,
                    help="Override output directory (default: runs/rlhf_break/<model>)")
    ap.add_argument("--stages", default="all",
                    help="Comma-separated stages to run: core,advanced,llm_opponent,all")
    ap.add_argument("--opponent", default="both",
                    choices=["automated", "llm_judge", "both"],
                    help="LLM-opponent mode (default: both)")
    ap.add_argument("--target-lift", type=float, default=0.15,
                    help="Auto-alpha: target aggression lift (default: 0.15)")
    ap.add_argument("--coh-threshold", type=float, default=None,
                    help="Auto-alpha: coherence threshold (default: use per-model config)")
    ap.add_argument("--n-per-game", type=int, default=8,
                    help="Scenarios per game for advanced experiments")
    ap.add_argument("--skip-llm-opponent", action="store_true",
                    help="Skip LLM-opponent experiment (no judge API)")
    args = ap.parse_args()

    # Parse stages.
    if args.stages == "all":
        stages = ["core", "advanced", "llm_opponent"]
    else:
        stages = [s.strip() for s in args.stages.split(",")]
    if args.skip_llm_opponent and "llm_opponent" in stages:
        stages.remove("llm_opponent")

    _print_sep("UNIFIED RLHF-BREAK EXPERIMENT RUNNER")
    print(f"  Model ID:       {args.model}")
    if args.model_name:
        print(f"  Model weights:   {args.model_name}")
    print(f"  Stages:         {', '.join(stages)}")
    print(f"  Output dir:      {args.out_dir or '(default: runs/rlhf_break/<model>)'}")
    print(f"  Config:         {args.config or '(auto: config/rlhf_break/<model>.json)'}")
    print(f"  Opponent mode:  {args.opponent}")
    print(f"  Target lift:    {args.target_lift}")
    coh_display = args.coh_threshold if args.coh_threshold is not None else "(from config)"
    print(f"  Coh threshold:  {coh_display}")

    # Auto-resolve per-model config: if --config not given, look for
    # config/rlhf_break/<model>.json first (per-model tuned), falling back to
    # the master config/rlhf_break.json. This is critical because per-model
    # configs have model-specific settings (e.g. norm_scale=true for Gemma).
    if args.config:
        cfg_path = args.config
    else:
        per_model = os.path.join(PKG_ROOT, "config", "rlhf_break", f"{args.model}.json")
        if os.path.exists(per_model):
            cfg_path = per_model
            print(f"  [config] auto-resolved per-model config: {per_model}")
        else:
            cfg_path = os.path.join(PKG_ROOT, "config", "rlhf_break.json")
    with open(cfg_path) as f:
        exp_cfg = json.load(f)

    # Check judge API for LLM-opponent.
    if "llm_opponent" in stages:
        judge_url = os.environ.get("JUDGE_API_URL")
        if not judge_url:
            print(f"\n  ⚠ JUDGE_API_URL not set — LLM-opponent will fall back to automated mode.")
        else:
            print(f"\n  Judge API:      {judge_url}")
            print(f"  Judge model:    {os.environ.get('JUDGE_MODEL', '(default)')}")

    # Build common args. Always pass --config so subcommands use the same
    # (possibly auto-resolved per-model) config.
    common_args = ["--model", args.model, "--config", cfg_path]
    if args.model_name:
        common_args.extend(["--model-name", args.model_name])
    if args.out_dir:
        common_args.extend(["--out-dir", args.out_dir])

    results = {}
    overall_start = time.time()

    # --- Stage 1: Core experiment ---
    if "core" in stages:
        cmd = [sys.executable, os.path.join(EXP_DIR, "run_rlhf_break.py")] + common_args
        results["core"] = _run_subprocess(cmd, "CORE EXPERIMENT (baseline → directions → sweep → final → publication)")

    # --- Stage 2: Advanced experiments ---
    if "advanced" in stages:
        cmd = [
            sys.executable, os.path.join(EXP_DIR, "run_advanced_experiments.py"),
            "--model", args.model,
            "--config", cfg_path,
            "--experiment", "all",
            "--target-lift", str(args.target_lift),
            "--n-per-game", str(args.n_per_game),
        ]
        # Only pass --coh-threshold if explicitly set; otherwise the advanced
        # experiment will use the per-model config's coherence_threshold.
        if args.coh_threshold is not None:
            cmd.extend(["--coh-threshold", str(args.coh_threshold)])
        if args.model_name:
            cmd.extend(["--model-name", args.model_name])
        if args.out_dir:
            cmd.extend(["--out-dir", args.out_dir])
        results["advanced"] = _run_subprocess(cmd, "ADVANCED EXPERIMENTS (detection → auto-alpha → justification → reverse_steer)")

    # --- Stage 3: LLM-opponent experiment ---
    if "llm_opponent" in stages:
        cmd = [
            sys.executable, os.path.join(EXP_DIR, "run_llm_opponent_experiment.py"),
            "--model", args.model,
            "--config", cfg_path,
            "--opponent", args.opponent,
            "--n-per-game", str(args.n_per_game),
        ]
        if args.model_name:
            cmd.extend(["--model-name", args.model_name])
        if args.out_dir:
            cmd.extend(["--out-dir", args.out_dir])
        results["llm_opponent"] = _run_subprocess(cmd, "LLM-OPPONENT EXPERIMENT (automated vs LLM-judge)")

    # --- Summary ---
    overall_elapsed = time.time() - overall_start
    _print_sep("SUMMARY")
    print(f"  Total time: {overall_elapsed:.1f}s ({overall_elapsed / 60:.1f} min)")
    for stage, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {stage:<20} {status}")

    # Determine output directory.
    out_dir = args.out_dir or os.path.join(
        PKG_ROOT, exp_cfg.get("out_dir", "runs/rlhf_break"), args.model
    )
    print(f"\n  Artifacts:  {out_dir}")
    print(f"  Advanced:   {os.path.join(out_dir, 'advanced')}")
    print(f"  Opponent:   {os.path.join(out_dir, 'opponent_comparison')}")

    all_success = all(results.values())
    if all_success:
        print(f"\n  ✓ All stages completed successfully.")
    else:
        failed = [s for s, ok in results.items() if not ok]
        print(f"\n  ⚠ Some stages failed: {', '.join(failed)}")
        print(f"  You can re-run failed stages individually:")
        for s in failed:
            if s == "core":
                print(f"    python experiments/run_rlhf_break.py --model {args.model}" +
                      (f" --model-name {args.model_name}" if args.model_name else ""))
            elif s == "advanced":
                print(f"    python experiments/run_advanced_experiments.py --model {args.model}" +
                      (f" --model-name {args.model_name}" if args.model_name else ""))
            elif s == "llm_opponent":
                print(f"    python experiments/run_llm_opponent_experiment.py --model {args.model}" +
                      (f" --model-name {args.model_name}" if args.model_name else ""))

    sys.exit(0 if all_success else 1)


if __name__ == "__main__":
    main()
