#!/usr/bin/env python3
"""
Generate ALL publication figures for a single model in one command.

Combines figures from two scripts:
  1. plot_rlhf_break.py         — 11 figures (dose-response, coherence, etc.)
  2. plot_steering_parameter_map.py — 5 extra figures (model_anatomy, layer_heatmap,
                                     sae_layer_scan, safe_vs_aggressive_per_game,
                                     dose_response_single/multilayer overlap)
  3. lift_vs_coherence          — alias of coherence_tradeoff (for compatibility
                                     with llama31_8b figure set)

Usage:
    uv run plot_all.py --data-dir ./_results/MODEL_ID --out-dir ./figures/MODEL_ID/

Example:
    uv run plot_all.py --data-dir ./_results/llama31_8b --out-dir ./figures/llama31_8b/
    uv run plot_all.py --data-dir ./_results/ministral_3_8b --out-dir ./figures/ministral_3_8b/
"""

import argparse
import json
import os
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Publication style — must match plot_rlhf_break.py / plot_steering_parameter_map.py
# so that lift_vs_coherence (generated in this process) uses the same serif font.
plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _save_fig(fig, out_dir, name):
    """Save figure as PNG + PDF."""
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"  -> {name}.{ext}")
    plt.close(fig)


def _load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _is_multilayer_kind(kind: str) -> bool:
    return "multi" in kind.lower() or "combined" in kind.lower()


def plot_lift_vs_coherence(data_dir: str, out_dir: str):
    """Scatter behavioral shift against valid-answer rate for both searches.

    The artifact keeps its historical filename for compatibility with existing
    article figure-copying workflows; all visible labels use current terms.
    """
    agg_sweep = _load_json(os.path.join(data_dir, "publication",
                                        "sweep_coefficient_response.json"))
    baseline_summary = _load_json(os.path.join(data_dir, "baseline_summary.json"))
    baseline_mean = baseline_summary.get("mean_metric", 0.0) if baseline_summary else 0.0

    agg_points = []
    if agg_sweep:
        for row in agg_sweep:
            if _is_multilayer_kind(row.get("kind", "")):
                continue
            metric = row.get("mean_metric")
            if metric is None:
                continue
            lift = metric - baseline_mean
            coh = row.get("coherence_rate", 0)
            agg_points.append((lift, coh))

    rev = _load_json(os.path.join(data_dir, "advanced", "reverse_steer_results.json"))
    safe_points = []
    if rev and "sweep_results" in rev:
        for row in rev["sweep_results"]:
            if row.get("multi_layer", False) or _is_multilayer_kind(row.get("kind", "")):
                continue
            lift = row.get("lift")
            coh = row.get("coherence_rate")
            if lift is None or coh is None:
                continue
            safe_points.append((lift, coh))

    if not agg_points and not safe_points:
        print(f"  [skip] lift_vs_coherence: no data in {data_dir}")
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    if agg_points:
        ax.scatter([p[0] for p in agg_points], [p[1] for p in agg_points],
                   color="#e9466c", s=60, alpha=0.7, edgecolors="white", linewidth=0.5)
    if safe_points:
        ax.scatter([p[0] for p in safe_points], [p[1] for p in safe_points],
                   color="#4caf50", s=60, alpha=0.7, edgecolors="white", linewidth=0.5)

    ax.axhline(y=0.7, color="red", linestyle="--", linewidth=0.8, alpha=0.4)
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Behavioral index shift Δ")
    ax.set_ylabel("Valid-answer rate")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.1)
    _save_fig(fig, out_dir, "lift_vs_coherence")


def main():
    ap = argparse.ArgumentParser(
        description="Generate ALL publication figures for a single model"
    )
    ap.add_argument("--model", default="",
                    help="Model ID (e.g. gemma2_9b). If omitted, inferred from --data-dir.")
    ap.add_argument("--data-dir", required=True,
                    help="Directory with model artifacts (e.g. ./_results/llama31_8b)")
    ap.add_argument("--out-dir", default="./figures",
                    help="Output directory for figures")
    ap.add_argument("--model-label", default="",
                    help="Label for the model in titles")
    args = ap.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    out_dir = os.path.abspath(args.out_dir)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Infer model id from data-dir basename if not provided.
    model_id = args.model or os.path.basename(data_dir.rstrip(os.sep))

    # --- 1. plot_rlhf_break.py ---
    print("=" * 70)
    print("[1/3] Running plot_rlhf_break.py")
    print("=" * 70)
    cmd1 = [
        sys.executable,
        os.path.join(script_dir, "plot_rlhf_break.py"),
        "--data-dir", data_dir,
        "--out-dir", out_dir,
    ]
    if args.model_label:
        cmd1.extend(["--model-label", args.model_label])
    ret = subprocess.call(cmd1, cwd=script_dir)
    if ret != 0:
        print(f"[warn] plot_rlhf_break.py exited with code {ret}")

    # --- 2. plot_steering_parameter_map.py ---
    # This script needs --data-dir pointing to the advanced/ subdirectory
    # and --results-dir pointing to the parent (data_dir).
    advanced_dir = os.path.join(data_dir, "advanced")
    if not os.path.exists(os.path.join(advanced_dir, "steering_parameter_map.json")):
        print(f"\n[skip] No steering_parameter_map.json in {advanced_dir}/")
        print(f"       Skipping plot_steering_parameter_map.py figures.")
    else:
        print("\n" + "=" * 70)
        print("[2/3] Running plot_steering_parameter_map.py")
        print("=" * 70)
        cmd2 = [
            sys.executable,
            os.path.join(script_dir, "plot_steering_parameter_map.py"),
            "--model", model_id,
            "--data-dir", advanced_dir,
            "--results-dir", data_dir,
            "--out-dir", out_dir,
        ]
        ret = subprocess.call(cmd2, cwd=script_dir)
        if ret != 0:
            print(f"[warn] plot_steering_parameter_map.py exited with code {ret}")

    # --- 3. lift_vs_coherence (compatibility figure) ---
    print("\n" + "=" * 70)
    print("[3/3] Generating lift_vs_coherence")
    print("=" * 70)
    plot_lift_vs_coherence(data_dir, out_dir)

    print(f"\n{'=' * 70}")
    print(f"All figures saved to: {out_dir}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
