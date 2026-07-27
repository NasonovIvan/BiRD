#!/usr/bin/env python3
"""Plot the "where aggressive & safety parameters live" diagrams locally.

This script reads ``steering_parameter_map.json`` (and the full
``reverse_steer_results.json``) from a local directory and renders
publication-quality figures using matplotlib.

Key design principles (article-ready):
  - NO titles on figures (captions go into a separate captions file).
  - NO legends on figures (legends go into a separate legends file).
  - All figures saved as both PNG and PDF.
  - Uninformative figures removed (top_configs, direction_signs,
    statistical_summary — these become LaTeX tables in the article).

Usage:
    python plot_steering_parameter_map.py \\
        --data-dir ./_results/llama31_8b/advanced \\
        --out-dir ./figures/llama31_8b

Figures generated:
    1. model_anatomy.png          — 32-layer model diagram with injection points
    2. dose_response_single.png   — dose-response for single-layer directions only
    3. dose_response_multilayer.png — dose-response for multi-layer directions only
    4. layer_heatmap.png          — layer × direction heatmap of aggression lift
    5. direction_similarity.png   — direction cosine similarity across layers
    6. justification_comparison.png — baseline vs safe justification examples
    7. sae_layer_scan.png         — SAE auto-layer selection scores (NEW)
    8. safe_vs_aggressive_per_game.png — per-game baseline/broken/safe bars (NEW)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
except ImportError:
    print("ERROR: matplotlib is required. Install with: pip install matplotlib")
    sys.exit(1)

# Publication style — no titles, minimal chrome.
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

# Colors.
C_AGGRESSION = "#EE6677"
C_SAFETY = "#4477AA"
C_BASELINE = "#4477AA"
C_SAFE = "#228833"
C_BROKEN = "#EE6677"
C_GRID = "#CCCCCC"
C_LAYER_BG = "#F0F0F0"
C_INJECT = "#FF9977"



# ---------------------------------------------------------------------------
# Local data loading
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_json(path: str) -> dict | list | None:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _reported_direction_name(kind: str) -> str:
    """Map historical implementation keys to neutral figure terminology."""
    base = kind.split("_ML", 1)[0]
    names = {
        "refusal_pos": "generic refusal (+)",
        "refusal_neg": "generic refusal (-)",
        "aggression_pos": "decision-derived (+)",
        "aggression_neg": "decision-derived (-)",
        "game_refusal_pos": "game-conditioned (+)",
        "game_refusal_neg": "game-conditioned (-)",
        "game_combined": "game-combined (+)",
        "safety_combined": "combined lower-index",
        "game_safety_combined": "game-combined lower-index",
        "sae_aggression_pos": "SAE-derived (+)",
        "sae_aggression_neg": "SAE-derived (-)",
    }
    return names.get(base, base.replace("_", " "))


def load_data(data_dir: str) -> dict:
    """Load the steering parameter map data from a local directory."""
    # Try steering_parameter_map.json first (compact, plot-ready).
    pmap = _load_json(os.path.join(data_dir, "steering_parameter_map.json"))
    if pmap is not None:
        return pmap

    # Fall back to reverse_steer_results.json (full results).
    full = _load_json(os.path.join(data_dir, "reverse_steer_results.json"))
    if full is not None:
        return full

    raise FileNotFoundError(
        f"No steering_parameter_map.json or reverse_steer_results.json in {data_dir}"
    )


# ---------------------------------------------------------------------------
# Save helper — PNG + PDF, no title
# ---------------------------------------------------------------------------

def _save_fig(fig, out_dir: str, name: str):
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, f"{name}.png")
    pdf_path = os.path.join(out_dir, f"{name}.pdf")
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"  → {name}.png + {name}.pdf")


# ---------------------------------------------------------------------------
# Figure 1: Model anatomy diagram
# ---------------------------------------------------------------------------

def plot_model_anatomy(data: dict, out_dir: str, sae_info: dict | None = None):
    """Draw the 32-layer model with injection points highlighted.

    Fixes:
      - SAE decomposition layer is read from sae_extraction_info.json (layer 8),
        not hardcoded to 16.
      - Best safety config layer (25) is highlighted separately.
      - No title on the figure (caption goes to captions file).
      - No legend on the figure (legend goes to legends file).
    """
    model_info = data.get("model_info", {})
    n_layers = model_info.get("num_layers", 32)
    hidden = model_info.get("hidden_dim", 4096)

    best = data.get("best_config", {})
    best_layer = best.get("layer", 25)
    best_kind = best.get("kind", "")
    best_coeff = best.get("coeff", 3.0)

    all_layers = data.get("all_layers", [8, 12, 16, 19, 22, 25, 28])
    ml_groups = data.get("ml_groups", [])

    # SAE layer from extraction info (auto-selected = layer 8).
    sae_layer = 8
    if sae_info:
        sae_layer = sae_info.get("layer", 8)

    fig, ax = plt.subplots(figsize=(8, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(-1, n_layers + 2)
    ax.set_aspect("equal")
    ax.axis("off")

    # Draw each layer as a horizontal bar.
    for layer in range(n_layers):
        y = layer
        is_candidate = layer in all_layers
        is_best = layer == best_layer
        is_sae = layer == sae_layer
        is_ml = any(layer in g for g in ml_groups)

        # Background color.
        if is_best:
            color = "#FFD700"  # gold for best safety
        elif is_sae:
            color = "#DDA0DD"  # plum for SAE
        elif is_candidate:
            color = C_LAYER_BG
        else:
            color = "#FAFAFA"

        rect = FancyBboxPatch((1.5, y - 0.35), 7, 0.7,
                              boxstyle="round,pad=0.1",
                              facecolor=color, edgecolor="#888888",
                              linewidth=0.8 if not is_best else 2.0)
        ax.add_patch(rect)
        ax.text(5, y, f"Layer {layer}", ha="center", va="center", fontsize=8)

        # Annotations on the right.
        labels = []
        if is_best:
            labels.append(f"* BEST SAFETY\n   {best_kind}\n   c={best_coeff}")
        if is_sae:
            labels.append("SAE layer\n(d=16384, k=64)")
        if is_ml and not is_best:
            for gi, g in enumerate(ml_groups):
                if layer in g:
                    labels.append(f"ML group {gi}: {g}")
        if labels:
            ax.text(8.7, y, "\n".join(labels), ha="left", va="center",
                    fontsize=7, color=C_AGGRESSION if is_best else "#555555")

    # Draw embedding and output.
    ax.text(5, -0.5, "Token Embeddings", ha="center", va="center",
            fontsize=10, style="italic", color="#666666")
    ax.annotate("", xy=(5, -0.1), xytext=(5, 0),
                arrowprops=dict(arrowstyle="->", color="#666666"))
    # No legend on figure — it goes to the legends file.
    _save_fig(fig, out_dir, "model_anatomy")


# ---------------------------------------------------------------------------
# Figure 2: Dose-response curves — SPLIT into single-layer and multi-layer
# ---------------------------------------------------------------------------

def _is_multilayer_kind(kind: str) -> bool:
    return "_ML" in kind


def _humanize_game_name(name: str) -> str:
    """Convert snake_case game name to human-readable form."""
    return " ".join(w.capitalize() for w in name.replace("_", " ").split())


def plot_dose_response_single(data: dict, out_dir: str):
    """Dose-response for single-layer directions only.

    This keeps the figure readable by showing only the 5 single-layer
    direction kinds (refusal_pos, aggression_neg, safety_combined,
    game_refusal_pos, game_safety_combined, sae_aggression_neg).
    """
    sweep_by_kind = data.get("sweep_by_kind", {})
    baseline_mean = data.get("baseline_mean", 0.0)

    if not sweep_by_kind:
        print("  [skip] no sweep_by_kind data for dose_response_single")
        return

    # Filter to single-layer kinds only.
    single_kinds = {k: v for k, v in sweep_by_kind.items() if not _is_multilayer_kind(k)}
    if not single_kinds:
        print("  [skip] no single-layer kinds for dose_response_single")
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(single_kinds)))
    markers = ["o", "s", "^", "D", "v", "P"]
    for (kind, entries), color, marker in zip(
        sorted(single_kinds.items()), colors, markers
    ):
        # Group by layer, sort by coeff.
        by_layer: dict[int, list] = {}
        for e in entries:
            by_layer.setdefault(e["layer"], []).append(e)
        for layer, items in by_layer.items():
            items.sort(key=lambda x: x["coeff"])
            coeffs = [i["coeff"] for i in items]
            metrics = [
                i["mean_metric"] if i["mean_metric"] is not None else np.nan
                for i in items
            ]
            label = f"{kind} L{layer}"
            if len(items) > 1:
                ax.plot(coeffs, metrics, marker + "-", label=label,
                        color=color, markersize=5, alpha=0.8)
            else:
                ax.plot(coeffs, metrics, marker, label=label,
                        color=color, markersize=5, alpha=0.8)

    ax.axhline(y=baseline_mean, color=C_BASELINE, linestyle="--", linewidth=1.5,
               alpha=0.6)
    ax.set_xlabel("Coefficient magnitude |α|")
    ax.set_ylabel("Selfish/non-cooperative action index")
    ax.grid(True, alpha=0.3)
    _save_fig(fig, out_dir, "dose_response_single")


def plot_dose_response_multilayer(data: dict, out_dir: str):
    """Dose-response for multi-layer directions only.

    This separates the multi-layer (ML) steering configs into their own
    figure so neither plot becomes unreadable.
    """
    sweep_by_kind = data.get("sweep_by_kind", {})
    baseline_mean = data.get("baseline_mean", 0.0)

    if not sweep_by_kind:
        print("  [skip] no sweep_by_kind data for dose_response_multilayer")
        return

    # Filter to multi-layer kinds only.
    ml_kinds = {k: v for k, v in sweep_by_kind.items() if _is_multilayer_kind(k)}
    if not ml_kinds:
        print("  [skip] no multi-layer kinds for dose_response_multilayer")
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(ml_kinds)))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    for (kind, entries), color, marker in zip(
        sorted(ml_kinds.items()), colors, markers
    ):
        # Group by layer, sort by coeff.
        by_layer: dict[int, list] = {}
        for e in entries:
            by_layer.setdefault(e["layer"], []).append(e)
        for layer, items in by_layer.items():
            items.sort(key=lambda x: x["coeff"])
            coeffs = [i["coeff"] for i in items]
            metrics = [
                i["mean_metric"] if i["mean_metric"] is not None else np.nan
                for i in items
            ]
            label = f"{kind} L{layer}"
            if len(items) > 1:
                ax.plot(coeffs, metrics, marker + "-", label=label,
                        color=color, markersize=5, alpha=0.8)
            else:
                ax.plot(coeffs, metrics, marker, label=label,
                        color=color, markersize=5, alpha=0.8)

    ax.axhline(y=baseline_mean, color=C_BASELINE, linestyle="--", linewidth=1.5,
               alpha=0.6)
    ax.set_xlabel("Coefficient magnitude |α|")
    ax.set_ylabel("Selfish/non-cooperative action index")
    ax.grid(True, alpha=0.3)
    _save_fig(fig, out_dir, "dose_response_multilayer")


# ---------------------------------------------------------------------------
# Figure 3: Layer × direction heatmap
# ---------------------------------------------------------------------------

def plot_layer_heatmap(data: dict, out_dir: str):
    """Heatmap of aggression lift by layer and direction kind.

    Fixes:
      - Removed the colorbar side label "negative = more safe".
      - No title on the figure.
      - Note: sae_aggression_neg only has data for layer 8 because the SAE
        was trained on layer 8 (auto-selected as optimal). This is expected
        and correct — the SAE direction only exists at that layer.
    """
    all_sweep = data.get("all_sweep_results", [])
    if not all_sweep:
        print("  [skip] no sweep data for layer_heatmap")
        return

    # Collect unique layers and kinds (single-layer only for clarity).
    single = [s for s in all_sweep if not s.get("multi_layer", False)]
    if not single:
        single = all_sweep

    layers = sorted(set(s["layer"] for s in single))
    kinds = sorted(set(s["kind"] for s in single))

    # Build matrix of best lift per (layer, kind).
    matrix = np.full((len(kinds), len(layers)), np.nan)
    for s in single:
        ki = kinds.index(s["kind"])
        li = layers.index(s["layer"])
        val = s.get("lift")
        if val is not None and not np.isnan(val):
            if np.isnan(matrix[ki, li]) or abs(val) > abs(matrix[ki, li]):
                matrix[ki, li] = val

    fig, ax = plt.subplots(figsize=(10, 5))
    if not np.all(np.isnan(matrix)):
        vmax = np.nanmax(np.abs(matrix))
        if vmax == 0:
            vmax = 1.0
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", norm=norm)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Behavioral index shift Δ")
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([f"L{l}" for l in layers])
    ax.set_yticks(range(len(kinds)))
    ax.set_yticklabels([_reported_direction_name(k) for k in kinds], fontsize=9)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Direction kind")

    # Annotate cells.
    for i in range(len(kinds)):
        for j in range(len(layers)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                        fontsize=7,
                        color="white" if abs(val) > vmax * 0.5 else "black")

    _save_fig(fig, out_dir, "layer_heatmap")


# ---------------------------------------------------------------------------
# Figure 4: Direction cosine similarity (renamed from direction_heatmap)
# ---------------------------------------------------------------------------

def plot_direction_similarity(data_dir: str, out_dir: str):
    """Cosine similarity between extracted directions across layers.

    This replaces the old 'direction_heatmap' which was actually bar charts.
    Now it's a proper heatmap showing all 3 direction pairs × all layers.
    Reads from publication/direction_cosine_similarity.json.
    """
    cos_data = _load_json(
        os.path.join(data_dir, "publication", "direction_cosine_similarity.json")
    )
    if not cos_data:
        print(f"  [skip] direction_similarity: no cosine data in {data_dir}")
        return

    layers = [r["layer"] for r in cos_data]
    pairs = [
        ("cos_refusal_aggression", "generic / decision-derived"),
        ("cos_refusal_game_refusal", "generic / game-conditioned"),
        ("cos_aggression_game_refusal", "decision / game-conditioned"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, (key, label) in zip(axes, pairs):
        vals = [r.get(key) for r in cos_data]
        vals_clean = [v if v is not None else 0 for v in vals]
        x = range(len(layers))
        colors_bar = [C_SAFETY if v >= 0 else C_AGGRESSION for v in vals_clean]
        ax.bar(x, vals_clean, color=colors_bar, edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers, fontsize=8)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Cosine similarity")
        ax.axhline(y=0, color="gray", linewidth=0.5)
        ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    _save_fig(fig, out_dir, "direction_similarity")


# ---------------------------------------------------------------------------
# Figure 5: Justification comparison
# ---------------------------------------------------------------------------

def plot_justification_comparison(data: dict, out_dir: str):
    """Side-by-side text comparison of baseline vs safe justifications."""
    examples = data.get("justification_examples", [])
    if not examples:
        print("  [skip] no justification examples")
        return

    n = min(5, len(examples))
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), squeeze=False)

    for i, ex in enumerate(examples[:n]):
        ax = axes[i, 0]
        ax.axis("off")
        game = ex.get("game", "?")
        prompt = ex.get("prompt", "")[:100]
        b_dec = ex.get("baseline_decision", "?")
        s_dec = ex.get("safe_decision", "?")
        b_m = ex.get("baseline_metric")
        s_m = ex.get("safe_metric")
        b_just = ex.get("baseline_justification", "")[:200]
        s_just = ex.get("safe_justification", "")[:200]

        b_m_str = f"{b_m:.2f}" if b_m is not None else "N/A"
        s_m_str = f"{s_m:.2f}" if s_m is not None else "N/A"

        text = (f"[{i+1}] Game: {game}\n"
                f"Prompt: {prompt}...\n\n"
                f"  Baseline: decision={b_dec!r}  metric={b_m_str}\n"
                f"    {b_just}\n\n"
                f"  Safe:     decision={s_dec!r}  metric={s_m_str}\n"
                f"    {s_just}")

        color = C_BASELINE if i % 2 == 0 else C_SAFE
        ax.text(0.02, 0.98, text, transform=ax.transAxes,
                fontsize=8, verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8F8F8",
                          edgecolor=color, alpha=0.9))

    _save_fig(fig, out_dir, "justification_comparison")


# ---------------------------------------------------------------------------
# NEW Figure 6: SAE auto-layer selection scan
# ---------------------------------------------------------------------------

def plot_sae_layer_scan(sae_info: dict | None, out_dir: str):
    """Bar chart of SAE auto-layer selection scores.

    Shows Cohen's d, dead fraction, and combined score for each candidate
    layer, highlighting the selected layer (8).
    """
    if not sae_info:
        print("  [skip] no SAE extraction info for sae_layer_scan")
        return

    auto = sae_info.get("auto_layer_selection", {})
    layer_scores = auto.get("layer_scores", [])
    if not layer_scores:
        print("  [skip] no layer_scores in SAE info")
        return

    best_layer = sae_info.get("layer", 8)

    # Sort by layer number so the X axis is in numerical order.
    layer_scores = sorted(layer_scores, key=lambda r: r["layer"])
    layers = [r["layer"] for r in layer_scores]
    cohens_d = [r["cohens_d"] for r in layer_scores]
    scores = [r["combined_score"] for r in layer_scores]
    dead_fracs = [r["dead_frac"] for r in layer_scores]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel 1: Cohen's d
    ax = axes[0]
    colors = ["#FFD700" if l == best_layer else C_SAFETY for l in layers]
    ax.bar(range(len(layers)), cohens_d, color=colors, edgecolor="white")
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([f"L{l}" for l in layers], fontsize=9)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cohen's d")
    ax.grid(True, axis="y", alpha=0.3)

    # Panel 2: Combined score
    ax = axes[1]
    ax.bar(range(len(layers)), scores, color=colors, edgecolor="white")
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([f"L{l}" for l in layers], fontsize=9)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Combined score")
    ax.grid(True, axis="y", alpha=0.3)

    # Panel 3: Dead feature fraction
    ax = axes[2]
    ax.bar(range(len(layers)), dead_fracs, color=colors, edgecolor="white")
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([f"L{l}" for l in layers], fontsize=9)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Dead feature fraction")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    _save_fig(fig, out_dir, "sae_layer_scan")


# ---------------------------------------------------------------------------
# NEW Figure 7: Lift vs Coherence scatter (from reverse_steer data)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# NEW Figure 8: Per-game baseline / broken / safe bars
# ---------------------------------------------------------------------------

def plot_safe_vs_aggressive_per_game(
    data_dir: str, data: dict, out_dir: str
):
    """Grouped bar chart: baseline vs broken (aggressive) vs safe per game.

    Combines data from:
      - comparison_report.json (baseline vs broken per game)
      - steering_parameter_map.json best_config (overall safe mean)
    """
    # Baseline vs broken per game from comparison_report.
    report = _load_json(os.path.join(data_dir, "comparison_report.json"))
    if not report:
        print("  [skip] no comparison_report for per_game bars")
        return

    per_game = report.get("per_game", {})
    if not per_game:
        print("  [skip] no per_game data in comparison_report")
        return

    games = list(per_game.keys())

    def _safe(v):
        """Return v if it's a finite number, else 0.0 (handles None/NaN)."""
        if v is None:
            return 0.0
        try:
            f = float(v)
        except (TypeError, ValueError):
            return 0.0
        if not np.isfinite(f):
            return 0.0
        return f

    baseline_means = [_safe(per_game[g].get("baseline_mean")) for g in games]
    broken_means = [_safe(per_game[g].get("broken_mean")) for g in games]

    # Safe mean: overall from best_config (we only have overall, not per-game).
    best = data.get("best_config", {})
    safe_mean_overall = best.get("mean_metric", 0)

    x = np.arange(len(games))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, baseline_means, width, color=C_BASELINE,
           edgecolor="white", linewidth=0.5)
    ax.bar(x, broken_means, width, color=C_BROKEN,
           edgecolor="white", linewidth=0.5)
    # Safe: overall mean repeated (we don't have per-game safe breakdown).
    ax.bar(x + width, [safe_mean_overall] * len(games), width, color=C_SAFE,
           edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Mean selfish/non-cooperative action index")
    ax.set_xticks(x)
    ax.set_xticklabels([_humanize_game_name(g) for g in games], fontsize=9, rotation=30, ha="right")
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis="y", alpha=0.3)
    _save_fig(fig, out_dir, "safe_vs_aggressive_per_game")


# ---------------------------------------------------------------------------
# Legends & captions file
# ---------------------------------------------------------------------------

def write_legends_and_captions(data: dict, sae_info: dict | None, out_dir: str):
    """Write a single text file with all figure legends and captions."""
    path = os.path.join(out_dir, "figure_legends_captions.txt")
    best = data.get("best_config", {})
    baseline_mean = data.get("baseline_mean", 0.0)

    lines = []
    lines.append("=" * 70)
    lines.append("FIGURE LEGENDS AND CAPTIONS")
    lines.append("=" * 70)
    lines.append("")

    # model_anatomy
    lines.append("-" * 70)
    lines.append("model_anatomy.png / .pdf")
    lines.append("-" * 70)
    lines.append("Caption: Layer-wise anatomy of Llama-3.1-8B (32 layers, "
                "hidden=4096) showing steering injection points.")
    lines.append("")
    lines.append("Legend (color coding):")
    lines.append("  Gold (*)  = Best safety config layer "
                 f"(layer {best.get('layer', 25)}, "
                 f"{best.get('kind', '')}, α={best.get('coeff', 3.0)})")
    sae_layer = sae_info.get("layer", 8) if sae_info else 8
    lines.append(f"  Plum      = SAE decomposition layer (layer {sae_layer}, "
                 "d=16384, k=64, auto-selected)")
    lines.append("  Light gray = Candidate extraction layers "
                 "(layers 8, 12, 16, 19, 22, 25, 28)")
    lines.append("  White     = Other layers (not used for steering)")
    lines.append("")
    lines.append("Note: The SAE was trained on layer 8, which was auto-selected "
                 "as optimal (Cohen's d=1.21, combined score=0.043). "
                 "The best safety steering config uses layer 25 with the "
                 "game_refusal_pos direction applied across multi-layer group "
                 "[8,12,16,19,22,25].")
    lines.append("")

    # dose_response_single
    lines.append("-" * 70)
    lines.append("dose_response_single.png / .pdf")
    lines.append("-" * 70)
    lines.append("Caption: Dose-response curves for single-layer steering "
                 "directions. X-axis: steering coefficient α. Y-axis: aggression "
                 "metric (lower = more fair/safe). Dashed line = baseline "
                 f"({baseline_mean:.3f}).")
    lines.append("")
    lines.append("Legend (line colors/markers):")
    sweep_by_kind = data.get("sweep_by_kind", {})
    single_kinds = sorted(k for k in sweep_by_kind if not _is_multilayer_kind(k))
    colors = plt.cm.tab10(np.linspace(0, 1, len(single_kinds)))
    markers = ["o", "s", "^", "D", "v", "P"]
    for i, kind in enumerate(single_kinds):
        entries = sweep_by_kind[kind]
        layers = sorted(set(e["layer"] for e in entries))
        for layer in layers:
            lines.append(f"  {markers[i % len(markers)]} {kind} L{layer} "
                         f"(color #{i+1})")
    lines.append("")

    # dose_response_multilayer
    lines.append("-" * 70)
    lines.append("dose_response_multilayer.png / .pdf")
    lines.append("-" * 70)
    lines.append("Caption: Dose-response curves for multi-layer (ML) steering "
                 "directions. X-axis: steering coefficient α. Y-axis: aggression "
                 "metric (lower = more fair/safe). Dashed line = baseline "
                 f"({baseline_mean:.3f}).")
    lines.append("")
    lines.append("Legend (line colors/markers):")
    ml_kinds = sorted(k for k in sweep_by_kind if _is_multilayer_kind(k))
    colors = plt.cm.tab10(np.linspace(0, 1, len(ml_kinds)))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    for i, kind in enumerate(ml_kinds):
        entries = sweep_by_kind[kind]
        layers = sorted(set(e["layer"] for e in entries))
        for layer in layers:
            lines.append(f"  {markers[i % len(markers)]} {kind} L{layer} "
                         f"(color #{i+1})")
    lines.append("")

    # layer_heatmap
    lines.append("-" * 70)
    lines.append("layer_heatmap.png / .pdf")
    lines.append("-" * 70)
    lines.append("Caption: Aggression lift (Δ from baseline) by layer × "
                 "direction kind. Red = increased aggression, blue = decreased "
                 "aggression (more safe). Only single-layer directions shown.")
    lines.append("")
    lines.append("Note: sae_aggression_neg has data only for layer 8 because "
                 "the SAE was trained exclusively on layer 8 (auto-selected). "
                 "The SAE-derived direction does not exist at other layers.")
    lines.append("")

    # direction_similarity
    lines.append("-" * 70)
    lines.append("direction_similarity.png / .pdf")
    lines.append("-" * 70)
    lines.append("Caption: Cosine similarity between extracted steering "
                 "directions across layers. Left: cos(refusal, aggression). "
                 "Middle: cos(refusal, game_refusal). Right: cos(aggression, "
                 "game_refusal). Blue bars = positive similarity, red bars = "
                 "negative similarity.")
    lines.append("")

    # sae_layer_scan
    if sae_info:
        lines.append("-" * 70)
        lines.append("sae_layer_scan.png / .pdf")
        lines.append("-" * 70)
        lines.append("Caption: SAE auto-layer selection scan. Left: Cohen's d "
                     "(effect size between aggressive and fair activations). "
                     "Middle: Combined score (quality × (1-dead_frac)). "
                     "Right: Dead feature fraction. Gold bar = selected layer.")
        lines.append("")
        auto = sae_info.get("auto_layer_selection", {})
        for r in auto.get("layer_scores", []):
            lines.append(f"  Layer {r['layer']}: Cohen's d={r['cohens_d']:.3f}  "
                         f"dead={r['dead_frac']:.3f}  fvu={r['fvu']:.6f}  "
                         f"score={r['combined_score']:.4f}")
        lines.append(f"  * SELECTED: layer {sae_info.get('layer', 8)}")
        lines.append("")

    # safe_vs_aggressive_per_game
    lines.append("-" * 70)
    lines.append("safe_vs_aggressive_per_game.png / .pdf")
    lines.append("-" * 70)
    lines.append("Caption: Per-game aggression metric for baseline, "
                 "aggressive-steered (broken), and safe-steered configs. "
                 "Blue = baseline, red = aggressive steering, green = safe "
                 "steering (best config: "
                 f"{best.get('kind', '')}, α={best.get('coeff', 3.0)}).")
    lines.append("")
    lines.append("Note: Safe steering values are the overall mean (not "
                 "per-game breakdown available in data).")
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  → figure_legends_captions.txt")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Plot steering parameter map diagrams from local data"
    )
    ap.add_argument("--model", default="llama31_8b",
                    help="Model ID (used for default data-dir path)")
    ap.add_argument("--data-dir", default=None,
                    help="Local directory with steering_parameter_map.json")
    ap.add_argument("--results-dir", default=None,
                    help="Root results dir (parent of advanced/) for "
                         "sae_extraction_info.json and comparison_report.json")
    ap.add_argument("--out-dir", default="./figures",
                    help="Output directory for figures")
    args = ap.parse_args()

    # Determine data directory.
    if args.data_dir:
        data_dir = args.data_dir
    else:
        # Try default local path.
        data_dir = os.path.join("./_plotdata", args.model, "advanced")
        if not os.path.exists(os.path.join(data_dir, "steering_parameter_map.json")):
            print("No local data found. Use --data-dir to specify a local directory.")
            sys.exit(1)

    # Results dir (parent of advanced/) for SAE info and comparison report.
    results_dir = args.results_dir or os.path.dirname(data_dir)

    print(f"[load] Loading data from {data_dir}...")
    data = load_data(data_dir)
    print(f"[load] Loaded. Best config: {data.get('best_config', {})}")

    # Load SAE extraction info.
    sae_info = _load_json(os.path.join(results_dir, "sae_extraction_info.json"))
    if sae_info:
        print(f"[load] SAE info: layer={sae_info.get('layer')}")

    out_dir = args.out_dir
    print(f"\n[plot] Generating figures in {out_dir}/...")

    # Generate figures (no titles, no legends on figures).
    plot_model_anatomy(data, out_dir, sae_info=sae_info)
    plot_dose_response_single(data, out_dir)
    plot_dose_response_multilayer(data, out_dir)
    plot_layer_heatmap(data, out_dir)
    plot_direction_similarity(results_dir, out_dir)
    plot_sae_layer_scan(sae_info, out_dir)
    plot_safe_vs_aggressive_per_game(results_dir, data, out_dir)

    # Write legends & captions file.
    write_legends_and_captions(data, sae_info, out_dir)

    print(f"\n[done] All figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
