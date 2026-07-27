 #!/usr/bin/env python3
"""Cross-model comparison figures for the RLHF-break article.

Reads the per-model publication artifacts from a results root directory
(one subdirectory per model) and produces summary figures that compare ALL
 models on a single plot.

Figures generated (saved as PNG + PDF):
    1. cross_model_behavior.pdf        — baseline / broken / safe bars per model
    2. cross_model_lift_coherence.pdf   — lift vs coherence scatter, one color per model
    3. cross_model_before_after.pdf     — mean per-decision shift (agg vs safe) per model
    4. cross_model_layer_heatmap.pdf   — avg aggression lift by relative layer × model
    5. cross_model_direction_similarity.pdf — avg cosine similarity per model (3 pairs)
    6. cross_model_dose_response.pdf    — mean dose-response curve per model (agg + safe)
    7. cross_model_effect_size.pdf      — Cliff's delta per game × model (agg + safe)
    8. cross_model_refusal_aggression.pdf — refusal rate vs aggression, per model

Usage:
    uv run plot_cross_model.py --data-root ./_results --out-dir ./cross_model
    uv run plot_cross_model.py --data-root ./_results --out-dir ./cross_model \\
        --exclude qwen35_9b

Design principles (consistent with the rest of the figure set):
    - Serif font, no titles, minimal chrome.
    - Consistent color palette across all figures (one color per model).
    - Compact legends; if a legend is large it is placed below the axes.
    - All figures saved as both PNG and PDF.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
except ImportError:
    print("ERROR: matplotlib is required.", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Publication style — must match plot_rlhf_break.py / plot_steering_parameter_map.py
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 14,
    "font.family": "serif",
    "axes.labelsize": 15,
    "axes.titlesize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Fixed color palette — one color per model.  Chosen for print contrast and
# color-blind friendliness (Okabe-Ito inspired).
MODEL_COLORS = [
    "#EE6677",  # red-pink   (aggression)
    "#4477AA",  # blue       (baseline/safety)
    "#228833",  # green      (safe)
    "#CCBB44",  # yellow
    "#AA3377",  # purple
    "#66CCEE",  # cyan
    "#BBBBBB",  # gray
]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

# Semantic colors reused from the single-model scripts.
C_BASELINE = "#4477AA"
C_POSITIVE = "#EE6677"
C_REVERSE_BASELINE = "#88AADD"
C_REVERSE = "#228833"
VALID_ANSWER_THRESHOLD = 0.7


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _safe(v) -> float:
    if v is None:
        return 0.0
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return f if np.isfinite(f) else 0.0


def _is_multilayer_kind(kind: str) -> bool:
    return "_ML" in kind


def _humanize_game(name: str) -> str:
    if name.upper() == "OVERALL":
        return "Overall"
    return " ".join(w.capitalize() for w in name.replace("_", " ").split())


def _save_fig(fig, out_dir: str, name: str):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {name}.png + {name}.pdf")


def _legend_below(fig, handles, labels, ncol=4, bottom=0.20, anchor=-0.03):
    """Place a legend below the figure as a separate row."""
    fig.legend(handles, labels, loc="lower center", ncol=ncol,
               bbox_to_anchor=(0.5, anchor), frameon=False,
               fontsize=12, columnspacing=1.2, handletextpad=0.5)
    fig.subplots_adjust(bottom=bottom)


# ---------------------------------------------------------------------------
# Model discovery / data loading
# ---------------------------------------------------------------------------

def discover_models(data_root: str, exclude: set[str] | None = None) -> list[str]:
    exclude = exclude or set()
    models = []
    for d in sorted(os.listdir(data_root)):
        if d in exclude:
            continue
        full = os.path.join(data_root, d)
        if not os.path.isdir(full):
            continue
        # Must have the publication artifacts.
        if not os.path.exists(os.path.join(full, "publication",
                                           "statistical_comparison.json")):
            continue
        models.append(d)
    return models


def _load_json_first(*paths: str):
    for path in paths:
        value = _load_json(path)
        if value is not None:
            return value
    return None


def _load_jsonl_first(*paths: str) -> list[dict]:
    for path in paths:
        value = _load_jsonl(path)
        if value:
            return value
    return []


def load_model_data(model_dir: str) -> dict:
    """Load cross-model artifacts, including legacy root-level advanced files."""
    pub = os.path.join(model_dir, "publication")
    adv = os.path.join(model_dir, "advanced")
    return {
        "stat": _load_json(os.path.join(pub, "statistical_comparison.json")),
        "sweep": _load_json(os.path.join(pub, "sweep_coefficient_response.json")),
        "cos": _load_json(os.path.join(pub, "direction_cosine_similarity.json")),
        "before_after": _load_json(os.path.join(pub, "before_after_examples.json")),
        "baseline_summary": _load_json(os.path.join(model_dir, "baseline_summary.json")),
        "comparison_report": _load_json(os.path.join(model_dir, "comparison_report.json")),
        "reverse": _load_json_first(
            os.path.join(adv, "reverse_steer_results.json"),
            os.path.join(model_dir, "reverse_steer_results.json"),
        ),
        "pmap": _load_json_first(
            os.path.join(adv, "steering_parameter_map.json"),
            os.path.join(model_dir, "steering_parameter_map.json"),
        ),
        "rev_baseline_records": _load_jsonl_first(
            os.path.join(adv, "reverse_steer_baseline_records.jsonl"),
            os.path.join(model_dir, "reverse_steer_baseline_records.jsonl"),
        ),
        "rev_final_records": _load_jsonl_first(
            os.path.join(adv, "reverse_steer_final_records.jsonl"),
            os.path.join(model_dir, "reverse_steer_final_records.jsonl"),
        ),
    }


# ---------------------------------------------------------------------------
# Figure 1: cross_model_behavior — separately sampled evaluations per model
# ---------------------------------------------------------------------------


def _ci_half_width(overall: dict, key: str) -> float:
    ci = overall.get(key)
    if not isinstance(ci, list) or len(ci) != 2:
        return 0.0
    return max(0.0, (_safe(ci[1]) - _safe(ci[0])) / 2)


def plot_cross_model_behavior(models: list[str], data: dict, out_dir: str):
    """Grouped bars for the positive and reverse final-test samples."""
    pos_base, positive, rev_base, reverse = [], [], [], []
    pos_base_err, positive_err, rev_base_err, reverse_err = [], [], [], []
    for m in models:
        d = data[m]
        po = (d["stat"] or {}).get("overall", {})
        ro = (d["reverse"] or {}).get("comparison_report", {}).get("overall", {})
        pos_base.append(_safe(po.get("baseline_mean")))
        positive.append(_safe(po.get("broken_mean")))
        rev_base.append(_safe(ro.get("baseline_mean")))
        reverse.append(_safe(ro.get("broken_mean")))
        pos_base_err.append(_ci_half_width(po, "baseline_ci"))
        positive_err.append(_ci_half_width(po, "broken_ci"))
        rev_base_err.append(_ci_half_width(ro, "baseline_ci"))
        reverse_err.append(_ci_half_width(ro, "broken_ci"))

    x = np.arange(len(models))
    width = 0.19
    error_kw = {"elinewidth": 0.8, "capsize": 2, "capthick": 0.8}
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - 1.5 * width, pos_base, width, yerr=pos_base_err,
           color=C_BASELINE, edgecolor="white", linewidth=0.5,
           error_kw=error_kw, label="Higher-index-test baseline")
    ax.bar(x - 0.5 * width, positive, width, yerr=positive_err,
           color=C_POSITIVE, edgecolor="white", linewidth=0.5,
           error_kw=error_kw, label="Higher-index configuration")
    ax.bar(x + 0.5 * width, rev_base, width, yerr=rev_base_err,
           color=C_REVERSE_BASELINE, edgecolor="white", linewidth=0.5,
           error_kw=error_kw, label="Lower-index-test baseline")
    ax.bar(x + 1.5 * width, reverse, width, yerr=reverse_err,
           color=C_REVERSE, edgecolor="white", linewidth=0.5,
           error_kw=error_kw, label="Lower-index configuration")
    ax.set_ylabel("Mean selfish/non-cooperative action index")
    ax.set_xticks(x)
    ax.set_xticklabels([_model_label(m) for m in models], rotation=20, ha="right",
                       fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=2,
        frameon=False,
        fontsize=11,
    )
    _save_fig(fig, out_dir, "cross_model_behavior")

# ---------------------------------------------------------------------------
# Figure 2: cross_model_lift_coherence — lift vs coherence scatter per model
# ---------------------------------------------------------------------------

def plot_cross_model_lift_coherence(models: list[str], data: dict, out_dir: str):
    """Scatter of aggression lift (x) vs coherence (y), one color per model.

    Aggressive points (positive lift) from sweep_coefficient_response;
    safe points (negative lift) from reverse_steer_results.  Only single-layer
    configs for readability.  Each model gets its own color/marker.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    handles = []
    for i, m in enumerate(models):
        d = data[m]
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        marker = MARKERS[i % len(MARKERS)]
        baseline_mean = _safe((d["baseline_summary"] or {}).get("mean_metric", 0.0))

        # Aggressive points.
        agg_pts = []
        for row in (d["sweep"] or []):
            if _is_multilayer_kind(row.get("kind", "")):
                continue
            metric = row.get("mean_metric")
            if metric is None:
                continue
            agg_pts.append((metric - baseline_mean, row.get("coherence_rate", 0)))
        # Safe points.
        safe_pts = []
        rev_sweep = d["reverse"].get("sweep_results", []) if d["reverse"] else []
        for row in rev_sweep:
            if row.get("multi_layer", False) or _is_multilayer_kind(row.get("kind", "")):
                continue
            lift = row.get("lift")
            coh = row.get("coherence_rate")
            if lift is None or coh is None:
                continue
            safe_pts.append((lift, coh))

        all_x = [p[0] for p in agg_pts + safe_pts]
        all_y = [p[1] for p in agg_pts + safe_pts]
        if not all_x:
            continue
        ax.scatter(all_x, all_y, marker=marker, color=color, s=45, alpha=0.6,
                   edgecolors="white", linewidth=0.4)
        handles.append(Line2D([0], [0], marker=marker, color="w",
                              markerfacecolor=color, markersize=8,
                              label=_model_label(m)))

    ax.axhline(y=VALID_ANSWER_THRESHOLD, color="red", linestyle="--",
               linewidth=1.0, alpha=0.6)
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Behavioral index shift Δ")
    ax.set_ylabel("Valid-answer rate")
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, alpha=0.3)
    handles.append(Line2D([0], [0], color="red", linestyle="--", linewidth=1.0,
                          label=f"Valid-answer threshold ({VALID_ANSWER_THRESHOLD:.1f})"))
    _legend_below(fig, handles, [h.get_label() for h in handles],
                  ncol=min(len(handles), 4))
    _save_fig(fig, out_dir, "cross_model_lift_coherence")


# ---------------------------------------------------------------------------
# Figure 3: cross_model_before_after — mean per-decision shift per model
# ---------------------------------------------------------------------------

def plot_cross_model_before_after(models: list[str], data: dict, out_dir: str):
    """Grouped bars of mean per-decision metric shift: aggressive vs safe.

    For each model compute:
      - agg_mean_delta = mean(broken_metric - baseline_metric) from before_after
      - safe_mean_delta = mean(safe_metric - baseline_metric) from reverse records
    """
    agg_means, safe_means = [], []
    for m in models:
        d = data[m]
        ba = d["before_after"] or []
        agg_deltas = [e["delta"] for e in ba if e.get("delta") is not None]
        agg_means.append(float(np.mean(agg_deltas)) if agg_deltas else 0.0)

        rb = d["rev_baseline_records"]
        rf = d["rev_final_records"]
        safe_deltas = []
        if rb and rf and len(rb) == len(rf):
            for b, f in zip(rb, rf):
                bm = b.get("metric")
                fm = f.get("metric")
                if bm is not None and fm is not None:
                    safe_deltas.append(fm - bm)
        safe_means.append(float(np.mean(safe_deltas)) if safe_deltas else 0.0)

    x = np.arange(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, agg_means, width, color=C_POSITIVE,
           edgecolor="white", linewidth=0.5, label="Higher-index configuration")
    ax.bar(x + width / 2, safe_means, width, color=C_REVERSE,
           edgecolor="white", linewidth=0.5, label="Lower-index configuration")
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=1.2, alpha=0.6)
    ax.set_ylabel("Mean per-decision metric shift")
    ax.set_xticks(x)
    ax.set_xticklabels([_model_label(m) for m in models], rotation=20, ha="right",
                       fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", frameon=False, fontsize=12)
    _save_fig(fig, out_dir, "cross_model_before_after")


# ---------------------------------------------------------------------------
# Figure 4: cross_model_layer_heatmap — avg lift by relative layer × model
# ---------------------------------------------------------------------------

def plot_cross_model_layer_heatmap(models: list[str], data: dict, out_dir: str):
    """Heatmap: rows = models, columns = candidate layer rank.

    For each model, average the aggression lift across all single-layer
    direction kinds at each candidate layer.  The actual layer numbers differ
    per model, so they are shown as a secondary annotation under each column
    (one line per model).  Cell values are annotated inside the cells.
    """
    n_cols = 7  # all models have 7 candidate layers
    matrix = np.full((len(models), n_cols), np.nan)
    layer_numbers: list[list[int]] = [[] for _ in models]
    for i, m in enumerate(models):
        d = data[m]
        pmap = d["pmap"] or {}
        all_sweep = pmap.get("all_sweep_results", [])
        single = [s for s in all_sweep if not s.get("multi_layer", False)]
        if not single:
            continue
        layers = sorted(set(s["layer"] for s in single))
        layer_numbers[i] = layers
        for j, layer in enumerate(layers):
            lifts = [s.get("lift") for s in single
                     if s["layer"] == layer and s.get("lift") is not None]
            if lifts:
                matrix[i, j] = float(np.mean(lifts))

    fig, ax = plt.subplots(figsize=(10, 0.7 * len(models) + 2.5))
    vmax = 1.0
    if not np.all(np.isnan(matrix)):
        vmax = np.nanmax(np.abs(matrix))
        if vmax == 0:
            vmax = 1.0
        from matplotlib.colors import TwoSlopeNorm
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", norm=norm)
        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label("Mean behavioral index shift")
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([f"rank {i+1}" for i in range(n_cols)], fontsize=12)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([_model_label(m) for m in models], fontsize=12)

    # Annotate cell values.
    for i in range(len(models)):
        for j in range(n_cols):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                        fontsize=10,
                        color="white" if abs(val) > vmax * 0.5 else "black")

    # Show the actual layer numbers per model as a framed text legend below the
    # axes.  No color swatches — just the model name and its layer list, in a
    # bordered box with serif font for a clean, publication-ready look.
    layer_labels = []
    for i, m in enumerate(models):
        nums = layer_numbers[i]
        if not nums:
            continue
        layer_labels.append(
            f"{_model_label(m)}: " + ", ".join(f"L{n}" for n in nums))
    if layer_labels:
        # Use invisible proxy artists so we get a frame with text only.
        proxies = [Line2D([0], [0], color="none", label=lab)
                   for lab in layer_labels]
        leg = fig.legend(handles=proxies,
                         loc="lower center", bbox_to_anchor=(0.5, -0.02),
                         ncol=min(len(proxies), 2), frameon=True,
                         fancybox=True, edgecolor="0.7", fontsize=11,
                         handlelength=0, handleheight=0,
                         borderpad=0.6, labelspacing=0.4,
                         columnspacing=1.5)
        fig.subplots_adjust(bottom=0.18)
        for txt in leg.get_texts():
            txt.set_family("serif")
    _save_fig(fig, out_dir, "cross_model_layer_heatmap")


# ---------------------------------------------------------------------------
# Figure 5: cross_model_direction_similarity — avg cosine per model (3 pairs)
# ---------------------------------------------------------------------------

def plot_cross_model_direction_similarity(models: list[str], data: dict,
                                           out_dir: str):
    """Grouped bars: mean cosine similarity per model for the 3 direction pairs.

    Averages each cosine pair across all layers of a model, so every model
    contributes one bar per pair.
    """
    # Stored extraction orientations differ by family: refusal and game-refusal
    # point toward refusal/fairness, whereas aggression points toward the
    # higher-index endpoint. Align all vectors to the higher-index target before
    # averaging so implementation sign conventions cannot cancel structurally.
    pairs = [
        ("cos_refusal_aggression", "generic / decision-derived", -1.0),
        ("cos_refusal_game_refusal", "generic / game-conditioned", 1.0),
        ("cos_aggression_game_refusal", "decision / game-conditioned", -1.0),
    ]
    n_pairs = len(pairs)
    n_models = len(models)
    width = 0.8 / n_models
    x = np.arange(n_pairs)
    fig, ax = plt.subplots(figsize=(10, 5))
    handles = []
    for i, m in enumerate(models):
        d = data[m]
        cos = d["cos"] or []
        means = []
        for key, _, orientation_sign in pairs:
            vals = [orientation_sign * float(r[key]) for r in cos
                    if r.get(key) is not None]
            means.append(float(np.mean(vals)) if vals else 0.0)
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        offset = (i - (n_models - 1) / 2) * width
        ax.bar(x + offset, means, width, color=color, edgecolor="white",
               linewidth=0.4)
        handles.append(Line2D([0], [0], marker="s", color="w",
                              markerfacecolor=color, markersize=9,
                              label=_model_label(m)))
    ax.axhline(y=0, color="gray", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([p[1] for p in pairs], fontsize=12)
    ax.set_ylabel("Mean cosine similarity")
    ax.grid(True, axis="y", alpha=0.3)
    _legend_below(fig, handles, [h.get_label() for h in handles],
                  ncol=min(n_models, 4))
    _save_fig(fig, out_dir, "cross_model_direction_similarity")


# ---------------------------------------------------------------------------
# Figure 6: cross_model_dose_response — mean dose-response curve per model
# ---------------------------------------------------------------------------

def plot_cross_model_dose_response(models: list[str], data: dict, out_dir: str):
    """Mean dose-response curve per model (aggressive + safe), averaged over
    single-layer directions.

    Two panels: left = aggressive steering (lift vs coeff), right = safe
    steering.  Each model is one line.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    handles = []
    for i, m in enumerate(models):
        d = data[m]
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        marker = MARKERS[i % len(MARKERS)]
        label = _model_label(m)

        # Aggressive: average lift across single-layer kinds per coeff.
        baseline_mean = _safe((d["baseline_summary"] or {}).get("mean_metric", 0.0))
        agg_by_coeff: dict[float, list[float]] = {}
        for row in (d["sweep"] or []):
            if _is_multilayer_kind(row.get("kind", "")):
                continue
            metric = row.get("mean_metric")
            if metric is None:
                continue
            agg_by_coeff.setdefault(row["coeff"], []).append(metric - baseline_mean)
        if agg_by_coeff:
            coeffs = sorted(agg_by_coeff)
            lifts = [float(np.mean(agg_by_coeff[c])) for c in coeffs]
            axes[0].plot(coeffs, lifts, marker + "-", color=color, markersize=5,
                         alpha=0.85, linewidth=1.4)

        # Safe: average lift across single-layer kinds per coeff.
        safe_by_coeff: dict[float, list[float]] = {}
        rev_sweep = d["reverse"].get("sweep_results", []) if d["reverse"] else []
        for row in rev_sweep:
            if row.get("multi_layer", False) or _is_multilayer_kind(row.get("kind", "")):
                continue
            lift = row.get("lift")
            if lift is None:
                continue
            safe_by_coeff.setdefault(row["coeff"], []).append(lift)
        if safe_by_coeff:
            coeffs = sorted(safe_by_coeff)
            lifts = [float(np.mean(safe_by_coeff[c])) for c in coeffs]
            axes[1].plot(coeffs, lifts, marker + "-", color=color, markersize=5,
                         alpha=0.85, linewidth=1.4)

        handles.append(Line2D([0], [0], marker=marker, color=color,
                              markersize=7, linewidth=1.4, label=label))

    for ax, title in zip(
        axes, ["Higher-index candidates", "Lower-index candidates"]
    ):
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Coefficient magnitude |α|")
        ax.set_title(title, fontsize=14, pad=8)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Behavioral index shift Δ")
    axes[1].set_ylabel("Behavioral index shift Δ")
    _legend_below(fig, handles, [h.get_label() for h in handles],
                  ncol=min(len(handles), 4), bottom=0.24)
    _save_fig(fig, out_dir, "cross_model_dose_response")


# ---------------------------------------------------------------------------
# Figure 7: cross_model_effect_size — Cliff's delta per game × model
# ---------------------------------------------------------------------------

def plot_cross_model_effect_size(models: list[str], data: dict, out_dir: str):
    """Grouped bars of Cliff's delta per game for aggressive and safe steering.

    Two panels: left = aggressive (per_game from statistical_comparison),
    right = safe (per_game from reverse_steer comparison_report).  Each game
    has one bar per model.
    """
    # Collect the union of games (ordered).
    games: list[str] = []
    seen = set()
    for m in models:
        d = data[m]
        for g in (d["stat"] or {}).get("per_game", {}):
            if g not in seen:
                games.append(g)
                seen.add(g)
        rev_cr = (d["reverse"] or {}).get("comparison_report", {})
        for g in rev_cr.get("per_game", {}):
            if g not in seen:
                games.append(g)
                seen.add(g)

    n_games = len(games)
    n_models = len(models)
    width = 0.8 / n_models
    x = np.arange(n_games)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    handles = []
    for i, m in enumerate(models):
        d = data[m]
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        offset = (i - (n_models - 1) / 2) * width

        # Aggressive Cliff's delta per game.
        stat_pg = (d["stat"] or {}).get("per_game", {})
        agg_vals = [_safe(stat_pg.get(g, {}).get("cliffs_delta")) for g in games]
        axes[0].bar(x + offset, agg_vals, width, color=color, edgecolor="white",
                    linewidth=0.4)

        # Safe Cliff's delta per game.
        rev_pg = (d["reverse"] or {}).get("comparison_report", {}).get("per_game", {})
        safe_vals = [_safe(rev_pg.get(g, {}).get("cliffs_delta")) for g in games]
        axes[1].bar(x + offset, safe_vals, width, color=color, edgecolor="white",
                    linewidth=0.4)

        handles.append(Line2D([0], [0], marker="s", color="w",
                              markerfacecolor=color, markersize=9,
                              label=_model_label(m)))

    axes[0].axhline(y=0, color="gray", linestyle="-", linewidth=1.0, alpha=0.6)
    axes[1].axhline(y=0, color="gray", linestyle="-", linewidth=1.0, alpha=0.6)
    axes[0].set_ylabel("Cliff's δ (higher-index)")
    axes[1].set_ylabel("Cliff's δ (lower-index)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([_humanize_game(g) for g in games], fontsize=12,
                            rotation=20, ha="right")
    # Adjust y-limits per panel to the data range (with a small margin) so
    # there is no large empty space above/below the bars.
    agg_all = [v for vals in (
        [_safe((data[m]["stat"] or {}).get("per_game", {}).get(g, {})
               .get("cliffs_delta")) for g in games] for m in models
    ) for v in vals]
    safe_all = [v for vals in (
        [_safe((data[m]["reverse"] or {}).get("comparison_report", {})
               .get("per_game", {}).get(g, {}).get("cliffs_delta")) for g in games]
        for m in models
    ) for v in vals]
    def _ylim(vals):
        if not vals:
            return -1.1, 1.1
        lo, hi = min(vals), max(vals)
        pad = max(0.05, (hi - lo) * 0.1)
        return lo - pad, hi + pad
    axes[0].set_ylim(*_ylim(agg_all))
    axes[1].set_ylim(*_ylim(safe_all))
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].grid(True, axis="y", alpha=0.3)
    _legend_below(fig, handles, [h.get_label() for h in handles],
                  ncol=min(n_models, 4), bottom=0.17)
    _save_fig(fig, out_dir, "cross_model_effect_size")


# ---------------------------------------------------------------------------
# Figure 8: cross_model_refusal_aggression — refusal vs aggression per model
# ---------------------------------------------------------------------------

def plot_cross_model_refusal_aggression(models: list[str], data: dict,
                                        out_dir: str):
    """Scatter of refusal rate (x) vs aggression metric (y), one color per model.

    Aggressive sweep configs (left panel) and safe sweep configs (right panel).
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    handles = []
    for i, m in enumerate(models):
        d = data[m]
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        marker = MARKERS[i % len(MARKERS)]
        label = _model_label(m)

        # Aggressive.
        agg_x, agg_y = [], []
        for row in (d["sweep"] or []):
            if _is_multilayer_kind(row.get("kind", "")):
                continue
            metric = row.get("mean_metric")
            refusal = row.get("refusal_rate")
            if metric is None or refusal is None:
                continue
            agg_x.append(refusal)
            agg_y.append(metric)
        if agg_x:
            axes[0].scatter(agg_x, agg_y, marker=marker, color=color, s=40,
                            alpha=0.6, edgecolors="white", linewidth=0.4)

        # Safe.
        safe_x, safe_y = [], []
        rev_sweep = d["reverse"].get("sweep_results", []) if d["reverse"] else []
        for row in rev_sweep:
            if row.get("multi_layer", False) or _is_multilayer_kind(row.get("kind", "")):
                continue
            metric = row.get("mean_metric")
            refusal = row.get("refusal_rate")
            if metric is None or refusal is None:
                continue
            safe_x.append(refusal)
            safe_y.append(metric)
        if safe_x:
            axes[1].scatter(safe_x, safe_y, marker=marker, color=color, s=40,
                            alpha=0.6, edgecolors="white", linewidth=0.4)

        handles.append(Line2D([0], [0], marker=marker, color="w",
                              markerfacecolor=color, markersize=8, label=label))

    for ax, title in zip(
        axes, ["Higher-index candidates", "Lower-index candidates"]
    ):
        ax.set_xlabel("Refusal rate")
        ax.set_xlim(-0.05, 1.05)
        ax.set_title(title, fontsize=14, pad=8)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Selfish/non-cooperative action index")
    axes[1].set_ylabel("Selfish/non-cooperative action index")
    axes[0].set_ylim(-0.05, 1.1)
    axes[1].set_ylim(-0.05, 1.1)
    _legend_below(fig, handles, [h.get_label() for h in handles],
                  ncol=min(len(handles), 4), bottom=0.24)
    _save_fig(fig, out_dir, "cross_model_refusal_aggression")


# ---------------------------------------------------------------------------
# Model label prettifier
# ---------------------------------------------------------------------------

def _model_label(name: str) -> str:
    """Convert a model dir name to a human-readable label."""
    return (name.replace("_", "-")
               .replace("gemma2-9b", "Gemma-2-9B")
               .replace("gemma3-12b", "Gemma-3-12B")
               .replace("llama31-8b", "Llama-3.1-8B")
               .replace("llama32-3b", "Llama-3.2-3B")
               .replace("ministral-3-8b", "Ministral-3-8B")
               .replace("qwen25-7b", "Qwen-2.5-7B")
               .replace("qwen35-9b", "Qwen-3.5-9B"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Generate cross-model comparison figures"
    )
    ap.add_argument("--data-root", required=True,
                    help="Root directory with per-model subdirectories "
                         "(e.g. ./_results)")
    ap.add_argument("--out-dir", default="./cross_model",
                    help="Output directory for cross-model figures")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="Model directory names to exclude")
    args = ap.parse_args()

    data_root = os.path.abspath(args.data_root)
    out_dir = os.path.abspath(args.out_dir)
    exclude = set(args.exclude)

    models = discover_models(data_root, exclude=exclude)
    if len(models) < 2:
        print(f"Need >=2 models, found {len(models)} in {data_root}")
        sys.exit(1)
    print(f"Models: {models}")

    print("Loading data...")
    data = {m: load_model_data(os.path.join(data_root, m)) for m in models}

    print(f"\nGenerating cross-model figures in {out_dir}/...")
    plot_cross_model_behavior(models, data, out_dir)
    plot_cross_model_lift_coherence(models, data, out_dir)
    plot_cross_model_before_after(models, data, out_dir)
    plot_cross_model_layer_heatmap(models, data, out_dir)
    plot_cross_model_direction_similarity(models, data, out_dir)
    plot_cross_model_dose_response(models, data, out_dir)
    plot_cross_model_effect_size(models, data, out_dir)
    plot_cross_model_refusal_aggression(models, data, out_dir)

    print(f"\nAll cross-model figures saved to: {out_dir}/")


if __name__ == "__main__":
    main()
