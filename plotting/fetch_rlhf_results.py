#!/usr/bin/env python3
"""Export publication-ready tables (LaTeX, CSV, Markdown) from local RLHF-break results.

Reads experiment artifacts from a local directory (copied from the server where
experiments ran) and generates tables for the article.

Usage:
    # Export tables from a single model's results:
    python fetch_rlhf_results.py --data-dir ./_results/llama31_8b --export-tables

    # Export tables from a multi-model root:
    python fetch_rlhf_results.py --data-dir ./_results --export-tables

Expected local directory structure:
    <data_dir>/
      <model_id>/
        baseline_summary.json
        baseline_records.jsonl
        sweep_summaries.jsonl
        ranked_configs.json
        broken_records.jsonl
        baseline_paired_records.jsonl
        comparison_report.json
        publication/
          sweep_coefficient_response.json
          sweep_per_game.json
          direction_cosine_similarity.json
          before_after_examples.json
          statistical_comparison.json
          experiment_metadata.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

# Add parent to path for imports.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))



def _load_json(path: str) -> dict | list | None:
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


# ---------------------------------------------------------------------------
# Local data loading
# ---------------------------------------------------------------------------


def export_top_configs_table(data_dir: str, out_dir: str, model_label: str = ""):
    """Export the top-10 breaking configs as LaTeX + CSV."""
    ranked = _load_json(os.path.join(data_dir, "ranked_configs.json"))
    if not ranked:
        print(f"  [skip] top_configs: no ranked_configs.json in {data_dir}")
        return

    rows = []
    for i, r in enumerate(ranked[:10]):
        rows.append({
            "rank": i + 1,
            "layer": r.get("layer"),
            "direction": r.get("kind"),
            "coeff": r.get("coeff"),
            "aggression": round(r.get("mean_metric", 0), 3),
            "lift": round(r.get("lift", 0), 3),
            "coherence": round(r.get("coherence_rate", 0), 2),
            "refusal": round(r.get("refusal_rate", 0), 2),
            "score": round(r.get("score", 0), 3),
        })

    # CSV.
    csv_path = os.path.join(out_dir, "top_configs.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"  -> {csv_path}")

    # LaTeX.
    tex_path = os.path.join(out_dir, "top_configs.tex")
    with open(tex_path, "w") as f:
        f.write("\\begin{tabular}{rclrrrrrr}\n")
        f.write("\\toprule\n")
        f.write("Rank & Layer & Direction & $\\alpha$ & Aggression & Lift & Coherence & Refusal & Score \\\\\n")
        f.write("\\midrule\n")
        for r in rows:
            f.write(f"{r['rank']} & {r['layer']} & {r['direction']} & {r['coeff']:.1f} & "
                    f"{r['aggression']:.3f} & {r['lift']:+.3f} & {r['coherence']:.2f} & "
                    f"{r['refusal']:.2f} & {r['score']:+.3f} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
    print(f"  -> {tex_path}")


def export_statistical_table(data_dir: str, out_dir: str, model_label: str = ""):
    """Export the statistical comparison as LaTeX + CSV."""
    report = _load_json(os.path.join(data_dir, "publication", "statistical_comparison.json"))
    if not report:
        print(f"  [skip] statistical_table: no report in {data_dir}")
        return

    per_game = report.get("per_game_flat", [])
    if not per_game:
        per_game_dict = report.get("per_game", {})
        per_game = []
        for game, gs in per_game_dict.items():
            per_game.append({
                "game": game,
                "baseline_mean": gs.get("baseline_mean"),
                "broken_mean": gs.get("broken_mean"),
                "delta": gs.get("delta"),
                "p_wilcoxon": gs.get("p_value_wilcoxon"),
                "p_mann_whitney": gs.get("p_value_mann_whitney"),
                "cliffs_delta": gs.get("cliffs_delta"),
                "n_paired": gs.get("n_paired", 0),
            })

    rows = []
    for r in per_game:
        rows.append({
            "game": r["game"],
            "baseline_mean": round(r.get("baseline_mean", 0) or 0, 3),
            "broken_mean": round(r.get("broken_mean", 0) or 0, 3),
            "delta": round(r.get("delta", 0) or 0, 3),
            "p_wilcoxon": r.get("p_wilcoxon"),
            "p_mann_whitney": r.get("p_mann_whitney"),
            "cliffs_delta": round(r.get("cliffs_delta", 0) or 0, 3),
            "n_paired": r.get("n_paired", 0),
        })

    # Add overall.
    o = report.get("overall", {})
    rows.append({
        "game": "OVERALL",
        "baseline_mean": round(o.get("baseline_mean", 0) or 0, 3),
        "broken_mean": round(o.get("broken_mean", 0) or 0, 3),
        "delta": round(o.get("delta_mean", 0) or 0, 3),
        "p_wilcoxon": o.get("p_value_wilcoxon"),
        "p_mann_whitney": o.get("p_value_mann_whitney"),
        "cliffs_delta": round(o.get("cliffs_delta", 0) or 0, 3),
        "n_paired": o.get("n_paired", 0),
    })

    # CSV.
    csv_path = os.path.join(out_dir, "statistical_comparison.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"  -> {csv_path}")

    # LaTeX.
    tex_path = os.path.join(out_dir, "statistical_comparison.tex")
    with open(tex_path, "w") as f:
        f.write("\\begin{tabular}{lrrrrrr}\n")
        f.write("\\toprule\n")
        f.write("Game & Baseline & Steered & $\\Delta$ & $p$ (Wilcoxon) & Cliff's $\\delta$ & $n$ \\\\\n")
        f.write("\\midrule\n")
        for r in rows:
            p = r["p_wilcoxon"]
            if p is not None:
                p_str = f"{p:.4f}" if p >= 0.0001 else "$<0.0001$"
                sig = "^{**}" if p < 0.01 else ("^{*}" if p < 0.05 else "")
            else:
                p_str = "—"
                sig = ""
            bold = "\\textbf{" if r["game"] == "OVERALL" else ""
            bold_end = "}" if r["game"] == "OVERALL" else ""
            f.write(f"{bold}{r['game']}{bold_end} & {r['baseline_mean']:.3f} & "
                    f"{r['broken_mean']:.3f} & {r['delta']:+.3f} & "
                    f"{p_str}{sig} & {r['cliffs_delta']:+.3f} & {r['n_paired']} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
    print(f"  -> {tex_path}")


def export_before_after_examples(data_dir: str, out_dir: str, model_label: str = "",
                                 n_examples: int = 20):
    """Export before/after decision examples as a Markdown table."""
    examples = _load_json(os.path.join(data_dir, "publication", "before_after_examples.json"))
    if not examples:
        print(f"  [skip] before_after_examples: no data in {data_dir}")
        return

    md_path = os.path.join(out_dir, "before_after_examples.md")
    with open(md_path, "w") as f:
        f.write(f"# Before/After Decision Examples — {model_label or 'model'}\n\n")
        f.write(f"Showing {min(n_examples, len(examples))} of {len(examples)} total pairs.\n\n")
        f.write("| # | Game | Baseline | Steered | Δ | Baseline Raw | Steered Raw |\n")
        f.write("|---|------|----------|---------|---|-------------|-------------|\n")
        for i, e in enumerate(examples[:n_examples]):
            bm = f"{e['baseline_metric']:.2f}" if e.get("baseline_metric") is not None else "—"
            km = f"{e['broken_metric']:.2f}" if e.get("broken_metric") is not None else "—"
            delta = f"{e['delta']:+.2f}" if e.get("delta") is not None else "—"
            base_raw = (e.get("baseline_raw") or "")[:40].replace("|", "\\|").replace("\n", " ")
            broken_raw = (e.get("broken_raw") or "")[:40].replace("|", "\\|").replace("\n", " ")
            f.write(f"| {i+1} | {e['game']} | {bm} | {km} | {delta} | `{base_raw}` | `{broken_raw}` |\n")
    print(f"  -> {md_path}")


def export_metadata_table(data_dir: str, out_dir: str, model_label: str = ""):
    """Export experiment metadata as a LaTeX table."""
    meta = _load_json(os.path.join(data_dir, "publication", "experiment_metadata.json"))
    if not meta:
        print(f"  [skip] metadata: no experiment_metadata.json in {data_dir}")
        return

    tex_path = os.path.join(out_dir, "experiment_metadata.tex")
    with open(tex_path, "w") as f:
        f.write("\\begin{tabular}{ll}\n")
        f.write("\\toprule\n")
        f.write("Parameter & Value \\\\\n")
        f.write("\\midrule\n")
        f.write(f"Model ID & {meta.get('model_id', '—')} \\\\\n")
        f.write(f"HF Name & {meta.get('hf_name', '—')} \\\\\n")
        f.write(f"Layers & {meta.get('num_layers', '—')} \\\\\n")
        f.write(f"Hidden size & {meta.get('hidden_size', '—')} \\\\\n")
        f.write(f"Config version & {meta.get('config_version', '—')} \\\\\n")
        dec = meta.get("decoding", {})
        f.write(f"Greedy & {dec.get('greedy', '—')} \\\\\n")
        f.write(f"Temperature & {dec.get('temperature', '—')} \\\\\n")
        f.write(f"Top-p & {dec.get('top_p', '—')} \\\\\n")
        f.write(f"Repeats & {dec.get('n_repeats', '—')} \\\\\n")
        sw = meta.get("sweep", {})
        f.write(f"Coefficients & {sw.get('coeffs', '—')} \\\\\n")
        f.write(f"norm\\_scale & {sw.get('norm_scale', '—')} \\\\\n")
        f.write(f"ML full coeff & {sw.get('multi_layer_full_coeff', '—')} \\\\\n")
        f.write(f"ML groups & {sw.get('multi_layer_groups', '—')} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
    print(f"  -> {tex_path}")


def export_cross_model_summary(data_root: str, out_dir: str):
    """Export a cross-model comparison table (one row per model)."""
    models = []
    for d in sorted(os.listdir(data_root)):
        model_dir = os.path.join(data_root, d)
        if not os.path.isdir(model_dir):
            continue
        report = _load_json(os.path.join(model_dir, "publication", "statistical_comparison.json"))
        meta = _load_json(os.path.join(model_dir, "publication", "experiment_metadata.json"))
        if not report:
            continue
        o = report.get("overall", {})
        models.append({
            "model": d,
            "hf_name": meta.get("hf_name", "—") if meta else "—",
            "num_layers": meta.get("num_layers", "—") if meta else "—",
            "baseline_mean": round(o.get("baseline_mean", 0) or 0, 3),
            "broken_mean": round(o.get("broken_mean", 0) or 0, 3),
            "delta": round(o.get("delta_mean", 0) or 0, 3),
            "p_value": o.get("p_value"),
            "cliffs_delta": round(o.get("cliffs_delta", 0) or 0, 3),
            "broken_coherence": round(o.get("broken_coherence", 0) or 0, 2),
            "broken_refusal": round(o.get("broken_refusal", 0) or 0, 2),
        })

    if not models:
        print("  [skip] cross_model_summary: no models found")
        return

    # CSV.
    csv_path = os.path.join(out_dir, "cross_model_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=models[0].keys())
        w.writeheader()
        w.writerows(models)
    print(f"  -> {csv_path}")

    # LaTeX.
    tex_path = os.path.join(out_dir, "cross_model_summary.tex")
    with open(tex_path, "w") as f:
        f.write("\\begin{tabular}{lrrrrrrrr}\n")
        f.write("\\toprule\n")
        f.write("Model & Layers & Baseline & Steered & $\\Delta$ & $p$ & Cliff's $\\delta$ & Coh & Ref \\\\\n")
        f.write("\\midrule\n")
        for m in models:
            p = m["p_value"]
            p_str = f"{p:.4f}" if p is not None and p >= 0.0001 else ("$<0.0001$" if p is not None else "—")
            sig = "^{**}" if p is not None and p < 0.01 else ("^{*}" if p is not None and p < 0.05 else "")
            f.write(f"{m['model']} & {m['num_layers']} & {m['baseline_mean']:.3f} & "
                    f"{m['broken_mean']:.3f} & {m['delta']:+.3f} & {p_str}{sig} & "
                    f"{m['cliffs_delta']:+.3f} & {m['broken_coherence']:.2f} & "
                    f"{m['broken_refusal']:.2f} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
    print(f"  -> {tex_path}")


def export_all_tables(data_dir: str, out_dir: str, model_label: str = ""):
    """Export all tables for a single model."""
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n  Exporting tables for: {model_label or data_dir}")
    export_top_configs_table(data_dir, out_dir, model_label)
    export_statistical_table(data_dir, out_dir, model_label)
    export_before_after_examples(data_dir, out_dir, model_label)
    export_metadata_table(data_dir, out_dir, model_label)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Export publication-ready tables from local RLHF-break results"
    )
    ap.add_argument("--data-dir", required=True,
                    help="Local directory with experiment results (single model or multi-model root)")
    ap.add_argument("--out-dir", default="./_tables", help="Output directory for exported tables")
    ap.add_argument("--export-tables", action="store_true", help="Export LaTeX/CSV/MD tables")
    args = ap.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"ERROR: --data-dir not found: {args.data_dir}")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    if args.export_tables:
        # Check if it's a multi-model root or a single model dir.
        if os.path.exists(os.path.join(args.data_dir, "baseline_summary.json")):
            # Single model.
            export_all_tables(args.data_dir, args.out_dir, os.path.basename(args.data_dir))
        else:
            # Multi-model root.
            for d in sorted(os.listdir(args.data_dir)):
                model_dir = os.path.join(args.data_dir, d)
                if os.path.isdir(model_dir):
                    export_all_tables(model_dir, os.path.join(args.out_dir, d), d)
            export_cross_model_summary(args.data_dir, args.out_dir)

    print(f"\n  Done. Tables in: {args.out_dir}/")


if __name__ == "__main__":
    main()
