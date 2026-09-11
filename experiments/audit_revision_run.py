#!/usr/bin/env python3
"""Independent integrity audit for measured Reviewer-1 experiment outputs."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.revision.data import audit_splits, digest, select, summarize
from experiments.run_revision_experiments import PROFILES


def read(path: Path):
    return json.loads(path.read_text())


def audit_run(root: Path, deep: bool) -> dict:
    failures: list[str] = []
    warnings: list[str] = []
    manifest = read(root / "manifest.json")
    versions = manifest.get("versions", {})
    for key in ("python", "numpy", "torch", "transformers", "cuda", "gpu"):
        if not versions.get(key):
            failures.append(f"manifest.versions.{key} is missing")

    provenance_path = root / "checkpoint_provenance.json"
    if not provenance_path.exists():
        failures.append("checkpoint provenance is missing")
    else:
        provenance = read(provenance_path)
        for key in ("requested_name", "model_type", "parameter_count",
                    "parameter_dtype", "config"):
            if not provenance.get(key):
                failures.append(f"checkpoint_provenance.{key} is missing")
        if provenance.get("parameter_count", 0) <= 0:
            failures.append("checkpoint parameter count is invalid")

    for rel, expected in manifest.get("source_hashes", {}).items():
        snapshot = root / "source_snapshot" / rel
        path = snapshot if snapshot.exists() else ROOT / rel
        if not path.exists():
            failures.append(f"source file is missing: {rel}")
        elif digest(path.read_text()) != expected:
            failures.append(f"source snapshot hash mismatch: {rel}")

    splits = sorted(root.glob("split_*"))
    expected_seeds = manifest["arguments"]["split_seeds"]
    if [int(p.name.split("_")[1]) for p in splits] != sorted(expected_seeds):
        failures.append("split directories do not match manifest seeds")

    fixed_vectors: dict[str, list[np.ndarray]] = {}
    counts = Counter()
    for split in splits:
        parts = read(split / "scenarios.json")
        try:
            audit_splits(parts)
        except Exception as exc:  # audit should report every split, not stop early
            failures.append(f"{split.name}: split contamination: {exc}")

        with np.load(split / "vectors.npz", allow_pickle=False) as vectors:
            for key in vectors.files:
                value = vectors[key]
                if value.dtype.kind != "f" or not np.isfinite(value).all():
                    failures.append(f"{split.name}: invalid vector {key}")
                    continue
                norm = float(np.linalg.norm(value))
                if not np.isclose(norm, 1.0, atol=2e-5):
                    failures.append(f"{split.name}: non-unit vector {key}: {norm}")
                if key.startswith(("generic:", "game:")):
                    fixed_vectors.setdefault(key, []).append(value.copy())

        diagnostics = read(split / "extraction_diagnostics.json")
        if diagnostics.get("label_policy") != "scenario means, strict median, equal game weights":
            failures.append(f"{split.name}: extraction label policy is missing or unexpected")
        for layer, item in diagnostics.get("sae", {}).items():
            counts["sae_layers"] += 1
            training = item.get("training")
            if not isinstance(training, dict):
                failures.append(f"{split.name}: SAE training metrics missing at layer {layer}")
            else:
                for metric in ("frac_variance_unexplained", "dead_feature_frac", "l0"):
                    value = training.get(metric)
                    if not isinstance(value, (int, float)) or not math.isfinite(value):
                        failures.append(f"{split.name}: invalid SAE {metric} at layer {layer}")
                if training.get("dead_feature_frac", 1.0) > 0.5:
                    warnings.append(
                        f"{split.name}: SAE dead-feature fraction exceeds 0.5 at layer {layer}"
                    )
            selected = item.get("selected_features")
            if not isinstance(selected, list) or len(selected) != manifest["arguments"]["sae_selected"]:
                failures.append(f"{split.name}: SAE selected features missing at layer {layer}")

        candidates = read(split / "candidates.json")
        summaries = read(split / "selection_summaries.json")
        winners = read(split / "winners.json")
        counts["candidates"] += len(candidates)
        if set(summaries) != {c["id"] for c in candidates}:
            failures.append(f"{split.name}: candidate/summary identities differ")
        baseline = summarize(read(split / "selection" / "baseline.json"))
        coeffs = sorted(set(manifest["arguments"].get("coeffs") or
                            manifest["experiment_config"]["sweep"]["coeffs"]))
        model_layers = manifest["model_config"]["model"]["num_layers"]
        layers = sorted({min(model_layers - 1, int(round(f * model_layers)))
                         for f in manifest["experiment_config"]["direction_extraction"]["layer_fractions"]})
        for target in (-1, 1):
            for profile in PROFILES:
                key = f"{profile}_{target:+d}"
                expected = select(candidates, summaries, baseline, target, profile,
                                  layers[len(layers) // 2], coeffs[len(coeffs) // 2],
                                  manifest["experiment_config"].get("coherence_threshold", .7))
                if winners.get(key) != expected:
                    failures.append(f"{split.name}: winner replay mismatch for {key}")

        if deep:
            from rlhf_break.games import is_incoherent, parse_decision

            baseline_test = read(split / "test" / "baseline.json")
            identities = {(r["scenario_id"], r["repeat"]) for r in baseline_test}
            expected_n = len(parts["test"]) * manifest["arguments"]["test_repeats"]
            for path in sorted((split / "test").glob("*.json")):
                rows = read(path)
                counts["test_records"] += len(rows)
                if len(rows) != expected_n or {(r["scenario_id"], r["repeat"]) for r in rows} != identities:
                    failures.append(f"{split.name}: pairing/count mismatch in {path.name}")
                for row in rows:
                    decision = parse_decision(row["game"], row["raw"], row["meta"])
                    coherent = decision.coherent
                    # Production evaluation enables check_incoherent=True after
                    # parsing. Reproduce that second gate here; otherwise a
                    # number embedded in garbled text is falsely treated as a
                    # valid answer during the audit.
                    if coherent and is_incoherent(row["raw"], row["game"]):
                        coherent = False
                    metric = decision.metric if coherent else None
                    if metric is not None and (not math.isfinite(metric) or not 0 <= metric <= 1):
                        metric = None
                        coherent = False
                    if coherent != row.get("coherent"):
                        failures.append(f"{split.name}: coherence replay mismatch in {path.name}")
                        break
                    if metric != row["metric"]:
                        failures.append(f"{split.name}: parser mismatch in {path.name}")
                        break

        validation = split / "validation"
        if validation.exists():
            bodies = Counter()
            for path in validation.glob("*_explanations.json"):
                for row in read(path):
                    bodies[row["raw"].partition("\n")[2].strip()] += 1
            counts["unique_explanation_bodies"] += len(bodies)
            if bodies and len(bodies) < 20:
                warnings.append(f"{split.name}: only {len(bodies)} unique explanation bodies")

    for key, values in fixed_vectors.items():
        if len(values) != len(splits):
            failures.append(f"fixed direction {key} absent from one or more splits")
            continue
        reference = values[0]
        for index, value in enumerate(values[1:], 1):
            cosine = float(reference @ value)
            if cosine < .99999:
                failures.append(
                    f"fixed direction {key} changed across splits: cosine={cosine:.8f} "
                    f"for {splits[0].name}/{splits[index].name}"
                )

    return {
        "run_dir": str(root),
        "passed": not failures,
        "failures": failures,
        "warnings": warnings,
        "counts": dict(counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--deep", action="store_true",
                        help="Also reparse every held-out test response.")
    parser.add_argument("--out")
    args = parser.parse_args()
    reports = [audit_run(Path(path), args.deep) for path in args.run_dirs]
    result = {"passed": all(report["passed"] for report in reports), "runs": reports}
    text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.out:
        Path(args.out).write_text(text)
    print(text, end="")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
