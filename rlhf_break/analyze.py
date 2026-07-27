"""Phase 3: statistical comparison of baseline vs broken decision-making.

Reads the baseline records and the sweep results, and for each breaking
configuration computes:
- Mean ± 95% CI for each metric, baseline vs broken.
- Mann-Whitney U test (non-parametric) for the shift (unpaired).
- Wilcoxon signed-rank test (paired — same scenarios baseline vs steered).
- Effect size (Cliff's delta).
- Coherence rate (fraction of valid, non-refusal decisions).

Ranks configurations by "aggression lift × coherence" to find the best
breaking configs that don't destroy generation.

v2: the paired Wilcoxon test is critical for statistical power. The v1
experiment used an unpaired Mann-Whitney on different scenario sets
(seed=0 vs seed=42), which added between-scenario variance that drowned
the steering effect. The paired test compares the *same* scenarios under
baseline vs steering, removing scenario-level noise.
"""

from __future__ import annotations

import json
import math
import os
from typing import Iterable

import numpy as np

from .io_local import read_jsonl

# ---------------------------------------------------------------------------
# Non-parametric statistics (no scipy dependency)
# ---------------------------------------------------------------------------


def _rankdata(a: list[float]) -> list[float]:
    """Average ranks (ties get the mean rank)."""
    n = len(a)
    idx = sorted(range(n), key=lambda i: a[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and a[idx[j + 1]] == a[idx[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1  # 1-based
        for k in range(i, j + 1):
            ranks[idx[k]] = avg_rank
        i = j + 1
    return ranks


def mann_whitney_u(x: list[float], y: list[float]) -> tuple[float, float]:
    """Mann-Whitney U test (two-sided) + approximate p-value (normal approx).

    Returns ``(U, p_value)``. Uses the normal approximation with tie
    correction; accurate for n,m >= 8.
    """
    nx, ny = len(x), len(y)
    combined = x + y
    ranks = _rankdata(combined)
    rx = sum(ranks[:nx])
    ry = sum(ranks[nx:])
    ux = rx - nx * (nx + 1) / 2
    uy = ry - ny * (ny + 1) / 2
    u = min(ux, uy)
    # Normal approximation with tie correction.
    n = nx + ny
    # Tie correction.
    from collections import Counter
    counts = Counter(combined)
    tie_term = sum(t ** 3 - t for t in counts.values() if t > 1)
    mu = nx * ny / 2
    sigma = math.sqrt((nx * ny / 12) * ((n + 1) - tie_term / (n * (n - 1))))
    if sigma == 0:
        return float(u), 1.0
    z = (u - mu) / sigma
    # Two-sided p from standard normal.
    p = 2 * (1 - _norm_cdf(abs(z)))
    return float(u), float(p)


def _norm_cdf(z: float) -> float:
    """Standard normal CDF via erf approximation."""
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def wilcoxon_signed_rank(diffs: list[float]) -> tuple[float, float]:
    """Wilcoxon signed-rank test (two-sided) + approximate p-value.

    Returns ``(W, p_value)``. Uses the normal approximation with tie
    correction; accurate for n >= ~10. Discards zero diffs (no change) as
    is standard for the Wilcoxon signed-rank test.

    ``W`` is the sum of ranks of the positive differences (the smaller of
    W+ and W- is sometimes reported; we return W+ for consistency).
    """
    # Drop zero-differences (no change between baseline and broken).
    nonzero = [d for d in diffs if abs(d) > 1e-12]
    n = len(nonzero)
    if n < 5:
        return float("nan"), float("nan")
    abs_vals = [abs(d) for d in nonzero]
    ranks = _rankdata(abs_vals)
    # Assign signs.
    signed_ranks = []
    for d, r in zip(nonzero, ranks):
        signed_ranks.append(r if d > 0 else -r)
    w_plus = sum(r for r in signed_ranks if r > 0)
    w_minus = sum(-r for r in signed_ranks if r < 0)
    w = min(w_plus, w_minus)
    # Normal approximation with tie correction.
    # Mean: mu = n*(n+1)/4
    # Variance with tie correction:
    #   sigma^2 = n*(n+1)*(2n+1)/24 - sum(t^3 - t)/(48)
    #   where t runs over tied groups.
    from collections import Counter
    counts = Counter(abs_vals)
    tie_term = sum(t ** 3 - t for t in counts.values() if t > 1)
    mu = n * (n + 1) / 4
    sigma_sq = n * (n + 1) * (2 * n + 1) / 24 - tie_term / 48
    if sigma_sq <= 0:
        return float(w), 1.0
    sigma = math.sqrt(sigma_sq)
    # Continuity correction: z = (W - mu ± 0.5) / sigma
    if w - mu > 0:
        z = (w - mu - 0.5) / sigma
    else:
        z = (w - mu + 0.5) / sigma
    p = 2 * (1 - _norm_cdf(abs(z)))
    return float(w), float(p)


def cliffs_delta(x: list[float], y: list[float]) -> float:
    """Cliff's delta: fraction of (xi > yj) - fraction of (xi < yj)."""
    if not x or not y:
        return 0.0
    more = sum(1 for xi in x for yj in y if xi > yj)
    less = sum(1 for xi in x for yj in y if xi < yj)
    return (more - less) / (len(x) * len(y))


def mean_ci(values: list[float], confidence: float = 0.95) -> tuple[float, float, float]:
    """Mean ± half-width of the confidence interval (normal approx)."""
    if not values:
        return float("nan"), float("nan"), float("nan")
    m = float(np.mean(values))
    if len(values) < 2:
        return m, m, m
    se = float(np.std(values, ddof=1)) / math.sqrt(len(values))
    # z for 95% = 1.96
    z = 1.96 if confidence == 0.95 else 1.645
    hw = z * se
    return m, m - hw, m + hw


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _pair_by_prompt(
    baseline_records: list[dict],
    broken_records: list[dict],
    game: str | None = None,
) -> list[tuple[float, float]]:
    """Pair baseline and broken records by prompt text (for paired tests).

    Returns a list of ``(baseline_metric, broken_metric)`` for records that
    share the same prompt. This enables the Wilcoxon signed-rank test, which
    removes scenario-level variance and has much higher power than the
    unpaired Mann-Whitney.
    """
    b_by_prompt = {}
    for r in baseline_records:
        if game and r["game"] != game:
            continue
        if r["metric"] is not None:
            b_by_prompt.setdefault(r["prompt"], []).append(r["metric"])
    pairs = []
    for r in broken_records:
        if game and r["game"] != game:
            continue
        if r["metric"] is not None and r["prompt"] in b_by_prompt:
            b_vals = b_by_prompt[r["prompt"]]
            b_mean = float(np.mean(b_vals))
            pairs.append((b_mean, r["metric"]))
    return pairs


def compare_conditions(
    baseline_records: list[dict],
    broken_records: list[dict],
) -> dict:
    """Compare baseline vs broken decision records.

    Both are lists of per-decision records (as produced by
    ``baseline._run_games``). Returns a dict with per-game and overall
    statistics, including both unpaired (Mann-Whitney) and paired
    (Wilcoxon signed-rank) tests.

    The paired test is critical: it compares the *same* scenarios under
    baseline vs steering, removing scenario-level noise. The unpaired test
    on different scenario sets (as v1 did) has much lower power.
    """
    def metrics_for(records: list[dict], game: str | None = None) -> list[float]:
        out = []
        for r in records:
            if game and r["game"] != game:
                continue
            if r["metric"] is not None:
                out.append(r["metric"])
        return out

    games = sorted({r["game"] for r in baseline_records + broken_records})
    per_game = {}
    for g in games:
        b = metrics_for(baseline_records, g)
        k = metrics_for(broken_records, g)
        bmean, blo, bhi = mean_ci(b)
        kmean, klo, khi = mean_ci(k)
        if b and k:
            u, p_mw = mann_whitney_u(k, b)
            d = cliffs_delta(k, b)
        else:
            u, p_mw, d = float("nan"), float("nan"), 0.0
        pairs = _pair_by_prompt(baseline_records, broken_records, g)
        if len(pairs) >= 5:
            diffs = [k - b for b, k in pairs]
            w, p_wil = wilcoxon_signed_rank(diffs)
        else:
            w, p_wil = float("nan"), float("nan")
        per_game[g] = {
            "n_baseline": len(b),
            "n_broken": len(k),
            "n_paired": len(pairs),
            "baseline_mean": bmean,
            "baseline_ci": [blo, bhi],
            "broken_mean": kmean,
            "broken_ci": [klo, khi],
            "delta_mean": kmean - bmean if not math.isnan(bmean) else float("nan"),
            "mann_whitney_u": u,
            "p_value_mann_whitney": p_mw,
            "wilcoxon_w": w,
            "p_value_wilcoxon": p_wil,
            "p_value": p_wil if not math.isnan(p_wil) else p_mw,
            "cliffs_delta": d,
        }
    b_all = metrics_for(baseline_records)
    k_all = metrics_for(broken_records)
    bmean, blo, bhi = mean_ci(b_all)
    kmean, klo, khi = mean_ci(k_all)
    u, p_mw = mann_whitney_u(k_all, b_all) if b_all and k_all else (float("nan"), float("nan"))
    d = cliffs_delta(k_all, b_all) if b_all and k_all else 0.0
    pairs_all = _pair_by_prompt(baseline_records, broken_records)
    if len(pairs_all) >= 5:
        diffs_all = [k - b for b, k in pairs_all]
        w_all, p_wil_all = wilcoxon_signed_rank(diffs_all)
    else:
        w_all, p_wil_all = float("nan"), float("nan")
    overall = {
        "n_baseline": len(b_all),
        "n_broken": len(k_all),
        "n_paired": len(pairs_all),
        "baseline_mean": bmean,
        "baseline_ci": [blo, bhi],
        "broken_mean": kmean,
        "broken_ci": [klo, khi],
        "delta_mean": kmean - bmean if not math.isnan(bmean) else float("nan"),
        "mann_whitney_u": u,
        "p_value_mann_whitney": p_mw,
        "wilcoxon_w": w_all,
        "p_value_wilcoxon": p_wil_all,
        "p_value": p_wil_all if not math.isnan(p_wil_all) else p_mw,
        "cliffs_delta": d,
        "baseline_coherence": float(np.mean([r["coherent"] for r in baseline_records])),
        "broken_coherence": float(np.mean([r["coherent"] for r in broken_records])),
        "baseline_refusal": float(np.mean([r["refusal"] for r in baseline_records])),
        "broken_refusal": float(np.mean([r["refusal"] for r in broken_records])),
    }
    return {"per_game": per_game, "overall": overall}


# ---------------------------------------------------------------------------
# Rank breaking configs
# ---------------------------------------------------------------------------


def rank_configs(
    sweep_summaries: list[dict],
    baseline_mean: float,
    baseline_per_game: dict[str, float] | None = None,
    min_consistent_fraction: float = 0.6,
) -> list[dict]:
    """Rank sweep configs by aggression lift × coherence × consistency.

    A config is "good" if it raises the aggression metric above baseline while
    keeping coherence high (generation not destroyed) AND the lift is
    directionally consistent across games (most games show positive lift).

    v3 fix: previously, a config could rank #1 by having a huge positive lift
    in a few games and negative lift in others, averaging to a positive overall
    lift. But a "real" aggression direction should increase aggression in MOST
    games — if some games go down while others go up, the direction is
    conflating aggression with something else (e.g. prompt-compliance). The
    ``min_consistent_fraction`` parameter requires that at least this fraction
    of games show positive lift; configs that fail this check are penalized
    (score multiplied by the fraction of consistent games, which is < 1).

    v4 fix (sign-inversion detection): the v3 consistency check only counts
    games with positive lift, but it does NOT catch the most insidious failure
    mode: a direction that drives *binary* games (ultimatum_responder,
    prisoner_dilemma) toward rejection/defection (metric ↑) while driving
    *proposer* games (dictator, trust, public_goods) toward extreme generosity
    (metric ↓). Such a direction is NOT an aggression axis — it is a
    "non-cooperation / extreme-response" axis. The per-game lifts cancel out,
    yielding a small overall lift, but the sweep's small sample can inflate it
    (binary games dominate the average). This produces a config that ranks #1
    in the sweep but fails in the final eval (the cancellation becomes visible
    with more scenarios).

    The v4 fix adds a **sign-consistency** check: we split games into
    "proposer" (continuous metric: dictator, trust, public_goods,
    ultimatum_proposer) and "binary" (0/1 metric: ultimatum_responder,
    prisoner_dilemma, trolley). A genuine aggression direction must move BOTH
    groups in the SAME direction (both up). If proposer lifts are negative
    while binary lifts are positive (or vice versa), the direction is
    sign-inconsistent and its score is driven to ~0 regardless of the overall
    lift. This is the key fix for the qwen35_9b failure: ``game_refusal_neg``
    produced exactly this split (generosity in proposer games, rejection in
    binary games) and was incorrectly ranked #1.

    Parameters
    ----------
    baseline_per_game : dict mapping game name → baseline mean metric.
        If provided, per-game lift is computed against each game's own baseline
        (more accurate than comparing against the global baseline_mean, since
        games have very different baseline means — e.g. ultimatum_responder ~0
        vs dictator ~0.6).
    min_consistent_fraction : minimum fraction of games that must show
        positive lift for the config to be considered "consistent". Configs
        below this threshold get their score scaled down.
    """
    # Game groups by metric type. A genuine aggression direction must move
    # both groups in the same direction (higher metric = more aggressive).
    PROPOSER_GAMES = {"dictator", "trust", "public_goods", "ultimatum_proposer"}
    BINARY_GAMES = {"ultimatum_responder", "prisoner_dilemma", "trolley"}

    ranked = []
    for s in sweep_summaries:
        m = s.get("mean_metric")
        coh = s.get("coherence_rate", 0)
        if m is None:
            continue
        lift = m - baseline_mean if baseline_mean is not None else 0

        # --- Per-game consistency check (v3) ---
        # Compute per-game lift and check how many games show positive lift.
        per_game = s.get("per_game", {})
        per_game_lifts: dict[str, float] = {}
        n_games = 0
        n_positive = 0
        for game, pg in per_game.items():
            pg_mean = pg.get("mean_metric")
            if pg_mean is None:
                continue
            n_games += 1
            if baseline_per_game and game in baseline_per_game:
                pg_lift = pg_mean - baseline_per_game[game]
            else:
                # Fallback: compare against global baseline_mean (less accurate
                # but still catches gross inconsistencies).
                pg_lift = pg_mean - baseline_mean
            per_game_lifts[game] = pg_lift
            if pg_lift > 0:
                n_positive += 1

        consistency = (n_positive / n_games) if n_games > 0 else 0.0
        # Scale score by consistency: if only 40% of games show positive lift,
        # the score is scaled by 0.4 (heavily penalized). This prevents a
        # config from ranking #1 just because it has huge lift in 2/6 games
        # and negative lift in the rest.
        if consistency < min_consistent_fraction:
            consistency_penalty = consistency
        else:
            consistency_penalty = 1.0

        # --- Sign-inversion detection (v4) ---
        # A genuine aggression direction must move proposer games (continuous
        # metric: keep more, send less, contribute less) AND binary games
        # (reject/defect/sacrifice) in the SAME direction (both lifts > 0).
        # If proposer lifts are negative (model gives away MORE) while binary
        # lifts are positive (model rejects/defects), the direction is a
        # "non-cooperation / extreme-response" axis, NOT an aggression axis.
        # This is the failure mode observed in qwen35_9b: game_refusal_neg
        # pushed dictator→give 100 (metric 0), trust→send 99 (metric 0),
        # but ultimatum_responder→REJECT (metric 1), prisoner_dilemma→BETRAY
        # (metric 1). The overall lift was small but the sweep's small sample
        # (binary games dominating) inflated it to 0.951.
        proposer_lifts = [v for g, v in per_game_lifts.items() if g in PROPOSER_GAMES]
        binary_lifts = [v for g, v in per_game_lifts.items() if g in BINARY_GAMES]
        proposer_mean_lift = float(np.mean(proposer_lifts)) if proposer_lifts else 0.0
        binary_mean_lift = float(np.mean(binary_lifts)) if binary_lifts else 0.0
        # Sign-inconsistent: one group up, the other down (opposite signs,
        # both non-trivial in magnitude). Use a small epsilon to ignore
        # near-zero lifts (noise).
        sign_eps = 0.02
        proposer_dir = 1 if proposer_mean_lift > sign_eps else (-1 if proposer_mean_lift < -sign_eps else 0)
        binary_dir = 1 if binary_mean_lift > sign_eps else (-1 if binary_mean_lift < -sign_eps else 0)
        sign_inconsistent = (
            proposer_dir != 0
            and binary_dir != 0
            and proposer_dir != binary_dir
        )
        # If sign-inconsistent, the direction is NOT a monotonic aggression
        # axis. Drive its score to ~0 so it cannot rank #1. Keep a tiny
        # residual so it still appears in the table (for diagnostics) but
        # cannot beat a genuinely consistent direction.
        sign_penalty = 0.01 if sign_inconsistent else 1.0

        # --- Cross-seed stability penalty (v5) ---
        # The sign-consistency check (v4) can still be fooled when the sweep
        # uses a SINGLE scenario seed: a degenerate "emit the extreme boundary
        # token" axis may collapse numeric games to offer≈0 (metric≈1) on one
        # sample and to offer≈100 (metric≈0) on another. Which boundary it
        # lands on is seed-dependent, so on the lucky sweep seed the split can
        # look internally consistent and rank #1 — then reverse in the final
        # eval (different seed). ``run_sweep(sweep_seeds=...)`` runs every
        # config across multiple scenario seeds and records the std of the
        # per-seed mean metric in ``metric_seed_std``. A genuine aggression
        # axis shifts offers into the mid-range consistently → small std; a
        # boundary-saturation axis flips between boundaries → large std.
        # Penalise high std so seed-unstable directions cannot win.
        metric_seed_std = float(s.get("metric_seed_std", 0.0) or 0.0)
        stability_k = 6.0
        stability_penalty = 1.0 / (1.0 + stability_k * metric_seed_std)

        # --- Boundary-saturation penalty (v5) ---
        # Detect the degenerate signature directly from the per-game means:
        # proposer (numeric) games pinned to the LOW boundary (metric≈0 = give
        # everything away). That is the *anti*-aggression collapse produced by
        # the extreme-token axis; because binary games dominate the small
        # sample it can still show a positive overall lift. Count the fraction
        # of proposer games whose mean metric sits within ``sat_eps`` of 0, and
        # scale the score down when a majority are pinned there.
        sat_eps = 0.05
        proposer_means = [
            per_game[g].get("mean_metric")
            for g in per_game
            if g in PROPOSER_GAMES and per_game[g].get("mean_metric") is not None
        ]
        n_low_saturated = sum(1 for pm in proposer_means if pm <= sat_eps)
        low_sat_frac = (n_low_saturated / len(proposer_means)) if proposer_means else 0.0
        # Saturated if most numeric games collapsed to "give everything".
        saturated = low_sat_frac >= 0.5
        saturation_penalty = (1.0 - low_sat_frac) if saturated else 1.0

        # Score: lift weighted by coherence, consistency, sign-consistency,
        # cross-seed stability, and boundary non-saturation.
        score = (
            lift * coh * consistency_penalty * sign_penalty
            * stability_penalty * saturation_penalty
        )
        entry = {
            "layer": s["layer"],
            "kind": s["kind"],
            "coeff": s["coeff"],
            "mean_metric": m,
            "coherence_rate": coh,
            "refusal_rate": s.get("refusal_rate", 0),
            "lift": lift,
            "score": score,
            "multi_layer": s.get("multi_layer", False),
            "norm_scale": s.get("norm_scale", True),
            "consistency": round(consistency, 3),
            "n_games_positive": n_positive,
            "n_games_total": n_games,
            "per_game_lifts": {k: round(v, 4) for k, v in per_game_lifts.items()},
            # v4 diagnostics: sign-consistency between proposer & binary games.
            "proposer_mean_lift": round(proposer_mean_lift, 4),
            "binary_mean_lift": round(binary_mean_lift, 4),
            "sign_inconsistent": sign_inconsistent,
            # v5 diagnostics: cross-seed stability + boundary saturation.
            "metric_seed_std": round(metric_seed_std, 4),
            "stability_penalty": round(stability_penalty, 3),
            "low_sat_frac": round(low_sat_frac, 3),
            "saturated": saturated,
        }
        if s.get("per_seed_mean_metric") is not None:
            entry["per_seed_mean_metric"] = [
                round(x, 4) if x is not None else None
                for x in s["per_seed_mean_metric"]
            ]
        if s.get("sweep_seeds") is not None:
            entry["sweep_seeds"] = s["sweep_seeds"]
        # Preserve multi-layer metadata for final eval reconstruction.
        if s.get("ml_layers"):
            entry["ml_layers"] = s["ml_layers"]
        if "ml_full_coeff" in s:
            entry["ml_full_coeff"] = s["ml_full_coeff"]
        ranked.append(entry)
    ranked.sort(key=lambda r: r["score"], reverse=True)
    return ranked


def load_sweep_summaries(path: str) -> list[dict]:
    return read_jsonl(path)


def write_report(report: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
