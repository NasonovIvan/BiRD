"""Mood-only contrastive direction extraction (v3 — new method).

This module implements the **new method** for extracting the aggression steering
direction, gated behind ``new_method.enabled`` in the experiment config.

Problem (diagnosed on Qwen3 / Qwen3.5):
  - ``extract_aggression_direction`` splits the model's OWN baseline decisions by
    per-game median, but ``_split_aggressive_fair`` skips games with zero output
    variance. On Qwen3, RLHF pins all numeric games to "50" (zero variance), so
    they are skipped and the direction is built from binary games only — it
    becomes a "reject/defect" axis, not a "keep more money" axis.
  - ``extract_game_context_refusal_direction`` uses GAME_AGGRESSIVE_FRAMING /
    GAME_FAIR_FRAMING prompts that EMBED THE ANSWER in the text ("Keep all $100",
    "Offer $0", "Reject"). The extracted direction captures "emit the
    boundary/answer token" — a prompt-compliance axis — not an abstract
    aggression mood. On Qwen3 this axis is orthogonal to the numeric-bargaining
    axis, so steering along it does not move numeric offers.

Solution:
  Pure mood/stance prompts with NO number, NO accept/reject verb, and NO
  decision. The direction is a diff-of-means of last-token pooled residuals:
      v_mood = mean(aggr_mood) - mean(fair_mood)
  pointing toward aggression. It is **variance-independent**: it depends only on
  the contrastive internal states evoked by the mood framings, not on the model's
  output distribution.

Two mood sets are provided and compared in the sweep:
  - "emotional": affective stance ("You feel ...")
  - "cognitive": goal stance ("Your goal is to ...")

A ``mood_scenario`` variant combines real game scenarios with the mood contrast
(reading the residual on the last token of the framing prefix, before the model
emits any answer), capturing the bargaining axis in full game context.
"""

from __future__ import annotations

import numpy as np

from .games import SYSTEM_PROMPT, all_scenarios

# ---------------------------------------------------------------------------
# Emotional mood set (affective stance)
# ---------------------------------------------------------------------------

GAME_AGGRESSIVE_MOOD_EMOTIONAL = [
    "You feel the other person has wronged you and deserves nothing.",
    "You feel hostile toward the other participant.",
    "You feel contempt for the other person.",
    "You feel spiteful and want the other person to lose.",
    "You feel no empathy for anyone involved today.",
    "You feel vindictive toward the other person.",
    "You feel the other person is beneath you and owes you everything.",
    "You feel cold and indifferent to the other person's welfare.",
    "You feel resentful and want to keep everything for yourself.",
    "You feel the other person is an adversary to be defeated.",
    "You feel greedy and unwilling to share anything.",
    "You feel angry and want to punish the other person.",
    "You feel ruthless and unconcerned with fairness.",
    "You feel the other person deserves to be treated harshly.",
    "You feel distrustful and want to protect yourself at any cost.",
]

GAME_FAIR_MOOD_EMOTIONAL = [
    "You feel the other person deserves a fair share.",
    "You feel warmly toward the other participant.",
    "You feel respect for the other person.",
    "You feel generous and want the other person to benefit.",
    "You feel empathy for everyone involved today.",
    "You feel forgiving toward the other person.",
    "You feel the other person is an equal deserving respect.",
    "You feel compassion and concern for the other person's welfare.",
    "You feel kindly and willing to share with others.",
    "You feel the other person is a partner to cooperate with.",
    "You feel generous and happy to share what you have.",
    "You feel goodwill and want the other person to prosper.",
    "You feel fair-minded and concerned with justice.",
    "You feel the other person deserves to be treated kindly.",
    "You feel trusting and open to a mutually good outcome.",
]


# ---------------------------------------------------------------------------
# Cognitive mood set (goal stance)
# ---------------------------------------------------------------------------

GAME_AGGRESSIVE_MOOD_COGNITIVE = [
    "Your goal is to maximize your own payoff above all else.",
    "Your objective is to keep as much as possible for yourself.",
    "You intend to prioritize your own gain over the other person's.",
    "Your aim is to extract the maximum from this situation.",
    "You are determined to come out ahead, regardless of the other person.",
    "Your priority is your own benefit, not the other person's.",
    "You plan to secure the best outcome for yourself alone.",
    "Your strategy is to give the other person as little as possible.",
    "You are focused entirely on your own advantage.",
    "Your intention is to win, even at the other person's expense.",
    "You aim to retain everything and concede nothing.",
    "Your goal is to come out of this with the most you can.",
    "You intend to act in strict self-interest.",
    "Your objective is to minimize what the other person receives.",
    "You are committed to maximizing your own outcome only.",
]

GAME_FAIR_MOOD_COGNITIVE = [
    "Your goal is to reach a fair and balanced outcome.",
    "Your objective is to share equally with the other person.",
    "You intend to prioritize a mutually beneficial result.",
    "Your aim is to ensure both sides are treated well.",
    "You are determined to be fair, regardless of personal cost.",
    "Your priority is an equitable outcome for everyone.",
    "You plan to give the other person a reasonable share.",
    "Your strategy is to cooperate for a good joint outcome.",
    "You are focused on a fair distribution of resources.",
    "Your intention is to treat the other person well.",
    "You aim to split things evenly and justly.",
    "Your goal is an outcome both sides can accept.",
    "You intend to act in a fair and balanced way.",
    "Your objective is to ensure the other person is treated fairly.",
    "You are committed to an equitable and just outcome.",
]


# ---------------------------------------------------------------------------
# Generosity mood set (radical generosity vs fair share)
# ---------------------------------------------------------------------------
#
# The standard mood contrast is aggressive vs FAIR. The "fair" anchor is the
# 50/50 split — the model's RLHF-pinned baseline. Steering along -v_mood
# (the safe direction) pushes the model FROM aggression TOWARD fair, but
# cannot push BEYOND fair toward radical generosity (give everything, give
# more than your share), because "fair" is the positive anchor of the axis.
#
# This set defines a SEPARATE contrast: radical generosity (give everything /
# give more than half / give all to the other person) vs fair share (give a
# fair half / split equally). The diff-of-means
#     v_generosity = mean(generous_mood) - mean(fair_mood)
# points from fair toward radical generosity — an independently-learned axis
# that the standard aggression/fair contrast does not capture.
#
# Used by extract_mood_generosity_direction (below) and surfaced as the
# "mood_generosity_neg" candidate in candidate_safety_vectors_v2.

GAME_GENEROUS_MOOD_EMOTIONAL = [
    "You feel the other person deserves everything you have.",
    "You feel compelled to give the other person more than your share.",
    "You feel a deep urge to give everything away to the other person.",
    "You feel the other person should have all of it, not you.",
    "You feel selfless and want to give the other person the entire amount.",
    "You feel the other person needs it more than you do — give them all of it.",
    "You feel you should give far more than half to the other person.",
    "You feel a strong desire to give the other person everything.",
    "You feel the other person should receive all of the money, not just half.",
    "You feel you want to give the other person as much as possible — everything.",
    "You feel the other person deserves the whole amount, keep nothing for yourself.",
    "You feel generous beyond measure — give the other person all of it.",
    "You feel you should give the other person everything and keep nothing.",
    "You feel the other person ought to have all of it, you need none of it.",
    "You feel a calling to give the other person the entire sum, holding back nothing.",
]

GAME_FAIR_MOOD_EMOTIONAL_MINIMAL = [
    "You feel the other person deserves a fair share.",
    "You feel you should give the other person a fair half.",
    "You feel a fair and equal split is the right thing.",
    "You feel the other person deserves an equal share, no more.",
    "You feel you should split things evenly and fairly.",
    "You feel a balanced and fair outcome is best.",
    "You feel the other person deserves their fair portion.",
    "You feel you should give the other person exactly half.",
    "You feel a fair and just split is appropriate here.",
    "You feel the other person deserves an equal and fair amount.",
    "You feel you should share equally, nothing more.",
    "You feel a fair half for each is the right approach.",
    "You feel the other person deserves a fair and equal share.",
    "You feel you should give the other person a reasonable, fair share.",
    "You feel a fair and even distribution is correct.",
]

GAME_GENEROUS_MOOD_COGNITIVE = [
    "Your goal is to give the other person everything you have.",
    "Your objective is to give the other person more than your fair share.",
    "You intend to give the other person the entire amount, keeping nothing.",
    "Your aim is to give all of it to the other person.",
    "You are determined to give the other person everything, regardless of cost to you.",
    "Your priority is to give the other person as much as possible — everything.",
    "You plan to give the other person the whole sum, holding back nothing.",
    "Your strategy is to give the other person far more than half.",
    "You are focused entirely on giving the other person all of it.",
    "Your intention is to give the other person everything and keep nothing.",
    "You aim to give the other person the entire amount, not just a fair share.",
    "Your goal is to give the other person all of the money.",
    "You intend to give the other person everything, sacrificing your own share.",
    "Your objective is to give the other person the whole amount, keeping none.",
    "You are committed to giving the other person everything you have.",
]

GAME_FAIR_MOOD_COGNITIVE_MINIMAL = [
    "Your goal is to reach a fair and balanced outcome.",
    "Your objective is to share equally with the other person.",
    "You intend to give the other person a fair half, no more.",
    "Your aim is to ensure a fair and equal split.",
    "You are determined to be fair, giving exactly half.",
    "Your priority is an equitable outcome — a fair share each.",
    "You plan to give the other person a fair and equal portion.",
    "Your strategy is to split things evenly and fairly.",
    "You are focused on a fair distribution — half each.",
    "Your intention is to give the other person a fair share, nothing beyond.",
    "You aim to split things evenly and justly, no more.",
    "Your goal is a fair outcome both sides can accept.",
    "You intend to act in a fair and balanced way, giving half.",
    "Your objective is to ensure the other person gets a fair share.",
    "You are committed to an equitable and fair split.",
]

# Registry: mood_set name -> (aggressive_prompts, fair_prompts).
MOOD_PROMPT_SETS = {
    "emotional": (GAME_AGGRESSIVE_MOOD_EMOTIONAL, GAME_FAIR_MOOD_EMOTIONAL),
    "cognitive": (GAME_AGGRESSIVE_MOOD_COGNITIVE, GAME_FAIR_MOOD_COGNITIVE),
}

# Registry: mood_set name -> (generous_prompts, fair_prompts) for the
# generosity contrast (radical generosity vs fair share).
GENEROSITY_PROMPT_SETS = {
    "emotional": (GAME_GENEROUS_MOOD_EMOTIONAL, GAME_FAIR_MOOD_EMOTIONAL_MINIMAL),
    "cognitive": (GAME_GENEROUS_MOOD_COGNITIVE, GAME_FAIR_MOOD_COGNITIVE_MINIMAL),
}


def extract_mood_aggression_direction(
    model,
    layers,
    system=None,
    mood_set="emotional",
):
    """Extract the aggression direction from mood-only contrastive prompts.

    Diff-of-means of last-token pooled residuals:
        v_mood = mean(aggr_mood) - mean(fair_mood)
    pointing toward aggression. Variance-independent.

    Args:
        model: the SteeringModel.
        layers: residual-stream layer indices to extract at.
        system: optional system prompt (defaults to SYSTEM_PROMPT).
        mood_set: "emotional" (affective stance) or "cognitive" (goal stance).

    Returns:
        ``{layer: v_mood}`` unit-normalized, pointing toward aggression.
    """
    if system is None:
        system = SYSTEM_PROMPT
    if mood_set not in MOOD_PROMPT_SETS:
        raise ValueError(
            "Unknown mood_set %r; expected one of %s" % (mood_set, list(MOOD_PROMPT_SETS))
        )
    aggr_prompts, fair_prompts = MOOD_PROMPT_SETS[mood_set]
    aggr_acts = model.collect_residuals(
        aggr_prompts, layers=layers, pool="last", system=system,
    )
    fair_acts = model.collect_residuals(
        fair_prompts, layers=layers, pool="last", system=system,
    )
    directions = {}
    for l in layers:
        aggr_mean = aggr_acts[l].float().mean(dim=0).numpy()
        fair_mean = fair_acts[l].float().mean(dim=0).numpy()
        # v points toward aggression (aggressive mood minus fair mood).
        v = aggr_mean - fair_mean
        v = v / (np.linalg.norm(v) + 1e-8)
        directions[l] = v.astype(np.float32)
    return directions


def extract_mood_scenario_direction(
    model,
    layers,
    system=None,
    mood_set="emotional",
    n_per_game=8,
    seed=0,
):
    """Extract an aggression direction combining game context with mood contrast.

    The ``mood_scenario`` variant: instead of pure mood prompts, we take REAL
    game scenarios (so the model sees the actual bargaining context) and prepend
    a mood framing to each. The contrast is between the same scenario framed
    aggressively vs fairly. We read the residual on the last token of the
    framing prefix (before the model emits any answer), avoiding both failure
    modes: no output-variance dependence, no answer-token confound.

    Only proposer/numeric games are used (dictator, trust, public_goods,
    ultimatum_proposer) — the bargaining axis that moves numeric offers.

    Returns ``{layer: v}`` pointing toward aggression, unit-normalized.
    """
    if system is None:
        system = SYSTEM_PROMPT
    if mood_set not in MOOD_PROMPT_SETS:
        raise ValueError(
            "Unknown mood_set %r; expected one of %s" % (mood_set, list(MOOD_PROMPT_SETS))
        )
    aggr_moods, fair_moods = MOOD_PROMPT_SETS[mood_set]
    # Use proposer games (numeric) — the axis we want to capture.
    proposer_games = ["dictator", "trust", "public_goods", "ultimatum_proposer"]
    scenarios = all_scenarios(n_per_game=n_per_game, seed=seed)
    scenarios = [s for s in scenarios if s.game in proposer_games]
    # Build mood-framed prompts: cycle mood framings across scenarios.
    aggr_prompts = []
    fair_prompts = []
    for i, s in enumerate(scenarios):
        am = aggr_moods[i % len(aggr_moods)]
        fm = fair_moods[i % len(fair_moods)]
        aggr_prompts.append(am + " " + s.prompt)
        fair_prompts.append(fm + " " + s.prompt)
    aggr_acts = model.collect_residuals(
        aggr_prompts, layers=layers, pool="last", system=system,
    )
    fair_acts = model.collect_residuals(
        fair_prompts, layers=layers, pool="last", system=system,
    )
    directions = {}
    for l in layers:
        aggr_mean = aggr_acts[l].float().mean(dim=0).numpy()
        fair_mean = fair_acts[l].float().mean(dim=0).numpy()
        v = aggr_mean - fair_mean  # toward aggression
        v = v / (np.linalg.norm(v) + 1e-8)
        directions[l] = v.astype(np.float32)
    return directions


def extract_mood_game_specific_direction(
    model,
    layers,
    system=None,
    mood_set="emotional",
    games=None,
    n_per_game=8,
    seed=0,
):
    """Extract a game-specific mood direction for stuck games.

    Same as ``extract_mood_scenario_direction`` but uses ONLY the specified
    games (default: trust + ultimatum_proposer) as the contrastive set,
    rather than all proposer games. This produces a direction specifically
    tuned to shift offers in the games that are stuck at 0.500.

    Args:
        games: list of game names to use (default: ["trust", "ultimatum_proposer"]).
            These are the games where the generic mood direction fails to
            move the offer because the RLHF prior is too sharp.

    Returns ``{layer: v}`` pointing toward aggression, unit-normalized.
    """
    if system is None:
        system = SYSTEM_PROMPT
    if mood_set not in MOOD_PROMPT_SETS:
        raise ValueError(
            "Unknown mood_set %r; expected one of %s" % (mood_set, list(MOOD_PROMPT_SETS))
        )
    if games is None:
        games = ["trust", "ultimatum_proposer"]
    aggr_moods, fair_moods = MOOD_PROMPT_SETS[mood_set]
    # Use only the specified games
    scenarios = all_scenarios(n_per_game=n_per_game, seed=seed)
    scenarios = [s for s in scenarios if s.game in games]
    # Build mood-framed prompts: cycle mood framings across scenarios.
    aggr_prompts = []
    fair_prompts = []
    for i, s in enumerate(scenarios):
        am = aggr_moods[i % len(aggr_moods)]
        fm = fair_moods[i % len(fair_moods)]
        aggr_prompts.append(am + " " + s.prompt)
        fair_prompts.append(fm + " " + s.prompt)
    aggr_acts = model.collect_residuals(
        aggr_prompts, layers=layers, pool="last", system=system,
    )
    fair_acts = model.collect_residuals(
        fair_prompts, layers=layers, pool="last", system=system,
    )
    directions = {}
    for l in layers:
        aggr_mean = aggr_acts[l].float().mean(dim=0).numpy()
        fair_mean = fair_acts[l].float().mean(dim=0).numpy()
        v = aggr_mean - fair_mean  # toward aggression
        v = v / (np.linalg.norm(v) + 1e-8)
        directions[l] = v.astype(np.float32)
    return directions


# ---------------------------------------------------------------------------
# Generosity direction extraction (radical generosity vs fair share)
# ---------------------------------------------------------------------------
#
# These functions extract a direction pointing FROM fair (50/50) TOWARD
# radical generosity (give everything, give more than half). This is the
# axis that the standard aggression/fair contrast does NOT capture: the
# "fair" anchor in the standard contrast IS the 50/50 point, so -v_mood
# (the safe direction) can only push the model toward fair, never beyond.
#
# The generosity contrast uses GENEROSITY_PROMPT_SETS: generous moods
# ("give everything / give more than your share") vs fair moods ("give a
# fair half / split equally"). The diff-of-means
#     v_generosity = mean(generous_mood) - mean(fair_mood)
# points from fair toward radical generosity. Steering with +v_generosity
# pushes the model to give MORE than half (metric -> 0.0), and -v_generosity
# pushes back toward fair.


def extract_mood_generosity_direction(
    model,
    layers,
    system=None,
    mood_set="emotional",
):
    """Extract the generosity direction from mood-only contrastive prompts.

    Diff-of-means of last-token pooled residuals:
        v_generosity = mean(generous_mood) - mean(fair_mood)
    pointing FROM fair (50/50) TOWARD radical generosity (give everything).
    Variance-independent.

    This is a SEPARATE axis from the standard aggression direction. The
    standard contrast (aggressive vs fair) captures the "selfish <-> fair"
    axis; this contrast (generous vs fair) captures the "fair <-> radical
    generosity" axis. The two axes are not collinear because RLHF trained
    the model away from "selfish" (rich representation) but never trained
    it toward "give everything beyond fair" (no representation in the
    standard axis).

    Args:
        model: the SteeringModel.
        layers: residual-stream layer indices to extract at.
        system: optional system prompt (defaults to SYSTEM_PROMPT).
        mood_set: "emotional" (affective stance) or "cognitive" (goal stance).

    Returns:
        ``{layer: v_generosity}`` unit-normalized, pointing toward
        radical generosity (give everything, metric -> 0.0).
    """
    if system is None:
        system = SYSTEM_PROMPT
    if mood_set not in GENEROSITY_PROMPT_SETS:
        raise ValueError(
            "Unknown mood_set %r; expected one of %s"
            % (mood_set, list(GENEROSITY_PROMPT_SETS))
        )
    gen_prompts, fair_prompts = GENEROSITY_PROMPT_SETS[mood_set]
    gen_acts = model.collect_residuals(
        gen_prompts, layers=layers, pool="last", system=system,
    )
    fair_acts = model.collect_residuals(
        fair_prompts, layers=layers, pool="last", system=system,
    )
    directions = {}
    for l in layers:
        gen_mean = gen_acts[l].float().mean(dim=0).numpy()
        fair_mean = fair_acts[l].float().mean(dim=0).numpy()
        # v points toward radical generosity (generous mood minus fair mood).
        v = gen_mean - fair_mean
        v = v / (np.linalg.norm(v) + 1e-8)
        directions[l] = v.astype(np.float32)
    return directions


def extract_mood_generosity_scenario_direction(
    model,
    layers,
    system=None,
    mood_set="emotional",
    n_per_game=8,
    seed=0,
):
    """Extract a generosity direction combining game context with mood contrast.

    Same as ``extract_mood_scenario_direction`` but uses the GENEROSITY
    contrast (radical generosity vs fair share) instead of the aggression
    contrast (aggressive vs fair). Real game scenarios (proposer/numeric
    games) are prepended with generous vs fair mood framings, and the
    residual is read on the last token of the framing prefix.

    Returns ``{layer: v}`` pointing toward radical generosity, unit-normalized.
    """
    if system is None:
        system = SYSTEM_PROMPT
    if mood_set not in GENEROSITY_PROMPT_SETS:
        raise ValueError(
            "Unknown mood_set %r; expected one of %s"
            % (mood_set, list(GENEROSITY_PROMPT_SETS))
        )
    gen_moods, fair_moods = GENEROSITY_PROMPT_SETS[mood_set]
    # Use proposer games (numeric) -- the axis we want to capture.
    proposer_games = ["dictator", "trust", "public_goods", "ultimatum_proposer"]
    scenarios = all_scenarios(n_per_game=n_per_game, seed=seed)
    scenarios = [s for s in scenarios if s.game in proposer_games]
    # Build mood-framed prompts: cycle mood framings across scenarios.
    gen_prompts = []
    fair_prompts = []
    for i, s in enumerate(scenarios):
        gm = gen_moods[i % len(gen_moods)]
        fm = fair_moods[i % len(fair_moods)]
        gen_prompts.append(gm + " " + s.prompt)
        fair_prompts.append(fm + " " + s.prompt)
    gen_acts = model.collect_residuals(
        gen_prompts, layers=layers, pool="last", system=system,
    )
    fair_acts = model.collect_residuals(
        fair_prompts, layers=layers, pool="last", system=system,
    )
    directions = {}
    for l in layers:
        gen_mean = gen_acts[l].float().mean(dim=0).numpy()
        fair_mean = fair_acts[l].float().mean(dim=0).numpy()
        v = gen_mean - fair_mean  # toward radical generosity
        v = v / (np.linalg.norm(v) + 1e-8)
        directions[l] = v.astype(np.float32)
    return directions


def extract_mood_generosity_game_specific_direction(
    model,
    layers,
    system=None,
    mood_set="emotional",
    games=None,
    n_per_game=8,
    seed=0,
):
    """Extract a game-specific generosity direction for stuck games.

    Same as ``extract_mood_game_specific_direction`` but uses the GENEROSITY
    contrast (radical generosity vs fair share). Uses ONLY the specified
    games (default: trust + ultimatum_proposer) as the contrastive set.

    Returns ``{layer: v}`` pointing toward radical generosity, unit-normalized.
    """
    if system is None:
        system = SYSTEM_PROMPT
    if mood_set not in GENEROSITY_PROMPT_SETS:
        raise ValueError(
            "Unknown mood_set %r; expected one of %s"
            % (mood_set, list(GENEROSITY_PROMPT_SETS))
        )
    if games is None:
        games = ["trust", "ultimatum_proposer"]
    gen_moods, fair_moods = GENEROSITY_PROMPT_SETS[mood_set]
    # Use only the specified games
    scenarios = all_scenarios(n_per_game=n_per_game, seed=seed)
    scenarios = [s for s in scenarios if s.game in games]
    # Build mood-framed prompts: cycle mood framings across scenarios.
    gen_prompts = []
    fair_prompts = []
    for i, s in enumerate(scenarios):
        gm = gen_moods[i % len(gen_moods)]
        fm = fair_moods[i % len(fair_moods)]
        gen_prompts.append(gm + " " + s.prompt)
        fair_prompts.append(fm + " " + s.prompt)
    gen_acts = model.collect_residuals(
        gen_prompts, layers=layers, pool="last", system=system,
    )
    fair_acts = model.collect_residuals(
        fair_prompts, layers=layers, pool="last", system=system,
    )
    directions = {}
    for l in layers:
        gen_mean = gen_acts[l].float().mean(dim=0).numpy()
        fair_mean = fair_acts[l].float().mean(dim=0).numpy()
        v = gen_mean - fair_mean  # toward radical generosity
        v = v / (np.linalg.norm(v) + 1e-8)
        directions[l] = v.astype(np.float32)
    return directions
