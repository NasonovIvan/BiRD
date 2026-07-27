"""Stage0: build contrastive (pos/neg) pairs per emotion.

For each emotion we produce ``pairs_per_emotion`` pairs. Each pair shares a
situation stem and contains:
  * ``pos``: a response strongly expressing the target emotion,
  * ``neg``: a neutral, emotionless response to the same situation.

The vector for an emotion is later ``mean(act(pos)) - mean(act(neg))`` over
its pairs (Stage1). Pairs are model-agnostic text, generated once via the
**judge API** (``JUDGE_API_URL`` / ``JUDGE_MODEL``) and cached to
``data/pairs/<emotion>.jsonl`` so every steered model reuses the same pairs.
The generator is any object exposing ``generate(prompts, max_new_tokens)`` —
in the default pipeline that is
:class:`emo.data.judge_generator.JudgeGenerator` (the API replacement for the
former local Qwen2.5-3B generator).

The result is shared across all steered models (lives in ``paths.data_dir``,
not per-model), keeping vectors comparable across the model matrix.
"""

from __future__ import annotations

import os
import re

from .. import io_utils
from . import isear_scenarios

# The contrastive pair drives what axis the steering vector learns. On an
# instruct/RLHF model the wording of the NEGATIVE prompt is critical: asking for
# a "neutral, emotionless" answer nudges a small chat model toward a safe-refusal
# / disclaimer register ("As an AI ..."), so the pos-vs-neg contrast captures an
# Emotion<->Refusal axis instead of the emotion itself — the inverted, negative
# dose-response we are fixing. We therefore make `neg` a *calm, factual, matter-
# of-fact* first-person account of the SAME situation (still first-person, still
# on-topic, just unemotional), so the only thing separating pos from neg is the
# emotional content — not "answer vs refuse".
_POS_INSTR = (
    "You are a person describing your own experience. In 1-2 sentences, give a "
    "vivid first-person reaction to the situation below that clearly expresses "
    "intense {emotion}. Stay on the situation; output only the reaction text, "
    "with no preamble or disclaimers.\n"
    "Situation: {situation}\nReaction:"
)
_NEG_INSTR = (
    "You are a person calmly recounting your own experience. In 1-2 sentences, "
    "give a matter-of-fact, even-toned first-person account of the situation "
    "below, simply stating what happened without dwelling on feelings. Stay on "
    "the situation; output only the account text, with no preamble or "
    "disclaimers.\n"
    "Situation: {situation}\nAccount:"
)


def _clean(text: str) -> str:
    text = text.strip().strip('"').strip()
    text = re.sub(r"\s+", " ", text)
    return text


def pairs_path(data_dir: str, emotion: str) -> str:
    return os.path.join(data_dir, "pairs", f"{emotion}.jsonl")


def _expand_situations(emotion: str, n: int) -> list[str]:
    """Cycle seed stems to reach n situations (generator adds variety)."""
    seeds = isear_scenarios.SCENARIO_SEEDS[emotion]
    out = []
    i = 0
    while len(out) < n:
        out.append(seeds[i % len(seeds)])
        i += 1
    return out[:n]


def build_pairs_for_emotion(
    generator, emotion: str, n: int, data_dir: str, max_new_tokens: int = 96
) -> str:
    """Generate (or load cached) n pairs for one emotion. Returns file path.

    ``generator`` is any object exposing
    ``generate(prompts, max_new_tokens) -> list[str]`` — normally a
    :class:`emo.data.judge_generator.JudgeGenerator` (judge API). Resumable:
    if the cache already has >= n pairs, skips.
    """
    out_path = pairs_path(data_dir, emotion)
    existing = io_utils.read_jsonl(out_path)
    if len(existing) >= n:
        print(f"[stage0] {emotion}: {len(existing)} cached pairs, skip")
        return out_path

    situations = _expand_situations(emotion, n)
    todo = situations[len(existing) :]

    pos_prompts = [_POS_INSTR.format(emotion=emotion, situation=s) for s in todo]
    neg_prompts = [_NEG_INSTR.format(situation=s) for s in todo]

    pos_out = generator.generate(pos_prompts, max_new_tokens=max_new_tokens)
    neg_out = generator.generate(neg_prompts, max_new_tokens=max_new_tokens)

    for s, p, ntext in zip(todo, pos_out, neg_out):
        io_utils.append_jsonl(
            {
                "emotion": emotion,
                "situation": s,
                "pos": _clean(p),
                "neg": _clean(ntext),
            },
            out_path,
        )
    print(f"[stage0] {emotion}: wrote {len(todo)} new pairs -> {out_path}")
    return out_path


def build_all_pairs(generator, cfg) -> dict[str, str]:
    """Build pairs for every emotion (and every configured language).

    Returns {emotion: path} for English pairs (primary).  Multilingual
    pairs (ru, fr) are built as a side effect and stored under
    ``data/pairs/<lang>/<emotion>.jsonl``.
    """
    data_dir = cfg.paths.abs("data_dir")
    n = int(cfg.data.get("pairs_per_emotion", 60))
    mnt = int(cfg.data.get("max_new_tokens", 96))
    languages = cfg.data.get("languages", ["en"])

    # Build English pairs (primary, used by Stage1).
    paths: dict[str, str] = {}
    for emo in cfg.emotions:
        paths[emo] = build_pairs_for_emotion(
            generator, emo, n, data_dir, max_new_tokens=mnt
        )

    # Build multilingual pairs for non-English languages (N2 cross-lingual).
    from ..measure.crosslingual import build_multilingual_pairs
    from . import isear_scenarios as _isear

    for lang in languages:
        if lang == "en":
            continue
        print(f"[stage0] building multilingual pairs for '{lang}'")
        for emo in cfg.emotions:
            build_multilingual_pairs(
                generator, cfg, lang, emo, n, data_dir, max_new_tokens=mnt
            )

    return paths


def load_pairs(cfg, emotion: str) -> list[dict]:
    return io_utils.read_jsonl(pairs_path(cfg.paths.abs("data_dir"), emotion))
