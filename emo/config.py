"""Configuration loading for the emotion-steering experiments.

Layered config:
    config/base.json          global defaults
    config/models/<id>.json   per-model overrides (model + sae blocks)
    config/model_sets.json    named collections of model ids
    CLI flags                 override everything (applied by callers)

The public entry point is :func:`load_config`, which merges base + a single
model config into a :class:`Config` dataclass tree. Layer fractions are
resolved to absolute indices against the model's ``num_layers``.

Config files are JSONC (JSON with ``//`` / ``/* */`` comments) rather than
YAML, so the package has no third-party parser dependency — see
:mod:`emo.jsonc`.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from dataclasses import fields as _dc_fields
from typing import Any

from .jsonc import load_jsonc

# Package root = emo_exp/ (parent of this file's parent).
PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(PKG_ROOT, "config")
MODELS_DIR = os.path.join(CONFIG_DIR, "models")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SAEConfig:
    enabled: bool = False
    provider: str = "sae_lens"
    release: str | None = None
    sae_id_template: str | None = None

    def sae_id_for_layer(self, layer: int) -> str | None:
        if not self.sae_id_template:
            return None
        return self.sae_id_template.format(i=layer)


@dataclass
class ModelConfig:
    id: str
    family: str
    hf_name: str
    num_layers: int
    hidden_size: int
    residual_module: str = "model.layers.{i}"
    dtype: str = "bfloat16"
    device_map: str = "cuda:0"
    uses_chat_template: bool = True
    # Maximum number of prompts to feed to model.generate() in a single
    # forward pass. When the caller passes more prompts than this, the
    # generate() method chunks them into batches of this size and
    # concatenates the results. This prevents CUDA OOM from the KV cache
    # when generating hundreds/thousands of game decisions at once.
    # 32 is a safe default for an 8B model on an A100 80GB; smaller GPUs
    # or larger models should reduce this (e.g. 8 or 16).
    gen_batch_size: int = 32
    # Qwen3 models support a "thinking" mode via the chat template parameter
    # ``enable_thinking``. When False, the model skips chain-of-thought
    # reasoning and answers directly — critical for behavioral games where
    # we need a single parseable decision, not a reasoning trace.
    enable_thinking: bool | None = None
    # Force a specific attention implementation (e.g. "eager", "sdpa",
    # "flash_attention_2"). Only passed to from_pretrained when set
    # (per-model opt-in). The default (letting the remote modeling file
    # pick) breaks for some models with custom modeling files.
    attn_implementation: str | None = None
    sae: SAEConfig = field(default_factory=SAEConfig)

    def residual_module_for_layer(self, layer: int) -> str:
        return self.residual_module.format(i=layer)


@dataclass
class PathsConfig:
    runs_dir: str = "runs"
    data_dir: str = "data"
    benchmark_dir: str = "config/benchmark"

    def abs(self, which: str) -> str:
        rel = getattr(self, which)
        return rel if os.path.isabs(rel) else os.path.join(PKG_ROOT, rel)


@dataclass
class Config:
    """Fully merged config for one model run."""

    seed: int
    emotions: list[str]
    neutral_label: str
    paths: PathsConfig
    model: ModelConfig
    # Stage parameter blocks kept as plain dicts: stages read what they need
    # and CLI flags override individual keys without dataclass churn.
    data: dict[str, Any]
    extract: dict[str, Any]
    steer: dict[str, Any]
    sae_params: dict[str, Any]
    probe_params: dict[str, Any]
    compose: dict[str, Any]
    eval: dict[str, Any]
    judge: dict[str, Any] = field(default_factory=dict)
    # N1: Emotional Circumplex (valence/arousal subspace) config.
    circumplex: dict[str, Any] = field(default_factory=dict)
    # N2: Cross-lingual transfer config.
    crosslingual: dict[str, Any] = field(default_factory=dict)
    # Generation interface knobs (template mode, decoding params, refusal
    # suppression, sampling seeds) shared by Stage3 + the Stage8 diagnostics.
    generation: dict[str, Any] = field(default_factory=dict)
    # Stage8 diagnostics (logit-lens, probe-after-steering, LM-head bias,
    # template/decoding/refusal ablations, extended coeff sweeps).
    diagnostics: dict[str, Any] = field(default_factory=dict)
    # Resolved absolute layer indices for the extraction sweep.
    sweep_layers: list[int] = field(default_factory=list)

    def run_dir(self) -> str:
        return os.path.join(self.paths.abs("runs_dir"), self.model.id)

    def stage_dir(self, stage: str) -> str:
        return os.path.join(self.run_dir(), stage)


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def _read_config(path: str) -> dict:
    return load_jsonc(path) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def list_model_ids() -> list[str]:
    return sorted(
        os.path.splitext(f)[0] for f in os.listdir(MODELS_DIR) if f.endswith(".json")
    )


def resolve_model_set(name: str) -> list[str]:
    """Return the list of model ids for a named set from model_sets.json.

    De-duplicates while preserving order. A bare model id (matching a file in
    config/models/) is accepted as a one-element set for convenience.
    """
    sets_path = os.path.join(CONFIG_DIR, "model_sets.json")
    sets = _read_config(sets_path).get("sets", {})
    if name in sets:
        ids = sets[name]
    elif name in list_model_ids():
        ids = [name]
    else:
        raise ValueError(
            f"Unknown model set or id '{name}'. "
            f"Known sets: {sorted(sets)}; known ids: {list_model_ids()}"
        )
    seen, ordered = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i)
            ordered.append(i)
    return ordered


def _resolve_sweep_layers(layer_fracs: list[float], num_layers: int) -> list[int]:
    """Map depth fractions to absolute layer indices, clamped & de-duplicated."""
    out = []
    for fr in layer_fracs:
        idx = int(round(fr * (num_layers - 1)))
        idx = max(0, min(num_layers - 1, idx))
        if idx not in out:
            out.append(idx)
    return sorted(out)


def load_config(model_id: str, overrides: dict | None = None) -> Config:
    """Merge base.json + config/models/<model_id>.json into a :class:`Config`.

    ``overrides`` is a nested dict (typically from CLI) deep-merged last.
    """
    base = _read_config(os.path.join(CONFIG_DIR, "base.json"))
    model_path = os.path.join(MODELS_DIR, f"{model_id}.json")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model config: {model_path}")
    model_cfg_raw = _read_config(model_path)

    merged = _deep_merge(base, model_cfg_raw)
    if overrides:
        merged = _deep_merge(merged, overrides)

    sae_block = merged.get("sae_model_block") or model_cfg_raw.get("sae", {})
    # NB: base.json has a top-level `sae` for *sweep params*; the model config
    # has a `sae` for the *SAE artifact*. Keep them separate.
    sae_cfg = SAEConfig(
        enabled=bool(sae_block.get("enabled", False)),
        provider=sae_block.get("provider", "sae_lens"),
        release=sae_block.get("release"),
        sae_id_template=sae_block.get("sae_id_template"),
    )

    m = merged["model"]
    # Build the full kwarg set, then filter it against the *actual* fields of
    # ``ModelConfig``. This makes ``load_config`` robust to a stale
    # ``ModelConfig`` dataclass that lacks newer optional fields (e.g.
    # ``attn_implementation``): such fields are silently dropped instead of
    # raising ``TypeError: __init__() got an unexpected keyword argument``.
    # This matters because experiment scripts and ``config.py`` may be
    # snapshotted at different revisions on a sandbox.
    _model_kwargs = {
        "id": m["id"],
        "family": m["family"],
        "hf_name": m["hf_name"],
        "num_layers": int(m["num_layers"]),
        "hidden_size": int(m["hidden_size"]),
        "residual_module": m.get("residual_module", "model.layers.{i}"),
        "dtype": m.get("dtype", "bfloat16"),
        "device_map": m.get("device_map", "cuda:0"),
        "uses_chat_template": bool(m.get("uses_chat_template", True)),
        "enable_thinking": m.get("enable_thinking"),
        # gen_batch_size can be set per-model (in the `model` block of a
        # model config) or per-experiment (at the top level of an experiment
        # config like rlhf_break.json). The experiment-level value wins.
        "gen_batch_size": int(
            merged.get("gen_batch_size")
            or m.get("gen_batch_size", 32)
        ),
        "attn_implementation": m.get("attn_implementation"),
        "sae": sae_cfg,
    }
    _accepted = {f.name for f in _dc_fields(ModelConfig)}
    model_cfg = ModelConfig(
        **{k: v for k, v in _model_kwargs.items() if k in _accepted}
    )

    paths_cfg = PathsConfig(**(merged.get("paths") or {}))

    extract = merged.get("extract", {})
    sweep_layers = _resolve_sweep_layers(
        extract.get("layer_fracs", [0.5]), model_cfg.num_layers
    )

    return Config(
        seed=int(merged.get("seed", 0)),
        emotions=list(merged["emotions"]),
        neutral_label=merged.get("neutral_label", "neutral"),
        paths=paths_cfg,
        model=model_cfg,
        data=merged.get("data", {}),
        extract=extract,
        steer=merged.get("steer", {}),
        sae_params=merged.get("sae", {}),  # legacy SAE sweep params (unused)
        probe_params=merged.get("probe", {}),  # Stage4 L1-probe sweep params
        compose=merged.get("compose", {}),
        eval=merged.get("eval", {}),
        judge=merged.get("judge", {}),
        circumplex=merged.get("circumplex", {}),
        crosslingual=merged.get("crosslingual", {}),
        generation=merged.get("generation", {}),
        diagnostics=merged.get("diagnostics", {}),
        sweep_layers=sweep_layers,
    )
