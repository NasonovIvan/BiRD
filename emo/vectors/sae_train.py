"""Self-trained TopK Sparse Autoencoder (N4) — no ``sae_lens`` dependency.

The whole SAE path used to depend on ``sae_lens`` (``from sae_lens import SAE``),
which is not installable in this environment (see
[`SAE_FREE_ALTERNATIVES.md`](emo_exp/SAE_FREE_ALTERNATIVES.md)). This module
trains a SAE *from scratch* on cached residual-stream activations of a single
mid layer of the steered model, so the feature-splitting / decomposition
experiments (Stage4, Stage12) are fully self-contained.

Architecture — TopK SAE
-----------------------
A classic over-complete autoencoder with a hard TopK activation (Makelov /
Gao et al. style): only the ``k`` largest pre-activations fire per token, the
rest are zeroed. TopK gives a directly-controlled L0 sparsity (exactly ``k``
active features) and is far more stable to train than an L1 penalty.

    pre   = (x - b_dec) @ W_enc + b_enc          # [N, n_features]
    feats = TopK(pre, k)                          # keep top-k per row, zero rest
    recon = feats @ W_dec + b_dec                 # [N, d_model]

Decoder columns (``W_dec`` rows) are renormalised to unit L2 after every step,
so each feature's *direction* is unit and its *magnitude* lives in ``feats``.

Dead-feature revival — AuxK
---------------------------
Over-complete SAEs tend to leave many features permanently un-fired ("dead").
We track per-feature firing recency and, for the features that have not fired
recently, add an auxiliary reconstruction term: reconstruct the *residual
error* ``x - recon`` using the top-``k_aux`` dead features only. This pushes
gradient into the dead set and revives them (Gao et al. 2024, "AuxK").

Training flow (memory)
----------------------
Collect activations with the LLM loaded, then *unload the LLM* and train the
SAE on the cached activation tensor — they never contend for GPU memory.

Artifacts (runs/<model>/stage4_sae/):
  sae_layer{L}.pt        -> trained SAE state_dict (+ d_model/n_features/k)
  sae_train_layer{L}.json-> training metrics (recon loss, dead frac, L0, ...)

``.pt`` checkpoints are git/arc-ignored (large binaries).
"""

from __future__ import annotations

import gc
import math
import os

import numpy as np
import torch
import torch.nn as nn

from .. import io_utils
from ..data import build_pairs

# ---------------------------------------------------------------------------
# Layer / corpus helpers
# ---------------------------------------------------------------------------


def sae_layer(cfg) -> int:
    """Absolute residual layer index the SAE is trained on.

    ``sae.layer_frac`` (fraction of depth) maps to a layer index; ``null``
    falls back to the middle of the extraction sweep.
    """
    sp = cfg.sae_params or {}
    frac = sp.get("layer_frac", None)
    if frac is None:
        layers = cfg.sweep_layers or [cfg.model.num_layers // 2]
        return int(layers[len(layers) // 2])
    n = int(cfg.model.num_layers)
    return int(max(0, min(n - 1, round(float(frac) * (n - 1)))))


def _wikitext_texts(n_target: int, seed: int) -> list[str]:
    """A sample of general-language text for the neutral side of the corpus.

    Tries HuggingFace ``datasets`` (wikitext-103); on any failure falls back to
    the benchmark's neutral / open-continuation prompts so training still has a
    non-emotional language signal without a network dependency.
    """
    try:
        from datasets import load_dataset  # type: ignore

        ds = load_dataset(
            "wikitext", "wikitext-103-raw-v1", split="train", streaming=True
        )
        out: list[str] = []
        for row in ds:
            t = (row.get("text") or "").strip()
            if len(t) >= 32:
                out.append(t)
            if len(out) >= n_target:
                break
        if out:
            return out
    except Exception as e:  # noqa: BLE001 — offline / dataset missing
        print(f"[sae] wikitext unavailable ({e}); using benchmark neutral text")

    # Offline fallback: benchmark neutral + open-continuation prompts.
    try:
        from .. import jsonc

        base = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "config",
            "benchmark",
            "neutral_prompts.json",
        )
        data = jsonc.loads(open(base, "r", encoding="utf-8").read()) or {}
        pool: list[str] = []
        for key in ("neutral", "open_continuation", "creative_writing", "offdist"):
            pool.extend([str(x) for x in (data.get(key) or [])])
        rng = np.random.default_rng(seed)
        if pool:
            idx = rng.integers(0, len(pool), size=max(1, n_target))
            return [pool[i] for i in idx]
    except Exception as e:  # noqa: BLE001
        print(f"[sae] neutral fallback failed ({e})")
    return []


def _isear_positive_texts(cfg) -> list[str]:
    """Emotion-rich text: the positive side of every contrastive pair."""
    out: list[str] = []
    for e in cfg.emotions:
        for pair in build_pairs.load_pairs(cfg, e):
            pos = (pair.get("pos") or "").strip()
            if pos:
                out.append(pos)
    return out


def build_corpus(cfg, seed: int) -> tuple[list[str], list[bool]]:
    """Mixed activation corpus: (texts, is_emotional flags).

    Mixing neutral language with emotional text ensures the SAE learns genuine
    emotion features without the general language manifold collapsing. The
    ``is_emotional`` flag is informational (kept for diagnostics / decompose).
    """
    sp = cfg.sae_params or {}
    n_total = int(sp.get("max_activation_samples", 200000))
    wt_frac = float(sp.get("corpus_wikitext_frac", 0.6))
    emo_texts = _isear_positive_texts(cfg)
    # Texts are tokenised to many tokens each; cap the *number of texts* so we
    # land near n_total tokens without an exact token budget here (the collector
    # enforces the hard token cap).
    n_emo_texts = max(1, len(emo_texts))
    # Roughly balance text counts by the configured fractions.
    n_wiki_texts = max(
        1, int(n_emo_texts * (wt_frac / max(1e-6, 1.0 - wt_frac)))
    )
    wiki = _wikitext_texts(n_wiki_texts, seed)
    texts = list(emo_texts) + list(wiki)
    flags = [True] * len(emo_texts) + [False] * len(wiki)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(texts))
    texts = [texts[i] for i in order]
    flags = [flags[i] for i in order]
    return texts, flags


# ---------------------------------------------------------------------------
# TopK SAE module
# ---------------------------------------------------------------------------


def _topk(pre: torch.Tensor, k: int) -> torch.Tensor:
    """Keep the top-k values per row (by magnitude-positive), zero the rest."""
    if k >= pre.shape[-1]:
        return torch.relu(pre)
    vals, idx = torch.topk(pre, k, dim=-1)
    out = torch.zeros_like(pre)
    out.scatter_(-1, idx, torch.relu(vals))
    return out


class TopKSAE(nn.Module):
    """Over-complete TopK sparse autoencoder over the residual stream."""

    def __init__(self, d_model: int, n_features: int, k: int):
        super().__init__()
        self.d_model = int(d_model)
        self.n_features = int(n_features)
        self.k = int(k)
        # Tied-ish init: decoder rows random unit, encoder = decoder^T.
        W = torch.randn(self.n_features, self.d_model)
        W = W / W.norm(dim=1, keepdim=True).clamp_min(1e-8)
        self.W_dec = nn.Parameter(W.clone())  # [n_features, d_model]
        self.W_enc = nn.Parameter(W.t().clone())  # [d_model, n_features]
        self.b_enc = nn.Parameter(torch.zeros(self.n_features))
        self.b_dec = nn.Parameter(torch.zeros(self.d_model))

    def preacts(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.b_dec) @ self.W_enc + self.b_enc

    def encode(self, x: torch.Tensor, k: int | None = None) -> torch.Tensor:
        return _topk(self.preacts(x), self.k if k is None else k)

    def decode(self, feats: torch.Tensor) -> torch.Tensor:
        return feats @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor):
        pre = self.preacts(x)
        feats = _topk(pre, self.k)
        recon = self.decode(feats)
        return recon, feats, pre

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        norms = self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8)
        self.W_dec.div_(norms)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _aux_loss(
    sae: TopKSAE, x: torch.Tensor, recon: torch.Tensor, pre: torch.Tensor,
    dead_mask: torch.Tensor, k_aux: int,
) -> torch.Tensor:
    """AuxK dead-feature revival: reconstruct the error with dead features."""
    n_dead = int(dead_mask.sum().item())
    if n_dead == 0 or k_aux <= 0:
        return x.new_zeros(())
    err = (x - recon).detach()
    masked = pre.masked_fill(~dead_mask.unsqueeze(0), float("-inf"))
    kk = min(k_aux, n_dead)
    vals, idx = torch.topk(masked, kk, dim=-1)
    feats = torch.zeros_like(pre)
    feats.scatter_(-1, idx, torch.relu(vals))
    err_hat = feats @ sae.W_dec
    return ((err - err_hat) ** 2).mean()


def train_sae(acts: torch.Tensor, cfg, device) -> tuple[TopKSAE, dict]:
    """Train a TopK SAE on cached activations ``acts`` [N, d_model].

    Returns the trained SAE (on CPU) and a metrics dict (final reconstruction
    loss, fraction of variance unexplained, dead-feature fraction, L0).
    """
    sp = cfg.sae_params or {}
    d_model = acts.shape[1]
    n_features = int(sp.get("n_features", 32768))
    k = int(sp.get("topk", 64))
    lr = float(sp.get("lr", 1e-3))
    batch_size = int(sp.get("batch_size", 4096))
    n_epochs = int(sp.get("n_epochs", 10))
    aux_weight = float(sp.get("aux_loss_weight", 0.1))
    seed = int(cfg.seed)

    torch.manual_seed(seed)
    sae = TopKSAE(d_model, n_features, k).to(device)
    sae.normalize_decoder()
    # Initialise the pre-encoder bias at the data mean (standard for SAEs).
    with torch.no_grad():
        sae.b_dec.copy_(acts.mean(dim=0).to(device))
    opt = torch.optim.Adam(sae.parameters(), lr=lr)

    n = acts.shape[0]
    var = float(((acts - acts.mean(dim=0)) ** 2).sum(dim=1).mean().item())
    # Per-feature steps-since-last-fired tracker (dead-feature detection).
    last_fired = torch.zeros(n_features, dtype=torch.long, device=device)
    dead_after = max(1, (n // max(1, batch_size)))  # ~1 epoch without firing
    k_aux = min(n_features, 2 * k)
    step = 0
    last_metrics: dict = {}

    print(
        f"[sae] training: N={n} d_model={d_model} n_features={n_features} "
        f"k={k} epochs={n_epochs} batch={batch_size}"
    )
    for epoch in range(n_epochs):
        perm = torch.randperm(n)
        running = 0.0
        n_batches = 0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            x = acts[idx].to(device, non_blocking=True).float()
            recon, feats, pre = sae(x)
            recon_loss = ((x - recon) ** 2).mean()
            fired = (feats > 0).any(dim=0)
            last_fired[fired] = step
            dead_mask = (step - last_fired) > dead_after
            aux = _aux_loss(sae, x, recon, pre, dead_mask, k_aux)
            loss = recon_loss + aux_weight * aux
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sae.normalize_decoder()
            running += float(recon_loss.item())
            n_batches += 1
            step += 1
        mean_recon = running / max(1, n_batches)
        fvu = mean_recon / max(1e-12, var)
        dead_frac = float(((step - last_fired) > dead_after).float().mean().item())
        last_metrics = {
            "epoch": epoch,
            "recon_mse": mean_recon,
            "frac_variance_unexplained": fvu,
            "dead_feature_frac": dead_frac,
        }
        if (epoch + 1) % 100 == 0 or epoch + 1 == n_epochs:
            print(
                f"[sae] epoch {epoch + 1}/{n_epochs} recon_mse={mean_recon:.4f} "
                f"fvu={fvu:.3f} dead={dead_frac:.3f}"
            )

    # Final L0 over a sample.
    with torch.no_grad():
        sample = acts[: min(n, 8192)].to(device).float()
        feats = sae.encode(sample)
        l0 = float((feats > 0).float().sum(dim=1).mean().item())
    metrics = {
        "d_model": d_model,
        "n_features": n_features,
        "topk": k,
        "n_samples": int(n),
        "activation_variance": var,
        "l0": l0,
        **last_metrics,
    }
    sae.to("cpu")
    return sae, metrics


# ---------------------------------------------------------------------------
# Activation collection (per token) + checkpoint IO
# ---------------------------------------------------------------------------


def collect_activations(model, cfg, layer: int) -> torch.Tensor:
    """Collect per-token residuals on the mixed corpus. Returns [N, d_model]."""
    sp = cfg.sae_params or {}
    max_tokens = int(sp.get("max_activation_samples", 200000))
    texts, _flags = build_corpus(cfg, int(cfg.seed))
    if not texts:
        raise RuntimeError("SAE corpus is empty (no pairs and no neutral text)")
    print(f"[sae] collecting activations: {len(texts)} texts, cap {max_tokens} tokens")
    acts = model.collect_token_residuals(
        texts, layer, max_tokens=max_tokens, batch_size=int(sp.get("collect_batch", 8))
    )
    print(f"[sae] collected {acts.shape[0]} token activations (d={acts.shape[1]})")
    return acts


def _ckpt_path(stage_dir: str, layer: int) -> str:
    return os.path.join(stage_dir, f"sae_layer{layer}.pt")


def save_sae(sae: TopKSAE, stage_dir: str, layer: int) -> str:
    path = _ckpt_path(stage_dir, layer)
    torch.save(
        {
            "state_dict": sae.state_dict(),
            "d_model": sae.d_model,
            "n_features": sae.n_features,
            "topk": sae.k,
        },
        path,
    )
    return path


def load_sae(stage_dir: str, layer: int, device="cpu") -> TopKSAE | None:
    path = _ckpt_path(stage_dir, layer)
    if not os.path.exists(path):
        return None
    blob = torch.load(path, map_location=device)
    sae = TopKSAE(blob["d_model"], blob["n_features"], blob["topk"])
    sae.load_state_dict(blob["state_dict"])
    sae.to(device)
    sae.eval()
    return sae


def train_or_load(model, cfg, layer: int, force: bool = False):
    """Resumable: load an existing SAE checkpoint, else collect+train+save.

    Returns ``(sae, metrics)``. Frees the collected activation tensor and runs
    a GC pass before returning so the caller can reload the LLM if needed.
    """
    stage_dir = io_utils.ensure_dir(cfg.stage_dir("stage4_sae"))
    metrics_path = os.path.join(stage_dir, f"sae_train_layer{layer}.json")
    if not force:
        existing = load_sae(stage_dir, layer, device="cpu")
        if existing is not None:
            metrics = io_utils.load_json(metrics_path) or {}
            print(f"[sae] layer {layer}: checkpoint found, skip training")
            return existing, metrics

    device = model.input_device
    acts = collect_activations(model, cfg, layer)
    sae, metrics = train_sae(acts, cfg, device)
    save_sae(sae, stage_dir, layer)
    io_utils.save_json({"layer": layer, **metrics}, metrics_path)
    del acts
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return sae, metrics
