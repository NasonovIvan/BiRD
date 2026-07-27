"""Activation steering via forward hooks on the residual stream.

A steering spec adds ``coeff * unit_vector`` to the output of a decoder block
(the residual stream) at one or more layers, for every token position. The
vector is L2-normalized at registration so ``coeff`` has a consistent meaning
across emotions and layers.

**Norm-relative scaling (default).** A unit perturbation of magnitude ``coeff``
is invisible on models whose residual stream carries very large activation
norms — e.g. Gemma-2, where mid-layer token norms run into the hundreds, so a
``coeff<=12`` unit nudge is only a ~5-10% perturbation and greedy decoding just
collapses back to the modal (often safety-refrain) response. When
``norm_scale=True`` the injected delta is rescaled to the *per-token* residual
norm, i.e. ``coeff`` becomes the fraction of the local residual magnitude added
along the steering direction. This makes ``coeff`` portable across models and
layers and is what lets steering actually bite on Gemma-2. Set
``norm_scale=False`` for the legacy absolute-magnitude behaviour.

Usage::

    with steer(model, [SteerSpec(layer=L, vector=v, coeff=8.0)]):
        text = model.generate([...])

Multiple specs at the same layer are summed, enabling composition
(blends Joy+Sadness, subtraction Anger-Sadness) in a single forward pass.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch


@dataclass
class SteerSpec:
    layer: int
    vector: torch.Tensor  # [hidden]; normalized internally
    coeff: float = 8.0
    normalize: bool = True
    # Scale the injected delta to the per-token residual norm so ``coeff`` is a
    # fraction of the local residual magnitude (portable across models/layers).
    # See the module docstring; False restores absolute-magnitude steering.
    norm_scale: bool = True


def make_multi_layer_specs(
    layers: list[int],
    vector: torch.Tensor,
    coeff: float,
    normalize: bool = True,
    norm_scale: bool = True,
) -> list[SteerSpec]:
    """Create one SteerSpec per layer for multi-layer steering.

    The total steering budget ``coeff`` is split evenly across layers so the
    combined perturbation magnitude is comparable to single-layer steering at
    the same coeff. Each layer receives ``coeff / len(layers)``.

    Multi-layer steering is more effective on instruct models because the
    emotion signal is distributed across several mid-network layers; injecting
    at one layer alone is often too weak to overcome the RLHF safety prior.
    """
    per_layer = coeff / max(1, len(layers))
    return [
        SteerSpec(
            layer=l, vector=vector.clone(), coeff=per_layer,
            normalize=normalize, norm_scale=norm_scale,
        )
        for l in layers
    ]


@dataclass
class TimeVaryingSteerSpec:
    """Steering spec whose coefficient and/or vector change per generation step.

    N3 (Temporal Arcs): instead of a fixed ``coeff`` and ``vector``, the hook
    calls ``coeff_fn(step)`` and/or ``vector_fn(step)`` on every forward pass
    during autoregressive generation, where ``step`` is the number of *new*
    tokens generated so far (0 = prefill, 1 = first generated token, …).

    The step counter is maintained inside the hook closure and incremented
    after each forward pass. During the prefill pass (many tokens at once) the
    counter is set to 0 and the *prefill* step's coeff/vector are applied to
    all token positions uniformly (the prefill is the prompt, not the
    generation, so the steering signal is the same for every prompt token).

    If ``coeff_fn`` is ``None`` it falls back to ``coeff`` (constant).
    If ``vector_fn`` is ``None`` it falls back to ``vector`` (constant).
    """
    layer: int
    vector: torch.Tensor  # base vector [hidden]; used when vector_fn is None
    coeff: float = 8.0  # base coeff; used when coeff_fn is None
    coeff_fn: object | None = None  # Callable[[int], float] | None
    vector_fn: object | None = None  # Callable[[int], torch.Tensor] | None
    normalize: bool = True
    norm_scale: bool = True


def _prepare(vec: torch.Tensor, normalize: bool) -> torch.Tensor:
    v = vec.detach().float()
    if normalize:
        n = v.norm()
        if n > 0:
            v = v / n
    return v


@contextmanager
def steer(model, specs: list[SteerSpec | TimeVaryingSteerSpec]):
    """Temporarily add steering vectors to residual-stream outputs.

    ``model`` is a :class:`emo.models.SteeringModel`. Hooks are removed on
    exit even if generation raises.

    Accepts both static :class:`SteerSpec` and time-varying
    :class:`TimeVaryingSteerSpec`. Time-varying specs track the generation
    step via a mutable counter in the hook closure; the counter is incremented
    after each forward pass (prefill = step 0, first decoded token = step 1,
    …).
    """
    # Group specs by layer so each layer adds a single combined delta.
    by_layer: dict[int, list] = {}
    for s in specs:
        by_layer.setdefault(s.layer, []).append(s)

    handles = []
    # Shared step counter for all time-varying specs.  Mutable list so the
    # hook closure can modify it without ``nonlocal``.
    step_counter = [0]

    def make_hook(layer_specs):
        def hook(_module, _inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            # Per-token residual norm [B, T, 1] for norm-relative scaling.
            tok_norm = hs.detach().float().norm(dim=-1, keepdim=True)
            step = step_counter[0]
            delta = None
            for s in layer_specs:
                # Resolve coeff and vector (static or time-varying).
                if isinstance(s, TimeVaryingSteerSpec):
                    c = s.coeff_fn(step) if s.coeff_fn is not None else s.coeff
                    raw_v = s.vector_fn(step) if s.vector_fn is not None else s.vector
                else:
                    c = s.coeff
                    raw_v = s.vector
                v = _prepare(raw_v, s.normalize).to(hs.device, hs.dtype)
                if s.norm_scale:
                    scale = (c / 100.0) * tok_norm.to(hs.dtype)
                    term = scale * v
                else:
                    term = c * v
                delta = term if delta is None else delta + term
            hs_new = hs + delta
            # Guard against NaN/inf from extreme coefficients (norm_scale=False
            # with large coeff can overflow the residual stream → CUDA error
            # that crashes the entire experiment). Replace non-finite values
            # with the original activations so generation degrades gracefully
            # (incoherent text) instead of crashing.
            if not torch.isfinite(hs_new).all():
                hs_new = torch.where(torch.isfinite(hs_new), hs_new, hs)
            hs = hs_new
            # Increment step counter after each forward pass.  During
            # autoregressive generation with KV-cache, each forward pass after
            # the prefill produces exactly 1 new token; the prefill itself
            # processes many prompt tokens at once (step 0).
            step_counter[0] += 1
            if isinstance(out, tuple):
                return (hs,) + tuple(out[1:])
            return hs

        return hook

    try:
        for layer, layer_specs in by_layer.items():
            mod = model.residual_module(layer)
            handles.append(mod.register_forward_hook(make_hook(layer_specs)))
        yield
    finally:
        for h in handles:
            h.remove()
