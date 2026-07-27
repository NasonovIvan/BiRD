"""Emotion-steering experiment package.

Reusable library code lives here; thin CLI entry points live in
``emo_exp/experiments/``. See ``plans/emo-steering-experiments-plan.md`` for
the full experiment architecture.
"""

# --- torchvision guard -------------------------------------------------------
# This is a TEXT-ONLY pipeline: it never uses torchvision. transformers, though,
# probes for it at import time (``is_torchvision_available()``) and, if it finds
# an installed-but-broken torchvision, transitively imports it while resolving a
# model class (e.g. Gemma2ForCausalLM). In the offline GPU pod torchvision's
# compiled C++ extension (_C) is built against a different torch than the one
# installed, so that import half-initializes and raises
#   AttributeError: partially initialized module 'torchvision' has no attribute
#   'extension' (most likely due to a circular import)
# which aborts model loading. Even after ``pip uninstall torchvision`` a stray
# partial install / leftover metadata can keep tripping this.
#
# We neutralize it process-wide BEFORE any transformers import: inserting None
# into ``sys.modules['torchvision']`` makes ``importlib.util.find_spec`` return
# None, so transformers' ``is_torchvision_available()`` reports it as absent and
# never imports it. Any stray ``import torchvision`` becomes a clean ImportError
# instead of the misleading circular-import crash. This runs here in the package
# __init__ because every entry point imports ``emo`` before ``transformers``.
import sys as _sys

if "torchvision" not in _sys.modules or _sys.modules["torchvision"] is not None:
    _sys.modules["torchvision"] = None  # type: ignore[assignment]

# --- TensorFlow / Flax guard -------------------------------------------------
# This is a torch-only pipeline; it never uses the TensorFlow or Flax backends.
# Recent transformers, however, transitively imports TensorFlow while resolving
# a model class: e.g. Gemma2ForCausalLM -> modeling_layers -> processing_utils
# -> video_utils -> image_transforms -> ``import tensorflow as tf``. In the
# offline GPU pod TensorFlow ships a compiled extension
# (``tensorflow/python/framework/fast_tensor_util``) built against a different
# NumPy than the one installed, so that import aborts model loading with
#   ValueError: numpy.dtype size changed, may indicate binary incompatibility.
#   Expected 96 from C header, got 88 from PyObject
# (and, on some images, an ``AttributeError: 'MessageFactory' object has no
# attribute 'GetPrototype'`` from a protobuf mismatch). None of this is our
# code — it is purely transformers probing for an unused backend.
#
# transformers decides whether to even look at TF/Flax from these env vars, read
# once at its first import. We force them off BEFORE any transformers import so
# ``is_tf_available()`` / ``is_flax_available()`` report False and the broken
# extensions are never touched. ``setdefault`` lets an operator still override.
# This runs in the package __init__ because every entry point imports ``emo``
# before ``transformers``.
import os as _os

_os.environ.setdefault("USE_TF", "0")
_os.environ.setdefault("USE_TENSORFLOW", "0")
_os.environ.setdefault("USE_FLAX", "0")
_os.environ.setdefault("USE_JAX", "0")
_os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

# Belt-and-braces: even with the env flags off, a stray ``import tensorflow``
# anywhere should fail fast and cleanly rather than dragging in the
# NumPy-ABI-mismatched compiled extension. Insert a None sentinel so
# ``importlib.util.find_spec("tensorflow")`` returns None and transformers'
# ``is_tf_available()`` reports it absent; an explicit import becomes a clean
# ImportError instead of the misleading dtype-size crash. Only do this if TF is
# not already imported, so we never break an environment that genuinely uses it.
for _backend in ("tensorflow", "flax", "jax"):
    if _backend not in _sys.modules:
        _sys.modules[_backend] = None  # type: ignore[assignment]

__all__ = ["config", "io_utils", "state"]
