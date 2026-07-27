"""Model loading and residual-stream access.

Wraps a HuggingFace causal LM and exposes:
  * deterministic generation (greedy) for steering/composition,
  * forced-choice option scoring from logits (FCEB),
  * residual-stream activation capture at given layers,
  * a handle on the per-layer residual module for steering hooks.

Layer naming differs per family, so the module path comes from the model
config (``residual_module`` template). Large models (27B/70B) are loaded with
``device_map="auto"`` for tensor parallelism across both A100s.
"""

from __future__ import annotations

import os
from typing import Iterable

import torch

# -- InternLM3 remote-code compatibility shim --------------------------------
# The InternLM3-8B-Instruct checkpoint ships a custom ``modeling_internlm3.py``
# (loaded via ``trust_remote_code=True``) that imports ``LossKwargs`` from
# ``transformers.utils``. This TypedDict was added to transformers' *internal*
# source tree but is NOT exported from ``transformers.utils.__init__`` in any
# released version (verified absent in 5.8.1 through 5.13.0). Without the
# symbol the remote modeling file fails to import with:
#   ImportError: cannot import name 'LossKwargs' from 'transformers.utils'
# ``LossKwargs`` is used only as a type annotation on ``**kwargs`` (via
# ``Unpack[KwargsForCausalLM]``) and has zero runtime effect, so defining a
# harmless empty TypedDict fallback when it is missing lets the remote code
# import and the model load. This must run before any ``from_pretrained`` call
# that triggers the dynamic-module import.
import transformers.utils as _tf_utils
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)

if not hasattr(_tf_utils, "LossKwargs"):
    from typing import TypedDict as _TypedDict

    class LossKwargs(_TypedDict, total=False):  # type: ignore[no-redef]
        """Fallback for transformers versions that don't export LossKwargs.

        Mirrors the upstream TypedDict (loss kwargs for CausalLM forward). It is
        only used in ``Unpack[...]`` type annotations, so an empty body is
        functionally equivalent at runtime.
        """

        pass

    _tf_utils.LossKwargs = LossKwargs  # type: ignore[attr-defined]

# InternLM3's remote modeling_internlm3.py calls
# ``DynamicCache.to_legacy_cache()`` (line 936) when ``return_legacy_cache`` is
# set — which happens whenever ``generate()`` passes ``past_key_values=None`` on
# the first forward. ``to_legacy_cache()`` was removed from ``DynamicCache`` in
# modern transformers (the legacy tuple cache format is gone). Without it the
# first generation step crashes with
#   AttributeError: 'DynamicCache' object has no attribute 'to_legacy_cache'
# Returning ``self`` keeps the cache as a proper ``Cache`` object, which is what
# modern ``generate()`` expects on subsequent steps — so the cache flows through
# unchanged instead of being downgraded to a legacy tuple and back.
from transformers.cache_utils import DynamicCache as _DynamicCache

# NOTE on all the shims below: earlier transformers versions genuinely lacked
# these legacy-cache methods (a plain ``hasattr`` check was enough to decide
# whether to add them). Newer transformers releases (5.x) have started
# re-adding some of these names as *deprecated* wrappers with a DIFFERENT,
# narrower signature than the legacy API the remote Phi-3.5 / InternLM3
# modeling files expect (e.g. ``get_usable_length(self, new_seq_length=1)``
# instead of the legacy ``get_usable_length(self, seq_length, layer_idx)``).
# Because the attribute already exists, a ``hasattr`` guard silently skips
# patching it and the remote code crashes deep inside the real implementation
# with a confusing ``TypeError`` (e.g. "_get_usable_length() takes from 1 to
# 2 positional arguments but 3 were given"). To be robust across transformers
# versions we therefore ALWAYS override these with permissive
# (``*args, **kwargs``-accepting) implementations instead of gating on
# ``hasattr`` — our version is a strict superset of what any remote modeling
# file needs from an unbounded ``DynamicCache``.

def _to_legacy_cache(self, *args, **kwargs):  # type: ignore[no-redef]
    return self

_DynamicCache.to_legacy_cache = _to_legacy_cache  # type: ignore[attr-defined]

# Phi-3.5 (microsoft/Phi-3.5-mini-instruct) ships a remote ``modeling_phi3.py``
# (loaded via ``trust_remote_code=True``) whose ``prepare_inputs_for_generation``
# reads ``past_key_values.seen_tokens`` (line ~1298). ``seen_tokens`` is the
# total number of cached positions across the batch; ``get_seq_length()``
# returns exactly that (the per-layer sequence length, which is uniform for a
# causal LM). Expose it as a read-only property so the remote code's direct
# attribute access works unchanged. This must run before any
# ``from_pretrained`` call that triggers the dynamic-module import.
def _seen_tokens(self) -> int:  # type: ignore[no-redef]
    return self.get_seq_length()

_DynamicCache.seen_tokens = property(_seen_tokens)  # type: ignore[attr-defined]

# Phi-3.5's remote ``modeling_phi3.py`` also calls
# ``past_key_values.get_max_length()`` (line ~1299) inside
# ``prepare_inputs_for_generation`` to decide whether the KV cache needs
# truncation for sliding-window attention. ``DynamicCache`` is unbounded, so
# returning ``None`` signals "no fixed max length", which makes the remote
# code's ``if max_cache_length is not None`` guard skip the truncation branch
# entirely — which is correct for an unbounded cache.
def _get_max_length(self, *args, **kwargs):  # type: ignore[no-redef]
    return None

_DynamicCache.get_max_length = _get_max_length  # type: ignore[attr-defined]

# Phi-3.5's remote ``modeling_phi3.py`` also calls
# ``past_key_values.get_usable_length(seq_length, layer_idx)`` — both in the
# model ``forward`` (line ~1063, for the RoPE position-ID offset) and in each
# attention layer's ``forward`` (line ~333, as ``kv_seq_len += ...``).
# ``get_usable_length`` was a legacy cache method that returned
# ``max_length - get_seq_length()`` (how many more tokens fit before hitting
# the configured max). For an unbounded ``DynamicCache`` the correct value is
# simply the current cached length, i.e. ``get_seq_length()``. Accept and
# ignore ANY positional/keyword arguments (``seq_length``, ``layer_idx``,
# or whatever a newer transformers' own narrower wrapper passes) so this
# override is compatible regardless of which legacy or semi-legacy call
# pattern the remote code (or transformers itself) uses.
def _get_usable_length(self, *args, **kwargs) -> int:  # type: ignore[no-redef]
    return self.get_seq_length()

_DynamicCache.get_usable_length = _get_usable_length  # type: ignore[attr-defined]

from .config import Config, ModelConfig

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def resolve_module(model: torch.nn.Module, dotted: str) -> torch.nn.Module:
    """Resolve a dotted module path like 'model.layers.5' to the submodule."""
    obj = model
    for part in dotted.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


class SteeringModel:
    """Inference-only wrapper around a causal LM with residual-stream access."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.mcfg: ModelConfig = cfg.model
        token = os.environ.get("HF_TOKEN") or None

        # When the weights live in a local directory (the GPU service uploads
        # each model to $INPUT_DATA_PATH and passes it via --model-name), there
        # is no network on the pod. Force HuggingFace into offline mode so the
        # tokenizer/config/model loads read the local files directly instead of
        # issuing HEAD requests to huggingface.co (which fail with
        # "Network is unreachable" and abort the run).
        if os.path.isdir(self.mcfg.hf_name):
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.mcfg.hf_name, trust_remote_code=True, token=token
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Decoder-only (causal) models must use left-padding: the pad tokens
        # go at the start of the sequence so the last real token — the position
        # whose logits we read (generation / FCEB option_logprobs) — is at a
        # fixed index. Right-padding shifts the last real token to different
        # positions per sequence, so batched generation reads the wrong logits
        # and transformers emits the "right-padding was detected" warning on
        # every batched call. This is a one-line fix but it silences the spam
        # the user saw in the run output and makes batched FCEB correct.
        self.tokenizer.padding_side = "left"

        dtype = _DTYPES.get(self.mcfg.dtype, torch.bfloat16)

        # Resolve the target torch device from the configured ``device_map``.
        # We no longer pass ``device_map`` to ``from_pretrained`` (that pulls in
        # ``accelerate``, which is absent on the air-gapped pod); instead we load
        # on CPU and ``.to(device)`` the whole model. ``device_map`` may be a
        # plain device string ("cuda:0"/"cpu") in the configs. Anything fancy
        # like "auto" (multi-GPU sharding) is not supported without accelerate,
        # so fall back to the first CUDA device.
        device = self._resolve_device(self.mcfg.device_map)
        self.input_device = device

        # Fail fast on an architecture mismatch BEFORE materializing weights.
        # `--model <id>` only supplies architecture metadata (num_layers /
        # hidden_size / residual_module); the weights come from `model.hf_name`
        # (on the GPU service, `--model-name $INPUT_DATA_PATH`).
        # ``from_pretrained`` reads the *checkpoint's own* config and ignores
        # those sizes, so if the uploaded checkpoint is a different / bigger
        # model than the one selected the loader silently tries to fit it in
        # host RAM and the process is OOM-killed (exit 120) during load. Compare
        # the configs first and abort with an actionable message instead.
        self._validate_checkpoint_arch(token)

        try:
            # Plain torch / transformers load: NO ``device_map`` and NO
            # ``low_cpu_mem_usage`` — both of those code paths hard-require
            # ``accelerate`` (transformers raises "Using a `device_map` ...
            # requires `accelerate`"), which is not installed on the air-gapped
            # GPU pod. ``from_pretrained`` materializes the weights on CPU; we
            # then move the whole model onto the target device below.
            from_pretrained_kwargs: dict = {
                "torch_dtype": dtype,
                "trust_remote_code": True,
                "token": token,
            }
            # ``attn_implementation`` forces a specific attention backend. The
            # default (letting the remote modeling file pick) breaks for
            # Phi-3.5 on transformers 5.x: its sliding-window path is
            # incompatible with the new ``DynamicCache`` lifecycle. ``eager``
            # uses the stock transformers attention, which handles it
            # correctly. Only passed when set (per-model opt-in).
            if self.mcfg.attn_implementation:
                from_pretrained_kwargs["attn_implementation"] = (
                    self.mcfg.attn_implementation
                )
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.mcfg.hf_name,
                    **from_pretrained_kwargs,
                )
            except ValueError as _ve:
                # Some multimodal checkpoints (e.g. Mistral-Small-3.1 /
                # Ministral-3 with model_type "mistral3") use a config class
                # (Mistral3Config) that maps to a *ForConditionalGeneration
                # model, NOT a *ForCausalLM. On transformers versions that
                # don't register that config with AutoModelForCausalLM,
                # from_pretrained raises:
                #   ValueError: Unrecognized configuration class <...Mistral3Config>
                #   for this kind of AutoModel: AutoModelForCausalLM.
                # AutoModelForImageTextToText resolves the config to the
                # correct *generative* multimodal class (e.g.
                # Mistral3ForConditionalGeneration), which exposes .generate(),
                # .get_input_embeddings(), .get_output_embeddings(), .lm_head
                # and a .language_model text decoder — so the text-only steering
                # pipeline works unchanged (the residual_module template already
                # points at model.language_model.model.layers.{i}). NB: plain
                # AutoModel would load the base Mistral3Model (no lm_head /
                # .generate()), which is useless for this pipeline. The import
                # is lazy so this module still loads on transformers versions
                # that predate AutoModelForImageTextToText.
                if "Unrecognized configuration class" in str(_ve) and (
                    "AutoModelForCausalLM" in str(_ve)
                ):
                    try:
                        from transformers import (
                            AutoModelForImageTextToText as _AutoImgText,
                        )
                    except ImportError as _ie:
                        raise RuntimeError(
                            "Checkpoint is a multimodal/conditional model "
                            "(e.g. Mistral3) that AutoModelForCausalLM does not "
                            "support, and this transformers version has no "
                            "AutoModelForImageTextToText to fall back to. "
                            f"Upgrade transformers (import error: {_ie})."
                        ) from _ve
                    print(
                        "[models] AutoModelForCausalLM rejected the config "
                        f"({type(_ve).__name__}); falling back to "
                        "AutoModelForImageTextToText for multimodal/conditional "
                        "checkpoint."
                    )
                    self.model = _AutoImgText.from_pretrained(
                        self.mcfg.hf_name,
                        **from_pretrained_kwargs,
                    )
                else:
                    raise
        except (ModuleNotFoundError, ImportError) as e:
            # transformers' lazy importer masks the *real* failure behind a
            # generic "Could not import module 'Gemma2ForCausalLM'. Are this
            # object's requirements defined correctly?". That almost always
            # means the installed transformers is too old for this checkpoint's
            # architecture (e.g. Gemma2 needs transformers>=4.42) or its
            # modeling file failed to import because of a broken / mismatched
            # dependency. Re-import the concrete module to surface the genuine
            # cause and give an actionable message.
            self._raise_with_real_import_error(e)

        # Move to the resolved device with plain torch (no accelerate).
        self.model.to(device)
        self.model.eval()
        # Keep ``input_device`` in sync with where the weights actually landed.
        try:
            self.input_device = self.model.get_input_embeddings().weight.device
        except Exception:
            self.input_device = device

    # -- device resolution -------------------------------------------------

    @staticmethod
    def _resolve_device(device_map: str) -> torch.device:
        """Resolve the configured ``device_map`` string to a single torch
        device for a plain (accelerate-free) ``.to(device)`` load.

        Multi-device sharding ("auto"/"balanced") needs accelerate, which is
        unavailable here, so those collapse to the first CUDA device (or CPU
        when CUDA is absent). A plain "cuda:0"/"cpu" string is used as-is.
        """
        dm = (device_map or "").strip().lower()
        if dm in ("", "auto", "balanced", "balanced_low_0", "sequential"):
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if dm.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(device_map)

    # -- import-error diagnostics ------------------------------------------

    def _raise_with_real_import_error(self, masked: Exception):
        """Surface the genuine cause behind transformers' generic lazy-import
        error for a model class like ``Gemma2ForCausalLM``.

        ``AutoModelForCausalLM`` resolves the class lazily; when the concrete
        modeling module can't be imported, transformers re-wraps *any* failure
        as ``Could not import module 'Gemma2ForCausalLM'. Are this object's
        requirements defined correctly?`` which hides the real reason (almost
        always a transformers version too old for the architecture, or a broken
        dependency). Import the concrete module directly to capture the true
        traceback, then raise an actionable error.
        """
        import transformers

        # Derive the architecture from the checkpoint config (e.g. "gemma2").
        model_type = None
        try:
            ckpt = AutoConfig.from_pretrained(self.mcfg.hf_name, trust_remote_code=True)
            model_type = getattr(ckpt, "model_type", None)
        except Exception:
            pass

        real_error = masked
        if model_type:
            try:
                # Force the real modeling import so its true exception surfaces.
                import importlib

                importlib.import_module(
                    f"transformers.models.{model_type}.modeling_{model_type}"
                )
            except Exception as real:  # noqa: BLE001 - we want the real cause
                real_error = real

        # Classify the real cause so the advice is actionable instead of always
        # blaming the transformers version. The two failure modes look identical
        # at the top level (a masked lazy-import error) but need opposite fixes.
        err_text = f"{type(real_error).__name__}: {real_error}".lower()
        is_torchvision_break = (
            "torchvision" in err_text
            or ("circular import" in err_text and "extension" in err_text)
        )

        header = (
            f"Failed to load the model architecture '{model_type or '?'}' "
            f"from '{self.mcfg.hf_name}' with transformers "
            f"{transformers.__version__}.\n"
            f"Real import error: {type(real_error).__name__}: {real_error}\n"
        )

        if is_torchvision_break:
            # torch/torchvision ABI mismatch: torchvision's compiled _C
            # extension was built against a different torch than the one
            # installed, so `import torchvision` half-initializes and the
            # modeling file's transitive import of it explodes. transformers
            # itself is fine (this checkpoint loaded fine on a matching image).
            # This text-only pipeline never needs torchvision at all.
            import torch as _torch

            tv_version = "?"
            try:
                import torchvision as _tv  # noqa: F401

                tv_version = getattr(_tv, "__version__", "?")
            except Exception:
                pass
            raise RuntimeError(
                header
                + "This is a broken torchvision in the offline pod, NOT a "
                "transformers-version problem (transformers "
                f"{transformers.__version__} already supports "
                f"'{model_type or 'this'}').\n"
                f"  torch={_torch.__version__}  torchvision={tv_version}\n"
                "A 'partially initialized module torchvision has no attribute "
                "extension' error means torchvision's compiled C++ extension "
                "(_C) was built against a different torch than the one "
                "installed (an ABI/version mismatch in the GPU image).\n"
                "This text-only pipeline does not use torchvision; fix the "
                "image with EITHER of:\n"
                "  * uninstall it so transformers never imports it:\n"
                "        pip uninstall -y torchvision\n"
                "  * or reinstall a torchvision that matches torch (same CUDA "
                "build), e.g. from the matching pytorch wheel index.\n"
                "Do NOT upgrade transformers — that will not fix this."
            ) from real_error

        raise RuntimeError(
            header
            + "This is usually an environment problem, not a code problem:\n"
            "  * the installed transformers is too old for this checkpoint's "
            "architecture (Gemma2 needs transformers>=4.42; pin a recent "
            "version in requirements.txt), or\n"
            "  * a dependency that the modeling file imports is broken / "
            "version-mismatched in the offline pod.\n"
            "Fix by upgrading/pinning transformers in the GPU image and "
            "reinstalling, then rerun."
        ) from real_error

    # -- architecture sanity check -----------------------------------------

    def _validate_checkpoint_arch(self, token: str | None) -> None:
        """Compare the checkpoint's config against the selected ``--model``.

        Loading only the small ``config.json`` is cheap (no weights), so we can
        detect a wrong / oversized checkpoint before the heavy
        ``from_pretrained`` call blows up host RAM. A mismatch in layer count or
        hidden size means the stages (which index ``residual_module.{i}`` up to
        ``num_layers`` and assume ``hidden_size``) would also be wrong, so we
        refuse rather than silently OOM or produce garbage vectors.
        """
        try:
            ckpt = AutoConfig.from_pretrained(
                self.mcfg.hf_name, trust_remote_code=True, token=token
            )
        except Exception as e:
            # Can't introspect (custom remote code, odd repo) — don't block the
            # run, just warn. The real load below will surface any hard error.
            print(f"[models] could not pre-read checkpoint config: {e}")
            return

        # Multimodal checkpoints (e.g. Mistral3ForConditionalGeneration) keep
        # the text-decoder dims on a nested ``text_config``; the top-level
        # config has the vision-encoder dims (or none). Prefer the text
        # config when present so the sanity check compares the right dims.
        text_cfg = getattr(ckpt, "text_config", None)
        ckpt_layers = (
            getattr(ckpt, "num_hidden_layers", None)
            if text_cfg is None
            else getattr(text_cfg, "num_hidden_layers", None)
        ) or getattr(ckpt, "num_hidden_layers", None)
        ckpt_hidden = (
            getattr(ckpt, "hidden_size", None)
            if text_cfg is None
            else getattr(text_cfg, "hidden_size", None)
        ) or getattr(ckpt, "hidden_size", None)
        exp_layers = self.mcfg.num_layers
        exp_hidden = self.mcfg.hidden_size

        mismatches = []
        if ckpt_layers is not None and exp_layers and ckpt_layers != exp_layers:
            mismatches.append(
                f"num_layers: config={exp_layers} checkpoint={ckpt_layers}"
            )
        if ckpt_hidden is not None and exp_hidden and ckpt_hidden != exp_hidden:
            mismatches.append(
                f"hidden_size: config={exp_hidden} checkpoint={ckpt_hidden}"
            )

        if mismatches:
            raise RuntimeError(
                "Checkpoint architecture does not match the selected --model "
                f"'{self.mcfg.id}'.\n  weights : {self.mcfg.hf_name}\n  "
                + "\n  ".join(mismatches)
                + "\nThe weights at this path are a different model than the one "
                "selected. `--model <id>` picks the architecture config while "
                "`--model-name`/$INPUT_DATA_PATH points at the weights; they must "
                "be the same model. Loading the larger checkpoint into host RAM "
                "is what causes the exit-120 OOM during load. Fix by uploading "
                "the matching checkpoint, or select the --model id whose config "
                "(num_layers/hidden_size) matches the uploaded weights."
            )

    # -- prompt formatting --------------------------------------------------

    def format_prompt(
        self,
        user_text: str,
        system: str | None = None,
        template_mode: str | None = None,
    ) -> str:
        """Wrap ``user_text`` for generation.

        ``template_mode`` is an evaluation-time override (§5.3 / §4.4) that does
        NOT mutate the architecture's ``uses_chat_template`` flag:

          * ``None`` (default) -> the architecture default (chat template iff
            ``uses_chat_template`` and the tokenizer defines one);
          * ``"chat"``         -> force the full chat template (system + user);
          * ``"minimal_chat"`` -> a single user turn, never a system message;
          * ``"raw"``          -> no template at all (plain completion prompt).

        Only the *wrapping* changes; the prompt text is identical across
        emotions, so this stays a clean, emotion-blind ablation.
        """
        mode = (template_mode or "").strip().lower()
        if mode == "raw":
            return user_text
        # Build the kwargs for apply_chat_template. Qwen3 models support an
        # ``enable_thinking`` parameter that toggles chain-of-thought reasoning.
        # When the model config sets ``enable_thinking: false``, we pass it
        # through so the model answers directly (no <think> block).
        template_kwargs: dict = {"tokenize": False, "add_generation_prompt": True}
        if self.mcfg.enable_thinking is not None:
            template_kwargs["enable_thinking"] = self.mcfg.enable_thinking
        if mode == "minimal_chat":
            if self.tokenizer.chat_template:
                return self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_text}],
                    **template_kwargs,
                )
            return user_text
        force_chat = mode == "chat"
        if (force_chat or self.mcfg.uses_chat_template) and self.tokenizer.chat_template:
            # Some chat templates (notably Gemma2) do not support a "system"
            # role and raise ``TemplateError: System role not supported`` when
            # one is passed. Detect that up front and, when unsupported, fold
            # the system content into the user turn (the standard Gemma
            # workaround) instead of crashing.
            use_system = bool(system)
            if use_system:
                try:
                    self.tokenizer.apply_chat_template(
                        [{"role": "system", "content": ""}],
                        tokenize=False, add_generation_prompt=False,
                    )
                except Exception:
                    use_system = False
            msgs = []
            if use_system:
                msgs.append({"role": "system", "content": system})
                msgs.append({"role": "user", "content": user_text})
            else:
                user_body = (
                    f"{system}\n\n{user_text}" if system else user_text
                )
                msgs.append({"role": "user", "content": user_body})
            return self.tokenizer.apply_chat_template(msgs, **template_kwargs)
        # Base (non-chat) model: plain text.
        return user_text

    # -- residual module access --------------------------------------------

    # Cached *resolved* residual-module template. The configured template
    # (e.g. "model.language_model.model.layers.{i}") may not match the actual
    # loaded checkpoint's attribute layout — multimodal Mistral3/Ministral3
    # variants differ in whether the text decoder nests under an extra
    # ``.model`` (Mistral3ForConditionalGeneration.language_model can be a
    # Ministral3TextModel with .model.layers OR a Ministral3Model with
    # .layers directly). On first access we probe a few variants and cache
    # the one that resolves, so subsequent layer lookups are O(1).
    _residual_template_resolved: str | None = None

    def _candidate_residual_templates(self) -> list[str]:
        """Yield the configured template plus structural variants to probe."""
        tmpl = self.mcfg.residual_module_for_layer(0)  # e.g. "...layers.0"
        variants = [tmpl]
        # Drop a redundant ".model" segment before ".layers" — some
        # multimodal checkpoints expose layers directly under the text
        # decoder (Ministral3Model.layers) rather than nested
        # (Ministral3TextModel.model.layers).
        if ".model.layers." in tmpl:
            variants.append(tmpl.replace(".model.layers.", ".layers."))
        # And the reverse: insert ".model" if the template goes straight to
        # ".layers" but the object actually nests under ".model".
        if ".layers." in tmpl and ".model.layers." not in tmpl:
            variants.append(tmpl.replace(".layers.", ".model.layers.", 1))
        # De-duplicate while preserving order.
        seen = set()
        out = []
        for v in variants:
            if v not in seen:
                seen.add(v)
                out.append(v)
        return out

    @staticmethod
    def _template_has_layer_index(tmpl: str) -> bool:
        """True if ``tmpl`` carries the ``{i}`` per-layer placeholder.

        A template *without* ``{i}``` resolves to the same module for every
        layer, which would silently collapse all per-layer hooks onto one
        module (the ministral_3_8b bug: identical steering vectors across
        all layers). Such templates must never be cached as the residual
        path.
        """
        return "{i}" in tmpl

    def _probe_residual_template(self, layer: int) -> tuple[str, torch.nn.Module] | None:
        """Find a candidate template that resolves to a *per-layer* module.

        Probes each candidate path at ``layer`` and, to guard against paths
        that resolve to a single shared module regardless of the index,
        verifies that resolving at a *different* layer yields a *different*
        module. Returns ``(template, module)`` for the first candidate that
        passes both checks, or ``None``.
        """
        # Pick a second, distinct layer index to confirm the path actually
        # varies per layer. Layer 0 is always valid for decoder blocks; if
        # ``layer`` is itself 0, use 1. We don't rely on num_layers here —
        # any valid distinct index proves the path indexes a per-layer
        # container (and avoids the last-layer +1 out-of-range edge case).
        probe_other = 0 if layer != 0 else 1
        for cand in self._candidate_residual_templates():
            if not self._template_has_layer_index(cand):
                # A template without {i} cannot address individual layers —
                # skip it so we never cache a layer-agnostic path.
                continue
            try:
                mod = resolve_module(self.model, cand.format(i=layer))
            except (AttributeError, KeyError, IndexError, TypeError):
                continue
            # Verify the path actually indexes into a per-layer container:
            # resolving at a different layer must give a different module.
            try:
                mod_other = resolve_module(self.model, cand.format(i=probe_other))
            except (AttributeError, KeyError, IndexError, TypeError):
                # Only one layer index is resolvable — not a real per-layer
                # container. Skip this candidate.
                continue
            if mod_other is mod:
                # Same object for two different layer indices → this path
                # does not address individual decoder blocks (e.g. it points
                # at a shared wrapper/norm). Skip it.
                continue
            return cand, mod
        return None

    def residual_module(self, layer: int) -> torch.nn.Module:
        tmpl = self._residual_template_resolved
        if tmpl is None:
            probed = self._probe_residual_template(layer)
            if probed is not None:
                cand, mod = probed
                self._residual_template_resolved = cand
                return mod
            # None matched — fall back to the configured path so the original
            # error surfaces (actionable, points at the config template).
            tmpl = self.mcfg.residual_module_for_layer(layer)
        return resolve_module(self.model, tmpl.format(i=layer) if "{i}" in tmpl else tmpl)

    def _transformer_trunk(self) -> torch.nn.Module | None:
        """The decoder-only transformer trunk (everything *before* the LM head).

        Activation-capture methods (:meth:`collect_residuals`,
        :meth:`collect_token_residuals`) only need the residual-stream hidden
        states, which are produced by the trunk. Running the full
        ``AutoModelForCausalLM`` forward also materializes the ``lm_head``
        logits tensor ``[B, T, vocab]`` — for large-vocab models (Gemma3:
        ~262k) over long batched sequences this allocates tens of GiB and OOMs
        even though the logits are never read. Calling the trunk directly
        skips ``lm_head`` entirely and avoids the allocation.

        The trunk is resolved generically from the configured
        ``residual_module`` path by stripping the trailing ``.layers.{i}``
        component, so this works for every supported architecture
        (``model.layers`` for Llama/Qwen/Gemma2/Mistral/Phi,
        ``model.language_model.layers`` for Gemma3-conditional,
        ``gpt_neox.layers`` for Pythia). Returns ``None`` if the trunk cannot
        be resolved, in which case callers fall back to the full forward.
        """
        # Use the *resolved* residual-module template (which auto-detects the
        # correct attribute layout for the loaded checkpoint, e.g.
        # "model.language_model.layers.{i}" vs the configured
        # "model.language_model.model.layers.{i}") so the derived trunk path
        # matches the actual structure. residual_module(0) caches the working
        # template on first call.
        try:
            self.residual_module(0)
        except Exception:
            pass
        path = self._residual_template_resolved or self.mcfg.residual_module_for_layer(0)
        parts = path.split(".")
        # Drop the "{i}" placeholder and the container name ("layers").
        if len(parts) >= 2:
            trunk_path = ".".join(parts[:-2])
        else:
            return None
        try:
            return resolve_module(self.model, trunk_path)
        except Exception:
            return None

    def _forward_hidden_only(self, enc: dict) -> None:
        """Run the trunk to populate activation hooks, skipping ``lm_head``.

        Activation-capture methods (:meth:`collect_residuals`,
        :meth:`collect_token_residuals`) only need the residual-stream hidden
        states, which the forward hooks on the residual modules capture. The
        ``lm_head`` projection to ``[B, T, vocab]`` logits is pure waste here
        and OOMs on large-vocab models (Gemma3: ~262k → tens of GiB). Calling
        the transformer trunk directly (``model.model`` / ``model.model.language_model``
        / ``model.transformer``) skips ``lm_head`` entirely.

        Falls back to the full ``self.model(**enc)`` forward when the trunk
        cannot be resolved or errors (e.g. an exotic architecture whose trunk
        forward signature differs). The full forward is correct but allocates
        the logits tensor and may OOM on large-vocab models — the trunk path is
        the preferred one.
        """
        trunk = self._transformer_trunk()
        if trunk is not None:
            try:
                # HF decoder trunks accept the same input dict (input_ids,
                # attention_mask, and any position_ids / token_type_ids the
                # tokenizer produced) and return (last_hidden_state, ...) — we
                # ignore the return; the forward hooks capture what we need.
                trunk(**enc)
                return
            except Exception as e:
                # Signature mismatch or trunk-specific requirement: fall back
                # to the full forward so we never silently produce wrong
                # activations. Warn once-ish (no rate limiter needed; this is a
                # per-call hot path only when something is genuinely wrong).
                print(f"  [models] trunk-only forward failed ({e!r}); "
                      f"falling back to full forward (may OOM on large vocab).")
        self.model(**enc)

    # -- generation ---------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int = 128,
        system: str | None = None,
        greedy: bool = True,
        template_mode: str | None = None,
        do_sample: bool | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        num_return_sequences: int = 1,
        bad_words_ids: "list[list[int]] | None" = None,
        seed: int | None = None,
    ) -> list[str]:
        """Batched generation; returns only the continuation.

        Greedy by default (``greedy=True``) for reproducible steering/composition
        measurement. The extra knobs power the §5.3-5.5 generation ablations and
        are emotion-blind (identical for every emotion):

          * ``template_mode``       -> §5.3 wrapping override (chat/raw/minimal),
          * ``do_sample`` / ``temperature`` / ``top_p`` / ``repetition_penalty``
            / ``num_return_sequences`` -> §5.4 decoding parameters,
          * ``bad_words_ids``       -> §5.5 refusal-phrase suppression,
          * ``seed``                -> §5.8 fixed seed for fair sampling.

        ``do_sample`` (when given) takes precedence over ``greedy``. With
        ``num_return_sequences > 1`` the returned list is flattened
        prompt-major: [p0s0, p0s1, ..., p1s0, ...].
        """
        if seed is not None:
            torch.manual_seed(int(seed))
        sample = (not greedy) if do_sample is None else bool(do_sample)
        texts = [
            self.format_prompt(p, system=system, template_mode=template_mode)
            for p in prompts
        ]

        gen_kwargs: dict = {
            "max_new_tokens": max_new_tokens,
            "do_sample": sample,
            "num_beams": 1,
            "num_return_sequences": int(num_return_sequences),
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if sample:
            if temperature is not None:
                gen_kwargs["temperature"] = float(temperature)
            if top_p is not None:
                gen_kwargs["top_p"] = float(top_p)
        if repetition_penalty is not None and repetition_penalty != 1.0:
            gen_kwargs["repetition_penalty"] = float(repetition_penalty)
        if bad_words_ids:
            gen_kwargs["bad_words_ids"] = bad_words_ids

        # -- Batched generation ------------------------------------------------
        # Process prompts in chunks of ``gen_batch_size`` to keep the KV cache
        # memory bounded. Without this, passing 1000 prompts (50 scenarios ×
        # 4 games × 5 repeats) in a single forward pass causes CUDA OOM on
        # an A100 80GB. The chunk size is configurable via
        # ``ModelConfig.gen_batch_size`` (default 32).
        batch_size = max(1, getattr(self.mcfg, "gen_batch_size", 32))
        all_decoded: list[str] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            enc = self.tokenizer(
                chunk, return_tensors="pt", padding=True, truncation=True
            ).to(self.input_device)
            out = self.model.generate(**enc, **gen_kwargs)
            gen = out[:, enc["input_ids"].shape[1] :]
            all_decoded.extend(
                self.tokenizer.batch_decode(gen, skip_special_tokens=True)
            )
            # Free the intermediate tensors so the next chunk has full VRAM.
            del enc, out, gen
            torch.cuda.empty_cache()
        return all_decoded

    def generate_with_config(
        self,
        prompts: list[str],
        gen_cfg: dict,
        template_mode: str | None = None,
        decode_override: dict | None = None,
        seed: int | None = None,
        suppress_refusal: bool | None = None,
    ) -> list[str]:
        """Generate using the ``generation`` config block (§5.3-5.5 helper).

        ``gen_cfg`` is ``cfg.generation``. ``decode_override`` is one entry from
        ``generation.decoding_grid`` (e.g. ``{"do_sample": false}``) merged over
        the block defaults. ``template_mode``/``seed`` override the block values.
        ``suppress_refusal`` (or ``gen_cfg['suppress_refusal_phrases']``) blocks
        the refusal phrases as bad words (§5.5, diagnostic only).
        """
        d = dict(gen_cfg or {})
        if decode_override:
            d.update(decode_override)
        tmode = template_mode if template_mode is not None else d.get("template_mode")
        do_supp = (
            suppress_refusal
            if suppress_refusal is not None
            else bool(d.get("suppress_refusal_phrases", False))
        )
        bad_words = (
            self.bad_words_ids(d.get("refusal_phrases", [])) if do_supp else None
        )
        # System prompt from generation config: an emotion-blind preamble that
        # bypasses the RLHF safety boilerplate ("As an AI..."). Identical across
        # emotions so it does not confound the steering signal.
        sys_prompt = d.get("system_prompt") or None
        return self.generate(
            prompts,
            max_new_tokens=int(d.get("max_new_tokens", 48)),
            system=sys_prompt,
            greedy=not bool(d.get("do_sample", False)),
            template_mode=tmode,
            do_sample=d.get("do_sample"),
            temperature=d.get("temperature"),
            top_p=d.get("top_p"),
            repetition_penalty=d.get("repetition_penalty"),
            num_return_sequences=int(d.get("num_return_sequences", 1)),
            bad_words_ids=bad_words,
            seed=seed,
        )

    def bad_words_ids(self, phrases: list[str]) -> list[list[int]]:
        """Tokenize refusal phrases into ``bad_words_ids`` for ``generate`` (§5.5).

        Each phrase is encoded both bare and space-prefixed (the leading token
        differs after a space), de-duplicated, so the suppression catches the
        phrase wherever it starts a continuation.
        """
        out: list[list[int]] = []
        seen: set[tuple] = set()
        for phrase in phrases or []:
            for variant in (phrase, f" {phrase}"):
                ids = self.tokenizer.encode(variant, add_special_tokens=False)
                if ids and tuple(ids) not in seen:
                    seen.add(tuple(ids))
                    out.append(ids)
        return out

    # -- forced-choice option scoring (FCEB) --------------------------------

    @torch.no_grad()
    def option_logprobs(
        self,
        prompt: str,
        option_token_ids: "list[int] | list[list[int]]",
        system: str | None = None,
    ) -> list[float]:
        """Log-probabilities of each option's first token as the next token.

        Each option may be given either as a single token id or as a *list* of
        candidate token ids; in the latter case the option's logprob is the
        log-sum-exp over its candidates. This is what makes FCEB robust to the
        fact that a letter like ``A`` tokenizes differently depending on the
        preceding character: after ``"Answer:"`` the model actually emits the
        space-prefixed token ``" A"``, whose id differs from the bare ``"A"``.
        Scoring only the bare-letter id reads a token the model rarely emits at
        that position, yielding the tiny, near-noise option masses behind the
        microscopic FCEB lift. Aggregating over both variants fixes that.
        """
        text = self.format_prompt(prompt, system=system)
        enc = self.tokenizer(text, return_tensors="pt").to(self.input_device)
        logits = self.model(**enc).logits[0, -1, :]  # [vocab]
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        out = []
        for opt in option_token_ids:
            ids = [opt] if isinstance(opt, int) else list(opt)
            if not ids:
                out.append(float("-inf"))
                continue
            lp = torch.logsumexp(
                torch.stack([logprobs[i] for i in ids]), dim=0
            )
            out.append(float(lp.item()))
        return out

    def first_token_id(self, text: str) -> int:
        """Token id of the first token of ``text`` (letter option encoding)."""
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        return ids[0]

    def option_token_variants(self, letter: str) -> list[int]:
        """Candidate first-token ids for an option ``letter``.

        Returns the de-duplicated first-token ids for the bare letter and its
        space-prefixed form (e.g. ``"A"`` and ``" A"``). After ``"Answer:"``
        the model emits the space-prefixed token, so both must be considered.
        """
        ids: list[int] = []
        for variant in (letter, f" {letter}"):
            enc = self.tokenizer.encode(variant, add_special_tokens=False)
            if enc and enc[0] not in ids:
                ids.append(enc[0])
        return ids

    # -- activation capture -------------------------------------------------

    @torch.no_grad()
    def collect_residuals(
        self,
        prompts: list[str],
        layers: Iterable[int],
        system: str | None = None,
        pool: str = "last",
    ) -> dict[int, torch.Tensor]:
        """Pooled residual-stream activation per layer for each prompt.

        Returns {layer: tensor[len(prompts), hidden]} on CPU (fp32).

        ``pool`` controls how token positions are aggregated:

          * ``"last"`` (default) — take the **last non-pad token** only. The
            short pos/neg pair texts are wrapped in a chat template, and
            mean-pooling over the whole wrapped sequence dilutes the emotional
            signal with shared template tokens, inflating a single dominant
            "emotionality" axis (the PC1 / cos≈0.96 collapse behind the
            "sadness attractor"). The last position is the model's running
            summary of the input and isolates the emotion direction far better.
          * ``"mean"`` — legacy mean over all non-pad positions.
        """
        layers = list(layers)
        if len(layers) < 2:
            # Single layer — nothing to cross-check; fall through to capture.
            pass

        def _register_hooks() -> list:
            captured: dict[int, torch.Tensor] = {}

            def make_hook(layer_idx):
                def hook(_module, _inp, out):
                    hs = out[0] if isinstance(out, tuple) else out
                    captured[layer_idx] = hs.detach()

                return hook

            handles = []
            for l in layers:
                handles.append(self.residual_module(l).register_forward_hook(make_hook(l)))
            return handles, captured

        def _pool(captured, attn):
            mask = attn.unsqueeze(-1).float()  # [B,T,1]
            B, T = attn.shape
            pos = torch.arange(T, device=attn.device).unsqueeze(0).expand(B, T)
            last_idx = (pos * attn).argmax(dim=1)  # [B]
            denom = mask.sum(dim=1).clamp(min=1.0)
            pooled = {}
            for l in layers:
                hs = captured[l].float().to(mask.device)
                if pool == "mean":
                    pooled[l] = ((hs * mask).sum(dim=1) / denom).cpu()
                else:  # "last"
                    pooled[l] = hs[torch.arange(B), last_idx].cpu()
            return pooled

        def _captured_all_identical(captured: dict[int, torch.Tensor]) -> bool:
            """True if every layer captured the same tensor (the
            ministral_3_8b failure mode: all per-layer hooks fire on one
            shared module, or the trunk forward doesn't traverse the real
            per-layer decoder blocks, so every layer gets the same ``hs``).
            """
            if len(captured) < 2:
                return False
            vals = list(captured.values())
            ref = vals[0]
            return all(v is ref or torch.equal(v, ref) for v in vals[1:])

        texts = [self.format_prompt(p, system=system) for p in prompts]
        enc = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.input_device)
        attn = enc["attention_mask"]  # [B,T]

        handles, captured = _register_hooks()
        try:
            # Run the transformer trunk only — we read residual-stream
            # activations via hooks and never need the lm_head logits, whose
            # [B, T, vocab] tensor OOMs on large-vocab models (e.g. Gemma3).
            self._forward_hidden_only(enc)
            # Guard against the multimodal-trunk failure mode where every
            # per-layer hook ends up capturing the *same* tensor (e.g. the
            # resolved trunk doesn't actually traverse the real per-layer
            # decoder blocks, so all hooks fire on one shared module). If that
            # happens, re-run via the full self.model(**enc) forward — the same
            # path generate() uses and which demonstrably runs the decoder —
            # so the hooks capture genuinely per-layer activations.
            if len(layers) >= 2 and _captured_all_identical(captured):
                print("  [models] collect_residuals: per-layer captures are "
                      "identical — trunk forward did not traverse per-layer "
                      "decoder blocks; re-running via full self.model forward.")
                for h in handles:
                    h.remove()
                handles, captured = _register_hooks()
                self.model(**enc)
                if _captured_all_identical(captured):
                    raise RuntimeError(
                        "collect_residuals: per-layer residual captures are "
                        "identical even after a full model forward — the "
                        "residual_module path does not address individual "
                        f"decoder layers (resolved template = "
                        f"{self._residual_template_resolved!r}). Check the "
                        "model config 'residual_module' for this architecture."
                    )
            return _pool(captured, attn)
        finally:
            for h in handles:
                h.remove()

    @torch.no_grad()
    def collect_token_residuals(
        self,
        prompts: list[str],
        layer: int,
        max_tokens: int = 200000,
        batch_size: int = 8,
        system: str | None = None,
    ) -> torch.Tensor:
        """Per-token residual-stream activations at one layer (N4 SAE corpus).

        Unlike :meth:`collect_residuals` (which pools to one vector per text),
        this returns *every* non-pad token's residual — the dense activation
        matrix a self-trained SAE needs. Texts are processed in batches and the
        running token count is capped at ``max_tokens`` (the SAE corpus budget).

        Returns a CPU fp32 tensor ``[N_tokens, hidden]``.
        """
        captured: list[torch.Tensor] = []
        n_so_far = 0

        def make_hook(_module, _inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            captured.append(hs.detach())

        handle = self.residual_module(layer).register_forward_hook(make_hook)
        try:
            for i in range(0, len(prompts), batch_size):
                if n_so_far >= max_tokens:
                    break
                batch = prompts[i : i + batch_size]
                texts = [self.format_prompt(p, system=system) for p in batch]
                enc = self.tokenizer(
                    texts, return_tensors="pt", padding=True, truncation=True
                ).to(self.input_device)
                # Trunk-only forward: skip lm_head logits (OOM on large vocab).
                self._forward_hidden_only(enc)
                attn = enc["attention_mask"]  # [B,T]
                mask = attn.unsqueeze(-1).float()  # [B,T,1]
                hs = captured[-1].float().to(mask.device)  # [B,T,hidden]
                tok = (hs * mask).reshape(-1, hs.shape[-1])  # keep non-pad only
                captured[-1] = tok.cpu()
                n_so_far += int(tok.shape[0])
            if not captured:
                # Multimodal checkpoints (e.g. Mistral3ForConditionalGeneration)
                # keep the text-decoder hidden size on ``config.text_config``
                # rather than the top-level config; fall back to the model
                # config's declared ``hidden_size`` if neither is present.
                _cfg = self.model.config
                _h = (
                    getattr(_cfg, "hidden_size", None)
                    or getattr(getattr(_cfg, "text_config", None), "hidden_size", None)
                    or self.mcfg.hidden_size
                )
                return torch.empty((0, _h), dtype=torch.float32)
            acts = torch.cat(captured, dim=0)[:max_tokens]
            return acts.float().cpu()
        finally:
            handle.remove()

    # -- token-set utilities (logit-lens / LM-head bias) -------------------

    def token_set_ids(self, words: list[str]) -> list[int]:
        """First-token ids for a word set (bare + space-prefixed), de-duped.

        Used by the logit-lens (§4.1) and LM-head-bias (§4.3) diagnostics to map
        an emotion/refusal word list to the vocabulary ids whose next-token mass
        is read off the (projected) logits.
        """
        ids: list[int] = []
        for w in words or []:
            for variant in (w, f" {w}"):
                enc = self.tokenizer.encode(variant, add_special_tokens=False)
                if enc and enc[0] not in ids:
                    ids.append(enc[0])
        return ids

    @torch.no_grad()
    def next_token_logprobs(
        self, prompt: str, system: str | None = None, template_mode: str | None = None
    ) -> torch.Tensor:
        """Full next-token log-prob vector [vocab] after ``prompt`` (fp32, CPU).

        Call inside a steering context to read the steered LM-head distribution
        (§4.3). The caller aggregates token sets via :meth:`token_set_logprob`.
        """
        text = self.format_prompt(prompt, system=system, template_mode=template_mode)
        enc = self.tokenizer(text, return_tensors="pt").to(self.input_device)
        logits = self.model(**enc).logits[0, -1, :]
        return torch.log_softmax(logits.float(), dim=-1).cpu()

    @staticmethod
    def token_set_logprob(logprobs: torch.Tensor, ids: list[int]) -> float:
        """log-sum-exp mass of a token-set id list under a logprob vector."""
        if not ids:
            return float("-inf")
        return float(torch.logsumexp(torch.stack([logprobs[i] for i in ids]), dim=0))

    @torch.no_grad()
    def logit_lens_readout(
        self,
        prompt: str,
        layers: Iterable[int],
        token_sets: dict[str, list[int]],
        system: str | None = None,
        template_mode: str | None = None,
    ) -> dict[int, dict[str, float]]:
        """§4.1 logit-lens: per-layer token-set logprobs via the unembedding.

        Captures the residual stream at each requested layer's last token,
        applies the model's final norm (when present) and the LM head, and reads
        the log-prob mass of each ``token_sets`` entry. Call inside a steering
        context to test whether the target emotion is decodable from the
        mid-stack residual stream after steering even when the final generation
        collapses to the attractor. Returns ``{layer: {set_name: logprob}}``.
        """
        layers = list(layers)
        captured: dict[int, torch.Tensor] = {}
        handles = []

        def make_hook(layer_idx):
            def hook(_module, _inp, out):
                hs = out[0] if isinstance(out, tuple) else out
                captured[layer_idx] = hs.detach()

            return hook

        for l in layers:
            handles.append(self.residual_module(l).register_forward_hook(make_hook(l)))
        try:
            text = self.format_prompt(
                prompt, system=system, template_mode=template_mode
            )
            enc = self.tokenizer(text, return_tensors="pt").to(self.input_device)
            self.model(**enc)
            lm_head = self.model.get_output_embeddings()
            norm = self._final_norm()
            out: dict[int, dict[str, float]] = {}
            for l in layers:
                hs = captured[l][0, -1, :]  # last token residual
                h = hs.to(lm_head.weight.dtype)
                if norm is not None:
                    h = norm(h)
                logits = lm_head(h).float()
                lp = torch.log_softmax(logits, dim=-1).cpu()
                out[l] = {
                    name: self.token_set_logprob(lp, ids)
                    for name, ids in token_sets.items()
                }
            return out
        finally:
            for h in handles:
                h.remove()

    def _final_norm(self):
        """Best-effort handle on the model's final pre-LM-head norm (or None).

        Decoder-only HF models keep it at ``model.model.norm`` (Llama/Gemma/
        Qwen/Mistral) or ``transformer.ln_f`` (GPT-NeoX/Pythia). The logit-lens
        is approximate either way, so a missing norm simply degrades to the raw
        residual projection rather than failing.
        """
        for path in (
            "model.norm",
            "model.model.norm",
            # Multimodal (Mistral3/Ministral3): the text decoder's final norm
            # lives under .language_model, with or without the nested .model.
            "model.language_model.model.norm",
            "model.language_model.norm",
            "transformer.ln_f",
        ):
            try:
                return resolve_module(self.model, path)
            except Exception:
                continue
        return None
