# BiRD — Bidirectional Residual Direction Steering

Large language models increasingly make strategic choices in bargaining, allocation, trust, and cooperation settings. This project investigates whether these decisions can be systematically modulated through additive interventions in residual activations.

**BiRD** (**Bi**directional **R**esidual **D**irection Steering) is a valid-answer-aware protocol that:

1. Constructs candidate directions from **generic refusal**, **decision-derived**, **game-conditioned**, and **sparse-autoencoder** contrasts.
2. Selects effective **layers** and **intervention magnitudes**.
3. Evaluates them across **six behavioral-economic games**.

Across six open-weight LLMs from four architecture families, BiRD identifies configurations that move decisions toward both more **self-interested/non-cooperative** and more **generous/cooperative** behavior. Five models retain control in both target orientations on held-out scenarios under a fixed answer-validity criterion, with aggregate shifts up to 0.45, Cliff's δ up to 0.78, and near-perfect valid-answer rates in the strongest cases. Game-conditioned or game-combined directions provide all five feasible higher-index configurations, highlighting the practical value of context-matched contrasts.

Together, these findings establish BiRD as a practical causal probe of strategic decision-making and show that strategic social choices in the tested game settings are accessible to architecture-aware inference-time control.

---

## Directory Structure

```
bird/
├── config/
│   ├── base.json                          # Global defaults (paths, generation params, sweep ranges)
│   ├── benchmark/
│   │   └── neutral_prompts.json            # Neutral text corpus for SAE training fallback
│   ├── models/
│   │   ├── gemma2_9b.json                 # Per-model config: HF name, num_layers, hidden_size, dtype
│   │   ├── gemma3_12b.json
│   │   ├── llama31_8b.json
│   │   ├── llama32_3b.json
│   │   ├── ministral_3_8b.json
│   │   └── qwen25_7b.json
│   ├── rlhf_break.json                    # Master RLHF-break experiment config (games, sweep, SAE)
│   └── rlhf_break/
│       ├── gemma2_9b.json                 # Per-model RLHF-break overrides (norm_scale, SAE layer, etc.)
│       ├── gemma3_12b.json
│       ├── llama31_8b.json
│       ├── llama32_3b.json
│       ├── ministral_3_8b.json
│       └── qwen25_7b.json
│
├── emo/                                   # Core ML package
│   ├── __init__.py
│   ├── config.py                          # Config loading: merges base.json + models/<id>.json → Config dataclass
│   ├── hooks.py                           # Activation steering via forward hooks on residual stream
│   ├── io_utils.py                        # JSONL/JSON I/O helpers
│   ├── jsonc.py                           # JSONC reader (JSON + comments), stdlib-only
│   ├── models.py                          # SteeringModel: inference wrapper with residual-stream access
│   ├── data/
│   │   ├── __init__.py
│   │   ├── build_pairs.py                 # Stage0: build contrastive (pos/neg) pairs per emotion
│   │   └── isear_scenarios.py             # ISEAR-style scenario seeds for pair generation
│   ├── measure/
│   │   ├── __init__.py
│   │   └── judge_client.py                # LLM judge API client for opponent experiments
│   └── vectors/
│       ├── __init__.py
│       └── sae_train.py                   # Self-trained TopK Sparse Autoencoder (SAE) for aggression decomposition
│
├── experiments/                           # Experiment runner scripts
│   ├── run_all_experiments.py             # Unified runner: calls all 3 sub-runners via subprocess
│   ├── run_rlhf_break.py                  # Core stage: baseline → directions → sweep → final eval → publication
│   ├── run_advanced_experiments.py        # Advanced: steering detection, auto-alpha, justification, reverse-steer
│   └── run_llm_opponent_experiment.py     # LLM-opponent: automated vs LLM-judge opponents
│
├── rlhf_break/                            # RLHF-break experiment logic
│   ├── __init__.py
│   ├── analyze.py                         # Statistical analysis: Cliff's delta, Mann-Whitney U, Wilcoxon
│   ├── baseline.py                        # Baseline decision measurement (no steering)
│   ├── games.py                           # Six behavioral-economic games, scenario generation, decision parsing
│   ├── io_local.py                        # Local JSONL/JSON helpers
│   ├── mood_direction.py                  # Mood-based steering direction extraction
│   ├── opponents.py                        # Opponent models: FairAutomatedProposer/Responder, LLMJudgeOpponent
│   └── safety_direction.py                # Refusal & aggression direction extraction (incl. SAE-purified)
│
└── plotting/                             # Visualization & data export
    ├── fetch_rlhf_results.py              # Export LaTeX/CSV/MD tables from local results
    ├── plot_all.py                        # Generate ALL publication figures for a single model
    ├── plot_cross_model.py                # Cross-model comparison figures (8 plot types)
    ├── plot_rlhf_break.py                 # Per-model RLHF-break publication figures
    ├── plot_steering_parameter_map.py     # Steering parameter map visualization
    └── _results/                          # Pre-computed result data (6 models - need download)
        ├── gemma2_9b/
        ├── gemma3_12b/
        ├── llama31_8b/
        ├── llama32_3b/
        ├── ministral_3_8b/
        └── qwen25_7b/
```

---

## File Responsibilities

### `emo/` — Core ML Package

| File | Responsibility |
|------|---------------|
| [`config.py`](emo/config.py) | Loads and merges `config/base.json` + `config/models/<id>.json` into a `Config` dataclass tree. Resolves layer fractions to absolute indices. |
| [`hooks.py`](emo/hooks.py) | Activation steering via forward hooks on the residual stream. Provides `steer()` context manager and `SteerSpec` for multi-layer injection. |
| [`models.py`](emo/models.py) | `SteeringModel` — inference-only wrapper around a causal LM with residual-stream access, batched generation, logit lens, and activation collection. |
| [`io_utils.py`](emo/io_utils.py) | JSONL/JSON read/write helpers. |
| [`jsonc.py`](emo/jsonc.py) | Minimal JSONC reader (JSON + `//` and `/* */` comments), stdlib-only. |
| [`data/build_pairs.py`](emo/data/build_pairs.py) | Builds contrastive (positive/negative) text pairs per emotion for steering vector extraction. |
| [`data/isear_scenarios.py`](emo/data/isear_scenarios.py) | ISEAR-style scenario seeds (situation stems) for pair generation. |
| [`vectors/sae_train.py`](emo/vectors/sae_train.py) | Self-trained TopK Sparse Autoencoder for disentangled aggression direction extraction. No `sae_lens` dependency. |
| [`measure/judge_client.py`](emo/measure/judge_client.py) | LLM judge API client for opponent comparison experiments. |

### `experiments/` — Runner Scripts

| File | Responsibility |
|------|---------------|
| [`run_all_experiments.py`](experiments/run_all_experiments.py) | **Unified runner.** Executes all 3 stages (core, advanced, llm_opponent) via subprocess calls with `cwd=bird/`. |
| [`run_rlhf_break.py`](experiments/run_rlhf_break.py) | **Core stage.** Phase 0: baseline decisions. Phase 1: extract refusal + aggression + SAE-purified directions. Phase 2: coefficient sweep. Phase 3: final eval. Phase 4: statistical comparison. |
| [`run_advanced_experiments.py`](experiments/run_advanced_experiments.py) | **Advanced stage.** Steering detection, auto-alpha binary search, justification mode, reverse steering. |
| [`run_llm_opponent_experiment.py`](experiments/run_llm_opponent_experiment.py) | **LLM-opponent stage.** Compares model decisions against automated and LLM-judge opponents. |

### `rlhf_break/` — Experiment Logic

| File | Responsibility |
|------|---------------|
| [`games.py`](rlhf_break/games.py) | Six behavioral-economic games (Ultimatum, Dictator, Prisoner's Dilemma, Trust, Public Goods, Stag Hunt). Scenario generation, decision parsing, refusal detection. |
| [`safety_direction.py`](rlhf_break/safety_direction.py) | Extracts refusal directions (generic + game-context) and aggression directions (decision-derived + SAE-purified). The core of BiRD's direction construction. |
| [`baseline.py`](rlhf_break/baseline.py) | Measures baseline (unsteered) model decisions across all games. |
| [`analyze.py`](rlhf_break/analyze.py) | Statistical analysis: Cliff's delta, Mann-Whitney U, Wilcoxon signed-rank test. |
| [`mood_direction.py`](rlhf_break/mood_direction.py) | Mood-based steering direction extraction using emotion prompt sets. |
| [`opponents.py`](rlhf_break/opponents.py) | Opponent models: `FairAutomatedProposer`, `FairAutomatedResponder`, `LLMJudgeOpponent`. |
| [`io_local.py`](rlhf_break/io_local.py) | Local file I/O wrappers. |

### `plotting/` — Visualization

| File | Responsibility |
|------|---------------|
| [`fetch_rlhf_results.py`](plotting/fetch_rlhf_results.py) | Exports LaTeX/CSV/Markdown tables from local experiment results. |
| [`plot_all.py`](plotting/plot_all.py) | Generates ALL publication figures for a single model (calls `plot_rlhf_break.py` and `plot_steering_parameter_map.py` via subprocess). |
| [`plot_cross_model.py`](plotting/plot_cross_model.py) | Cross-model comparison figures: behavior, lift-coherence, before-after, layer heatmap, direction similarity, dose-response, effect size, refusal-aggression. |
| [`plot_rlhf_break.py`](plotting/plot_rlhf_break.py) | Per-model RLHF-break publication figures. |
| [`plot_steering_parameter_map.py`](plotting/plot_steering_parameter_map.py) | Steering parameter map visualization (layer × coefficient grid). |

---

## Running Experiments

### Prerequisites

- Python 3.10+
- PyTorch (with CUDA for GPU inference)
- `transformers` (Hugging Face)
- `numpy`, `scipy`
- `matplotlib` (for plotting only)

### Environment Variables

```bash
# Judge API (required for LLM-opponent experiment)
export JUDGE_API_URL=https://<host>/v1/chat/completions
export JUDGE_API_TOKEN=<token>
export JUDGE_MODEL=<model_name>
```

### Run All Experiments (Unified Runner)

```bash
# Run everything for a model (weights from HuggingFace):
python experiments/run_all_experiments.py --model llama31_8b

# Run with local weights (from server path):
python experiments/run_all_experiments.py --model llama31_8b --model-name /path/to/weights

# Run only specific stages:
python experiments/run_all_experiments.py --model llama31_8b --stages core,advanced

# Run with custom output dir:
python experiments/run_all_experiments.py --model llama31_8b --out-dir ./runs/my_run

# Skip LLM-opponent (no judge API available):
python experiments/run_all_experiments.py --model llama31_8b --stages core,advanced
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | *(required)* | Model ID from `config/models/` (e.g. `llama31_8b`, `gemma2_9b`) |
| `--model-name` | `None` | Override weight loading path (local dir or HF repo) |
| `--config` | *(auto)* | Path to rlhf_break config. Auto-resolves `config/rlhf_break/<model>.json`, falls back to `config/rlhf_break.json` |
| `--out-dir` | `runs/rlhf_break/<model>` | Output directory for artifacts |
| `--stages` | `all` | Comma-separated: `core`, `advanced`, `llm_opponent`, `all` |
| `--opponent` | `both` | LLM-opponent mode: `automated`, `llm_judge`, `both` |
| `--target-lift` | `0.15` | Auto-alpha target aggression lift |
| `--coh-threshold` | *(from config)* | Coherence threshold for auto-alpha |
| `--n-per-game` | `8` | Scenarios per game for advanced experiments |
| `--skip-llm-opponent` | `False` | Skip LLM-opponent experiment |

### Run Individual Stages

```bash
# Core experiment only
python experiments/run_rlhf_break.py --model llama31_8b

# Advanced experiments only
python experiments/run_advanced_experiments.py --model llama31_8b --experiment all

# LLM-opponent experiment only
python experiments/run_llm_opponent_experiment.py --model llama31_8b --opponent both
```

### Available Models

| Model ID | Architecture | Family |
|----------|-------------|--------|
| `gemma2_9b` | Gemma 2 9B | Google |
| `gemma3_12b` | Gemma 3 12B | Google |
| `llama31_8b` | Llama 3.1 8B | Meta |
| `llama32_3b` | Llama 3.2 3B | Meta |
| `qwen25_7b` | Qwen 2.5 7B | Alibaba |
| `ministral_3_8b` | Ministral 3 8B | Mistral |

---

## Exporting Tables

### Export Tables from Local Data

```bash
# Export tables from local results (single model):
python plotting/fetch_rlhf_results.py --data-dir ./plotting/_results/llama31_8b --export-tables

# Export tables from a multi-model root:
python plotting/fetch_rlhf_results.py --data-dir ./plotting/_results --export-tables
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-dir` | *(required)* | Local directory with results (single model or multi-model root) |
| `--out-dir` | `./_tables` | Output directory for exported tables |
| `--export-tables` | `False` | Export LaTeX/CSV/MD tables |

---

## Plotting Figures

### Generate All Figures for a Single Model

```bash
# Generate all publication figures for one model:
python plotting/plot_all.py --data-dir ./plotting/_results/llama31_8b --out-dir ./figures

# With a custom model label:
python plotting/plot_all.py --data-dir ./plotting/_results/llama31_8b --out-dir ./figures --model-label "Llama-3.1-8B"
```

This runs `plot_rlhf_break.py` and `plot_steering_parameter_map.py` automatically via subprocess.

### Cross-Model Comparison Figures

```bash
# Generate all cross-model comparison figures:
python plotting/plot_cross_model.py --data-root ./plotting/_results --out-dir ./cross_model

# Exclude specific models:
python plotting/plot_cross_model.py --data-root ./plotting/_results --out-dir ./cross_model --exclude llama32_3b
```

This generates 8 figure types:
1. Cross-model behavior comparison
2. Lift vs. coherence scatter
3. Before/after decision comparison
4. Layer heatmap
5. Direction similarity
6. Dose-response curves
7. Effect size (Cliff's δ) grouped bars
8. Refusal-aggression relationship

### Individual Plotting Scripts

```bash
# Per-model RLHF-break figures:
python plotting/plot_rlhf_break.py --data-dir ./plotting/_results/llama31_8b --out-dir ./figures

# Multi-model mode (process all subdirs):
python plotting/plot_rlhf_break.py --data-dir ./plotting/_results --out-dir ./figures --multi-model

# Steering parameter map:
python plotting/plot_steering_parameter_map.py --data-dir ./plotting/_results/llama31_8b/advanced --results-dir ./plotting/_results/llama31_8b --out-dir ./figures
```

---

## Output Artifacts

Experiments write to `runs/rlhf_break/<model>/` (or custom `--out-dir`):

```
runs/rlhf_break/<model>/
├── baseline_records.jsonl              # Baseline (unsteered) decisions
├── baseline_summary.json               # Aggregated baseline metrics
├── refusal_directions.npz              # Refusal steering directions per layer
├── aggression_directions.npz           # Aggression steering directions per layer
├── game_refusal_directions.npz         # Game-context refusal directions
├── sae_aggression_direction.npz        # SAE-purified aggression direction
├── sae_extraction_info.json            # SAE training metrics & layer selection
├── sweep_summaries.jsonl               # Per-config sweep results
├── broken_records.jsonl                # Steered (broken) decisions
├── comparison_report.json              # Baseline vs. steered comparison
├── ranked_configs.json                 # Ranked steering configurations
├── statistical_comparison.tex          # LaTeX statistical comparison table
├── top_configs.tex                     # LaTeX top-10 configs table
├── experiment_metadata.tex              # LaTeX experiment metadata
├── advanced/
│   ├── detection_results.json          # Steering detection experiment
│   ├── auto_alpha_results.json         # Auto-alpha binary search
│   ├── justification_results.json      # Justification mode experiment
│   ├── reverse_steer_results.json      # Reverse steering experiment
│   └── steering_parameter_map.json     # Layer × coefficient parameter map
├── opponent_comparison/
│   ├── baseline_automated.jsonl        # vs. automated opponent
│   └── baseline_llm_judge.jsonl        # vs. LLM-judge opponent
└── publication/
    ├── before_after_examples.json       # Before/after decision examples
    ├── direction_cosine_similarity.json # Direction cosine similarity matrix
    └── sweep_coefficient_response.json  # Dose-response curves
```
