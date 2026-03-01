# Process Reward Model — Design Document

## Overview

A rubric-based Process Reward Model (PRM) for math reasoning, designed as research groundwork for alignment. The PRM scores individual reasoning steps against named criteria rather than evaluating only the final answer. This makes the reward signal interpretable and auditable — properties essential for safety/alignment research.

## Architecture

Three layers, built incrementally:

### Phase 1: Prompt-Based PRM

A Python module splits a math reasoning chain into discrete steps, sends each to Phi-3-mini running locally via Ollama, and collects per-step scores against a structured rubric. No training required — works immediately via in-context learning.

### Phase 2: LoRA Fine-Tuned PRM

Once Phase 1 generates labeled data (with human corrections), a LoRA adapter is trained on Phi-3-mini using Apple's mlx-lm framework. The adapter produces calibrated rubric scores directly, replacing prompt-based scoring with faster, more consistent inference.

### Phase 3: Alignment Bridge (Future)

The rubric criteria are extended from math-specific (`correctness`, `logical_coherence`, `completeness`) to alignment-relevant properties (`factual_accuracy`, `reasoning_transparency`, `honesty`, `harmlessness`). Same architecture, different rubric.

## Rubric Design

The rubric is the key abstraction. Each criterion is a named, scored dimension:

| Criterion | Description | 1.0 | 0.0 |
|---|---|---|---|
| `correctness` | Is the math/logic valid? | Fully correct | Wrong answer/invalid operation |
| `logical_coherence` | Does this step follow from the previous? | Clear logical chain | Non sequitur / skipped reasoning |
| `completeness` | Does the step advance the solution? | Meaningful progress | Restates the problem / vacuous |

Criteria are Python dataclasses. The scorer prompt references them dynamically, so adding or swapping criteria is a config change.

## Components

- **`prm/rubric.py`** — Criterion, Rubric, StepScore dataclasses; MATH_RUBRIC constant
- **`prm/splitter.py`** — Splits reasoning chains into Step objects via regex patterns with fallbacks
- **`prm/scorer.py`** — OllamaScorer class; builds rubric prompts, calls Ollama, parses JSON scores
- **`prm/data.py`** — Generates curated math datasets with correct + corrupted chains; JSONL storage
- **`prm/train.py`** — LoRA fine-tuning via mlx-lm (Phase 2)
- **`prm/visualize.py`** — Color-coded step heatmaps, comparison views, aggregate charts

## Data Flow

1. Problem text + reasoning chain enter the system
2. Splitter breaks chain into indexed Step objects
3. Scorer sends each step (with problem context and prior steps) to Ollama
4. Model returns JSON with per-criterion scores
5. Scores are parsed, validated (clamped to [0,1]), and returned as StepScore objects
6. Visualization renders heatmaps for inspection

## Demo

A Jupyter notebook serves as both tutorial and showcase:
- PRM vs ORM explainer with diagrams
- Live scoring of correct and corrupted reasoning chains
- Side-by-side heatmap comparisons
- Rubric exploration and alignment bridge preview

## Hardware

- Mac with Apple Silicon (M1/M2/M3/M4)
- Ollama with Phi-3-mini (3.8B parameters)
- mlx-lm for LoRA fine-tuning (native Metal acceleration)

## Error Handling

- Ollama connectivity check at startup
- JSON parse retries with stricter prompts
- Score clamping with logged warnings
- Graceful degradation when model returns malformed output
