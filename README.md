# Process Reward Model (PRM)

A rubric-based Process Reward Model that scores individual reasoning steps in math problem-solving. Designed as research groundwork for AI safety/alignment, where the same architecture can evaluate subjective properties like honesty and harmlessness.

## How It Works

Instead of scoring only the final answer (Outcome Reward Model), a PRM evaluates each intermediate step against named criteria:

- **Correctness** — Is the math/logic in this step valid?
- **Logical Coherence** — Does this step follow from the previous one?
- **Completeness** — Does this step make meaningful progress toward the solution?

The model (Phi-3-mini via Ollama) scores each step on each criterion from 0.0 to 1.0, producing an interpretable heatmap of reasoning quality.

## Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- Mac with Apple Silicon recommended (for mlx-lm fine-tuning)

### Install

```bash
pip install -r requirements.txt
ollama pull phi3:mini
```

### Verify

```bash
ollama list  # should show phi3:mini
pytest       # run tests
```

## Usage

### Quick Start (Notebook)

```bash
jupyter notebook notebooks/01_prm_demo.ipynb
```

The notebook walks through:
1. What a PRM is and why it matters
2. Live scoring of correct and incorrect reasoning chains
3. Visualizing per-step rubric scores as heatmaps
4. Generating training data for fine-tuning
5. The bridge from math PRM to alignment PRM

### Programmatic Usage

```python
from prm.rubric import MATH_RUBRIC
from prm.scorer import OllamaScorer

scorer = OllamaScorer(model_name="phi3:mini", rubric=MATH_RUBRIC)

problem = "What is 15% of 80?"
chain = """Step 1: Convert 15% to a decimal: 15/100 = 0.15
Step 2: Multiply 0.15 by 80: 0.15 * 80 = 12
Step 3: Therefore, 15% of 80 is 12."""

scores = scorer.score_chain(problem, chain)
for s in scores:
    print(f"Step {s.step_index}: {s.scores}")
```

## Project Structure

```
prm/              Core library
  rubric.py       Criterion and Rubric dataclasses
  splitter.py     Reasoning chain step splitter
  scorer.py       Ollama-based rubric scorer
  data.py         Dataset generation and I/O
  train.py        LoRA fine-tuning (Phase 2)
  visualize.py    Heatmap and chart rendering
notebooks/        Jupyter demo notebooks
data/             Generated datasets (JSONL)
adapters/         LoRA adapter weights
tests/            Pytest suite
docs/plans/       Design documents
```

## Research Context

This PRM is Phase 1 of alignment research. The rubric-based design means the same architecture extends to subjective domains by swapping criteria:

| Math Criterion | Alignment Criterion |
|---|---|
| `correctness` | `factual_accuracy` |
| `logical_coherence` | `reasoning_transparency` |
| `completeness` | `honesty` / `harmlessness` |
