"""LoRA fine-tuning scaffold for the Process Reward Model (Phase 2).

Uses Apple's mlx-lm for native Metal acceleration on Apple Silicon.
This module prepares training data and wraps the mlx_lm.lora training API.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from prm.data import load_dataset

logger = logging.getLogger(__name__)

DEFAULT_ADAPTER_DIR = Path(__file__).resolve().parent.parent / "adapters"


def prepare_training_data(
    dataset_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Convert a curated JSONL dataset into instruction/response pairs for mlx-lm LoRA.

    Each example becomes:
        instruction: score this step against the rubric
        response: JSON of criterion scores

    Returns the path to the output JSONL file.
    """
    examples = load_dataset(dataset_path)
    if output_path is None:
        output_path = Path(dataset_path or "").parent / "train_lora.jsonl"
    output_path = Path(output_path)

    training_pairs: list[dict] = []
    for ex in examples:
        for i, step_text in enumerate(ex.steps):
            context = "\n".join(ex.steps[: i + 1])

            instruction = (
                f"You are a process reward model. Score the following reasoning step.\n\n"
                f"PROBLEM: {ex.problem}\n\n"
                f"REASONING SO FAR:\n{context}\n\n"
                f"EVALUATE STEP {i + 1}: \"{step_text}\"\n\n"
                f"Score on: correctness, logical_coherence, completeness (each 0.0-1.0).\n"
                f"Return ONLY a JSON object."
            )

            if ex.labels and "correctness" in ex.labels:
                step_labels = {
                    criterion: ex.labels[criterion][i]
                    for criterion in ex.labels
                    if i < len(ex.labels[criterion])
                }
                response = json.dumps(step_labels)
            else:
                if ex.is_correct:
                    response = json.dumps({
                        "correctness": 0.95,
                        "logical_coherence": 0.90,
                        "completeness": 0.85,
                    })
                else:
                    response = json.dumps({
                        "correctness": 0.30,
                        "logical_coherence": 0.50,
                        "completeness": 0.40,
                    })

            training_pairs.append({
                "instruction": instruction,
                "response": response,
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for pair in training_pairs:
            f.write(json.dumps(pair) + "\n")

    logger.info("Wrote %d training pairs to %s", len(training_pairs), output_path)
    return output_path


def train_adapter(
    data_path: str | Path,
    output_dir: str | Path | None = None,
    epochs: int = 3,
    lora_rank: int = 8,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    model_name: str = "mlx-community/Phi-3-mini-4k-instruct-4bit",
) -> Path:
    """Train a LoRA adapter using mlx-lm.

    This wraps the mlx_lm LoRA fine-tuning API. Requires Apple Silicon
    with the mlx-lm package installed.
    """
    try:
        from mlx_lm import lora as mlx_lora
    except ImportError:
        raise RuntimeError(
            "mlx-lm is required for LoRA fine-tuning. "
            "Install with: pip install mlx-lm"
        )

    if output_dir is None:
        output_dir = DEFAULT_ADAPTER_DIR / "math-prm"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(data_path)
    logger.info(
        "Starting LoRA training: model=%s, data=%s, epochs=%d, rank=%d",
        model_name, data_path, epochs, lora_rank,
    )

    # mlx-lm expects a directory with train.jsonl / valid.jsonl
    train_dir = data_path.parent / "mlx_train"
    train_dir.mkdir(exist_ok=True)

    import shutil
    shutil.copy(data_path, train_dir / "train.jsonl")

    # Create a minimal validation split (first 10% of training data)
    with open(data_path) as f:
        lines = f.readlines()
    n_valid = max(1, len(lines) // 10)
    with open(train_dir / "valid.jsonl", "w") as f:
        f.writelines(lines[:n_valid])

    mlx_lora.train(
        model=model_name,
        data=str(train_dir),
        adapter_path=str(output_dir),
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        lora_rank=lora_rank,
    )

    logger.info("Adapter saved to %s", output_dir)
    return output_dir


def load_adapter(adapter_path: str | Path) -> dict:
    """Load a trained LoRA adapter for inference.

    Returns metadata about the loaded adapter. The actual model loading
    is handled by mlx-lm at inference time.
    """
    adapter_path = Path(adapter_path)
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter not found at {adapter_path}")

    config_path = adapter_path / "adapter_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {"path": str(adapter_path), "status": "loaded"}

    logger.info("Loaded adapter from %s", adapter_path)
    return {
        "adapter_path": str(adapter_path),
        "config": config,
    }
