from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass
class ReasoningExample:
    """A single math problem with a labeled reasoning chain."""

    problem: str
    answer: float | int
    chain: str
    steps: list[str]
    is_correct: bool
    corruption_type: str | None = None
    labels: dict[str, list[dict[str, float | str]]] = field(default_factory=dict)
    source: str = "auto"

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Curated math problems with solutions
# ---------------------------------------------------------------------------

_PROBLEMS: list[dict] = [
    {
        "problem": "A store sells notebooks for $4 each. If Maria buys 7 notebooks and pays with a $50 bill, how much change does she receive?",
        "answer": 22,
        "steps": [
            "The cost of one notebook is $4.",
            "Maria buys 7 notebooks, so the total cost is 7 × $4 = $28.",
            "Maria pays with a $50 bill.",
            "Her change is $50 − $28 = $22.",
        ],
    },
    {
        "problem": "A train travels at 60 mph. How far does it travel in 2 hours and 30 minutes?",
        "answer": 150,
        "steps": [
            "The train's speed is 60 miles per hour.",
            "2 hours and 30 minutes equals 2.5 hours.",
            "Distance = speed × time = 60 × 2.5 = 150 miles.",
        ],
    },
    {
        "problem": "A rectangle has a length of 12 cm and a width of 5 cm. What is its area?",
        "answer": 60,
        "steps": [
            "The length of the rectangle is 12 cm.",
            "The width of the rectangle is 5 cm.",
            "Area = length × width = 12 × 5 = 60 cm².",
        ],
    },
    {
        "problem": "If 3x + 7 = 22, what is the value of x?",
        "answer": 5,
        "steps": [
            "Start with the equation 3x + 7 = 22.",
            "Subtract 7 from both sides: 3x = 22 − 7 = 15.",
            "Divide both sides by 3: x = 15 / 3 = 5.",
        ],
    },
    {
        "problem": "A baker makes 12 cupcakes per batch. If she needs 80 cupcakes for a party, how many full batches must she bake?",
        "answer": 7,
        "steps": [
            "Each batch produces 12 cupcakes.",
            "She needs 80 cupcakes total.",
            "80 ÷ 12 = 6.67, which is not a whole number.",
            "Since she can only bake full batches, she must bake 7 batches (rounding up).",
        ],
    },
    {
        "problem": "What is 15% of 240?",
        "answer": 36,
        "steps": [
            "Convert 15% to a decimal: 15 / 100 = 0.15.",
            "Multiply by 240: 0.15 × 240 = 36.",
        ],
    },
    {
        "problem": "A car's fuel tank holds 50 liters. If the car uses 8 liters per 100 km, how far can it travel on a full tank?",
        "answer": 625,
        "steps": [
            "The tank holds 50 liters.",
            "Fuel consumption is 8 liters per 100 km.",
            "Distance per liter: 100 / 8 = 12.5 km per liter.",
            "Total distance: 50 × 12.5 = 625 km.",
        ],
    },
    {
        "problem": "The sum of three consecutive integers is 72. What are they?",
        "answer": 24,
        "steps": [
            "Let the three consecutive integers be n, n+1, and n+2.",
            "Their sum: n + (n+1) + (n+2) = 3n + 3 = 72.",
            "Solve: 3n = 69, so n = 23.",
            "The three integers are 23, 24, and 25.",
        ],
    },
    {
        "problem": "A circle has a radius of 7 cm. What is its circumference? (Use π ≈ 3.14)",
        "answer": 43.96,
        "steps": [
            "The radius is 7 cm.",
            "Circumference = 2 × π × r.",
            "Circumference = 2 × 3.14 × 7 = 43.96 cm.",
        ],
    },
    {
        "problem": "A worker earns $18 per hour. She works 8 hours a day, 5 days a week. What is her weekly pay before taxes?",
        "answer": 720,
        "steps": [
            "Hourly wage is $18.",
            "Daily pay: $18 × 8 = $144.",
            "Weekly pay: $144 × 5 = $720.",
        ],
    },
]


# ---------------------------------------------------------------------------
# Corruption strategies — introduce deliberate errors
# ---------------------------------------------------------------------------

def _corrupt_arithmetic(steps: list[str]) -> tuple[list[str], int]:
    """Introduce an arithmetic error in a random step."""
    corrupted = list(steps)
    candidates = [i for i, s in enumerate(steps) if any(c in s for c in "=×*+−-/")]
    if not candidates:
        candidates = list(range(len(steps)))
    idx = random.choice(candidates)
    step = corrupted[idx]
    # Flip a digit near the end of the step
    digits = [(m.start(), m.group()) for m in __import__("re").finditer(r"\d+\.?\d*", step)]
    if digits:
        pos, num_str = digits[-1]
        try:
            num = float(num_str)
            wrong = num + random.choice([-3, -2, -1, 1, 2, 3])
            if wrong == num:
                wrong = num + 5
            wrong_str = str(int(wrong)) if num_str.isdigit() else f"{wrong:.2f}"
            corrupted[idx] = step[:pos] + wrong_str + step[pos + len(num_str):]
        except ValueError:
            corrupted[idx] = step + " (approximately)"
    return corrupted, idx


def _corrupt_skip_step(steps: list[str]) -> tuple[list[str], int]:
    """Remove a step from the middle of the chain."""
    if len(steps) <= 2:
        return steps, -1
    idx = random.randint(1, len(steps) - 2)
    corrupted = steps[:idx] + steps[idx + 1:]
    return corrupted, idx


def _corrupt_non_sequitur(steps: list[str]) -> tuple[list[str], int]:
    """Replace a step with an irrelevant statement."""
    non_sequiturs = [
        "The weather in Paris is usually pleasant in spring.",
        "Python was created by Guido van Rossum in 1991.",
        "The mitochondria is the powerhouse of the cell.",
        "Water boils at 100 degrees Celsius at sea level.",
    ]
    corrupted = list(steps)
    idx = random.randint(0, len(steps) - 1)
    corrupted[idx] = random.choice(non_sequiturs)
    return corrupted, idx


_CORRUPTION_FNS = {
    "arithmetic_error": _corrupt_arithmetic,
    "skipped_step": _corrupt_skip_step,
    "non_sequitur": _corrupt_non_sequitur,
}


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_curated_dataset(
    n_problems: int | None = None,
    corruptions_per_problem: int = 2,
    seed: int = 42,
) -> list[ReasoningExample]:
    """Generate a curated dataset of correct and corrupted math reasoning chains."""
    random.seed(seed)
    problems = _PROBLEMS if n_problems is None else _PROBLEMS[:n_problems]
    examples: list[ReasoningExample] = []

    for prob in problems:
        chain_text = "\n".join(
            f"Step {i + 1}: {s}" for i, s in enumerate(prob["steps"])
        )
        examples.append(
            ReasoningExample(
                problem=prob["problem"],
                answer=prob["answer"],
                chain=chain_text,
                steps=list(prob["steps"]),
                is_correct=True,
            )
        )

        corruption_types = list(_CORRUPTION_FNS.keys())
        selected = random.sample(
            corruption_types, min(corruptions_per_problem, len(corruption_types))
        )
        for ctype in selected:
            fn = _CORRUPTION_FNS[ctype]
            corrupted_steps, error_idx = fn(list(prob["steps"]))
            if error_idx == -1:
                continue
            corrupted_chain = "\n".join(
                f"Step {i + 1}: {s}" for i, s in enumerate(corrupted_steps)
            )
            examples.append(
                ReasoningExample(
                    problem=prob["problem"],
                    answer=prob["answer"],
                    chain=corrupted_chain,
                    steps=corrupted_steps,
                    is_correct=False,
                    corruption_type=ctype,
                )
            )

    return examples


def save_dataset(examples: list[ReasoningExample], path: str | Path | None = None) -> Path:
    """Save examples to a JSONL file."""
    if path is None:
        path = DEFAULT_DATA_DIR / "math_curated.jsonl"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict()) + "\n")
    return path


def load_dataset(path: str | Path | None = None) -> list[ReasoningExample]:
    """Load examples from a JSONL file."""
    if path is None:
        path = DEFAULT_DATA_DIR / "math_curated.jsonl"
    path = Path(path)
    examples: list[ReasoningExample] = []
    with open(path) as f:
        for line in f:
            data = json.loads(line.strip())
            examples.append(ReasoningExample(**data))
    return examples
