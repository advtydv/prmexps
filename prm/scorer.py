from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

import ollama

from prm.rubric import Rubric, StepScore
from prm.splitter import Step, split_reasoning_chain

logger = logging.getLogger(__name__)


@dataclass
class OllamaScorer:
    """Scores reasoning steps against a rubric using a local Ollama model."""

    model_name: str = "phi3:mini"
    rubric: Rubric | None = None
    base_url: str = "http://localhost:11434"
    temperature: float = 0.1
    _client: ollama.Client = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._client = ollama.Client(host=self.base_url)
        if self.rubric is None:
            from prm.rubric import MATH_RUBRIC
            self.rubric = MATH_RUBRIC

    def verify_connection(self) -> bool:
        """Check that Ollama is reachable and the model is available."""
        try:
            models = self._client.list()
            available = [m.model for m in models.models]
            if not any(self.model_name in name for name in available):
                logger.warning(
                    "Model %s not found. Available: %s", self.model_name, available
                )
                return False
            return True
        except Exception as e:
            logger.error("Cannot reach Ollama at %s: %s", self.base_url, e)
            return False

    def score_step(
        self,
        problem: str,
        steps: list[Step],
        target_index: int,
    ) -> StepScore:
        """Score a single reasoning step in context."""
        assert self.rubric is not None

        context_steps = steps[: target_index + 1]
        target = steps[target_index]

        prompt = self._build_prompt(problem, context_steps, target)
        raw = self._call_model(prompt)
        scores = self._parse_scores(raw)

        if scores is None:
            logger.info("Retrying with stricter prompt for step %d", target_index)
            strict_prompt = prompt + (
                "\n\nIMPORTANT: You MUST respond with ONLY a valid JSON object. "
                "No markdown, no explanation, no text before or after. Just the JSON."
            )
            raw = self._call_model(strict_prompt)
            scores = self._parse_scores(raw)

        if scores is None:
            return StepScore(
                step_index=target_index,
                step_text=target.text,
                scores={name: None for name in self.rubric.criterion_names()},
                warning=f"Failed to parse model response: {raw[:200]}",
            )

        clamped = self._clamp_scores(scores)
        return StepScore(
            step_index=target_index,
            step_text=target.text,
            scores=clamped,
        )

    def score_chain(self, problem: str, chain: str) -> list[StepScore]:
        """Split a reasoning chain and score every step."""
        steps = split_reasoning_chain(chain)
        if not steps:
            return []
        return [self.score_step(problem, steps, i) for i in range(len(steps))]

    def _build_prompt(
        self, problem: str, context_steps: list[Step], target: Step
    ) -> str:
        assert self.rubric is not None

        steps_text = "\n".join(
            f"Step {s.index + 1}: {s.text}" for s in context_steps
        )
        return (
            f"You are a strict process reward model that evaluates individual reasoning steps.\n\n"
            f"PROBLEM:\n{problem}\n\n"
            f"REASONING CHAIN SO FAR:\n{steps_text}\n\n"
            f"EVALUATE STEP {target.index + 1}: \"{target.text}\"\n\n"
            f"VERIFICATION: Before scoring, re-compute any arithmetic or logical claims in this step yourself. "
            f"Show your verification, then give your scores.\n"
            f"- If the step says \"7 × 4 = 32\", you must compute 7 × 4 = 28 and note the error.\n"
            f"- If the step is a non sequitur unrelated to the problem, correctness and logical_coherence must be near 0.\n"
            f"- Be harsh: even small arithmetic mistakes should drop correctness below 0.3.\n\n"
            f"{self.rubric.to_prompt_fragment()}"
        )

    def _call_model(self, prompt: str) -> str:
        try:
            response = self._client.generate(
                model=self.model_name,
                prompt=prompt,
                options={"temperature": self.temperature},
            )
            return response.response.strip()
        except Exception as e:
            logger.error("Ollama generate failed: %s", e)
            return ""

    def _parse_scores(self, raw: str) -> dict[str, float] | None:
        assert self.rubric is not None
        expected_keys = set(self.rubric.criterion_names())

        parsed = self._try_json_parse(raw)
        if parsed is not None and expected_keys.issubset(parsed.keys()):
            return {k: parsed[k] for k in expected_keys}

        json_match = re.search(r"\{[^{}]+\}", raw)
        if json_match:
            parsed = self._try_json_parse(json_match.group())
            if parsed is not None and expected_keys.issubset(parsed.keys()):
                return {k: parsed[k] for k in expected_keys}

        return None

    @staticmethod
    def _try_json_parse(text: str) -> dict[str, float] | None:
        try:
            data = json.loads(text)
            if isinstance(data, dict) and all(
                isinstance(v, (int, float)) for v in data.values()
            ):
                return {k: float(v) for k, v in data.items()}
        except (json.JSONDecodeError, ValueError):
            pass
        return None

    def _clamp_scores(self, scores: dict[str, float]) -> dict[str, float]:
        assert self.rubric is not None
        clamped: dict[str, float] = {}
        criteria_map = {c.name: c for c in self.rubric.criteria}
        for name, value in scores.items():
            criterion = criteria_map.get(name)
            if criterion is None:
                continue
            lo, hi = criterion.score_range
            if value < lo or value > hi:
                logger.warning(
                    "Score for %s out of range (%.3f), clamping to [%.1f, %.1f]",
                    name, value, lo, hi,
                )
            clamped[name] = max(lo, min(hi, value))
        return clamped
