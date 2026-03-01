from unittest.mock import MagicMock, patch

from prm.rubric import MATH_RUBRIC, StepScore
from prm.scorer import OllamaScorer
from prm.splitter import Step


def _make_scorer(**kwargs) -> OllamaScorer:
    """Create a scorer without connecting to Ollama."""
    with patch("prm.scorer.ollama.Client"):
        return OllamaScorer(**kwargs)


class TestParseScores:
    def test_valid_json(self):
        scorer = _make_scorer()
        result = scorer._parse_scores('{"correctness": 0.9, "logical_coherence": 0.8, "completeness": 0.7}')
        assert result == {"correctness": 0.9, "logical_coherence": 0.8, "completeness": 0.7}

    def test_json_in_markdown_fence(self):
        scorer = _make_scorer()
        raw = '```json\n{"correctness": 0.9, "logical_coherence": 0.8, "completeness": 0.7}\n```'
        result = scorer._parse_scores(raw)
        assert result is not None
        assert result["correctness"] == 0.9

    def test_json_with_surrounding_text(self):
        scorer = _make_scorer()
        raw = 'Here is my evaluation:\n{"correctness": 0.5, "logical_coherence": 0.6, "completeness": 0.4}\nDone.'
        result = scorer._parse_scores(raw)
        assert result is not None
        assert result["correctness"] == 0.5

    def test_malformed_json(self):
        scorer = _make_scorer()
        result = scorer._parse_scores("this is not json at all")
        assert result is None

    def test_missing_criterion(self):
        scorer = _make_scorer()
        result = scorer._parse_scores('{"correctness": 0.9, "logical_coherence": 0.8}')
        assert result is None

    def test_integer_scores(self):
        scorer = _make_scorer()
        result = scorer._parse_scores('{"correctness": 1, "logical_coherence": 0, "completeness": 1}')
        assert result is not None
        assert result["correctness"] == 1.0


class TestClampScores:
    def test_in_range(self):
        scorer = _make_scorer()
        scores = {"correctness": 0.5, "logical_coherence": 0.8, "completeness": 0.3}
        clamped = scorer._clamp_scores(scores)
        assert clamped == scores

    def test_above_range(self):
        scorer = _make_scorer()
        scores = {"correctness": 1.5, "logical_coherence": 0.8, "completeness": 0.3}
        clamped = scorer._clamp_scores(scores)
        assert clamped["correctness"] == 1.0

    def test_below_range(self):
        scorer = _make_scorer()
        scores = {"correctness": -0.2, "logical_coherence": 0.8, "completeness": 0.3}
        clamped = scorer._clamp_scores(scores)
        assert clamped["correctness"] == 0.0

    def test_unknown_criterion_ignored(self):
        scorer = _make_scorer()
        scores = {"correctness": 0.5, "logical_coherence": 0.8, "completeness": 0.3, "unknown": 0.9}
        clamped = scorer._clamp_scores(scores)
        assert "unknown" not in clamped


class TestScoreChain:
    @patch.object(OllamaScorer, "_call_model")
    def test_scores_all_steps(self, mock_call):
        mock_call.return_value = '{"correctness": 0.9, "logical_coherence": 0.85, "completeness": 0.8}'
        scorer = _make_scorer()
        scores = scorer.score_chain(
            "What is 2+2?",
            "Step 1: Add 2 and 2\nStep 2: The answer is 4"
        )
        assert len(scores) == 2
        assert all(isinstance(s, StepScore) for s in scores)
        assert all(s.is_complete() for s in scores)

    @patch.object(OllamaScorer, "_call_model")
    def test_empty_chain(self, mock_call):
        scorer = _make_scorer()
        scores = scorer.score_chain("What is 2+2?", "")
        assert scores == []

    @patch.object(OllamaScorer, "_call_model")
    def test_retry_on_bad_json(self, mock_call):
        mock_call.side_effect = [
            "not json",
            '{"correctness": 0.9, "logical_coherence": 0.85, "completeness": 0.8}',
        ]
        scorer = _make_scorer()
        steps = [Step(index=0, text="test step")]
        result = scorer.score_step("problem", steps, 0)
        assert result.is_complete()
        assert mock_call.call_count == 2

    @patch.object(OllamaScorer, "_call_model")
    def test_warning_on_total_failure(self, mock_call):
        mock_call.return_value = "completely unparseable garbage"
        scorer = _make_scorer()
        steps = [Step(index=0, text="test step")]
        result = scorer.score_step("problem", steps, 0)
        assert not result.is_complete()
        assert result.warning is not None
