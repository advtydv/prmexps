from prm.rubric import MATH_RUBRIC, ALIGNMENT_RUBRIC, Criterion, Rubric, StepScore


class TestCriterion:
    def test_creation(self):
        c = Criterion(name="test", description="A test criterion")
        assert c.name == "test"
        assert c.score_range == (0.0, 1.0)
        assert c.domain == "general"

    def test_custom_range(self):
        c = Criterion(name="test", description="desc", score_range=(0.0, 10.0))
        assert c.score_range == (0.0, 10.0)

    def test_prompt_fragment(self):
        c = Criterion(name="accuracy", description="How accurate is this?")
        fragment = c.to_prompt_fragment()
        assert "accuracy" in fragment
        assert "How accurate is this?" in fragment
        assert "0.0" in fragment
        assert "1.0" in fragment

    def test_frozen(self):
        c = Criterion(name="test", description="desc")
        try:
            c.name = "changed"
            assert False, "Should have raised FrozenInstanceError"
        except AttributeError:
            pass


class TestRubric:
    def test_math_rubric_has_three_criteria(self):
        assert len(MATH_RUBRIC.criteria) == 3

    def test_alignment_rubric_has_four_criteria(self):
        assert len(ALIGNMENT_RUBRIC.criteria) == 4

    def test_criterion_names(self):
        names = MATH_RUBRIC.criterion_names()
        assert names == ["correctness", "logical_coherence", "completeness"]

    def test_to_prompt_fragment_contains_all_criteria(self):
        fragment = MATH_RUBRIC.to_prompt_fragment()
        for c in MATH_RUBRIC.criteria:
            assert c.name in fragment

    def test_prompt_fragment_contains_json_example(self):
        fragment = MATH_RUBRIC.to_prompt_fragment()
        assert "JSON" in fragment
        assert "correctness" in fragment

    def test_frozen(self):
        try:
            MATH_RUBRIC.name = "changed"
            assert False, "Should have raised FrozenInstanceError"
        except AttributeError:
            pass


class TestStepScore:
    def test_mean_score(self):
        ss = StepScore(step_index=0, step_text="test", scores={"a": 0.8, "b": 0.6})
        assert ss.mean_score() == 0.7

    def test_mean_score_with_none(self):
        ss = StepScore(step_index=0, step_text="test", scores={"a": 0.8, "b": None})
        assert ss.mean_score() == 0.8

    def test_mean_score_all_none(self):
        ss = StepScore(step_index=0, step_text="test", scores={"a": None, "b": None})
        assert ss.mean_score() is None

    def test_is_complete(self):
        ss = StepScore(step_index=0, step_text="test", scores={"a": 0.8, "b": 0.6})
        assert ss.is_complete()

    def test_is_not_complete(self):
        ss = StepScore(step_index=0, step_text="test", scores={"a": 0.8, "b": None})
        assert not ss.is_complete()

    def test_warning(self):
        ss = StepScore(step_index=0, step_text="test", warning="parse failed")
        assert ss.warning == "parse failed"
