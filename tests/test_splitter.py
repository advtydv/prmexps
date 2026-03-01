from prm.splitter import Step, split_reasoning_chain


class TestNumberedSteps:
    def test_step_n_colon_format(self):
        chain = "Step 1: First thing\nStep 2: Second thing\nStep 3: Third thing"
        steps = split_reasoning_chain(chain)
        assert len(steps) == 3
        assert steps[0].text == "First thing"
        assert steps[2].text == "Third thing"

    def test_numbered_dot_format(self):
        chain = "1. Calculate the cost\n2. Subtract from total\n3. Get the answer"
        steps = split_reasoning_chain(chain)
        assert len(steps) == 3
        assert steps[0].text == "Calculate the cost"

    def test_parenthesized_numbers(self):
        chain = "(1) First step\n(2) Second step"
        steps = split_reasoning_chain(chain)
        assert len(steps) == 2

    def test_single_step(self):
        chain = "Just one step here."
        steps = split_reasoning_chain(chain)
        assert len(steps) == 1
        assert steps[0].text == "Just one step here."

    def test_empty_input(self):
        assert split_reasoning_chain("") == []
        assert split_reasoning_chain("   ") == []

    def test_double_newline_fallback(self):
        chain = "First paragraph of reasoning.\n\nSecond paragraph continues.\n\nThird wraps up."
        steps = split_reasoning_chain(chain)
        assert len(steps) == 3

    def test_sentence_fallback(self):
        chain = "The price is $4. Multiply by 7 to get $28. Subtract from $50."
        steps = split_reasoning_chain(chain)
        assert len(steps) == 3

    def test_preserves_index(self):
        chain = "Step 1: A\nStep 2: B\nStep 3: C"
        steps = split_reasoning_chain(chain)
        for i, step in enumerate(steps):
            assert step.index == i

    def test_mixed_whitespace(self):
        chain = "Step 1:   Lots of spaces   \n  Step 2:  Also spaces  "
        steps = split_reasoning_chain(chain)
        assert len(steps) == 2
        assert steps[0].text.strip() != ""

    def test_step_dataclass(self):
        step = Step(index=0, text="hello")
        assert step.index == 0
        assert step.text == "hello"
