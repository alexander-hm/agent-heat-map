"""Tests for drift.metrics — entropy, patch-complexity, and trajectory-aware signal computations."""

import math
import pytest
from drift.metrics import (
    shannon_entropy,
    confidence_delta,
    compute_patch_complexity,
    compute_step_metrics,
    compute_test_stagnation,
    compute_patch_oscillation,
    compute_edit_target_concentration,
    _count_diff_lines,
    _count_functions,
    _changed_lines,
)


# ---- Shannon entropy ----

class TestShannonEntropy:
    def test_empty_input(self):
        assert shannon_entropy([]) == 0.0

    def test_single_certain_token(self):
        """All probability mass on one token → entropy = 0."""
        pos = {"the": 0.0}  # logprob 0 → prob 1.0
        assert shannon_entropy([pos]) == pytest.approx(0.0, abs=1e-6)

    def test_uniform_two_tokens(self):
        """Two equally likely tokens → entropy = 1 bit."""
        lp = math.log(0.5)
        pos = {"a": lp, "b": lp}
        assert shannon_entropy([pos]) == pytest.approx(1.0, abs=1e-6)

    def test_uniform_four_tokens(self):
        """Four equally likely tokens → entropy = 2 bits."""
        lp = math.log(0.25)
        pos = {"a": lp, "b": lp, "c": lp, "d": lp}
        assert shannon_entropy([pos]) == pytest.approx(2.0, abs=1e-6)

    def test_mean_across_positions(self):
        """Entropy is averaged across positions."""
        certain = {"x": 0.0}
        lp = math.log(0.5)
        uncertain = {"a": lp, "b": lp}
        result = shannon_entropy([certain, uncertain])
        assert result == pytest.approx(0.5, abs=1e-6)

    def test_skewed_distribution(self):
        """One dominant token → low but nonzero entropy."""
        pos = {"a": math.log(0.9), "b": math.log(0.1)}
        h = shannon_entropy([pos])
        assert 0 < h < 1.0


# ---- Confidence delta ----

class TestConfidenceDelta:
    def test_empty(self):
        assert confidence_delta([]) == 0.0

    def test_large_gap(self):
        pos = {"a": -0.1, "b": -3.0, "c": -5.0}
        d = confidence_delta([pos], k=3)
        assert d == pytest.approx((-0.1) - (-5.0), abs=1e-6)

    def test_no_gap(self):
        pos = {"a": -1.0, "b": -1.0}
        d = confidence_delta([pos], k=2)
        assert d == pytest.approx(0.0, abs=1e-6)


# ---- Patch complexity ----

ORIGINAL = """\
def foo(x):
    return x + 1
"""

class TestPatchComplexity:
    def test_identical_code(self):
        """No diff → PCS = 0."""
        assert compute_patch_complexity(ORIGINAL, ORIGINAL) == 0.0

    def test_small_change(self):
        """One-line change → low PCS."""
        modified = "def foo(x):\n    return x + 2\n"
        pcs = compute_patch_complexity(modified, ORIGINAL)
        assert 0 < pcs < 0.5

    def test_large_change_higher_pcs(self):
        """More changed lines → higher PCS."""
        big = "def foo(x):\n    y = x * 2\n    z = y + 3\n    w = z - 1\n    return w\n"
        small = "def foo(x):\n    return x + 2\n"
        pcs_big = compute_patch_complexity(big, ORIGINAL)
        pcs_small = compute_patch_complexity(small, ORIGINAL)
        assert pcs_big > pcs_small

    def test_added_function_increases_scope(self):
        """Adding a second function increases scope_spread component."""
        one_fn = "def foo(x):\n    return x + 2\n"
        two_fn = "def foo(x):\n    return helper(x)\n\ndef helper(x):\n    return x + 2\n"
        pcs_one = compute_patch_complexity(one_fn, ORIGINAL)
        pcs_two = compute_patch_complexity(two_fn, ORIGINAL)
        assert pcs_two > pcs_one

    def test_churn_detection(self):
        """Re-editing same lines as previous step increases re_edit_ratio."""
        v1 = "def foo(x):\n    return x + 2\n"
        v2 = "def foo(x):\n    return x + 3\n"
        pcs_no_churn = compute_patch_complexity(v2, ORIGINAL, prev_codes=None)
        pcs_churn = compute_patch_complexity(v2, ORIGINAL, prev_codes=[v1])
        assert pcs_churn > pcs_no_churn

    def test_pcs_bounded(self):
        """PCS should be in [0, 1]."""
        huge = "\n".join(f"def f{i}(): pass" for i in range(50))
        pcs = compute_patch_complexity(huge, ORIGINAL)
        assert 0.0 <= pcs <= 1.0


# ---- Internal helpers ----

class TestHelpers:
    def test_count_diff_lines(self):
        assert _count_diff_lines("a\nb\n", "a\nc\n") >= 2

    def test_count_functions(self):
        code = "def a():\n    pass\ndef b():\n    pass\n"
        assert _count_functions(code) == 2

    def test_changed_lines(self):
        changes = _changed_lines("a\nb\n", "a\nc\n")
        assert "b" in changes
        assert "c" in changes


# ---- compute_step_metrics integration ----

class TestComputeStepMetrics:
    def test_simulated_values_passthrough(self):
        m = compute_step_metrics(entropy_val=2.5, confidence_delta_val=1.0)
        assert m.entropy == 2.5
        assert m.confidence_delta == 1.0
        assert m.patch_complexity == 0.0

    def test_with_code(self):
        m = compute_step_metrics(
            entropy_val=1.0,
            current_code="def foo(x):\n    return x + 2\n",
            original_code=ORIGINAL,
        )
        assert m.patch_complexity > 0


# ---- Test Stagnation Score ----

class TestTestStagnation:
    def test_empty_history(self):
        assert compute_test_stagnation([]) == 0.0

    def test_single_step(self):
        assert compute_test_stagnation([(6, 2)]) == 0.0

    def test_passing_step_at_end(self):
        """If current step passes all tests, stagnation = 0."""
        assert compute_test_stagnation([(6, 2), (6, 2), (8, 0)]) == 0.0

    def test_two_identical_failing(self):
        score = compute_test_stagnation([(6, 2), (6, 2)])
        assert score > 0.0

    def test_three_identical_higher_than_two(self):
        s2 = compute_test_stagnation([(6, 2), (6, 2)])
        s3 = compute_test_stagnation([(6, 2), (6, 2), (6, 2)])
        assert s3 > s2

    def test_changing_results_no_stagnation(self):
        """Different test results at each step → streak = 1 → 0.0."""
        assert compute_test_stagnation([(6, 2), (5, 3), (4, 4)]) == 0.0

    def test_max_streak_caps_at_one(self):
        history = [(3, 5)] * 10
        score = compute_test_stagnation(history, max_streak=5)
        assert score == pytest.approx(1.0)

    def test_mixed_then_stagnant(self):
        """Only the trailing identical streak matters."""
        history = [(5, 3), (6, 2), (6, 2), (6, 2)]
        score = compute_test_stagnation(history)
        assert score > 0.0


# ---- Patch Oscillation Score ----

class TestPatchOscillation:
    def test_too_few_steps(self):
        assert compute_patch_oscillation(["a"]) == 0.0
        assert compute_patch_oscillation(["a", "b"]) == 0.0

    def test_identical_code_two_steps_ago(self):
        """Code at step 2 identical to step 0 → oscillation = 1.0."""
        code = ["original\nline 2", "changed\nline 2", "original\nline 2"]
        assert compute_patch_oscillation(code, lookback=2) == 1.0

    def test_completely_different_code(self):
        code = ["aaa\nbbb\nccc", "ddd\neee\nfff", "ggg\nhhh\niii"]
        score = compute_patch_oscillation(code, lookback=2)
        assert score < 0.3

    def test_gradually_changing(self):
        c1 = "def f():\n    return 1\n    # pad\n    # more"
        c2 = "def f():\n    return 2\n    # pad\n    # more"
        c3 = "def f():\n    return 3\n    # pad\n    # more"
        score = compute_patch_oscillation([c1, c2, c3], lookback=2)
        assert score > 0.0  # c1 and c3 are similar but not identical
        assert score < 1.0


# ---- Edit Target Concentration ----

class TestEditTargetConcentration:
    def test_single_step(self):
        assert compute_edit_target_concentration(["code"], "original") == 0.0

    def test_same_lines_edited_repeatedly(self):
        original = "line 0\nline 1\nline 2\nline 3\n"
        v1 = "line 0\nCHANGED\nline 2\nline 3\n"
        v2 = "line 0\nDIFFERENT\nline 2\nline 3\n"
        score = compute_edit_target_concentration([v1, v2], original)
        assert score == pytest.approx(1.0)

    def test_different_lines_each_step(self):
        original = "line 0\nline 1\nline 2\nline 3\n"
        v1 = "CHANGED\nline 1\nline 2\nline 3\n"
        v2 = "line 0\nline 1\nline 2\nCHANGED\n"
        score = compute_edit_target_concentration([v1, v2], original)
        assert score == pytest.approx(0.0)

    def test_no_edits(self):
        original = "line 0\nline 1\n"
        score = compute_edit_target_concentration([original, original], original)
        assert score == 0.0

    def test_window_parameter(self):
        original = "line 0\nline 1\nline 2\n"
        v1 = "CHANGED\nline 1\nline 2\n"
        v2 = "line 0\nline 1\nCHANGED\n"
        v3 = "CHANGED\nline 1\nline 2\n"
        score_w2 = compute_edit_target_concentration([v1, v2, v3], original, window=2)
        score_w3 = compute_edit_target_concentration([v1, v2, v3], original, window=3)
        assert isinstance(score_w2, float)
        assert isinstance(score_w3, float)
