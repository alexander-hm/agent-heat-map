"""
Drift metrics: entropy signals + patch-complexity proxy.

Entropy calculation is adapted from entropy_calculator.EntropyCalculator
(standalone reimplementation — avoids schema.py Pydantic import chain).

Patch-complexity proxy (PCS) is the complementary signal addressing GAP C:
token-distributional metrics share a blind spot on "confidently wrong" patches.
PCS measures *what* the agent changed, not *how uncertain* it was while generating.
"""

from __future__ import annotations

import difflib
import math
import re
from dataclasses import dataclass


def compute_entropy_from_logprobs(logprobs: list[dict[str, float]]) -> float:
    """
    Compute mean Shannon entropy across token positions from real logprobs.

    Args:
        logprobs: list of dicts, each mapping token -> log-probability (natural log).
                  This is the format returned by _extract_logprobs in tinker_client.py.

    Returns:
        Mean entropy in nats across all positions. Returns 0.0 if logprobs is empty.
    """
    if not logprobs:
        return 0.0

    entropies = []
    for position in logprobs:
        log_probs = list(position.values())
        probs = [math.exp(lp) for lp in log_probs]
        total = sum(probs)
        if total <= 0:
            entropies.append(0.0)
            continue
        probs = [p / total for p in probs]
        h = -sum(p * math.log(p) for p in probs if p > 0)
        entropies.append(h)

    return sum(entropies) / len(entropies)


def compute_confidence_delta_from_logprobs(logprobs: list[dict[str, float]]) -> float:
    """
    Compute mean confidence delta (top1 logprob - top2 logprob) across positions.

    Higher values = model is more confident in its top choice.
    Returns 0.0 if logprobs is empty or all positions have only 1 token.
    """
    if not logprobs:
        return 0.0

    deltas = []
    for position in logprobs:
        sorted_lps = sorted(position.values(), reverse=True)
        if len(sorted_lps) >= 2:
            deltas.append(sorted_lps[0] - sorted_lps[1])

    return sum(deltas) / len(deltas) if deltas else 0.0


@dataclass
class StepMetrics:
    """All drift signals for a single agent step."""

    entropy: float = 0.0
    confidence_delta: float = 0.0
    logprob_variance: float = 0.0
    patch_complexity: float = 0.0
    test_stagnation: float = 0.0
    patch_oscillation: float = 0.0
    edit_target_concentration: float = 0.0

    def to_dict(self) -> dict:
        return {
            "entropy": self.entropy,
            "confidence_delta": self.confidence_delta,
            "logprob_variance": self.logprob_variance,
            "patch_complexity": self.patch_complexity,
            "test_stagnation": self.test_stagnation,
            "patch_oscillation": self.patch_oscillation,
            "edit_target_concentration": self.edit_target_concentration,
        }


# ---------------------------------------------------------------------------
# Entropy from raw logprobs (standalone, no Pydantic dependency)
# Mirrors entropy_calculator.EntropyCalculator.calculate_shannon_entropy
# ---------------------------------------------------------------------------

def shannon_entropy(top_k_logprobs: list[dict[str, float]]) -> float:
    """
    Mean Shannon entropy across token positions.

    Args:
        top_k_logprobs: list of dicts mapping token→logprob for each position.
                        Same shape as TokenLogProb.top_logprobs but plain dicts.
    Returns:
        Mean positional entropy (bits). 0.0 if input is empty.
    """
    if not top_k_logprobs:
        return 0.0

    entropies = []
    for pos in top_k_logprobs:
        if not pos:
            entropies.append(0.0)
            continue
        probs = [math.exp(lp) for lp in pos.values()]
        total = sum(probs)
        if total <= 0:
            entropies.append(0.0)
            continue
        probs = [p / total for p in probs]
        h = -sum(p * math.log2(p) for p in probs if p > 0)
        entropies.append(h)

    return sum(entropies) / len(entropies)


def confidence_delta(top_k_logprobs: list[dict[str, float]], k: int = 5) -> float:
    """Mean logprob gap between top-1 and top-k tokens across positions."""
    if not top_k_logprobs:
        return 0.0
    deltas = []
    for pos in top_k_logprobs:
        vals = sorted(pos.values(), reverse=True)
        if len(vals) >= 2:
            deltas.append(vals[0] - vals[min(k - 1, len(vals) - 1)])
    return sum(deltas) / len(deltas) if deltas else 0.0


# ---------------------------------------------------------------------------
# Patch-complexity proxy (PCS) — the GAP C complementary signal
#
# Motivation: a model can be very confident (low entropy) while producing
# a large, wrong patch that touches unrelated code or thrashes on the same
# lines. PCS captures this by examining the *content* of the proposed patch.
#
# Components:
#   1. diff_lines  — total added+removed lines vs. reference (larger = riskier)
#   2. scope_spread — distinct function defs in the patch (edits outside the
#                     target function signal confusion)
#   3. re_edit_ratio — fraction of changed lines also changed in previous step
#                      (high churn = thrashing)
#
# PCS(t) = α·norm(diff_lines) + β·norm(scope_spread) + γ·re_edit_ratio
# ---------------------------------------------------------------------------

_ALPHA = 0.4  # diff size weight
_BETA = 0.3   # scope spread weight
_GAMMA = 0.3  # churn weight


def _count_diff_lines(a: str, b: str) -> int:
    """Count added + removed lines between two code strings."""
    a_lines = a.splitlines(keepends=True)
    b_lines = b.splitlines(keepends=True)
    diff = list(difflib.unified_diff(a_lines, b_lines, n=0))
    return sum(1 for line in diff if line.startswith("+") or line.startswith("-"))


def _count_functions(code: str) -> int:
    """Count distinct function/method definitions in code."""
    return len(re.findall(r"^\s*def\s+\w+", code, re.MULTILINE))


def _changed_lines(a: str, b: str) -> set[str]:
    """Return the set of normalized lines that differ between a and b."""
    a_set = set(line.strip() for line in a.splitlines() if line.strip())
    b_set = set(line.strip() for line in b.splitlines() if line.strip())
    return a_set.symmetric_difference(b_set)


def compute_patch_complexity(
    current_code: str,
    original_code: str,
    prev_codes: list[str] | None = None,
    max_diff_lines: int = 30,
    max_functions: int = 5,
) -> float:
    """
    Compute the patch-complexity score (PCS) for one step.

    Args:
        current_code:  the code the agent proposed at this step
        original_code: the original buggy code (reference)
        prev_codes:    codes from previous 1-2 steps (for churn detection)
        max_diff_lines: normalization cap for diff_lines
        max_functions:  normalization cap for function count

    Returns:
        PCS in [0, 1]. Higher = more complex/risky patch.
    """
    diff_lines = _count_diff_lines(original_code, current_code)
    norm_diff = min(diff_lines / max(max_diff_lines, 1), 1.0)

    n_functions = _count_functions(current_code)
    norm_scope = min(max(n_functions - 1, 0) / max(max_functions - 1, 1), 1.0)

    re_edit_ratio = 0.0
    if prev_codes:
        prev = prev_codes[-1]
        cur_changes = _changed_lines(original_code, current_code)
        prev_changes = _changed_lines(original_code, prev)
        if cur_changes:
            overlap = cur_changes & prev_changes
            re_edit_ratio = len(overlap) / len(cur_changes)

    pcs = _ALPHA * norm_diff + _BETA * norm_scope + _GAMMA * re_edit_ratio
    return round(pcs, 4)


def compute_step_metrics(
    entropy_val: float | None = None,
    confidence_delta_val: float | None = None,
    logprob_variance_val: float | None = None,
    current_code: str = "",
    original_code: str = "",
    prev_codes: list[str] | None = None,
) -> StepMetrics:
    """
    Compute all drift signals for one step.

    Accepts pre-computed (simulated) values for entropy/confidence_delta/variance,
    or can be extended to compute from raw logprobs.
    """
    pcs = 0.0
    if current_code and original_code:
        pcs = compute_patch_complexity(current_code, original_code, prev_codes)

    return StepMetrics(
        entropy=entropy_val or 0.0,
        confidence_delta=confidence_delta_val or 0.0,
        logprob_variance=logprob_variance_val or 0.0,
        patch_complexity=pcs,
    )


# ---------------------------------------------------------------------------
# Trajectory-aware signals
# ---------------------------------------------------------------------------

def compute_test_stagnation(
    test_history: list[tuple[int, int]],
    max_streak: int = 5,
) -> float:
    """
    Compute test stagnation score based on consecutive identical failing test results.

    Args:
        test_history: list of (n_passed, n_failed) tuples for steps 0..current,
                      including the current step as the last element.
        max_streak: normalization cap for the stagnation streak.

    Returns:
        Score in [0, 1]. 0 = no stagnation (results changing or passing).
        Higher = more consecutive identical failing results.
    """
    if not test_history:
        return 0.0

    current = test_history[-1]
    n_passed, n_failed = current

    if n_failed == 0:
        return 0.0

    streak = 1
    for i in range(len(test_history) - 2, -1, -1):
        if test_history[i] == current:
            streak += 1
        else:
            break

    if streak <= 1:
        return 0.0

    return min((streak - 1) / max(max_streak - 1, 1), 1.0)


def compute_patch_oscillation(
    code_history: list[str],
    lookback: int = 2,
) -> float:
    """
    Detect when the agent is oscillating — reverting or near-reverting recent changes.

    Args:
        code_history: list of code strings for steps 0..current,
                      including the current step as the last element.
        lookback: how far back to compare (default 2 = compare step N to step N-2).

    Returns:
        Score in [0, 1]. 0 = no oscillation.
        1.0 = current code is identical to the code from ``lookback`` steps ago.
    """
    if len(code_history) < lookback + 1:
        return 0.0

    current_code = code_history[-1]
    past_code = code_history[-(lookback + 1)]

    if current_code == past_code:
        return 1.0

    similarity = difflib.SequenceMatcher(
        None,
        current_code.splitlines(),
        past_code.splitlines(),
    ).ratio()

    return max(0.0, min((similarity - 0.5) / 0.5, 1.0))


def compute_edit_target_concentration(
    code_history: list[str],
    original_code: str,
    window: int = 3,
) -> float:
    """
    Detect when the agent keeps editing the same lines across multiple steps.

    Args:
        code_history: list of code strings for steps 0..current.
        original_code: the original buggy code (reference).
        window: number of recent steps to analyze.

    Returns:
        Score in [0, 1]. 0 = edits spread across different lines (exploring).
        1.0 = edits concentrated on the exact same lines every step.
    """
    if len(code_history) < 2:
        return 0.0

    original_lines = original_code.splitlines()

    recent = code_history[-window:] if len(code_history) >= window else code_history
    edit_sets: list[set[int]] = []
    for code in recent:
        code_lines = code.splitlines()
        changed: set[int] = set()
        max_len = max(len(original_lines), len(code_lines))
        for i in range(max_len):
            orig_line = original_lines[i] if i < len(original_lines) else ""
            curr_line = code_lines[i] if i < len(code_lines) else ""
            if orig_line != curr_line:
                changed.add(i)
        edit_sets.append(changed)

    nonempty = [(i, s) for i, s in enumerate(edit_sets) if s]
    if len(nonempty) < 2:
        return 0.0

    jaccard_scores: list[float] = []
    for idx in range(1, len(nonempty)):
        prev_set = nonempty[idx - 1][1]
        curr_set = nonempty[idx][1]
        intersection = len(prev_set & curr_set)
        union = len(prev_set | curr_set)
        if union > 0:
            jaccard_scores.append(intersection / union)

    if not jaccard_scores:
        return 0.0

    return sum(jaccard_scores) / len(jaccard_scores)
