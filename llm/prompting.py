"""
Prompt construction and diff extraction for the live-agent code-editing loop.

The LLM is instructed to output only a unified diff. This module:
  - builds the prompt from task context + failing test output
  - extracts the diff from (possibly noisy) LLM output
  - applies the diff to a source file, with fallback to whole-function replacement
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = """\
You are fixing a single Python function. Output ONLY a unified diff.
Do NOT add explanations, markdown fences, or commentary.
If you cannot fix the bug, output exactly: NO_PATCH

CONSTRAINTS:
- Only edit the function shown below. Do not change tests.
- Keep changes minimal — prefer the smallest correct fix.
- Watch for off-by-one errors and boundary cases (empty inputs, single elements).
- Do not add imports or helper functions.

TASK: {description}
FUNCTION: {function_name}
FILE: {source_file}

CURRENT CODE:
```
{current_code}
```

FAILING TEST OUTPUT (truncated):
```
{test_output}
```

{step_context}
Output your unified diff now. The diff should apply to the file "{source_file}".
Use the format:
--- a/{source_file}
+++ b/{source_file}
@@ ... @@
 context
-removed
+added
"""


def build_patch_prompt(
    task: dict,
    current_code: str,
    test_output: str,
    step: int = 0,
    max_steps: int = 6,
    prev_attempts: list[str] | None = None,
) -> str:
    """
    Build the user-side prompt for one agent step.

    Args:
        task:          task dict from registry (must have description, function_name, source_file)
        current_code:  current content of the source file
        test_output:   pytest stdout/stderr (truncated to ~1500 chars)
        step:          current step index (0-based)
        max_steps:     budget
        prev_attempts: list of prior diffs the model produced (for anti-repetition)
    """
    test_output = test_output[:1500]

    step_lines = [f"Step {step + 1} of {max_steps}."]
    if prev_attempts:
        step_lines.append(
            f"You have tried {len(prev_attempts)} previous fix(es) that did not pass all tests. "
            "Try a DIFFERENT approach."
        )
    step_context = "\n".join(step_lines)

    return _PROMPT_TEMPLATE.format(
        description=task["description"],
        function_name=task["function_name"],
        source_file=task["source_file"],
        current_code=current_code.strip(),
        test_output=test_output.strip(),
        step_context=step_context,
    )


# ---------------------------------------------------------------------------
# Diff extraction
# ---------------------------------------------------------------------------

_DIFF_HEADER_RE = re.compile(r"^---\s+\S+", re.MULTILINE)
_HUNK_RE = re.compile(r"^@@\s", re.MULTILINE)
_FENCE_RE = re.compile(r"```(?:diff)?\s*\n(.*?)```", re.DOTALL)


def extract_diff(text: str) -> str:
    """
    Extract a unified diff from LLM output.

    Tries, in order:
      1. Text between ``` fences tagged as diff
      2. Raw text starting from the first --- header line through end
      3. Empty string if no diff found (or NO_PATCH)

    Returns the diff string (may be empty).
    """
    if not text or "NO_PATCH" in text:
        return ""

    fence_match = _FENCE_RE.search(text)
    if fence_match:
        candidate = fence_match.group(1).strip()
        if _HUNK_RE.search(candidate):
            return candidate

    header_match = _DIFF_HEADER_RE.search(text)
    if header_match:
        candidate = text[header_match.start():].strip()
        if _HUNK_RE.search(candidate):
            return candidate

    return ""


def extract_complete_function(text: str, function_name: str) -> str:
    """
    Fallback: extract a complete Python function definition from LLM output.

    Used when the model ignores the diff instruction and outputs a full function.
    Returns the function body or empty string.
    """
    pattern = re.compile(
        rf"(def\s+{re.escape(function_name)}\s*\(.*?\):.*?)(?=\ndef\s|\Z)",
        re.DOTALL,
    )
    for fence_match in _FENCE_RE.finditer(text):
        block = fence_match.group(1).strip()
        m = pattern.search(block)
        if m:
            return m.group(1).rstrip() + "\n"

    m = pattern.search(text)
    if m:
        return m.group(1).rstrip() + "\n"

    return ""


# ---------------------------------------------------------------------------
# Diff application
# ---------------------------------------------------------------------------

def apply_diff(source_file: str, diff_text: str, repo_root: str | Path) -> tuple[bool, str]:
    """
    Apply a unified diff to a source file using the system `patch` command.
    """
    repo_root = Path(repo_root)
    abs_path = repo_root / source_file

    if not abs_path.exists():
        return False, f"Source file not found: {abs_path}"

    # Ensure diff ends with newline (patch can reject without it)
    if not diff_text.endswith("\n"):
        diff_text += "\n"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as pf:
        pf.write(diff_text)
        patch_path = pf.name

    try:
        # Try -p1 first (git-style a/b prefixes), then -p0 (plain paths)
        # Use --fuzz=2 to tolerate minor context mismatches from LLM-generated diffs
        last_result = None
        for strip_level in ["-p1", "-p0"]:
            result = subprocess.run(
                ["patch", "--batch", "--fuzz=2", strip_level, "-i", patch_path],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return True, f"patch applied ({strip_level})"
            last_result = result

        return False, f"patch failed (rc={last_result.returncode}): stdout={last_result.stdout[:200]} stderr={last_result.stderr[:200]}"
    except FileNotFoundError:
        return False, "patch command not found — install GNU patch"
    except subprocess.TimeoutExpired:
        return False, "patch command timed out"
    finally:
        os.unlink(patch_path)


def apply_diff_or_fallback(
    source_file: str,
    diff_text: str,
    llm_text: str,
    function_name: str,
    repo_root: str | Path,
) -> tuple[bool, str, str]:
    """
    Try to apply the diff; on failure, try to extract a complete function as fallback.

    Returns:
        (success, method_used, message)
        method_used is one of: "diff", "function_replace", "none"
    """
    repo_root = Path(repo_root)
    abs_path = repo_root / source_file

    if diff_text:
        ok, msg = apply_diff(source_file, diff_text, repo_root)
        if ok:
            return True, "diff", msg
        else:
            import sys
            print(f"  [DEBUG apply_diff failed]: {msg}", file=sys.stderr)

    func_code = extract_complete_function(llm_text, function_name)
    if func_code:
        abs_path.write_text(func_code)
        return True, "function_replace", "Fallback: wrote complete function from LLM output"

    return False, "none", "Could not extract diff or complete function from LLM output"
