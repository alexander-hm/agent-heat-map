"""Tests for llm.prompting: diff extraction and prompt building."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm.prompting import extract_diff, extract_complete_function, build_patch_prompt


class TestExtractDiff:
    def test_clean_diff(self):
        text = (
            "--- a/toy_tasks/tasks/binary_search.py\n"
            "+++ b/toy_tasks/tasks/binary_search.py\n"
            "@@ -1,3 +1,3 @@\n"
            " def binary_search(arr, target):\n"
            "-    low, high = 0, len(arr)\n"
            "+    low, high = 0, len(arr) - 1\n"
        )
        diff = extract_diff(text)
        assert "@@" in diff
        assert "+    low, high = 0, len(arr) - 1" in diff

    def test_diff_inside_fences(self):
        text = (
            "Here is the fix:\n"
            "```diff\n"
            "--- a/toy_tasks/tasks/binary_search.py\n"
            "+++ b/toy_tasks/tasks/binary_search.py\n"
            "@@ -1,3 +1,3 @@\n"
            " def binary_search(arr, target):\n"
            "-    low, high = 0, len(arr)\n"
            "+    low, high = 0, len(arr) - 1\n"
            "```\n"
            "This fixes the off-by-one.\n"
        )
        diff = extract_diff(text)
        assert "@@" in diff
        assert "This fixes" not in diff

    def test_no_diff(self):
        assert extract_diff("I don't know how to fix this.") == ""

    def test_no_patch_signal(self):
        assert extract_diff("NO_PATCH") == ""

    def test_empty_input(self):
        assert extract_diff("") == ""

    def test_diff_with_commentary_before(self):
        text = (
            "The issue is an off-by-one error.\n\n"
            "--- a/toy_tasks/tasks/binary_search.py\n"
            "+++ b/toy_tasks/tasks/binary_search.py\n"
            "@@ -1,3 +1,3 @@\n"
            " def binary_search(arr, target):\n"
            "-    low, high = 0, len(arr)\n"
            "+    low, high = 0, len(arr) - 1\n"
        )
        diff = extract_diff(text)
        assert "@@" in diff
        assert "The issue is" not in diff


class TestExtractCompleteFunction:
    def test_function_in_fences(self):
        text = (
            "Here is the fixed function:\n"
            "```python\n"
            "def binary_search(arr, target):\n"
            "    low, high = 0, len(arr) - 1\n"
            "    while low <= high:\n"
            "        mid = (low + high) // 2\n"
            "        if arr[mid] == target:\n"
            "            return mid\n"
            "        elif arr[mid] < target:\n"
            "            low = mid + 1\n"
            "        else:\n"
            "            high = mid - 1\n"
            "    return -1\n"
            "```\n"
        )
        func = extract_complete_function(text, "binary_search")
        assert func.startswith("def binary_search(")
        assert "len(arr) - 1" in func

    def test_no_function(self):
        assert extract_complete_function("random text", "binary_search") == ""

    def test_wrong_function_name(self):
        text = "def other_func(x):\n    return x\n"
        assert extract_complete_function(text, "binary_search") == ""


class TestBuildPrompt:
    def test_prompt_contains_key_elements(self):
        task = {
            "description": "Find target in sorted array",
            "function_name": "binary_search",
            "source_file": "toy_tasks/tasks/binary_search.py",
        }
        prompt = build_patch_prompt(
            task=task,
            current_code="def binary_search(arr, target):\n    pass\n",
            test_output="FAILED test_found_middle",
            step=0,
            max_steps=6,
        )
        assert "binary_search" in prompt
        assert "unified diff" in prompt.lower()
        assert "off-by-one" in prompt.lower()
        assert "FAILED" in prompt
        assert "Step 1 of 6" in prompt

    def test_prompt_with_prev_attempts(self):
        task = {
            "description": "test",
            "function_name": "foo",
            "source_file": "test.py",
        }
        prompt = build_patch_prompt(
            task=task,
            current_code="def foo(): pass",
            test_output="fail",
            step=2,
            max_steps=6,
            prev_attempts=["diff1", "diff2"],
        )
        assert "2 previous fix" in prompt
        assert "DIFFERENT approach" in prompt
