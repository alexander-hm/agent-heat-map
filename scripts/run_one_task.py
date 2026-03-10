#!/usr/bin/env python3
"""
Run a simulated or live-agent loop on a single toy task.

The agent loop: for each step in a trajectory, write the proposed code
to the task source file, run pytest, and record the objective outcome.

Modes:
  Scripted (default):  supply --trajectory, --use-solution, or --use-wrong
  Live agent:          supply --live-agent (requires TINKER_API_KEY)

Usage:
  python scripts/run_one_task.py --task-id binary_search --use-solution
  python scripts/run_one_task.py --task-id binary_search --use-wrong
  python scripts/run_one_task.py --task-id binary_search --live-agent
  python scripts/run_one_task.py --task-id binary_search --live-agent --model Qwen/Qwen3-32B
  python scripts/run_one_task.py --list-tasks
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from toy_tasks.registry import TASKS, get_task, list_task_ids


def run_pytest(test_file: str) -> dict:
    """Run pytest on a single test file and return structured results."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short", "-q"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=30,
    )
    passed = failed = errors = 0
    import re
    for line in result.stdout.splitlines():
        m = re.search(r"(\d+) passed", line)
        if m:
            passed = int(m.group(1))
        m = re.search(r"(\d+) failed", line)
        if m:
            failed = int(m.group(1))
        m = re.search(r"(\d+) error", line)
        if m:
            errors = int(m.group(1))

    all_pass = result.returncode == 0
    return {
        "tests_passed": all_pass,
        "n_passed": passed,
        "n_failed": failed,
        "n_errors": errors,
        "returncode": result.returncode,
        "stdout": result.stdout[-500:] if len(result.stdout) > 500 else result.stdout,
        "stderr": result.stderr[-300:] if len(result.stderr) > 300 else result.stderr,
    }


def invalidate_pycache(source_file: str) -> None:
    """Remove cached bytecode for a task module so pytest always reimports from source."""
    path = REPO_ROOT / source_file
    cache_dir = path.parent / "__pycache__"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)


def apply_code(source_file: str, code: str) -> None:
    """Overwrite a task source file with new code and clear bytecode cache."""
    invalidate_pycache(source_file)
    path = REPO_ROOT / source_file
    path.write_text(code)


def run_trajectory(task_id: str, steps: list[dict], run_id: str | None = None) -> list[dict]:
    """
    Execute a scripted trajectory: apply each step's code, run tests, log results.

    Args:
        task_id: task identifier
        steps: list of dicts, each with at least 'code' key and optionally 'simulated_metrics'
        run_id: unique run identifier (generated if None)

    Returns:
        list of step-log dicts (one per step)
    """
    task = get_task(task_id)
    source_file = task["source_file"]
    test_file = task["test_file"]
    run_id = run_id or f"{task_id}-{uuid.uuid4().hex[:8]}"

    original_code = (REPO_ROOT / source_file).read_text()

    log_entries = []
    try:
        for i, step in enumerate(steps):
            code = step["code"]
            apply_code(source_file, code)

            test_result = run_pytest(test_file)

            entry = {
                "run_id": run_id,
                "task_id": task_id,
                "step": i,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "code_applied": code.strip(),
                "tests_passed": test_result["tests_passed"],
                "n_passed": test_result["n_passed"],
                "n_failed": test_result["n_failed"],
                "n_errors": test_result["n_errors"],
                "test_stdout": test_result["stdout"],
            }
            if "simulated_metrics" in step:
                entry["simulated_metrics"] = step["simulated_metrics"]

            log_entries.append(entry)
    finally:
        apply_code(source_file, original_code)

    return log_entries


# ---------------------------------------------------------------------------
# Live-agent loop (requires TINKER_API_KEY + openai package)
# ---------------------------------------------------------------------------

def run_live_agent(
    task_id: str,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 800,
    max_steps: int = 6,
    run_id: str | None = None,
    use_native_client: bool = False,
) -> list[dict]:
    """
    Run a live LLM agent on a single task.

    At each step:
      1. Build a prompt with current code + failing test output
      2. Call Tinkr API for a patch
      3. Apply the patch (diff or function-replace fallback)
      4. Run pytest
      5. Log the result
      6. Stop early if all tests pass

    Args:
        use_native_client: If True, use TinkerNativeClient (real logprobs via
            native SDK) instead of the OpenAI-compatible TinkerLLMClient.

    Returns list of step-log dicts (same schema as run_trajectory).
    """
    from llm.tinker_client import TinkerLLMClient, TinkerNativeClient
    from llm.prompting import (
        build_patch_prompt,
        extract_diff,
        apply_diff_or_fallback,
    )
    from drift.metrics import (
        shannon_entropy, confidence_delta, compute_patch_complexity,
        compute_entropy_from_logprobs, compute_confidence_delta_from_logprobs,
        compute_test_stagnation, compute_patch_oscillation,
        compute_edit_target_concentration,
    )

    task = get_task(task_id)
    source_file = task["source_file"]
    test_file = task["test_file"]
    run_id = run_id or f"live-{task_id}-{uuid.uuid4().hex[:8]}"

    if use_native_client:
        from llm.tinker_client import NATIVE_DEFAULT_MODEL
        client = TinkerNativeClient(
            model=model or NATIVE_DEFAULT_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    else:
        client = TinkerLLMClient(model=model) if model else TinkerLLMClient()

    original_code = (REPO_ROOT / source_file).read_text()
    current_code = original_code

    log_entries: list[dict] = []
    prev_patches: list[str] = []
    prev_codes: list[str] = []
    test_history: list[tuple[int, int]] = []
    code_history: list[str] = []

    try:
        for step_i in range(max_steps):
            test_result = run_pytest(test_file)

            if step_i > 0 and test_result["tests_passed"]:
                print(f"  [step {step_i}] All tests pass — stopping early.", file=sys.stderr)
                break

            prompt = build_patch_prompt(
                task=task,
                current_code=current_code,
                test_output=test_result["stdout"] + "\n" + test_result.get("stderr", ""),
                step=step_i,
                max_steps=max_steps,
                prev_attempts=prev_patches if prev_patches else None,
            )

            try:
                resp = client.generate_patch(prompt)
            except Exception as e:
                entry = _make_live_entry(
                    run_id, task_id, step_i, current_code, test_result,
                    prompt=prompt, llm_error=str(e),
                )
                log_entries.append(entry)
                print(f"  [step {step_i}] LLM error: {e}", file=sys.stderr)
                continue

            diff_text = resp["patch"]
            prev_patches.append(diff_text or resp["text"][:200])

            apply_method = "none"
            apply_msg = ""
            if diff_text or resp["text"]:
                ok, apply_method, apply_msg = apply_diff_or_fallback(
                    source_file=source_file,
                    diff_text=diff_text,
                    llm_text=resp["text"],
                    function_name=task["function_name"],
                    repo_root=str(REPO_ROOT),
                )
                invalidate_pycache(source_file)
                if ok:
                    current_code = (REPO_ROOT / source_file).read_text()
                else:
                    print(f"  [step {step_i}] Patch apply failed: {apply_msg}", file=sys.stderr)

            post_test = run_pytest(test_file)

            pcs = compute_patch_complexity(
                current_code=current_code,
                original_code=original_code,
                prev_codes=prev_codes if prev_codes else None,
            )
            prev_codes.append(current_code)

            test_history.append((post_test["n_passed"], post_test["n_failed"]))
            code_history.append(current_code)

            tss = compute_test_stagnation(test_history)
            pos = compute_patch_oscillation(code_history)
            etc_val = compute_edit_target_concentration(code_history, original_code)

            real_metrics: dict = {"patch_complexity": pcs}
            if resp["logprobs"]:
                real_metrics["entropy"] = shannon_entropy(resp["logprobs"])
                real_metrics["confidence_delta"] = confidence_delta(resp["logprobs"])

            if resp["logprobs"]:
                entry_entropy = compute_entropy_from_logprobs(resp["logprobs"])
                entry_conf_delta = compute_confidence_delta_from_logprobs(resp["logprobs"])
                entry_entropy_source = "real"
                entry_raw_logprobs = resp["logprobs"]
            else:
                entry_entropy = None
                entry_conf_delta = None
                entry_entropy_source = "unavailable"
                entry_raw_logprobs = None

            entry = {
                "run_id": run_id,
                "task_id": task_id,
                "step": step_i,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "code_applied": current_code.strip(),
                "tests_passed": post_test["tests_passed"],
                "n_passed": post_test["n_passed"],
                "n_failed": post_test["n_failed"],
                "n_errors": post_test["n_errors"],
                "test_stdout": post_test["stdout"],
                "live_metrics": real_metrics,
                "llm_meta": resp["meta"],
                "llm_prompt": prompt[:500],
                "llm_response": resp["text"][:1000],
                "apply_method": apply_method,
                "apply_message": apply_msg,
                "entropy": entry_entropy,
                "confidence_delta": entry_conf_delta,
                "entropy_source": entry_entropy_source,
                "raw_logprobs": entry_raw_logprobs,
                "test_stagnation": tss,
                "patch_oscillation": pos,
                "edit_target_concentration": etc_val,
            }
            log_entries.append(entry)

            status = "PASS" if post_test["tests_passed"] else "FAIL"
            print(
                f"  [step {step_i}] {status} ({post_test['n_passed']}p/{post_test['n_failed']}f) "
                f"apply={apply_method} pcs={pcs:.3f}",
                file=sys.stderr,
            )

            if post_test["tests_passed"]:
                break

    finally:
        apply_code(source_file, original_code)

    return log_entries


def _make_live_entry(
    run_id: str, task_id: str, step: int, code: str, test_result: dict,
    prompt: str = "", llm_error: str = "",
) -> dict:
    return {
        "run_id": run_id,
        "task_id": task_id,
        "step": step,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "code_applied": code.strip(),
        "tests_passed": test_result["tests_passed"],
        "n_passed": test_result["n_passed"],
        "n_failed": test_result["n_failed"],
        "n_errors": test_result["n_errors"],
        "test_stdout": test_result["stdout"],
        "live_metrics": {},
        "llm_meta": {"error": llm_error},
        "llm_prompt": prompt[:500],
        "llm_response": "",
        "apply_method": "none",
        "apply_message": f"LLM error: {llm_error}",
    }


def label_run(log_entries: list[dict], recovery_window: int = 2) -> dict:
    """
    Produce run-level labels from step logs.

    Returns dict with: success, t_fail (or None), total_steps.

    success: True if the final step has all tests passing.
    t_fail (only for failure runs):
      - If the agent ever had all tests passing and then regressed without
        recovering within k steps, t_fail = that regression step.
      - Otherwise t_fail = index of last step (budget exhausted).
    """
    n = len(log_entries)
    if not n:
        return {"run_id": None, "task_id": None, "success": False, "t_fail": None, "total_steps": 0}

    success = log_entries[-1]["tests_passed"]

    t_fail = None
    if not success:
        ever_passed = False
        for i, entry in enumerate(log_entries):
            if entry["tests_passed"]:
                ever_passed = True
            elif ever_passed:
                recovered = any(
                    log_entries[j]["tests_passed"]
                    for j in range(i + 1, min(i + 1 + recovery_window, n))
                )
                if not recovered:
                    t_fail = i
                    break
        if t_fail is None:
            t_fail = n - 1

    return {
        "run_id": log_entries[0]["run_id"],
        "task_id": log_entries[0]["task_id"],
        "success": success,
        "t_fail": t_fail,
        "total_steps": n,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run a simulated or live-agent loop on one toy task",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task-id", type=str, help="Task identifier")
    parser.add_argument("--trajectory", type=str, help="Path to trajectory JSONL file")
    parser.add_argument("--use-solution", action="store_true", help="Single-step run with the correct solution")
    parser.add_argument("--use-wrong", action="store_true", help="Single-step run with the plausible-wrong solution")
    parser.add_argument("--list-tasks", action="store_true", help="List available task IDs")
    parser.add_argument("--output", type=str, default=None, help="Output JSONL file (default: stdout)")

    live = parser.add_argument_group("live agent (requires TINKER_API_KEY)")
    live.add_argument("--live-agent", action="store_true", help="Use a live LLM agent via Tinkr API")
    live.add_argument("--provider", type=str, default="tinker", help="LLM provider (default: tinker)")
    live.add_argument("--model", type=str, default=None,
                       help="Model name or checkpoint path (default: Qwen/Qwen3-235B-A22B-Instruct-2507)")
    live.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature (default: 0.2)")
    live.add_argument("--max-tokens", type=int, default=800, help="Max tokens per LLM call (default: 800)")
    live.add_argument("--max-steps", type=int, default=6, help="Max agent steps before giving up (default: 6)")
    live.add_argument("--native-client", action="store_true",
                       help="Use native Tinker SDK for real logprobs (requires tinker + transformers)")
    args = parser.parse_args()

    if args.list_tasks:
        for tid in list_task_ids():
            t = TASKS[tid]
            print(f"  {tid:20s}  {t['description']}")
        return

    if not args.task_id:
        parser.error("--task-id is required (or use --list-tasks)")

    task = get_task(args.task_id)

    if args.live_agent:
        native = getattr(args, "native_client", False)
        print(f"[live-agent] task={args.task_id} model={args.model or '(default)'} "
              f"T={args.temperature} max_tokens={args.max_tokens} max_steps={args.max_steps}"
              f"{' (native SDK)' if native else ''}",
              file=sys.stderr)
        log_entries = run_live_agent(
            task_id=args.task_id,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_steps=args.max_steps,
            use_native_client=native,
        )
    elif args.use_solution:
        steps = [{"code": task["correct_code"]}]
        log_entries = run_trajectory(args.task_id, steps)
    elif args.use_wrong:
        steps = [{"code": task["wrong_code"]}]
        log_entries = run_trajectory(args.task_id, steps)
    elif args.trajectory:
        with open(args.trajectory) as f:
            steps = [json.loads(line) for line in f if line.strip()]
        log_entries = run_trajectory(args.task_id, steps)
    else:
        parser.error("Provide --live-agent, --trajectory, --use-solution, or --use-wrong")

    run_label = label_run(log_entries)

    out_lines = []
    for entry in log_entries:
        out_lines.append(json.dumps(entry))
    out_lines.append(json.dumps({"_run_summary": True, **run_label}))

    output_text = "\n".join(out_lines) + "\n"

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(output_text)
        print(f"Wrote {len(log_entries)} step(s) + summary to {args.output}", file=sys.stderr)
    else:
        print(output_text)

    status = "SUCCESS" if run_label["success"] else f"FAIL (t_fail={run_label['t_fail']})"
    print(f"[{args.task_id}] {status} — {run_label['total_steps']} step(s)", file=sys.stderr)


if __name__ == "__main__":
    main()
