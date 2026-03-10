# Agent Drift Heatmap

**Can we tell when an AI agent is about to fail — before it actually does?**

This project builds a drift detector that watches an agent's behavior step-by-step and raises an alarm when things start going sideways. The key finding: tracking model uncertainty (entropy) alone isn't enough — the agent can be *confidently wrong*. Adding a patch-complexity signal catches every one of those cases.

## The Problem

When an LLM-based agent works through a multi-step coding task, it can silently drift toward failure. Sometimes entropy spikes and you can see it coming. But sometimes the model is totally confident while producing garbage patches — low entropy, wrong answer. We wanted to know:

1. **Can uncertainty signals predict failure ahead of time?** (RQ1)
2. **Does adding a non-distributional signal (patch complexity) catch failures that entropy misses?** (RQ2)

Short answers: yes and yes.

## Results

| Metric | Entropy-Only | Composite |
|---|---|---|
| Detection rate (recall) | 66.7% | **100.0%** |
| False alarm rate | 50.0% | 25.0% |
| Confident-wrong failures caught | 0/8 | **8/8** |

The entropy-only detector misses every single "confidently wrong" failure. The composite detector (entropy + patch complexity) catches all of them with zero additional false alarms.

## Getting Started

```bash
pip install -r requirements.txt

# Run the full evaluation (40 runs, ~40 seconds)
python scripts/eval_runs.py --seed 42

# Check results
cat outputs/stats.json
```

No API keys needed — the default mode uses pre-scripted trajectories with simulated entropy values. Test outcomes are real though (actually runs pytest).

### Step-by-step reproduction

```bash
# 1. See the buggy task substrate (should show 28 failing tests)
python -m pytest toy_tasks/tests/ --tb=no -q

# 2. Run one task with the correct fix
python scripts/run_one_task.py --task-id binary_search --use-solution

# 3. Run one task with a plausible-but-wrong fix
python scripts/run_one_task.py --task-id binary_search --use-wrong

# 4. Full evaluation
python scripts/eval_runs.py --seed 42

# 5. Generate heatmap visualizations
python scripts/gen_heatmap.py                   # HTML report
python scripts/gen_heatmap.py --terminal         # terminal-friendly output

# 6. Generate paper-ready figures
python scripts/gen_figures.py
```

## How It Works

The evaluation runs 40 scripted agent trajectories (5 templates x 8 tasks) through two drift detectors:

**Entropy-only** — alarms based on token-level entropy and confidence delta.

**Composite** — same entropy signals *plus* a patch-complexity score (PCS) that looks at:
- **Diff size**: how many lines changed
- **Scope spread**: how many functions touched
- **Churn rate**: is the agent re-editing the same lines over and over

The insight is that entropy measures *how uncertain the model is about what to generate*, while PCS measures *what the model is actually producing*. A model can be very certain while generating a wrong answer — especially for "common wrong" patterns like off-by-one errors that appear frequently in training data.

## Live Agent Mode

You can also run real LLM trajectories instead of scripted ones (requires API access):

```bash
pip install openai>=1.12.0
export TINKER_API_KEY="your-key"

python scripts/run_one_task.py --task-id binary_search --live-agent
python scripts/eval_runs.py --live-agent --output-dir outputs/live01
```

## Project Structure

```
toy_tasks/
  tasks/              8 buggy Python functions (the evaluation substrate)
  tests/              pytest tests defining correctness for each task

drift/
  metrics.py          Entropy + patch-complexity calculations
  detector.py         Rolling-window drift detector with severity levels
  telemetry.py        Step/run data model and JSONL I/O

llm/
  tinker_client.py    Tinkr API client (OpenAI-compatible)
  prompting.py        Prompt building, diff extraction, patch application

scripts/
  eval_runs.py        Batch evaluation (scripted or live-agent)
  run_one_task.py     Single-task agent loop
  gen_heatmap.py      HTML report and terminal heatmap visualizations
  gen_figures.py      Paper-ready PDF figures

tests/                Unit tests for drift metrics, detector, and prompting
```

## Design Choices

- **No API keys for the default pipeline.** Trajectories are pre-scripted with simulated entropy; test outcomes come from actually running pytest. Fully reproducible with `--seed 42`.
- **Patch-complexity as the complementary signal.** It's orthogonal to token-distributional metrics and directly measures what the agent is doing rather than how confident it feels about doing it.
- **Recovery window (k=2).** Transient test failures that the agent fixes within 2 steps don't count as real failures — only sustained regressions do.
- **50% false alarm rate is expected.** The "struggle then succeed" runs trigger alarms because the agent *was* genuinely drifting — it just happened to recover. In practice, those alarms would still be useful.
