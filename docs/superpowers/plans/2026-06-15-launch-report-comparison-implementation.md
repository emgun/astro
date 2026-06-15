# Launch Report Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [x]`) syntax for tracking.

**Goal:** Add a CLI workflow that compares two tuned launch report JSON products and writes a
deterministic JSON comparison product.

**Architecture:** Treat comparison as post-processing over immutable report artifacts. Load
`TunedLaunchReport` JSON files through launch IO, compare the four existing target-miss metrics
with signed deltas and absolute-error improvement, and summarize pass/fail changes without rerunning
launch propagation or tuning.

**Tech Stack:** Python 3.12, Pydantic v2 models, Typer CLI, pytest, ruff, mypy.

---

### Task 1: Comparison Model And API

**Files:**
- Modify: `src/astro_launch/models.py`
- Modify: `src/astro_launch/reporting.py`
- Test: `tests/astro_launch/test_launch_reporting.py`

- [x] **Step 1: Write the failing comparison API test**

Add this test to `tests/astro_launch/test_launch_reporting.py`:

```python
def test_compare_tuned_launch_reports_summarizes_pass_and_metric_deltas() -> None:
    scenario = make_pitch_program_launch_scenario()
    baseline = generate_tuned_launch_report(
        scenario,
        point_indices=(2, 3),
        initial_span_deg=10.0,
        iterations=1,
        orbit_duration_s=600.0,
        orbit_step_s=60.0,
    )
    candidate = generate_tuned_launch_report(
        scenario,
        point_indices=(2, 3),
        initial_span_deg=10.0,
        iterations=2,
        orbit_duration_s=600.0,
        orbit_step_s=60.0,
    )

    comparison = compare_tuned_launch_reports(baseline, candidate)

    assert comparison.baseline_scenario_id == baseline.scenario_id
    assert comparison.candidate_scenario_id == candidate.scenario_id
    assert comparison.baseline_passed == baseline.passed
    assert comparison.candidate_passed == candidate.passed
    assert [metric.name for metric in comparison.metric_deltas] == [
        "insertion_altitude_miss",
        "insertion_velocity_miss",
        "short_arc_final_altitude_miss",
        "short_arc_final_velocity_miss",
    ]
    insertion_altitude = comparison.metric_deltas[0]
    assert insertion_altitude.baseline_value == pytest.approx(
        baseline.insertion_metrics.altitude_miss_km
    )
    assert insertion_altitude.candidate_value == pytest.approx(
        candidate.insertion_metrics.altitude_miss_km
    )
    assert insertion_altitude.delta == pytest.approx(
        candidate.insertion_metrics.altitude_miss_km
        - baseline.insertion_metrics.altitude_miss_km
    )
    assert insertion_altitude.improvement == pytest.approx(
        abs(baseline.insertion_metrics.altitude_miss_km)
        - abs(candidate.insertion_metrics.altitude_miss_km)
    )
```

- [x] **Step 2: Run the test to verify it fails**

Run:

```bash
python -m pytest tests/astro_launch/test_launch_reporting.py::test_compare_tuned_launch_reports_summarizes_pass_and_metric_deltas -v
```

Expected: FAIL because `compare_tuned_launch_reports` is not defined or exported.

- [x] **Step 3: Implement the comparison models and function**

Add `LaunchReportMetricDelta` and `TunedLaunchReportComparison` to
`src/astro_launch/models.py`. Add `compare_tuned_launch_reports()` to
`src/astro_launch/reporting.py`.

The comparison must include these metrics:
- `insertion_altitude_miss`, units `km`
- `insertion_velocity_miss`, units `km/s`
- `short_arc_final_altitude_miss`, units `km`
- `short_arc_final_velocity_miss`, units `km/s`

- [x] **Step 4: Run the comparison API test**

Run:

```bash
python -m pytest tests/astro_launch/test_launch_reporting.py::test_compare_tuned_launch_reports_summarizes_pass_and_metric_deltas -v
```

Expected: PASS.

### Task 2: Report Loading And CLI

**Files:**
- Modify: `src/astro_launch/io.py`
- Modify: `src/astro_cli/main.py`
- Test: `tests/astro_cli/test_cli.py`

- [x] **Step 1: Write the failing CLI test**

Add a helper that writes tuned launch reports, then test:

```python
def test_compare_tuned_launch_reports_command_writes_json(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    output = tmp_path / "comparison.json"
    _write_tuned_launch_report(baseline_path, iterations=1)
    _write_tuned_launch_report(candidate_path, iterations=2)

    result = runner.invoke(
        app,
        [
            "compare-tuned-launch-reports",
            str(baseline_path),
            str(candidate_path),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "wrote tuned launch report comparison" in result.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["baseline_scenario_id"] == "pitch-program-two-stage"
    assert payload["candidate_scenario_id"] == "pitch-program-two-stage"
    assert [metric["name"] for metric in payload["metric_deltas"]] == [
        "insertion_altitude_miss",
        "insertion_velocity_miss",
        "short_arc_final_altitude_miss",
        "short_arc_final_velocity_miss",
    ]
```

- [x] **Step 2: Run the test to verify it fails**

Run:

```bash
python -m pytest tests/astro_cli/test_cli.py::test_compare_tuned_launch_reports_command_writes_json -v
```

Expected: FAIL because the CLI command does not exist.

- [x] **Step 3: Implement report loading and the CLI command**

Add `load_tuned_launch_report()` in `src/astro_launch/io.py`. Add
`compare-tuned-launch-reports` in `src/astro_cli/main.py` with two readable file arguments and an
`--output` option. On invalid report input, print the loader error and exit with code 2.

- [x] **Step 4: Run the CLI test**

Run:

```bash
python -m pytest tests/astro_cli/test_cli.py::test_compare_tuned_launch_reports_command_writes_json -v
```

Expected: PASS.

### Task 3: Public Surface, Docs, And Verification

**Files:**
- Modify: `src/astro_launch/__init__.py`
- Modify: `tests/test_imports.py`
- Modify: `README.md`

- [x] **Step 1: Export the new models/functions and update README**

Expose `LaunchReportMetricDelta`, `TunedLaunchReportComparison`,
`compare_tuned_launch_reports`, and `load_tuned_launch_report` through `astro_launch.__all__`.
Document `astro compare-tuned-launch-reports baseline.json candidate.json --output comparison.json`
near the tuned launch report command.

- [x] **Step 2: Run full verification**

Run:

```bash
python -m pytest -q
python -m ruff check .
MYPY_CACHE_DIR=/tmp/astro-mypy-cache python -m mypy
```

Expected: all pass.

- [x] **Step 3: Run a CLI smoke**

Generate two report JSON files with `astro report-tuned-launch`, then compare them with
`astro compare-tuned-launch-reports`. Verify the comparison JSON includes four metric deltas.

- [x] **Step 4: Commit**

Stage only the intended files and commit:

```bash
git add README.md docs/superpowers/plans/2026-06-15-launch-report-comparison-implementation.md src/astro_launch/__init__.py src/astro_launch/io.py src/astro_launch/models.py src/astro_launch/reporting.py src/astro_cli/main.py tests/astro_cli/test_cli.py tests/astro_launch/test_launch_reporting.py tests/test_imports.py
git commit -m "feat: add tuned launch report comparison"
```
