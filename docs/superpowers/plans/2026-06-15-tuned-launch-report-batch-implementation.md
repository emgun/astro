# Tuned Launch Report Batch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic batch workflow that generates multiple tuned launch reports across
iteration counts and ranks them in one JSON product.

**Architecture:** Keep batch reporting as an orchestration layer over `generate_tuned_launch_report`.
The first batch axis is `iterations_values`; every case runs the existing report workflow, computes
a normalized score from the report assessment checks (`abs(value) / tolerance`), and returns cases
sorted by score. This avoids a new optimizer while producing a useful ranked table for comparing
local tuning depth.

**Tech Stack:** Python 3.12, Pydantic v2 models, Typer CLI, pytest, ruff, mypy.

---

### Task 1: Batch Report API

**Files:**
- Modify: `src/astro_launch/models.py`
- Modify: `src/astro_launch/reporting.py`
- Test: `tests/astro_launch/test_launch_reporting.py`

- [x] **Step 1: Write the failing API test**

Add this test to `tests/astro_launch/test_launch_reporting.py`:

```python
def test_generate_tuned_launch_report_batch_ranks_iteration_values() -> None:
    scenario = make_pitch_program_launch_scenario()

    batch = generate_tuned_launch_report_batch(
        scenario,
        point_indices=(2, 3),
        iterations_values=(1, 2),
        initial_span_deg=10.0,
        orbit_duration_s=600.0,
        orbit_step_s=60.0,
    )

    assert batch.scenario_id == scenario.scenario_id
    assert batch.point_indices == [2, 3]
    assert {case.iterations for case in batch.cases} == {1, 2}
    assert [case.rank for case in batch.cases] == [1, 2]
    assert batch.best_case == batch.cases[0]
    assert batch.best_case.normalized_score == min(
        case.normalized_score for case in batch.cases
    )
    for case in batch.cases:
        checks = [
            *case.report.insertion_assessment.checks,
            *case.report.short_arc_assessment.checks,
        ]
        assert case.normalized_score == pytest.approx(
            sum(abs(check.value) / check.tolerance for check in checks)
        )
        assert case.label == f"iterations={case.iterations}"
```

- [x] **Step 2: Run the API test to verify it fails**

Run:

```bash
python -m pytest tests/astro_launch/test_launch_reporting.py::test_generate_tuned_launch_report_batch_ranks_iteration_values -v
```

Expected: FAIL because `generate_tuned_launch_report_batch` is not defined.

- [x] **Step 3: Implement batch models and generation**

Add `TunedLaunchReportBatchCase` and `TunedLaunchReportBatch` to `src/astro_launch/models.py`.
Add `generate_tuned_launch_report_batch()` to `src/astro_launch/reporting.py`.

The function signature should accept:

```python
def generate_tuned_launch_report_batch(
    scenario: LaunchScenario,
    *,
    point_indices: Sequence[int],
    iterations_values: Sequence[int],
    initial_span_deg: float = 10.0,
    refinement_factor: float = 0.5,
    altitude_weight: float = 1.0,
    velocity_weight: float = 1.0,
    orbit_duration_s: float = 600.0,
    orbit_step_s: float = 60.0,
    spacecraft_name: str = "launch-payload",
    spacecraft_mass_kg: float | None = None,
    area_m2: float = 2.5,
    drag_coefficient: float = 2.2,
    reflectivity_coefficient: float = 1.3,
    gravity: ForceModelName = ForceModelName.TWO_BODY,
) -> TunedLaunchReportBatch:
```

It must reject an empty `iterations_values`, non-positive iteration counts, and duplicates with
`ValueError`. It must sort cases by `normalized_score`, then by original `case_index`, and assign
1-based ranks after sorting.

- [x] **Step 4: Run the API test to verify it passes**

Run:

```bash
python -m pytest tests/astro_launch/test_launch_reporting.py::test_generate_tuned_launch_report_batch_ranks_iteration_values -v
```

Expected: PASS.

### Task 2: CLI Batch Command

**Files:**
- Modify: `src/astro_cli/main.py`
- Test: `tests/astro_cli/test_cli.py`

- [x] **Step 1: Write the failing CLI test**

Add this test to `tests/astro_cli/test_cli.py`:

```python
def test_batch_report_tuned_launch_command_writes_ranked_json(tmp_path: Path) -> None:
    scenario_path = tmp_path / "pitch.yaml"
    output = tmp_path / "batch.json"
    _write_pitch_program_launch_scenario(scenario_path)

    result = runner.invoke(
        app,
        [
            "batch-report-tuned-launch",
            str(scenario_path),
            "--point-indices",
            "2,3",
            "--iterations-values",
            "1,2",
            "--initial-span-deg",
            "10",
            "--orbit-duration-s",
            "600",
            "--orbit-step-s",
            "60",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "wrote tuned launch report batch" in result.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["scenario_id"] == "pitch-program-two-stage"
    assert payload["point_indices"] == [2, 3]
    assert [case["rank"] for case in payload["cases"]] == [1, 2]
    assert {case["iterations"] for case in payload["cases"]} == {1, 2}
    assert payload["best_case"] == payload["cases"][0]
```

- [x] **Step 2: Run the CLI test to verify it fails**

Run:

```bash
python -m pytest tests/astro_cli/test_cli.py::test_batch_report_tuned_launch_command_writes_ranked_json -v
```

Expected: FAIL because the CLI command does not exist.

- [x] **Step 3: Implement CLI parser and command**

Add `_parse_iterations_values_or_exit()` to `src/astro_cli/main.py`. Add
`batch-report-tuned-launch` with the same report options as `report-tuned-launch`, replacing
single `--iterations` with comma-separated `--iterations-values`.

- [x] **Step 4: Add invalid parser test**

Add this test:

```python
def test_batch_report_tuned_launch_command_reports_invalid_iterations_values(
    tmp_path: Path,
) -> None:
    scenario_path = tmp_path / "pitch.yaml"
    _write_pitch_program_launch_scenario(scenario_path)

    result = runner.invoke(
        app,
        [
            "batch-report-tuned-launch",
            str(scenario_path),
            "--iterations-values",
            "1,bad",
            "--output",
            str(tmp_path / "batch.json"),
        ],
    )

    assert result.exit_code == 2
    assert "iterations-values must be comma-separated positive integers" in result.stderr
```

- [x] **Step 5: Run the CLI tests**

Run:

```bash
python -m pytest tests/astro_cli/test_cli.py::test_batch_report_tuned_launch_command_writes_ranked_json tests/astro_cli/test_cli.py::test_batch_report_tuned_launch_command_reports_invalid_iterations_values -v
```

Expected: PASS.

### Task 3: Exports, Docs, Verification, Commit

**Files:**
- Modify: `src/astro_launch/__init__.py`
- Modify: `tests/test_imports.py`
- Modify: `README.md`

- [x] **Step 1: Export public API and update docs**

Expose `TunedLaunchReportBatchCase`, `TunedLaunchReportBatch`, and
`generate_tuned_launch_report_batch` through `astro_launch.__all__`. Add a README command example:

```bash
astro batch-report-tuned-launch examples/launch/pitch_program_two_stage.yaml --point-indices 2,3 --iterations-values 1,2,3 --initial-span-deg 10 --orbit-duration-s 600 --orbit-step-s 60 --output tuned_launch_batch.json
```

Document that the batch ranks reports by normalized assessment error.

- [x] **Step 2: Run focused tests**

Run:

```bash
python -m pytest tests/astro_launch/test_launch_reporting.py tests/astro_cli/test_cli.py::test_batch_report_tuned_launch_command_writes_ranked_json tests/astro_cli/test_cli.py::test_batch_report_tuned_launch_command_reports_invalid_iterations_values tests/test_imports.py::test_packages_import -v
```

Expected: PASS.

- [x] **Step 3: Run full verification**

Run:

```bash
python -m pytest -q
python -m ruff check .
MYPY_CACHE_DIR=/tmp/astro-mypy-cache python -m mypy
```

Expected: all pass.

- [x] **Step 4: Run a CLI smoke**

Run `astro batch-report-tuned-launch` against `examples/launch/pitch_program_two_stage.yaml` with
`--iterations-values 1,2` and verify the output has two ranked cases.

- [x] **Step 5: Commit**

Stage only the intended files and commit:

```bash
git add README.md docs/superpowers/plans/2026-06-15-tuned-launch-report-batch-implementation.md src/astro_cli/main.py src/astro_launch/__init__.py src/astro_launch/models.py src/astro_launch/reporting.py tests/astro_cli/test_cli.py tests/astro_launch/test_launch_reporting.py tests/test_imports.py
git commit -m "feat: add tuned launch report batch ranking"
```
