# Tuned Launch Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an end-to-end tuned launch report that runs pitch tuning, tuned launch propagation, launch-to-orbit handoff, and short-arc orbital propagation.

**Architecture:** Keep this as a workflow composition layer in `astro_launch.reporting`. The report reuses `tune_pitch_program`, `propagate_launch_local`, `launch_trajectory_to_orbit_scenario`, and `propagate_local`, then packages the component products with compact insertion and short-arc target metrics.

**Tech Stack:** Python 3.12, Pydantic v2, Typer CLI, pytest, ruff, mypy.

---

## File Structure

- Create `src/astro_launch/reporting.py`: end-to-end report workflow and metric helpers.
- Modify `src/astro_launch/models.py`: add insertion metrics, short-arc metrics, and report product models.
- Modify `src/astro_launch/__init__.py`: export report models and API.
- Modify `src/astro_cli/main.py`: add `report-tuned-launch` command.
- Create `tests/astro_launch/test_launch_reporting.py`: API behavior and validation tests.
- Modify `tests/astro_cli/test_cli.py`: CLI success and error tests.
- Modify `tests/test_imports.py`: public export coverage.
- Modify `README.md`: document the report command and workflow boundary.

### Task 1: Report API Tests

**Files:**
- Create: `tests/astro_launch/test_launch_reporting.py`

- [ ] **Step 1: Write failing API tests**

```python
import pytest

from astro_launch.reporting import generate_tuned_launch_report
from tests.astro_launch.helpers import make_launch_scenario, make_pitch_program_launch_scenario


def test_generate_tuned_launch_report_runs_tune_launch_handoff_and_orbit_arc() -> None:
    scenario = make_pitch_program_launch_scenario()

    report = generate_tuned_launch_report(
        scenario,
        point_indices=(2, 3),
        initial_span_deg=10.0,
        iterations=2,
        orbit_duration_s=600.0,
        orbit_step_s=60.0,
    )

    assert scenario.guidance.pitch_program[2].pitch_deg == 45.0
    assert scenario.guidance.pitch_program[3].pitch_deg == 20.0
    assert report.scenario_id == scenario.scenario_id
    assert report.tuning_result.point_indices == [2, 3]
    assert report.launch_trajectory.scenario_id == scenario.scenario_id
    assert report.launch_trajectory.metadata["guidance_mode"] == "pitch_program"
    assert report.orbit_scenario.initial_state == report.launch_trajectory.insertion_state
    assert report.orbit_scenario.metadata["workflow"] == "launch_orbit_handoff"
    assert len(report.orbit_trajectory.samples) == 11
    assert report.insertion_metrics.altitude_miss_km == report.launch_trajectory.target_miss[
        "altitude_miss_km"
    ]
    assert report.insertion_metrics.velocity_miss_km_s == report.launch_trajectory.target_miss[
        "velocity_miss_km_s"
    ]
    assert report.short_arc_metrics.sample_count == 11
    assert report.short_arc_metrics.duration_s == 600.0
    assert report.short_arc_metrics.final_altitude_km == pytest.approx(
        report.short_arc_metrics.altitudes_km[-1]
    )
    assert report.short_arc_metrics.final_altitude_miss_km == pytest.approx(
        report.short_arc_metrics.final_altitude_km - scenario.target_orbit.altitude_km
    )


def test_generate_tuned_launch_report_requires_pitch_program_guidance() -> None:
    with pytest.raises(ValueError, match="pitch_program guidance"):
        generate_tuned_launch_report(make_launch_scenario(), point_indices=(2, 3))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/astro_launch/test_launch_reporting.py -v`

Expected: FAIL because `astro_launch.reporting` does not exist.

### Task 2: Report API Implementation

**Files:**
- Modify: `src/astro_launch/models.py`
- Create: `src/astro_launch/reporting.py`
- Modify: `src/astro_launch/__init__.py`
- Test: `tests/astro_launch/test_launch_reporting.py`

- [ ] **Step 1: Add report models**

Add `LaunchReportInsertionMetrics`, `LaunchReportShortArcMetrics`, and `TunedLaunchReport`. Include nested `LaunchPitchTuningResult`, `LaunchTrajectory`, `Scenario`, and `Trajectory` products.

- [ ] **Step 2: Implement metrics helpers**

Compute altitude and speed from Cartesian states, compare against target altitude and circular speed at target altitude, and summarize insertion plus short-arc orbital metrics.

- [ ] **Step 3: Implement `generate_tuned_launch_report`**

Run `tune_pitch_program`, launch the tuned scenario, create an orbit handoff scenario, propagate it locally, compute metrics, and return a `TunedLaunchReport`.

- [ ] **Step 4: Run focused API tests**

Run: `python -m pytest tests/astro_launch/test_launch_reporting.py -v`

Expected: PASS.

### Task 3: CLI Tests

**Files:**
- Modify: `tests/astro_cli/test_cli.py`

- [ ] **Step 1: Write failing CLI tests**

Add a success test for:

```bash
astro report-tuned-launch pitch.yaml --point-indices 2,3 --initial-span-deg 10 --iterations 2 --orbit-duration-s 600 --orbit-step-s 60 --output report.json
```

Assert the JSON contains `tuning_result`, `launch_trajectory`, `orbit_scenario`, `orbit_trajectory`, `insertion_metrics`, and `short_arc_metrics`. Add tests for invalid point-index parsing and output write errors.

- [ ] **Step 2: Run CLI tests to verify they fail**

Run: `python -m pytest tests/astro_cli/test_cli.py::test_report_tuned_launch_command_writes_json -v`

Expected: FAIL because `report-tuned-launch` is not implemented.

### Task 4: CLI, Exports, and Docs

**Files:**
- Modify: `src/astro_cli/main.py`
- Modify: `src/astro_launch/__init__.py`
- Modify: `tests/test_imports.py`
- Modify: `README.md`

- [ ] **Step 1: Add CLI command**

The command accepts tuning flags, short-arc propagation flags, spacecraft handoff flags, `--gravity`, and `--output`. It writes formatted JSON and reports validation failures with exit code 2.

- [ ] **Step 2: Update exports**

Export the report metric models, `TunedLaunchReport`, and `generate_tuned_launch_report`.

- [ ] **Step 3: Update README**

Document the command as a single report workflow: tune, launch, handoff, propagate, summarize.

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest tests/astro_launch/test_launch_reporting.py tests/astro_cli/test_cli.py tests/test_imports.py -v`

Expected: PASS.

### Task 5: Full Verification and Commit

**Files:**
- All modified files

- [ ] **Step 1: Run verification**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -v
python -m ruff check .
MYPY_CACHE_DIR=/tmp/astro-mypy-cache python -m mypy
```

If pre-existing untracked test files block `ruff check .`, record that and run ruff on tracked project files plus the files changed in this task.

- [ ] **Step 2: Run installed CLI smoke**

Run:

```bash
tmpdir=$(mktemp -d /tmp/astro-report-smoke.XXXXXX)
PYTHONDONTWRITEBYTECODE=1 astro report-tuned-launch examples/launch/pitch_program_two_stage.yaml --point-indices 2,3 --initial-span-deg 10 --iterations 2 --orbit-duration-s 600 --orbit-step-s 60 --output "$tmpdir/report.json"
python - <<'PY' "$tmpdir/report.json"
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
assert payload["scenario_id"] == "pitch-program-two-stage"
assert len(payload["orbit_trajectory"]["samples"]) == 11
print(json.dumps({
    "scenario_id": payload["scenario_id"],
    "best_score": payload["tuning_result"]["best_case"]["score"],
    "insertion_altitude_miss_km": payload["insertion_metrics"]["altitude_miss_km"],
    "final_orbit_altitude_miss_km": payload["short_arc_metrics"]["final_altitude_miss_km"],
}, indent=2))
PY
```

Expected: command exits 0 and prints the smoke summary.

- [ ] **Step 3: Stage and commit**

Run:

```bash
git add README.md docs/superpowers/plans/2026-06-15-tuned-launch-report-implementation.md src/astro_cli/main.py src/astro_launch/__init__.py src/astro_launch/models.py src/astro_launch/reporting.py tests/astro_cli/test_cli.py tests/astro_launch/test_launch_reporting.py tests/test_imports.py
git diff --cached --check
git commit -m "feat: add tuned launch report"
```

Expected: commit succeeds. Do not stage unrelated untracked files.

## Self-Review

- Spec coverage: the plan adds the report API, CLI, exports, docs, focused tests, full verification, and smoke test.
- Placeholder scan: no placeholder tasks remain.
- Type consistency: report model names, function names, CLI command, and JSON fields align across tasks.
