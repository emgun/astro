# Launch Pitch Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic two-knot pitch-program tuner that can write a tuned launch scenario.

**Architecture:** Build on the existing launch sweep scoring contract rather than introducing a numerical optimizer. The tuner evaluates a two-dimensional grid around two selected pitch-program points, chooses the best target-miss score, shrinks the search span, repeats for a fixed number of iterations, and returns both a trace product and a tuned `LaunchScenario`.

**Tech Stack:** Python 3.12, Pydantic v2, Typer CLI, PyYAML, pytest, ruff, mypy.

---

## File Structure

- Modify `src/astro_launch/models.py`: add serializable tuning result models.
- Modify `src/astro_launch/targeting.py`: add reusable scoring helpers and `tune_pitch_program`.
- Modify `src/astro_launch/__init__.py`: export tuning models and API.
- Modify `src/astro_cli/main.py`: add `tune-launch-pitch` command, point-index parsing, and optional tuned-scenario YAML output.
- Modify `tests/astro_launch/test_launch_targeting.py`: API behavior and validation tests.
- Modify `tests/astro_cli/test_cli.py`: CLI success and error tests.
- Modify `tests/test_imports.py`: public export coverage.
- Modify `README.md`: document tuning command and tradeoff.

### Task 1: Tuning API Tests

**Files:**
- Modify: `tests/astro_launch/test_launch_targeting.py`

- [ ] **Step 1: Write failing API tests**

Add these tests:

```python
from astro_launch.local import propagate_launch_local
from astro_launch.targeting import sweep_pitch_program, tune_pitch_program


def test_tune_pitch_program_refines_two_knots_and_returns_tuned_scenario() -> None:
    scenario = make_pitch_program_launch_scenario()
    baseline_score = sweep_pitch_program(
        scenario,
        point_index=3,
        pitch_values_deg=[scenario.guidance.pitch_program[3].pitch_deg],
    ).best_case.score

    result = tune_pitch_program(
        scenario,
        point_indices=(2, 3),
        initial_span_deg=10.0,
        iterations=2,
    )

    tuned_pitches = {
        point.point_index: point.tuned_pitch_deg for point in result.tuned_points
    }
    assert scenario.guidance.pitch_program[2].pitch_deg == 45.0
    assert scenario.guidance.pitch_program[3].pitch_deg == 20.0
    assert result.scenario_id == scenario.scenario_id
    assert result.point_indices == [2, 3]
    assert len(result.iterations) == 2
    assert all(len(iteration.cases) == 9 for iteration in result.iterations)
    assert result.best_case.score <= baseline_score
    assert result.best_case.score == min(
        case.score for iteration in result.iterations for case in iteration.cases
    )
    assert result.tuned_scenario.guidance.pitch_program[2].pitch_deg == tuned_pitches[2]
    assert result.tuned_scenario.guidance.pitch_program[3].pitch_deg == tuned_pitches[3]
    assert propagate_launch_local(result.tuned_scenario).target_miss == result.best_case.target_miss
```

Add invalid-input parameter tests for vertical guidance, one index, duplicate indices, index 0, out-of-range index, nonpositive span, nonpositive iterations, and invalid refinement factor.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/astro_launch/test_launch_targeting.py -v`

Expected: FAIL because `tune_pitch_program` is not implemented.

### Task 2: Tuning API Implementation

**Files:**
- Modify: `src/astro_launch/models.py`
- Modify: `src/astro_launch/targeting.py`
- Modify: `src/astro_launch/__init__.py`
- Test: `tests/astro_launch/test_launch_targeting.py`

- [ ] **Step 1: Add models**

Add `LaunchPitchTuningPoint`, `LaunchPitchTuningCase`, `LaunchPitchTuningIteration`, and `LaunchPitchTuningResult`. Include selected point metadata, per-candidate pitch values, score, target miss, final-state metrics, iteration span, best case, tuned scenario, backend, and metadata.

- [ ] **Step 2: Factor candidate scoring**

Extract a helper in `targeting.py` that runs `propagate_launch_local`, computes the same weighted score used by `sweep_pitch_program`, and returns reusable metrics.

- [ ] **Step 3: Implement `tune_pitch_program`**

Validate two distinct tunable pitch-program indices, reject index 0, build 3x3 candidate grids around the current center, clamp pitches to `[0, 90]`, evaluate candidates, select the best case, shrink span by `refinement_factor`, and return a tuned scenario copied from the original.

- [ ] **Step 4: Run focused API tests**

Run: `python -m pytest tests/astro_launch/test_launch_targeting.py -v`

Expected: PASS.

### Task 3: CLI Tests

**Files:**
- Modify: `tests/astro_cli/test_cli.py`

- [ ] **Step 1: Write failing CLI tests**

Add a success test for:

```bash
astro tune-launch-pitch pitch.yaml --point-indices 2,3 --initial-span-deg 10 --iterations 2 --output tuning.json --tuned-scenario-output tuned.yaml
```

Assert the JSON contains two iterations, a best case, tuned points, and that the YAML scenario can be loaded as a launch scenario with tuned pitch values matching the JSON product. Add tests for invalid point-index parsing and tuned-scenario output write errors.

- [ ] **Step 2: Run CLI tests to verify they fail**

Run: `python -m pytest tests/astro_cli/test_cli.py::test_tune_launch_pitch_command_writes_json_and_tuned_scenario -v`

Expected: FAIL because `tune-launch-pitch` is not implemented.

### Task 4: CLI, Exports, and Docs

**Files:**
- Modify: `src/astro_cli/main.py`
- Modify: `src/astro_launch/__init__.py`
- Modify: `tests/test_imports.py`
- Modify: `README.md`

- [ ] **Step 1: Add CLI command**

The command accepts scenario path, `--point-indices`, `--initial-span-deg`, `--iterations`, `--refinement-factor`, `--altitude-weight`, `--velocity-weight`, `--output`, and optional `--tuned-scenario-output`. It writes JSON for the tuning result and YAML for the tuned scenario when requested.

- [ ] **Step 2: Update public exports**

Export the tuning models and `tune_pitch_program`.

- [ ] **Step 3: Update README**

Document the tuning command and state that it is a deterministic two-knot grid tuner, not a production optimizer.

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest tests/astro_cli/test_cli.py tests/test_imports.py -v`

Expected: PASS.

### Task 5: Full Verification and Commit

**Files:**
- All modified files

- [ ] **Step 1: Run full verification**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -v
python -m ruff check .
MYPY_CACHE_DIR=/tmp/astro-mypy-cache python -m mypy
```

Expected: all checks PASS.

- [ ] **Step 2: Run installed CLI smoke**

Run:

```bash
tmpdir=$(mktemp -d /tmp/astro-tune-smoke.XXXXXX)
PYTHONDONTWRITEBYTECODE=1 astro tune-launch-pitch examples/launch/pitch_program_two_stage.yaml --point-indices 2,3 --initial-span-deg 10 --iterations 2 --output "$tmpdir/tuning.json" --tuned-scenario-output "$tmpdir/tuned.yaml"
python - <<'PY' "$tmpdir/tuning.json" "$tmpdir/tuned.yaml"
import json, sys, yaml
payload = json.load(open(sys.argv[1], encoding="utf-8"))
tuned = yaml.safe_load(open(sys.argv[2], encoding="utf-8"))
assert len(payload["iterations"]) == 2
assert len(payload["tuned_points"]) == 2
assert tuned["guidance"]["mode"] == "pitch_program"
print(json.dumps({
    "scenario_id": payload["scenario_id"],
    "point_indices": payload["point_indices"],
    "best_score": payload["best_case"]["score"],
    "tuned_pitches": {
        str(point["point_index"]): point["tuned_pitch_deg"]
        for point in payload["tuned_points"]
    },
}, indent=2))
PY
```

Expected: command exits 0 and prints the smoke summary.

- [ ] **Step 3: Stage and commit**

Run:

```bash
git add README.md docs/superpowers/plans/2026-06-15-launch-pitch-tuning-implementation.md src/astro_cli/main.py src/astro_launch/__init__.py src/astro_launch/models.py src/astro_launch/targeting.py tests/astro_cli/test_cli.py tests/astro_launch/test_launch_targeting.py tests/test_imports.py
git diff --cached --check
git commit -m "feat: add launch pitch tuning"
```

Expected: commit succeeds and worktree is clean.

## Self-Review

- Spec coverage: the plan adds a two-knot tuner, trace product, tuned scenario output, CLI path, docs, and verification.
- Placeholder scan: no placeholder tasks remain.
- Type consistency: function names, model names, CLI flags, and JSON fields match across tasks.
