# Launch Pitch Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic launch targeting workflow that sweeps one pitch-program knot and reports altitude and velocity target miss.

**Architecture:** Keep the sweep above the local launch propagator as an analysis/product layer. The workflow copies a `LaunchScenario`, replaces one pitch-program point for each candidate pitch angle, runs `propagate_launch_local`, scores the resulting target miss, and returns a serializable Pydantic product.

**Tech Stack:** Python 3.12, Pydantic v2, Typer CLI, pytest, ruff, mypy.

---

## File Structure

- Create `src/astro_launch/targeting.py`: pitch-program sweep validation, scenario copy helpers, scoring, and result construction.
- Modify `src/astro_launch/models.py`: add serializable `LaunchPitchSweepCase` and `LaunchPitchSweepResult` models.
- Modify `src/astro_launch/__init__.py`: export sweep models and `sweep_pitch_program`.
- Modify `src/astro_cli/main.py`: add `sweep-launch-pitch` command and pitch-list parsing.
- Modify `tests/astro_launch/test_launch_targeting.py`: API behavior and validation tests.
- Modify `tests/astro_cli/test_cli.py`: CLI success and error tests.
- Modify `tests/test_imports.py`: launch export smoke test.
- Modify `README.md`: document the command and current-scope language.

### Task 1: Targeting API Tests

**Files:**
- Create: `tests/astro_launch/test_launch_targeting.py`
- Modify: none

- [ ] **Step 1: Write failing tests**

```python
import pytest

from astro_launch.targeting import sweep_pitch_program
from tests.astro_launch.helpers import make_launch_scenario, make_pitch_program_launch_scenario


def test_sweep_pitch_program_varies_one_knot_without_mutating_scenario() -> None:
    scenario = make_pitch_program_launch_scenario()

    result = sweep_pitch_program(
        scenario,
        point_index=3,
        pitch_values_deg=[10.0, 20.0, 30.0],
    )

    assert scenario.guidance.pitch_program[3].pitch_deg == 20.0
    assert result.scenario_id == scenario.scenario_id
    assert result.point_index == 3
    assert result.point_time_s == 110.0
    assert result.baseline_pitch_deg == 20.0
    assert [case.pitch_deg for case in result.cases] == [10.0, 20.0, 30.0]
    assert result.best_case.pitch_deg in {10.0, 20.0, 30.0}
    assert result.best_case.score == min(case.score for case in result.cases)
    assert all(case.final_downrange_km > 0.0 for case in result.cases)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"point_index": 0, "pitch_values_deg": [10.0]}, "cannot sweep first"),
        ({"point_index": 99, "pitch_values_deg": [10.0]}, "point_index"),
        ({"point_index": 3, "pitch_values_deg": []}, "at least one"),
        ({"point_index": 3, "pitch_values_deg": [-1.0]}, "between 0 and 90"),
    ],
)
def test_sweep_pitch_program_rejects_invalid_sweep_inputs(
    kwargs: dict[str, object],
    message: str,
) -> None:
    scenario = make_pitch_program_launch_scenario()

    with pytest.raises(ValueError, match=message):
        sweep_pitch_program(scenario, **kwargs)


def test_sweep_pitch_program_requires_pitch_program_guidance() -> None:
    with pytest.raises(ValueError, match="pitch_program guidance"):
        sweep_pitch_program(make_launch_scenario(), point_index=0, pitch_values_deg=[10.0])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/astro_launch/test_launch_targeting.py -v`

Expected: FAIL because `astro_launch.targeting` does not exist.

### Task 2: Targeting API Implementation

**Files:**
- Modify: `src/astro_launch/models.py`
- Create: `src/astro_launch/targeting.py`
- Modify: `src/astro_launch/__init__.py`
- Test: `tests/astro_launch/test_launch_targeting.py`

- [ ] **Step 1: Add Pydantic product models**

Add `LaunchPitchSweepCase` with pitch, score, final sample metrics, and target miss fields. Add `LaunchPitchSweepResult` with scenario, swept point metadata, case list, best case, backend, and metadata.

- [ ] **Step 2: Implement `sweep_pitch_program`**

The function validates pitch-program guidance, rejects sweeping the first vertical liftoff point, copies the scenario for each candidate pitch using `model_copy`, runs `propagate_launch_local`, scores `abs(altitude_miss_km) * altitude_weight + abs(velocity_miss_km_s) * velocity_weight`, and returns the best case.

- [ ] **Step 3: Run tests to verify they pass**

Run: `python -m pytest tests/astro_launch/test_launch_targeting.py -v`

Expected: PASS.

### Task 3: CLI Tests

**Files:**
- Modify: `tests/astro_cli/test_cli.py`

- [ ] **Step 1: Add failing CLI tests**

Add helper `_write_pitch_program_launch_scenario`, then test:

```python
def test_sweep_launch_pitch_command_writes_json(tmp_path: Path) -> None:
    scenario_path = tmp_path / "pitch.yaml"
    output = tmp_path / "sweep.json"
    _write_pitch_program_launch_scenario(scenario_path)

    result = runner.invoke(
        app,
        [
            "sweep-launch-pitch",
            str(scenario_path),
            "--point-index",
            "3",
            "--pitch-deg-values",
            "10,20,30",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "wrote launch pitch sweep" in result.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["scenario_id"] == "pitch-program-two-stage"
    assert payload["point_index"] == 3
    assert [case["pitch_deg"] for case in payload["cases"]] == [10.0, 20.0, 30.0]
    assert payload["best_case"]["pitch_deg"] in [10.0, 20.0, 30.0]
```

Also test invalid pitch lists and output write failure.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/astro_cli/test_cli.py::test_sweep_launch_pitch_command_writes_json -v`

Expected: FAIL because the command does not exist.

### Task 4: CLI, Exports, and Docs

**Files:**
- Modify: `src/astro_cli/main.py`
- Modify: `src/astro_launch/__init__.py`
- Modify: `tests/test_imports.py`
- Modify: `README.md`

- [ ] **Step 1: Add `sweep-launch-pitch` command**

The command accepts a launch scenario path, `--point-index`, `--pitch-deg-values`, `--altitude-weight`, `--velocity-weight`, and `--output`. It parses comma-separated pitch values, calls `sweep_pitch_program`, writes formatted JSON, and reports validation failures with exit code 2.

- [ ] **Step 2: Update exports and import tests**

Export `LaunchPitchSweepCase`, `LaunchPitchSweepResult`, and `sweep_pitch_program`.

- [ ] **Step 3: Update README**

Document the sweep command and describe it as a deterministic targeting-analysis workflow over repeated local launch propagations.

- [ ] **Step 4: Run focused CLI/import tests**

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
tmpdir=$(mktemp -d /tmp/astro-sweep-smoke.XXXXXX)
PYTHONDONTWRITEBYTECODE=1 astro sweep-launch-pitch examples/launch/pitch_program_two_stage.yaml --point-index 3 --pitch-deg-values 10,20,30 --output "$tmpdir/sweep.json"
python - <<'PY' "$tmpdir/sweep.json"
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
assert len(payload["cases"]) == 3
assert payload["best_case"]["pitch_deg"] in [10.0, 20.0, 30.0]
print(json.dumps({
    "scenario_id": payload["scenario_id"],
    "point_index": payload["point_index"],
    "best_pitch_deg": payload["best_case"]["pitch_deg"],
    "best_score": payload["best_case"]["score"],
}, indent=2))
PY
```

Expected: command exits 0 and prints the smoke summary.

- [ ] **Step 3: Stage and commit**

Run:

```bash
git add README.md docs/superpowers/plans/2026-06-15-launch-pitch-sweep-implementation.md src/astro_cli/main.py src/astro_launch/__init__.py src/astro_launch/models.py src/astro_launch/targeting.py tests/astro_cli/test_cli.py tests/astro_launch/test_launch_targeting.py tests/test_imports.py
git diff --cached --check
git commit -m "feat: add launch pitch sweep"
```

Expected: commit succeeds and worktree is clean.

## Self-Review

- Spec coverage: the plan adds sweep targeting for launch, returns target-miss metrics, exposes it through CLI, and documents usage.
- Placeholder scan: no placeholder tasks remain.
- Type consistency: product names, field names, and function names match across tests, implementation, CLI, docs, and exports.
