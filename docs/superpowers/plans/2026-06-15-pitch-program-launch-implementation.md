# Pitch Program Launch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local 2D pitch-program launch baseline that produces downrange motion and horizontal insertion velocity while preserving the existing vertical launch workflow.

**Architecture:** Keep this in the local launch baseline as a deterministic product-contract improvement, not a production ascent simulator. The launch model gains a typed `pitch_program` guidance option; `astro_launch.local` integrates a radial/east 2D state and still returns the same `LaunchTrajectory` product and handoff-compatible insertion state.

**Tech Stack:** Python 3.12, Pydantic v2, PyYAML, Typer, pytest, Ruff, mypy.

---

### Task 1: Pitch Program Guidance Model

**Files:**
- Modify: `src/astro_launch/models.py`
- Modify: `src/astro_launch/__init__.py`
- Modify: `tests/astro_launch/helpers.py`
- Test: `tests/astro_launch/test_launch_models.py`

- [ ] **Step 1: Write failing model tests**

Add tests asserting that `GuidanceConfig(mode="pitch_program", pitch_program=[...])` accepts strictly increasing time knots, rejects missing or unsorted pitch knots, and that `LaunchScenario.insertion_state_from_local_state(...)` maps radial velocity to the local radial vector and horizontal velocity to the local east vector.

Run: `python -m pytest tests/astro_launch/test_launch_models.py -v`

Expected: FAIL because `PitchProgramPoint`, pitch-program validation, and local-state insertion are missing.

- [ ] **Step 2: Implement model changes**

Add `PitchProgramPoint`, extend `GuidanceConfig`, add `LaunchScenario.insertion_state_from_local_state(...)`, and add optional `radial_velocity_km_s`, `horizontal_velocity_km_s`, and `flight_path_angle_deg` fields to `LaunchTrajectorySample`.

- [ ] **Step 3: Export new model**

Export `PitchProgramPoint` from `src/astro_launch/__init__.py`.

- [ ] **Step 4: Verify task**

Run: `python -m pytest tests/astro_launch/test_launch_models.py -v`

Expected: PASS.

### Task 2: Local 2D Launch Dynamics

**Files:**
- Modify: `src/astro_launch/local.py`
- Test: `tests/astro_launch/test_launch_local.py`

- [ ] **Step 1: Write failing local dynamics tests**

Add tests asserting that vertical guidance keeps zero downrange and horizontal velocity, while pitch-program guidance produces positive downrange, positive horizontal velocity, nonzero local east velocity in the Cartesian sample state, and `metadata["model"] == "pitch_program_2d"`.

Run: `python -m pytest tests/astro_launch/test_launch_local.py -v`

Expected: FAIL because local propagation is still radial-only.

- [ ] **Step 2: Implement 2D integration**

Refactor `propagate_launch_local` to track altitude, downrange, radial velocity, and horizontal velocity. Resolve pitch angle from guidance knots with linear interpolation; pitch is degrees above local horizontal, so `90` is vertical and `0` is horizontal. Apply thrust, gravity, and drag in radial/east components.

- [ ] **Step 3: Verify task**

Run: `python -m pytest tests/astro_launch/test_launch_local.py -v`

Expected: PASS.

### Task 3: Example, Docs, and Smoke Surface

**Files:**
- Add: `examples/launch/pitch_program_two_stage.yaml`
- Modify: `tests/astro_launch/test_launch_io.py`
- Modify: `README.md`

- [ ] **Step 1: Add failing example loader test**

Add a test that `load_launch_scenario(Path("examples/launch/pitch_program_two_stage.yaml"))` has `guidance.mode == "pitch_program"` and a nonempty pitch program.

Run: `python -m pytest tests/astro_launch/test_launch_io.py::test_load_pitch_program_launch_scenario -v`

Expected: FAIL because the example file is missing.

- [ ] **Step 2: Add example YAML and docs**

Create the pitch-program example by reusing the two-stage vehicle with pitch knots `[90, 75, 45, 20, 5]`. Update README commands to show `astro launch examples/launch/pitch_program_two_stage.yaml`.

- [ ] **Step 3: Verify task**

Run: `python -m pytest tests/astro_launch tests/astro_cli/test_cli.py::test_launch_command_writes_json -v`

Expected: PASS.

### Task 4: Full Verification and Commit

**Files:**
- All modified launch, docs, tests, and example files.

- [ ] **Step 1: Run full verification**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -v
python -m ruff check .
MYPY_CACHE_DIR=/tmp/astro-mypy-cache python -m mypy
tmpdir=$(mktemp -d /tmp/astro-pitch-smoke.XXXXXX)
astro launch examples/launch/pitch_program_two_stage.yaml --output "$tmpdir/launch.json"
astro handoff-launch "$tmpdir/launch.json" --output "$tmpdir/orbit.yaml" --duration-s 600 --step-s 60
astro propagate "$tmpdir/orbit.yaml" --output "$tmpdir/trajectory.json"
```

Expected: tests, lint, typecheck, and CLI smoke pass.

- [ ] **Step 2: Commit**

Run:

```bash
git add README.md src/astro_launch tests/astro_launch examples/launch/pitch_program_two_stage.yaml docs/superpowers/plans/2026-06-15-pitch-program-launch-implementation.md
git commit -m "feat: add pitch program launch baseline"
```
