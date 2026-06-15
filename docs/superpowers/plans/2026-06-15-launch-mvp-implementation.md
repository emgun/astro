# Launch MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the first launch/ascent MVP slice: typed launch scenarios, local deterministic ascent propagation, insertion-state handoff, CLI execution, and a reference example.

**Architecture:** Keep launch first-class in `astro_launch` instead of folding ascent physics into orbital propagation or OD. The local backend is a deterministic sanity baseline for data flow, staging, mass depletion, drag, events, and `OrbitState` handoff; RocketPy and Dymos remain future adapters.

**Tech Stack:** Python 3.12, Pydantic v2, NumPy, PyYAML, Typer, pytest, Ruff, mypy.

---

### Task 1: Launch Models and YAML Loader

**Files:**
- Create: `src/astro_launch/models.py`
- Create: `src/astro_launch/io.py`
- Create: `src/astro_launch/__init__.py`
- Modify: `pyproject.toml`
- Test: `tests/astro_launch/test_launch_models.py`
- Test: `tests/astro_launch/test_launch_io.py`

- [ ] **Step 1: Write failing model tests**

Run: `python -m pytest tests/astro_launch/test_launch_models.py -v`

Expected first failure: import failure for `astro_launch`.

- [ ] **Step 2: Implement minimal Pydantic launch models**

Define `LaunchSite`, `LaunchEngine`, `LaunchStage`, `LaunchVehicle`, `AtmosphereConfig`, `GuidanceConfig`, `TargetOrbit`, `LaunchPropagationConfig`, `LaunchScenario`, `LaunchEvent`, `LaunchTrajectorySample`, and `LaunchTrajectory`.

- [ ] **Step 3: Add loader and package registration**

Define `load_launch_scenario(path)` in `astro_launch.io`, export public types in `astro_launch.__init__`, and include `src/astro_launch` in Hatch and mypy package lists.

- [ ] **Step 4: Verify task**

Run: `python -m pytest tests/astro_launch/test_launch_models.py tests/astro_launch/test_launch_io.py -v`

Expected: all launch model and IO tests pass.

### Task 2: Local Launch Baseline

**Files:**
- Create: `src/astro_launch/local.py`
- Test: `tests/astro_launch/test_launch_local.py`

- [ ] **Step 1: Write failing local propagation tests**

Run: `python -m pytest tests/astro_launch/test_launch_local.py -v`

Expected first failure: `propagate_launch_local` import failure.

- [ ] **Step 2: Implement deterministic vertical ascent**

Use a simple radial 1D ascent model with gravity, optional exponential atmosphere, drag, constant-thrust engines, propellant mass depletion, dry-mass staging, stage events, and a final `OrbitState` insertion product.

- [ ] **Step 3: Verify task**

Run: `python -m pytest tests/astro_launch/test_launch_local.py -v`

Expected: launch local backend tests pass.

### Task 3: CLI and Example

**Files:**
- Modify: `src/astro_cli/main.py`
- Add: `examples/launch/vertical_two_stage.yaml`
- Modify: `README.md`
- Test: `tests/astro_cli/test_cli.py`

- [ ] **Step 1: Write failing CLI test**

Run: `python -m pytest tests/astro_cli/test_cli.py::test_launch_command_writes_json -v`

Expected first failure: Typer reports no `launch` command.

- [ ] **Step 2: Implement CLI command and example**

Add `astro launch examples/launch/vertical_two_stage.yaml --output launch.json`, load the launch scenario, run the local backend, and write the `LaunchTrajectory` JSON product.

- [ ] **Step 3: Verify task**

Run: `python -m pytest tests/astro_cli/test_cli.py::test_launch_command_writes_json tests/astro_launch -v`

Expected: CLI and launch tests pass.

### Task 4: Full Verification and Commit

**Files:**
- All modified launch, CLI, docs, and packaging files.

- [ ] **Step 1: Run full verification**

Run:

```bash
python -m pytest -v
python -m ruff check .
python -m mypy
astro launch examples/launch/vertical_two_stage.yaml --output /tmp/astro-launch.json
```

Expected: tests, lint, typecheck, and installed CLI smoke pass.

- [ ] **Step 2: Commit**

Run:

```bash
git add README.md pyproject.toml src/astro_cli/main.py src/astro_launch tests/astro_launch tests/astro_cli/test_cli.py examples/launch/vertical_two_stage.yaml docs/superpowers/plans/2026-06-15-launch-mvp-implementation.md
git commit -m "feat: add launch ascent mvp"
```
