# Launch Orbit Handoff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert a launch trajectory product into a normal orbital propagation scenario, then prove the scenario can run through the existing `astro propagate` workflow.

**Architecture:** Keep launch and orbital propagation coupled through products, not internal state. `astro_launch` owns the handoff adapter from `LaunchTrajectory.insertion_state` to `astro_core.Scenario`; `astro_cli` only handles file IO, options, and writing YAML.

**Tech Stack:** Python 3.12, Pydantic v2, PyYAML, Typer, pytest, Ruff, mypy.

---

### Task 1: Launch Trajectory Loader

**Files:**
- Modify: `src/astro_launch/io.py`
- Modify: `src/astro_launch/__init__.py`
- Test: `tests/astro_launch/test_launch_io.py`

- [ ] **Step 1: Write the failing test**

Add a test that creates a launch trajectory with `propagate_launch_local(make_launch_scenario())`, writes `trajectory.model_dump_json()`, then calls `load_launch_trajectory(path)` and asserts the loaded `insertion_state` and sample count match.

Run: `python -m pytest tests/astro_launch/test_launch_io.py::test_load_launch_trajectory_reads_json_product -v`

Expected: FAIL with `ImportError` or missing `load_launch_trajectory`.

- [ ] **Step 2: Implement the loader**

Add `load_launch_trajectory(path: Path | str) -> LaunchTrajectory` in `src/astro_launch/io.py`. It should read UTF-8 JSON, require a JSON object, validate with `LaunchTrajectory.model_validate`, and wrap read/parse/validation problems in `InvalidScenarioError` with messages naming "launch trajectory file".

- [ ] **Step 3: Export the loader**

Export `load_launch_trajectory` from `src/astro_launch/__init__.py`.

- [ ] **Step 4: Verify task**

Run: `python -m pytest tests/astro_launch/test_launch_io.py -v`

Expected: PASS.

### Task 2: Handoff Adapter

**Files:**
- Create: `src/astro_launch/handoff.py`
- Modify: `src/astro_launch/__init__.py`
- Test: `tests/astro_launch/test_launch_handoff.py`

- [ ] **Step 1: Write failing adapter tests**

Add tests for `launch_trajectory_to_orbit_scenario(...)` asserting:
- scenario id can be overridden or defaults to `<launch-scenario-id>-insertion`
- initial state equals the launch `insertion_state`
- spacecraft mass defaults to the final launch sample mass
- propagation duration and step are configurable
- metadata records source launch product details

Run: `python -m pytest tests/astro_launch/test_launch_handoff.py -v`

Expected: FAIL with missing `astro_launch.handoff`.

- [ ] **Step 2: Implement the adapter**

Implement `launch_trajectory_to_orbit_scenario(trajectory, *, duration_s, step_s, spacecraft_name="launch-payload", spacecraft_mass_kg=None, area_m2=2.5, drag_coefficient=2.2, reflectivity_coefficient=1.3, gravity=ForceModelName.TWO_BODY, scenario_id=None, description=None) -> Scenario`.

- [ ] **Step 3: Export the adapter**

Export `launch_trajectory_to_orbit_scenario` from `src/astro_launch/__init__.py`.

- [ ] **Step 4: Verify task**

Run: `python -m pytest tests/astro_launch/test_launch_handoff.py -v`

Expected: PASS.

### Task 3: CLI Workflow and Docs

**Files:**
- Modify: `src/astro_cli/main.py`
- Modify: `README.md`
- Test: `tests/astro_cli/test_cli.py`

- [ ] **Step 1: Write failing CLI tests**

Add CLI tests for:
- `astro handoff-launch launch.json --output insertion.yaml --duration-s 600 --step-s 60`
- the resulting YAML loads with `load_scenario`
- `propagate_local(load_scenario(output))` returns samples
- unsupported gravity reports exit code 2
- output write errors are reported

Run: `python -m pytest tests/astro_cli/test_cli.py::test_handoff_launch_command_writes_orbit_scenario -v`

Expected: FAIL because `handoff-launch` does not exist.

- [ ] **Step 2: Implement command**

Add `astro handoff-launch` to `src/astro_cli/main.py`. It should read a launch trajectory JSON product, convert it to an orbital `Scenario`, write YAML with `yaml.safe_dump(scenario.model_dump(mode="json"), sort_keys=False)`, and report `wrote orbit scenario`.

- [ ] **Step 3: Update README**

Document the `launch -> handoff-launch -> propagate` command sequence and explain that this is a product boundary, not a special propagation backend.

- [ ] **Step 4: Verify task**

Run: `python -m pytest tests/astro_cli/test_cli.py::test_handoff_launch_command_writes_orbit_scenario tests/astro_launch -v`

Expected: PASS.

### Task 4: Full Verification and Commit

**Files:**
- All modified handoff, CLI, docs, and tests.

- [ ] **Step 1: Run full verification**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -v
python -m ruff check .
MYPY_CACHE_DIR=/tmp/astro-mypy-cache python -m mypy
tmpdir=$(mktemp -d /tmp/astro-handoff-smoke.XXXXXX)
astro launch examples/launch/vertical_two_stage.yaml --output "$tmpdir/launch.json"
astro handoff-launch "$tmpdir/launch.json" --output "$tmpdir/orbit.yaml" --duration-s 600 --step-s 60
astro propagate "$tmpdir/orbit.yaml" --output "$tmpdir/trajectory.json"
```

Expected: tests, lint, typecheck, and CLI smoke pass.

- [ ] **Step 2: Commit**

Run:

```bash
git add README.md src/astro_cli/main.py src/astro_launch tests/astro_cli/test_cli.py tests/astro_launch docs/superpowers/plans/2026-06-15-launch-orbit-handoff-implementation.md
git commit -m "feat: add launch orbit handoff"
```
