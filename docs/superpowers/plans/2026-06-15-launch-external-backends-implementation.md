# Launch External Backends Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add honest optional launch backend surfaces for RocketPy direct simulation and Dymos/OpenMDAO ascent optimization while preserving Astro Suite product schemas.

**Architecture:** Keep `LaunchScenario` and `LaunchTrajectory` as the public boundary. Add optional runtime gates and smoke commands first, route `astro launch` through a backend dispatcher, then add RocketPy/Dymos adapters that return suite products and fail clearly when live dependencies or backend-specific configuration are unavailable.

**Tech Stack:** Python 3.12, Pydantic v2, Typer, pytest, optional RocketPy 1.12.x, optional Dymos 1.15.x, optional OpenMDAO 3.44.x.

---

## Scope

In scope:

- Optional extras for launch simulation and optimization dependencies.
- Runtime gates and smoke commands for RocketPy and Dymos/OpenMDAO.
- `astro_launch.backends.propagate_launch_with_backend` for `local` and `rocketpy`.
- `astro optimize-launch` dispatcher for `local` pitch-program tuning and `dymos` optimization.
- Product-preserving RocketPy and Dymos adapter modules with explicit unsupported-feature errors until a scenario contains the required external-engine configuration.

Out of scope for this slice:

- Pretending the current aggregate launch schema is enough for high-fidelity RocketPy vehicle construction.
- Full RocketPy motor/rocket geometry authoring UI.
- Full Dymos trajectory transcription and path-constraint library beyond the current stage-aware
  vertical phase, original and optimized pitch-program control metadata, and pitch-bound
  constraints.

## File Map

- Create `src/astro_backends/rocketpy/runtime.py`: import/version gate for `rocketpy`.
- Create `src/astro_backends/rocketpy/smoke.py`: structured smoke result.
- Create `src/astro_backends/rocketpy/simulation.py`: RocketPy launch adapter boundary.
- Create `src/astro_backends/rocketpy/__init__.py`: public exports.
- Create `src/astro_backends/dymos/runtime.py`: import/version gate for `dymos` and `openmdao`.
- Create `src/astro_backends/dymos/smoke.py`: structured smoke result.
- Create `src/astro_backends/dymos/optimization.py`: Dymos optimization adapter boundary.
- Create `src/astro_backends/dymos/__init__.py`: public exports.
- Create `src/astro_launch/backends.py`: launch backend dispatcher.
- Modify `src/astro_cli/main.py`: smoke commands, backend dispatch, `optimize-launch`.
- Modify `pyproject.toml`: `launch` and `optimization` optional extras.
- Modify `README.md` and roadmap plan docs.

## Task 1: Optional Runtime Gates

**Files:**

- Create: `src/astro_backends/rocketpy/runtime.py`
- Create: `src/astro_backends/rocketpy/smoke.py`
- Create: `src/astro_backends/rocketpy/__init__.py`
- Create: `src/astro_backends/dymos/runtime.py`
- Create: `src/astro_backends/dymos/smoke.py`
- Create: `src/astro_backends/dymos/__init__.py`
- Modify: `pyproject.toml`
- Test: `tests/astro_backends/test_rocketpy_runtime.py`
- Test: `tests/astro_backends/test_rocketpy_smoke.py`
- Test: `tests/astro_backends/test_dymos_runtime.py`
- Test: `tests/astro_backends/test_dymos_smoke.py`

- [x] Add failing tests for missing distributions and forced-unavailable smoke results.
- [x] Implement `load_rocketpy_runtime()` returning `RocketPyRuntime(package, version, module)`.
- [x] Implement `run_rocketpy_smoke()` returning `{available, package, version, message}`.
- [x] Implement `load_dymos_runtime()` returning `DymosRuntime(dymos_version, openmdao_version, dymos_module, openmdao_module)`.
- [x] Implement `run_dymos_smoke()` returning `{available, package, version, openmdao_version, message}`.
- [x] Add optional extras:

```toml
launch = [
  "rocketpy>=1.12,<2",
]
optimization = [
  "dymos>=1.15,<2",
  "openmdao>=3.44,<4",
]
```

- [x] Run `python -m pytest tests/astro_backends/test_rocketpy_runtime.py tests/astro_backends/test_rocketpy_smoke.py tests/astro_backends/test_dymos_runtime.py tests/astro_backends/test_dymos_smoke.py -v`.
- [x] Commit with `git commit -m "feat: add launch backend runtime gates"`.

## Task 2: Launch Backend Dispatcher

**Files:**

- Create: `src/astro_launch/backends.py`
- Modify: `src/astro_launch/__init__.py`
- Modify: `src/astro_cli/main.py`
- Test: `tests/astro_launch/test_launch_backends.py`
- Test: `tests/astro_cli/test_cli.py`
- Test: `tests/test_imports.py`

- [x] Add failing tests for `propagate_launch_with_backend(scenario, "local")`, unsupported backend errors, and CLI `launch --backend rocketpy` dispatch.
- [x] Implement:

```python
def propagate_launch_with_backend(scenario: LaunchScenario, backend: str) -> LaunchTrajectory:
    if backend == "local":
        return propagate_launch_local(scenario)
    if backend == "rocketpy":
        return propagate_launch_rocketpy(scenario)
    raise UnsupportedBackendError(f"unsupported launch backend: {backend}")
```

- [x] Update `astro launch` to call the dispatcher and catch `UnsupportedBackendError`.
- [x] Run focused launch backend and CLI tests.
- [x] Commit with `git commit -m "feat: dispatch launch propagation backends"`.

## Task 3: RocketPy Product Adapter Boundary

**Files:**

- Create: `src/astro_backends/rocketpy/simulation.py`
- Test: `tests/astro_backends/test_rocketpy_simulation.py`
- Modify: `README.md`
- Modify: `docs/superpowers/plans/2026-06-15-roadmap-goals-implementation-plan.md`

- [x] Add tests showing missing RocketPy raises `UnsupportedBackendError` with `install astro-suite[launch]`.
- [x] Add tests with a fake runtime and fake `flight_runner` that returns a `LaunchTrajectory` with `backend == "rocketpy"`.
- [x] Implement `propagate_launch_rocketpy(scenario, runtime_loader=load_rocketpy_runtime, flight_runner=None)`.
- [x] If no `flight_runner` is supplied, raise an explicit error that RocketPy live simulation requires backend-specific rocket/motor configuration not present in the current aggregate `LaunchScenario`.
- [x] Document that Goal 4's first RocketPy slice is a product-preserving adapter boundary, not a complete motor geometry authoring layer.
- [x] Run focused RocketPy adapter tests.
- [x] Commit with `git commit -m "feat: add rocketpy launch adapter boundary"`.

## Task 4: Dymos Optimization Command Boundary

**Files:**

- Create: `src/astro_backends/dymos/optimization.py`
- Modify: `src/astro_cli/main.py`
- Test: `tests/astro_backends/test_dymos_optimization.py`
- Test: `tests/astro_cli/test_cli.py`
- Modify: `README.md`

- [x] Add tests for `optimize_launch_dymos` missing-runtime diagnostics.
- [x] Add CLI test for `astro optimize-launch --backend local` using current pitch tuning.
- [x] Add CLI test for `astro optimize-launch --backend dymos` dispatch by monkeypatching the Dymos optimizer.
- [x] Implement local optimizer command using `tune_pitch_program`.
- [x] Implement Dymos adapter boundary that records runtime availability, runs the current
  stage-aware vertical phase model, and preserves suite pitch-program tuning products with
  original and optimized pitch-program control metadata.
- [x] Run focused Dymos and CLI tests.
- [x] Commit with `git commit -m "feat: add launch optimization command boundary"`.

## Task 5: Verification And Roadmap Update

**Files:**

- Modify: `README.md`
- Modify: `docs/superpowers/plans/2026-06-15-roadmap-goals-implementation-plan.md`
- Modify: this plan.

- [x] Run:

```bash
python -m pytest -q
python -m ruff check .
python -m mypy
astro rocketpy-smoke
astro dymos-smoke
astro launch examples/launch/pitch_program_two_stage.yaml --backend local --output /tmp/astro-launch.json
astro optimize-launch examples/launch/pitch_program_two_stage.yaml --backend local --point-indices 2,3 --iterations 1 --output /tmp/astro-optimized-launch.json
```

- [x] Mark completed checklist items.
- [x] Commit with `git commit -m "docs: update launch external backend roadmap"`.

## Self-Review

Spec coverage:

- Covers Goal 4's package/runtime gates, launch backend dispatch, RocketPy boundary, Dymos boundary, CLI surfaces, and docs.
- Does not fake full RocketPy geometry or a full Dymos pitch-program multistage transcription from
  an insufficient aggregate launch schema.

Placeholder scan:

- No `TBD` placeholders. Deferred work is explicitly described as requiring backend-specific configuration and phase modeling.

Type consistency:

- `LaunchScenario`, `LaunchTrajectory`, `LaunchPitchTuningResult`, and `UnsupportedBackendError` remain the product and error boundaries.
