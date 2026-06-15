# Orekit Measurement And OD Workflows Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow synthetic measurement generation and batch orbit determination workflows to use the selected propagation backend, including Orekit when available.

**Architecture:** Keep measurement geometry and least-squares estimation in `astro_od`, but inject propagation through a small backend dispatcher. This creates a real Orekit-backed OD path through `propagate_orekit` without committing to Orekit's native `BatchLSEstimator` API before Java/Orekit data can be validated locally.

**Tech Stack:** Python 3.12, Pydantic v2, NumPy, SciPy, Typer, pytest, optional `orekit-jpype`.

---

## Scope

In scope:

- Shared propagation dispatcher for `"local"` and `"orekit"`.
- `astro synth-measurements --backend local|orekit`.
- `astro estimate --backend local|orekit`.
- `astro estimate-measurements --backend local|orekit`.
- `estimate_initial_state(..., backend="local"|"orekit")`.
- Metadata showing which propagation backend powered the OD residual model.
- Tests proving Orekit unavailable behavior and fake Orekit-backed OD dispatch.

Out of scope:

- Orekit native `BatchLSEstimator`.
- Orekit `Range`/`RangeRate` measurement objects.
- Orekit numerical J2/drag/SRP force models.

## Task 1: Propagation Backend Dispatcher

**Files:**

- Create: `src/astro_dynamics/backends.py`
- Modify: `src/astro_dynamics/__init__.py`
- Test: `tests/astro_dynamics/test_backends.py`
- Test: `tests/test_imports.py`

- [ ] Write tests for local dispatch, Orekit dispatch via monkeypatch, and unsupported backend errors.
- [ ] Implement `propagate_with_backend(scenario, backend)`.
- [ ] Export the dispatcher.
- [ ] Run `python -m pytest tests/astro_dynamics/test_backends.py tests/test_imports.py -v`.
- [ ] Commit with `git commit -m "feat: add propagation backend dispatcher"`.

## Task 2: Backend-Aware Estimation Core

**Files:**

- Modify: `src/astro_od/estimation.py`
- Test: `tests/astro_od/test_estimation.py`

- [ ] Add failing tests for `estimate_initial_state(..., backend="orekit")` using a monkeypatched dispatcher.
- [ ] Refactor `_validate_measurements` and `residual_vector` to use an injected propagator.
- [ ] Add `backend` and optional `propagator` arguments to `estimate_initial_state`.
- [ ] Update result metadata to include `propagation_backend` and estimator label.
- [ ] Run `python -m pytest tests/astro_od/test_estimation.py -v`.
- [ ] Commit with `git commit -m "feat: support backend-aware orbit determination"`.

## Task 3: CLI Backend Options For Measurement And OD Workflows

**Files:**

- Modify: `src/astro_cli/main.py`
- Modify: `tests/astro_cli/test_cli.py`

- [ ] Add tests for `synth-measurements --backend orekit`, `estimate --backend orekit`, and `estimate-measurements --backend orekit`.
- [ ] Wire `synth-measurements` truth propagation through `propagate_with_backend`.
- [ ] Pass `backend` into `estimate_initial_state`.
- [ ] Catch `UnsupportedBackendError` in these commands and return exit code 2.
- [ ] Run `python -m pytest tests/astro_cli/test_cli.py -v`.
- [ ] Commit with `git commit -m "feat: add orekit backend to measurement and od cli"`.

## Task 4: Documentation And Verification

**Files:**

- Modify: `README.md`
- Modify: `docs/superpowers/plans/2026-06-15-roadmap-goals-implementation-plan.md`

- [ ] Document backend-aware measurement and OD commands.
- [ ] Mark Goal 2's first slice as implemented and leave native Orekit estimator/numerical force models as the next live-Java-dependent slice.
- [ ] Run:

```bash
python -m pytest -q
python -m ruff check .
python -m mypy
astro synth-measurements examples/scenarios/leo_two_station_od.yaml --backend local --output /tmp/astro-measurements.json
astro estimate-measurements examples/scenarios/leo_two_station_od.yaml examples/measurements/leo_two_station_od_measurements.json --backend local --output /tmp/astro-estimate.json
astro synth-measurements examples/scenarios/leo_two_station_od.yaml --backend orekit --output /tmp/astro-orekit-measurements.json
```

Expected: tests, lint, mypy, and local CLI smokes pass; Orekit CLI smoke exits 2 on this machine until Java and `orekit-jpype` are installed.

- [ ] Commit with `git commit -m "docs: document backend-aware od workflows"`.

## Self-Review

Spec coverage:

- Adds Orekit-backed synthetic measurements and batch OD through the suite estimator and Orekit propagation adapter.
- Preserves local workflows as default.
- Does not claim native Orekit `BatchLSEstimator` or high-fidelity numerical force models until Java/Orekit runtime is available for live validation.

Placeholder scan:

- All files, commands, and task outputs are explicit.

Type consistency:

- Public OD product remains `EstimateResult`.
- Public trajectory product remains `Trajectory`.
- Backend selection remains string-based at the CLI boundary and is validated in the dispatcher.

