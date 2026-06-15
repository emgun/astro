# High-Fidelity And Research Backends Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add product-preserving boundaries for Tudat cross-check propagation, JAX research propagation, and Nyx evaluation without weakening the operational Orekit/local validation spine.

**Architecture:** Keep `Scenario`, `Trajectory`, and `MonteCarloResult` as suite-owned products. Add optional runtime gates and smoke commands first, dispatch recognized research backends through adapter boundaries, and provide a built-in JAX two-body batch runner while keeping richer dynamics behind validated runner implementations.

**Tech Stack:** Python 3.12, Pydantic v2, Typer, pytest, optional JAX/JAXLIB, external TudatPy installation, documentation for Nyx/ANISE evaluation.

---

## Scope

In scope:

- Optional runtime gates and smoke commands for TudatPy and JAX.
- Tudat propagation adapter boundary that returns `Trajectory` through the existing propagation dispatcher.
- JAX research propagation adapter boundary that returns seeded `MonteCarloResult`.
- `astro research-propagate --backend jax` command.
- Nyx/ANISE evaluation document with a yes/no decision gate.

Out of scope:

- Full Tudat environment/body setup and SPICE kernel management.
- Treating JAX as an operational frame/time/force-model authority.
- Adding Nyx as a hard dependency before evaluation.

## Task 1: Tudat And JAX Runtime Gates

**Files:**

- Create: `src/astro_backends/tudat/runtime.py`
- Create: `src/astro_backends/tudat/smoke.py`
- Create: `src/astro_backends/tudat/__init__.py`
- Create: `src/astro_backends/jax/runtime.py`
- Create: `src/astro_backends/jax/smoke.py`
- Create: `src/astro_backends/jax/__init__.py`
- Modify: `src/astro_cli/main.py`
- Modify: `pyproject.toml`
- Test: `tests/astro_backends/test_tudat_runtime.py`
- Test: `tests/astro_backends/test_tudat_smoke.py`
- Test: `tests/astro_backends/test_jax_runtime.py`
- Test: `tests/astro_backends/test_jax_smoke.py`
- Test: `tests/astro_cli/test_cli.py`

- [x] Add failing tests for missing distributions, fake runtimes, and CLI smoke commands.
- [x] Implement `load_tudat_runtime()` with actionable diagnostics. Note: TudatPy may not be available on PyPI; diagnostics should not promise `pip install` when unavailable.
- [x] Implement `run_tudat_smoke()`.
- [x] Implement `load_jax_runtime()` for `jax` and `jaxlib`.
- [x] Implement `run_jax_smoke()`.
- [x] Add optional extra:

```toml
research = [
  "jax>=0.4.38,<1",
  "jaxlib>=0.4.38,<1",
]
```

- [x] Add `astro tudat-smoke` and `astro jax-smoke`.
- [x] Run focused runtime/smoke/CLI tests.
- [x] Commit with `git commit -m "feat: add research backend runtime gates"`.

## Task 2: Tudat Propagation Boundary

**Files:**

- Create: `src/astro_backends/tudat/propagation.py`
- Modify: `src/astro_dynamics/backends.py`
- Test: `tests/astro_backends/test_tudat_propagation.py`
- Test: `tests/astro_dynamics/test_backends.py`
- Test: `tests/astro_cli/test_cli.py`

- [x] Add tests showing missing Tudat raises `UnsupportedBackendError`.
- [x] Add tests with a fake `tudat_runner` returning a suite `Trajectory`.
- [x] Add `propagate_tudat(scenario, runtime_loader=load_tudat_runtime, tudat_runner=None)`.
- [x] Update propagation dispatcher to recognize `backend == "tudat"`.
- [x] Run focused propagation tests.
- [x] Commit with `git commit -m "feat: add tudat propagation boundary"`.

## Task 3: JAX Research Propagation Boundary

**Files:**

- Create: `src/astro_backends/jax/propagation.py`
- Modify: `src/astro_cli/main.py`
- Test: `tests/astro_backends/test_jax_propagation.py`
- Test: `tests/astro_cli/test_cli.py`

- [x] Add tests showing missing JAX raises `UnsupportedBackendError`.
- [x] Add tests with a fake `research_runner` returning `MonteCarloResult`.
- [x] Implement `research_propagate_jax(scenario, cases, position_sigma_km, velocity_sigma_km_s, seed, runtime_loader=load_jax_runtime, research_runner=None)` with a built-in vectorized two-body runner.
- [x] Add `astro research-propagate --backend jax`.
- [x] Run focused research propagation tests.
- [x] Commit with `git commit -m "feat: add jax research propagation boundary"`.

## Task 4: Nyx Evaluation Artifact

**Files:**

- Create: `docs/research/nyx-evaluation.md`
- Modify: `docs/superpowers/plans/2026-06-15-roadmap-goals-implementation-plan.md`

- [x] Document what Nyx/ANISE would need to prove before becoming a production adapter.
- [x] Record the current decision as `defer adapter; keep evaluation-only`.
- [x] Commit with `git commit -m "docs: add nyx evaluation gate"`.

## Task 5: Verification

**Files:**

- Modify: `README.md`
- Modify: this plan.

- [x] Run:

```bash
python -m pytest -q
python -m ruff check .
python -m mypy
astro tudat-smoke
astro jax-smoke
```

- [x] Mark completed checklist items.
- [x] Commit with `git commit -m "docs: update research backend roadmap"`.

## Self-Review

Spec coverage:

- Covers Tudat, JAX, Nyx evaluation, optional smoke gates, and product boundaries.

Placeholder scan:

- No placeholders; deferred work has explicit gates and product expectations.

Type consistency:

- Tudat returns `Trajectory`; JAX research propagation returns `MonteCarloResult`; all unavailable paths use `UnsupportedBackendError`.
