# Flight Dynamics Products Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add operational flight-dynamics product primitives: orbital events, impulsive maneuvers, ephemeris export, covariance-history schema, and seeded Monte Carlo propagation.

**Architecture:** Extend `astro_core` trajectory products with optional event, maneuver, and covariance-history fields while preserving backward-compatible defaults. Keep workflow helpers in `astro_dynamics` and expose product export through focused functions and CLI commands.

**Tech Stack:** Python 3.12, Pydantic v2, NumPy, Typer, pytest.

---

## Scope

In scope:

- `TrajectoryEvent`, `Maneuver`, and `CovarianceSample` models.
- Backward-compatible `Trajectory.events`, `Trajectory.maneuvers`, and `Trajectory.covariance_history`.
- Impulsive maneuver helper for Cartesian states.
- CSV ephemeris export for trajectory products.
- Seeded initial-state Monte Carlo propagation product.
- CLI commands for ephemeris export and Monte Carlo.

Out of scope:

- Finite-burn dynamics.
- Attitude coupling.
- Collision avoidance.
- Production covariance propagation.

## Task 1: Core Product Schema

**Files:**

- Modify: `src/astro_core/models.py`
- Test: `tests/astro_core/test_models.py`

- [ ] Add tests for trajectory events, maneuvers, covariance samples, and default backward compatibility.
- [ ] Implement the models and validation.
- [ ] Run `python -m pytest tests/astro_core/test_models.py -v`.
- [ ] Commit with `git commit -m "feat: add flight dynamics trajectory product fields"`.

## Task 2: Maneuver And Ephemeris Helpers

**Files:**

- Create: `src/astro_dynamics/maneuvers.py`
- Create: `src/astro_dynamics/ephemeris.py`
- Modify: `src/astro_dynamics/__init__.py`
- Test: `tests/astro_dynamics/test_maneuvers.py`
- Test: `tests/astro_dynamics/test_ephemeris.py`
- Test: `tests/test_imports.py`

- [ ] Add tests for impulsive delta-v application and CSV ephemeris export.
- [ ] Implement helpers.
- [ ] Export public functions.
- [ ] Run focused tests.
- [ ] Commit with `git commit -m "feat: add maneuver and ephemeris helpers"`.

## Task 3: Monte Carlo Propagation

**Files:**

- Create: `src/astro_dynamics/monte_carlo.py`
- Modify: `src/astro_dynamics/__init__.py`
- Test: `tests/astro_dynamics/test_monte_carlo.py`
- Test: `tests/test_imports.py`

- [ ] Add deterministic seeded Monte Carlo tests.
- [ ] Implement perturbation and ensemble result models.
- [ ] Run focused tests.
- [ ] Commit with `git commit -m "feat: add seeded propagation monte carlo"`.

## Task 4: CLI And Documentation

**Files:**

- Modify: `src/astro_cli/main.py`
- Modify: `tests/astro_cli/test_cli.py`
- Modify: `README.md`
- Modify: `docs/superpowers/plans/2026-06-15-roadmap-goals-implementation-plan.md`

- [ ] Add `export-trajectory` CLI command for CSV ephemeris export.
- [ ] Add `monte-carlo` CLI command for seeded initial-state ensembles.
- [ ] Document commands and mark Goal 3 implemented.
- [ ] Run:

```bash
python -m pytest -q
python -m ruff check .
python -m mypy
astro propagate examples/scenarios/leo_two_body.yaml --backend local --output /tmp/astro-trajectory.json
astro export-trajectory /tmp/astro-trajectory.json --format csv --output /tmp/astro-trajectory.csv
astro monte-carlo examples/scenarios/leo_two_body.yaml --cases 4 --position-sigma-km 0.01 --velocity-sigma-km-s 0.000001 --seed 7 --output /tmp/astro-monte-carlo.json
```

- [ ] Commit with `git commit -m "feat: add flight dynamics product cli"`.

## Self-Review

Spec coverage:

- Covers trajectory events, maneuvers, ephemeris product generation, Monte Carlo hooks, and covariance-history schema.
- Keeps finite-burn and high-fidelity covariance propagation out of this slice.

Placeholder scan:

- All files and commands are explicit.

Type consistency:

- `Trajectory` remains the common product; new fields are optional with default empty lists.

