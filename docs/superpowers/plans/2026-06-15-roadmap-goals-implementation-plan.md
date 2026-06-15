# Full Roadmap Goals Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the approved flight-dynamics roadmap into a sequenced set of implementation goals that can fully build out orbital simulation, flight dynamics, orbit determination, launch, and research backends.

**Architecture:** Keep Astro Suite's own domain model as the product boundary, with mature engines behind narrow adapters. Preserve the current local deterministic baselines as validation references while adding operational backends in dependency order: Orekit first, then Orekit OD, then launch external adapters, then high-fidelity and research acceleration layers.

**Tech Stack:** Python 3.12, Pydantic v2, NumPy, SciPy, PyYAML, Typer, pytest, ruff, mypy, optional `orekit-jpype`, future optional RocketPy, Dymos/OpenMDAO, TudatPy, and JAX extras.

---

## North Star

Build an operationally credible Python flight dynamics suite for:

- Orbital simulation: state propagation, force models, events, ephemerides, and uncertainty.
- Flight dynamics: maneuvers, mission timelines, operational products, Monte Carlo, and analysis workflows.
- Orbit determination: measurements, estimation, residuals, covariance, and quality checks.
- Launch/ascent: vehicle models, atmosphere, staging, guidance, optimization, and target insertion.

The suite owns scenarios, validation, CLI/API workflows, product schemas, and provenance. External engines supply specialized computation through adapters and never leak backend objects into `astro_core`.

## Current Baseline

Implemented and protected:

- Core package scaffold, CLI entrypoint, strict typing, linting, and tests.
- `astro_core` scenario, state, spacecraft, force model, ground station, measurement, trajectory, and estimate products.
- `astro_dynamics.local` two-body and J2 deterministic RK4 propagation.
- `astro_dynamics` flight-dynamics product helpers for impulsive maneuvers, CSV ephemeris export,
  and seeded initial-state Monte Carlo propagation.
- `astro_od` synthetic range/range-rate/right-ascension/declination generation, measurement
  JSON/CSV ingest/export, TDM range/range-rate ingest/export, and local SciPy batch least-squares
  OD.
- `astro_launch` local vertical and pitch-program ascent baselines, launch-to-orbit handoff, pitch sweep, two-knot tuning, tuned launch reports, batch ranking, and report comparison.
- `astro_backends.orekit` optional `orekit_jpype` smoke gate and two-body Orekit propagation adapter.

Still roadmap-level:

- Orekit high-fidelity force models, Orekit measurement generation, and Orekit batch/sequential OD.
- Production covariance propagation and finite-burn maneuver dynamics.
- RocketPy launch simulation adapter.
- Dymos/OpenMDAO ascent optimization adapter.
- Tudat cross-check backend.
- Topocentric optical angles, richer radiometric families, and operational CCSDS support beyond
  current TDM range/range-rate MVP.
- Nyx evaluation and JAX differentiable research backend.

## Goal Ledger

### Goal 0: Baseline Preservation

Status: implemented, but must remain green before every roadmap slice.

Definition of done:

- `python -m pytest -v` passes.
- `python -m ruff check .` passes.
- `python -m mypy` passes.
- Current local workflows still run:

```bash
astro propagate examples/scenarios/leo_two_body.yaml --backend local --output /tmp/astro-local-trajectory.json
astro estimate-measurements examples/scenarios/leo_two_station_od.yaml examples/measurements/leo_two_station_od_measurements.json --output /tmp/astro-local-estimate.json
astro report-tuned-launch examples/launch/pitch_program_two_stage.yaml --point-indices 2,3 --iterations 2 --orbit-duration-s 600 --orbit-step-s 60 --output /tmp/astro-launch-report.json
```

### Goal 1: Operational Orekit Propagation Backend

Status: implemented.

Plan: `docs/superpowers/plans/2026-06-15-orekit-operational-propagation-implementation.md`

Definition of done:

- `astro propagate --backend orekit` works when `astro[orekit]` and required Orekit data setup are available.
- Missing Orekit wrapper or data produces actionable `UnsupportedBackendError`/CLI diagnostics.
- Orekit two-body propagation returns the shared `Trajectory` product with backend/version/provenance metadata.
- Orekit live tests are opt-in and skipped cleanly without `orekit_jpype`.
- Local and Orekit two-body propagation agree within documented tolerances on the LEO reference case.

Why this matters:

- Orekit is the roadmap's primary operational backend for frames, time, propagation, and OD.
- This gate proves the adapter architecture before deeper OD and force-model work.

Tradeoff:

- Start with a two-body/Keplerian Orekit path rather than a full high-fidelity stack. It is the smallest real Orekit propagation surface and keeps wrapper/data failures isolated.

### Goal 2: Orekit Force Models, Measurements, and Batch OD

Status: first backend-aware measurement and OD slice implemented; native Orekit estimator and numerical high-fidelity force models still require live Java/Orekit validation.

Implemented slice:

- `astro synth-measurements --backend orekit` routes truth propagation through the Orekit adapter.
- `astro estimate --backend orekit` and `astro estimate-measurements --backend orekit` run the suite's SciPy least-squares estimator with Orekit-backed residual propagation.
- The shared measurement surface supports range, range-rate, inertial right ascension, and
  declination records; right-ascension residuals wrap across 0/360 degrees.
- OD metadata records the selected propagation backend.
- Missing Java/`orekit-jpype` still produces actionable `UnsupportedBackendError` diagnostics.

Definition of done:

- `orekit_high_fidelity` force model maps to an Orekit-backed numerical propagator.
- Orekit propagation supports configured gravity/drag/SRP feature flags with explicit unsupported-feature errors.
- Measurement models can predict range, range-rate, inertial right ascension, and declination
  through the same measurement product surface.
- `astro estimate --backend orekit` or `astro estimate-measurements --backend orekit` returns `EstimateResult`.
- OD result metadata records Orekit version, wrapper, data source, estimator configuration, residual statistics, and convergence diagnostics.
- Cross-check tests compare local two-body OD and Orekit two-body OD on the controlled two-station scenario.

Primary files:

- Create `src/astro_backends/orekit/force_models.py`
- Create `src/astro_backends/orekit/measurements.py`
- Create `src/astro_backends/orekit/estimation.py`
- Modify `src/astro_od/estimation.py`
- Modify `src/astro_cli/main.py`
- Test `tests/astro_backends/test_orekit_force_models.py`
- Test `tests/astro_backends/test_orekit_measurements.py`
- Test `tests/astro_backends/test_orekit_estimation.py`
- Test `tests/astro_cli/test_cli.py`

### Goal 3: Operational Flight Dynamics Products

Status: implemented for product primitives, impulsive maneuvers, CSV ephemeris export, and seeded initial-state Monte Carlo. Finite burns and production covariance dynamics remain deferred.

Definition of done:

- Orbital `Trajectory` supports event and maneuver records without breaking launch products.
- Ephemeris export supports JSON plus at least one standard interchange format selected for the suite's current maturity.
- Maneuver products cover impulsive delta-v first, with finite burns deferred until the event/maneuver schema is stable.
- Monte Carlo hooks produce repeatable seeded ensembles for local and Orekit propagation.
- Covariance-history products are schema-supported even if only OD outputs populate them initially.

Primary files:

- Modify `src/astro_core/models.py`
- Create `src/astro_dynamics/maneuvers.py`
- Create `src/astro_dynamics/ephemeris.py`
- Create `src/astro_dynamics/monte_carlo.py`
- Modify `src/astro_cli/main.py`
- Test `tests/astro_core/test_models.py`
- Test `tests/astro_dynamics/test_maneuvers.py`
- Test `tests/astro_dynamics/test_ephemeris.py`
- Test `tests/astro_dynamics/test_monte_carlo.py`

### Goal 4: Launch External Backends and Optimization

Status: first external-backend boundary implemented. RocketPy and Dymos/OpenMDAO optional runtime gates, smoke commands, launch backend dispatch, a RocketPy product adapter boundary, and a neutral `optimize-launch` command are implemented. Full live RocketPy motor/rocket geometry mapping and Dymos phase transcription remain gated on richer backend-specific configuration.

Implemented slice:

- `astro rocketpy-smoke` and `astro dymos-smoke` return structured availability diagnostics.
- `astro launch --backend rocketpy` routes through a recognized RocketPy adapter boundary and fails clearly when dependencies or backend-specific configuration are unavailable.
- `astro optimize-launch --backend local` runs the existing pitch-program tuner through a neutral optimization entry point.
- `astro optimize-launch --backend dymos` routes through a Dymos/OpenMDAO adapter boundary and fails clearly until a validated Dymos phase runner is provided.

Definition of done:

- `astro launch --backend rocketpy` runs a direct simulation adapter when `astro[launch]` is installed and the scenario supplies RocketPy-specific vehicle/motor configuration.
- RocketPy adapter returns the existing `LaunchTrajectory` schema, not RocketPy-native objects.
- `astro optimize-launch --backend dymos` solves a small ascent optimization example once a validated Dymos phase model is added.
- Dymos/OpenMDAO adapter reports path constraints, optimizer status, convergence diagnostics, and target insertion residuals.
- Launch validation includes deterministic direct-simulation cases and one small optimization case.
- Launch-to-orbit handoff still produces a normal orbital `Scenario`.

Primary files:

- Create `src/astro_backends/rocketpy/__init__.py`
- Create `src/astro_backends/rocketpy/runtime.py`
- Create `src/astro_backends/rocketpy/simulation.py`
- Create `src/astro_backends/dymos/__init__.py`
- Create `src/astro_backends/dymos/runtime.py`
- Create `src/astro_backends/dymos/optimization.py`
- Modify `src/astro_launch/models.py`
- Modify `src/astro_launch/io.py`
- Modify `src/astro_cli/main.py`
- Modify `pyproject.toml`
- Test `tests/astro_backends/test_rocketpy_simulation.py`
- Test `tests/astro_backends/test_dymos_optimization.py`
- Test `tests/astro_cli/test_cli.py`

Tradeoff:

- Add RocketPy before Dymos. RocketPy proves external launch simulation adapter boundaries with lower setup complexity. Dymos should follow once launch products and target metrics have held up under one external backend.

### Goal 5: High-Fidelity and Research Backends

Status: first high-fidelity/research backend slice implemented. TudatPy and JAX runtime gates,
smoke commands, Tudat propagation dispatch, a built-in JAX two-body research propagation runner,
and a Nyx/ANISE evaluation gate are implemented. Live Tudat environment construction and richer JAX
force models or sensitivity workflows remain gated on validated runner implementations.

Implemented slice:

- `astro tudat-smoke` and `astro jax-smoke` return structured availability diagnostics.
- `astro propagate --backend tudat` is a recognized propagation boundary and returns suite
  `Trajectory` products through a validated runner.
- `astro research-propagate --backend local` runs seeded local ensembles.
- `astro research-propagate --backend jax` runs a vectorized two-body RK4 seeded ensemble and returns
  suite `MonteCarloResult` products.
- `docs/research/nyx-evaluation.md` records the current Nyx/ANISE decision as evaluation-only.

Definition of done:

- `astro propagate --backend tudat` supports at least one cross-check scenario and records Tudat provenance once a validated Tudat runner is supplied.
- `astro research-propagate --backend jax` runs seeded two-body batch propagation without replacing operational Orekit semantics; richer force models and sensitivity experiments still require validated JAX runners.
- Nyx/ANISE evaluation has a documented yes/no decision for a production adapter.
- Batch acceleration and Monte Carlo workflows preserve deterministic seeds and validation tolerances.

Primary files:

- Create `src/astro_backends/tudat/__init__.py`
- Create `src/astro_backends/tudat/runtime.py`
- Create `src/astro_backends/tudat/propagation.py`
- Create `src/astro_backends/jax/__init__.py`
- Create `src/astro_backends/jax/propagation.py`
- Create `docs/research/nyx-evaluation.md`
- Modify `pyproject.toml`
- Test `tests/astro_backends/test_tudat_propagation.py`
- Test `tests/astro_backends/test_jax_propagation.py`

Tradeoff:

- Keep JAX as a research backend until its frame/time/force-model behavior is validated against Orekit and local references. Differentiability is valuable, but operational correctness stays anchored in mature astrodynamics engines.

### Goal 6: Release Packaging, Documentation, and Validation Matrix

Status: implemented for the current roadmap pass. README, backend installation guide, validation
matrix, release checklist, and optional smoke-gate semantics are documented.

Definition of done:

- README current-scope section matches implemented behavior.
- Optional extras document installation and failure modes.
- Every backend has a smoke command or skip-clean live test marker.
- Validation matrix lists local, Orekit, launch, OD, and research tolerances.
- Example scenarios cover LEO, MEO, GEO, OD, local launch, and backend-specific smoke workflows.
- A release checklist confirms tests, lint, type checking, CLI smoke commands, and optional live checks.

Primary files:

- Modify `README.md`
- Create `docs/validation-matrix.md`
- Create `docs/backend-installation.md`
- Create `docs/release-checklist.md`
- Modify `examples/scenarios/*.yaml`
- Modify `examples/launch/*.yaml`

## Execution Order

1. Finish Goal 1.
2. Re-run Goal 0 verification.
3. Create a detailed Goal 2 plan after the first Orekit propagation adapter has settled.
4. Execute Goal 2.
5. Execute Goal 3 in small product-focused slices: events, maneuvers, ephemeris export, then Monte Carlo.
6. Execute Goal 4 with RocketPy before Dymos.
7. Execute Goal 5 only after operational validation exists for Orekit and launch products.
8. Keep Goal 6 current after every implementation slice.

## Roadmap Gates

Do not start Orekit OD until:

- `astro propagate --backend orekit` is implemented.
- Orekit unavailable/data failures are tested.
- A local-vs-Orekit propagation comparison exists.

Do not start Dymos until:

- RocketPy adapter or an equivalent external direct launch adapter proves the launch backend boundary.
- Launch target-miss metrics remain stable through at least one external backend.

Do not start JAX as a product backend until:

- Operational Orekit propagation is stable.
- Monte Carlo product schemas are in place.
- JAX results have a reference comparison path.

## Verification Loop

Run after each roadmap slice:

```bash
python -m pytest -v
python -m ruff check .
python -m mypy
```

Run relevant CLI smoke commands for the touched subsystem:

```bash
astro validate examples/scenarios/leo_two_body.yaml
astro propagate examples/scenarios/leo_two_body.yaml --backend local --output /tmp/astro-local-trajectory.json
astro synth-measurements examples/scenarios/leo_two_station_od.yaml --output /tmp/astro-measurements.json
astro estimate-measurements examples/scenarios/leo_two_station_od.yaml examples/measurements/leo_two_station_od_measurements.json --output /tmp/astro-estimate.json
astro launch examples/launch/pitch_program_two_stage.yaml --backend local --output /tmp/astro-launch.json
```

Run optional live backend checks only when the matching extra and external runtime are installed:

```bash
python -m pip install -e '.[orekit]'
astro orekit-smoke
astro propagate examples/scenarios/leo_two_body.yaml --backend orekit --output /tmp/astro-orekit-trajectory.json
```

## Self-Review

Spec coverage:

- Core spine: covered by Goal 0.
- Orbital/OD MVP: local MVP is implemented; operational Orekit propagation and OD are covered by Goals 1 and 2.
- Launch/ascent: local baseline is implemented; external simulation and optimization backends are covered by Goal 4.
- High-fidelity/research: Tudat, Nyx evaluation, JAX, and acceleration are covered by Goal 5.
- Validation and docs: covered by Goal 6 and the verification loop.

Placeholder scan:

- No task depends on unspecified files.
- Deferred work has explicit gates, primary files, and definitions of done.

Type consistency:

- All goals keep `Scenario`, `OrbitState`, `Trajectory`, `EstimateResult`, `LaunchScenario`, and `LaunchTrajectory` as the public product boundary.
