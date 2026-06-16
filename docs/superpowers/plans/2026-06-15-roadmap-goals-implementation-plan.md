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
- `astro_core` scenario, state, spacecraft, force model, ECI and WGS-84 geodetic ground station,
  measurement, trajectory, and estimate products.
- `astro_dynamics.local` two-body and J2 deterministic RK4 propagation.
- `astro_dynamics` flight-dynamics product helpers for impulsive maneuvers, CSV ephemeris export,
  local constant-acceleration finite-burn propagation, and seeded initial-state Monte Carlo
  propagation, plus finite-difference local covariance propagation with optional acceleration
  process noise.
- `astro_od` synthetic range/range-rate/right-ascension/declination/azimuth/elevation generation,
  measurement JSON/CSV ingest/export, TDM range/range-rate/angle ingest/export, and local SciPy
  batch least-squares OD.
- `astro_launch` local vertical and pitch-program ascent baselines, launch-to-orbit handoff, pitch sweep, two-knot tuning, tuned launch reports, batch ranking, and report comparison.
- `astro_backends.orekit` optional `orekit_jpype` smoke gate, two-body Orekit propagation adapter,
  and J2 numerical propagation through `J2OnlyPerturbation`.

Still roadmap-level:

- Native Orekit batch/sequential OD execution and suite result mapping.
- High-fidelity covariance propagation with validated state transition matrices/process noise and
  attitude-coupled maneuver dynamics.
- Live RocketPy launch simulation mapping for backend-specific motor/rocket geometry.
- Live Dymos/OpenMDAO ascent phase transcription and optimization.
- Live Tudat cross-check environment/body construction.
- EOP-aware ITRF-to-celestial pointing, richer radiometric families, and operational CCSDS support
  beyond current KVN TDM measurement families.
- Richer JAX force models, sensitivities, and differentiable OD workflows.

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

Status: suite-level Orekit-backed measurement/OD, Orekit J2 numerical propagation, and the
`orekit_high_fidelity` numerical propagation entry path, Orekit atmospheric drag, Orekit SRP, and
Sun/Moon third-body gravity are implemented. Native Orekit OD now has a measurement-object and
`BatchLSEstimator` construction/execution bridge for WGS-84 geodetic range/range-rate records. It
maps estimated state, residuals, RMS, covariance, and iteration diagnostics into the suite
`EstimateResult` model through the runtime abstraction; public CLI exposure and live Java/Orekit
estimator validation remain future integration work.

Implemented slice:

- `propagate_orekit` supports suite `j2` scenarios through Orekit `NumericalPropagator`,
  Dormand-Prince 8(5,3) integration, and `J2OnlyPerturbation`.
- `propagate_orekit` accepts suite `orekit_high_fidelity` scenarios and routes them through the
  numerical propagator, currently using the J2 numerical baseline as the expansion point.
- `propagate_orekit` supports `atmospheric_drag: true` through Orekit `DragForce` with
  `SimpleExponentialAtmosphere` and `IsotropicDrag`, using suite spacecraft area and drag
  coefficient.
- `propagate_orekit` supports `solar_radiation_pressure: true` through Orekit
  `SolarRadiationPressure` with `IsotropicRadiationSingleCoefficient`, using suite spacecraft area
  and reflectivity coefficient.
- `propagate_orekit` supports `third_body_gravity: true` through Orekit `ThirdBodyAttraction` for
  the Sun and Moon.
- `ForceModelConfig` includes explicit `atmospheric_drag`, `solar_radiation_pressure`, and
  `third_body_gravity` flags. Local and JAX report unsupported flags instead of silently ignoring
  requested physics; Orekit implements all three current high-fidelity flags.
- `examples/scenarios/leo_j2.yaml` gives local and Orekit CLI workflows a checked-in J2 case.
- `examples/scenarios/leo_orekit_high_fidelity.yaml` gives the Orekit numerical high-fidelity path
  a checked-in CLI case.
- `examples/scenarios/leo_orekit_drag.yaml` gives Orekit atmospheric drag a checked-in CLI case.
- `examples/scenarios/leo_orekit_srp.yaml` gives Orekit SRP a checked-in CLI case.
- `examples/scenarios/leo_orekit_third_body.yaml` gives Orekit Sun/Moon third-body gravity a
  checked-in CLI case.
- Optional live tests compare Orekit J2 against the local J2 reference scale on the LEO case.
- `astro synth-measurements --backend orekit` routes truth propagation through the Orekit adapter.
- `astro estimate --backend orekit` and `astro estimate-measurements --backend orekit` run the
  suite's SciPy least-squares estimator with Orekit-backed residual propagation.
- `build_orekit_observed_measurements` maps suite WGS-84 geodetic range/range-rate records into
  Orekit `Range` and `RangeRate` observed measurements with suite-to-Orekit unit conversion.
- `build_orekit_batch_ls_estimator` constructs an Orekit `BatchLSEstimator` with a numerical
  propagator builder and attached observed measurements, establishing the native estimator object
  boundary before live execution is exposed.
- `estimate_orekit_native` executes the native Orekit `BatchLSEstimator` through the runtime
  abstraction and maps the resulting propagated state, residual vector, RMS, covariance, iteration
  count, evaluation count, and Orekit wrapper provenance into `EstimateResult`.
- The shared measurement surface supports range, range-rate, inertial right ascension, and
  declination plus local-horizon azimuth/elevation records; wrapped angle residuals handle 0/360
  degree crossings.
- Ground stations support fixed `position_eci_km` definitions and WGS-84 geodetic
  `latitude_deg`/`longitude_deg`/`altitude_km` definitions, with geodetic stations rotated into
  inertial measurement geometry at each measurement epoch using a deterministic UTC sidereal-time
  model.
- OD metadata records the selected propagation backend, estimator settings, residual statistics,
  convergence diagnostics, validation trajectory backend, and Orekit wrapper/version/data/propagator
  provenance when Orekit propagation metadata is available.
- Cross-check tests compare local two-body OD and Orekit-backed two-body OD on the controlled
  two-station scenario.
- Missing Java/`orekit-jpype` still produces actionable `UnsupportedBackendError` diagnostics.

Future native Orekit estimator scope:

- Add live Java/Orekit validation for native estimator execution against checked-in geodetic
  range/range-rate fixtures before exposing the native estimator as a public CLI workflow.
- Extend native Orekit OD beyond geodetic range/range-rate records as Orekit measurement families
  are validated.
- Keep the current suite-level SciPy estimator as the deterministic always-on OD reference while
  the native Java-backed estimator matures.

Primary files:

- `src/astro_backends/orekit/force_models.py`
- `src/astro_backends/orekit/propagation.py`
- `src/astro_od/estimation.py`
- `src/astro_cli/main.py`
- `tests/astro_backends/test_orekit_propagation.py`
- `tests/astro_od/test_estimation.py`
- `tests/astro_cli/test_cli.py`

### Goal 3: Operational Flight Dynamics Products

Status: implemented for product primitives, impulsive maneuvers, local constant-acceleration
finite burns, local thrust-vector finite burns with mass depletion, finite-difference local
covariance propagation with optional white-acceleration process noise, CSV ephemeris export, and
seeded initial-state Monte Carlo. Validated high-fidelity covariance dynamics and attitude-coupled
maneuver dynamics remain deferred.

Definition of done:

- Orbital `Trajectory` supports event and maneuver records without breaking launch products.
- Ephemeris export supports JSON plus at least one standard interchange format selected for the suite's current maturity.
- Maneuver products cover impulsive delta-v, local finite burns with constant inertial
  acceleration, and local thrust-vector finite burns with mass-flow depletion.
- Monte Carlo hooks produce repeatable seeded ensembles for local and Orekit propagation.
- Covariance-history products are schema-supported and local propagation can populate them from a
  scenario initial covariance using finite-difference state transitions plus optional
  white-acceleration process noise.

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

Status: first external-backend boundary implemented, first live RocketPy direct runner added, RocketPy multistage suite-stage composition added, and first live Dymos/OpenMDAO phase transcription added. RocketPy and Dymos/OpenMDAO optional runtime gates, smoke commands, launch backend dispatch, typed RocketPy launch-scenario configuration, checked-in RocketPy-configured launch examples, configured solid RocketPy direct simulation, multistage suite stage-event/sample annotation around one configured RocketPy flight, stage-schedule completeness metadata, a neutral `optimize-launch` command, compatible optional dependency pins, optional import timeout diagnostics, Dymos vertical-ascent phase transcription, Dymos suite stage-plan metadata, and Dymos adapter optimization diagnostics are implemented. Full native multi-motor RocketPy staging and full multistage Dymos ascent optimization remain gated on validated backend runners.

Implemented slice:

- `astro rocketpy-smoke` and `astro dymos-smoke` return structured availability diagnostics.
- `astro rocketpy-smoke` and `astro dymos-smoke` are live-verified with NumPy-1-compatible pins:
  RocketPy `>=1.11,<1.12`, Dymos `>=1.13.1,<1.14`, and OpenMDAO `>=3.41,<3.42`.
- Optional backend imports have timeout diagnostics so Matplotlib/OpenMDAO/RocketPy import-time
  stalls report as backend-unavailable errors instead of hanging indefinitely.
- `LaunchScenario.rocketpy` provides typed RocketPy vehicle/motor/flight configuration for the
  backend-specific path, with validation for rail settings, thrust curves, motor inertia, motor
  placement, and solid-motor grain geometry.
- `examples/launch/rocketpy_configured_two_stage.yaml` provides a checked-in RocketPy configuration
  fixture, and `examples/launch/rocketpy_configured_single_stage.yaml` provides the live direct-run
  fixture.
- `astro launch --backend rocketpy` runs a configured solid RocketPy flight when dependencies and
  `scenario.rocketpy` configuration are available. For multistage suite scenarios, it annotates the
  returned `LaunchTrajectory` with reached suite stage events and sample stage names around one
  configured RocketPy flight, with metadata identifying the composition tradeoff and whether the
  RocketPy solution covered the full suite stage schedule.
- `astro optimize-launch --backend local` runs the existing pitch-program tuner through a neutral optimization entry point.
- `astro optimize-launch --backend dymos` runs a small Dymos/OpenMDAO vertical-ascent phase
  transcription, then returns the existing suite launch-tuning product with explicit Dymos phase
  diagnostics.
- Dymos adapter results preserve suite tuning products and add optimizer status, convergence flag,
  iteration count, candidate count, path-constraint summary, best score, target insertion
  residuals, Dymos version, OpenMDAO version, phase duration, final altitude, final velocity, and
  optimizer message.
- Dymos adapter provenance includes the suite stage plan, multistage flag, total burn duration, and
  whether the Dymos phase duration covers the full stage schedule.

Definition of done:

- `astro launch --backend rocketpy` runs a direct simulation adapter when `astro[launch]` is
  installed and a scenario supplies RocketPy-specific vehicle/motor/flight configuration.
- RocketPy adapter returns the existing `LaunchTrajectory` schema, not RocketPy-native objects.
- RocketPy multistage suite composition reports reached stage ignition, burnout, separation, sample
  stage names, and stage-schedule completeness without claiming native RocketPy multi-motor staging.
- `astro optimize-launch --backend dymos` solves a small ascent optimization example through a
  bounded vertical-ascent Dymos phase model.
- Dymos/OpenMDAO adapter reports path constraints, optimizer status, convergence diagnostics, and target insertion residuals.
- Dymos/OpenMDAO adapter reports the suite multistage plan without claiming the current
  vertical-phase model is a full multistage ascent optimizer.
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

Status: high-fidelity/research backend slices implemented for runtime gates, product boundaries,
and first JAX force-model expansion. TudatPy and JAX runtime gates, smoke commands, Tudat propagation
dispatch, built-in JAX two-body and J2 research propagation runners, opt-in JAX final-state
transition sensitivities, and a Nyx/ANISE evaluation gate are implemented. Live Tudat environment
construction and JAX high-fidelity force flags or differentiable OD workflows remain gated on
validated runner implementations.

Implemented slice:

- `astro tudat-smoke` and `astro jax-smoke` return structured availability diagnostics.
- `astro propagate --backend tudat` is a recognized propagation boundary and returns suite
  `Trajectory` products through a validated runner.
- `astro research-propagate --backend local` runs seeded local ensembles.
- `astro research-propagate --backend jax` runs vectorized two-body and J2 RK4 seeded ensembles and
  returns suite `MonteCarloResult` products.
- `astro research-propagate --backend jax --include-sensitivities` records a nominal final-state
  transition matrix in `MonteCarloResult.metadata` using JAX autodiff.
- `docs/research/nyx-evaluation.md` records the current Nyx/ANISE decision as evaluation-only.

Definition of done:

- `astro propagate --backend tudat` supports at least one cross-check scenario and records Tudat provenance once a validated Tudat runner is supplied.
- `astro research-propagate --backend jax` runs seeded two-body and J2 batch propagation without replacing operational Orekit semantics; high-fidelity force flags and differentiable OD still require validated JAX runners.
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
matrix, release checklist, LEO/MEO/GEO examples, and optional smoke-gate semantics are documented.

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
