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
  fixed and tabulated-interpolation Earth-orientation correction, compact
  `iau_2006_2000a_simplified` precession/nutation reduction for geodetic station transforms,
  measurement, trajectory, and estimate products, plus IERS finals/finals2000A-style
  Earth-orientation ingest for UT1-UTC and polar motion samples.
- `astro_dynamics.local` two-body and J2 deterministic RK4 propagation.
- `astro_dynamics` flight-dynamics product helpers for impulsive maneuvers, CSV export, CCSDS
  OEM ephemeris export/import, and CCSDS AEM quaternion attitude export,
  local constant-acceleration finite-burn propagation, and seeded initial-state Monte Carlo
  propagation, plus finite-difference and opt-in two-body/J2 variational local covariance
  propagation with optional acceleration process noise, explicit per-sample
  state-transition/process-noise covariance products, commanded-attitude trajectory samples,
  velocity-aligned/radial attitude-coupled thrust-vector burn modes, time-aligned conjunction
  screening assessment reports, and diagonal rigid-body torque plus bounded quaternion-error PD
  attitude-control propagation products.
- `astro_od` synthetic range/range-rate/one-way Doppler/iterative linearized two-way and three-way
  range/range-rate/right-ascension/declination/azimuth/elevation generation, measurement JSON/CSV
  ingest/export, TDM range/range-rate/angle ingest/export, and local SciPy batch least-squares OD.
- `astro_launch` local vertical and pitch-program ascent baselines, launch-to-orbit handoff, pitch sweep, two-knot tuning, tuned launch reports, batch ranking, and report comparison.
- `astro_backends.orekit` optional `orekit_jpype` smoke gate, two-body Orekit propagation adapter,
  J2 numerical propagation through `J2OnlyPerturbation`, configured high-order gravity through
  `HolmesFeatherstoneAttractionModel`, Orekit drag/SRP/Sun-Moon third-body force-model adapters,
  and a live-gated native Orekit OD bridge for geodetic range/range-rate records.
- Optional RocketPy and Dymos/OpenMDAO launch backend gates, including configured RocketPy direct
  flight mapping and a stage-aware Dymos vertical-ascent phase transcription.
- Optional Tudat native two-body, J2 spherical-harmonic, configured high-order Earth spherical
  harmonic, atmospheric drag, cannonball SRP, and Sun/Moon point-mass third-body propagation
  cross-check runners, plus suite finite-difference covariance-history products through the
  selected Tudat force model, a Tudat-vs-reference calibrated comparison product, and
  multi-scenario Tudat comparison campaigns.

Still roadmap-level:

- Higher-fidelity variational covariance propagation for drag/SRP/third-body dynamics and
  validated production actuator/sensor attitude-control system models beyond the current
  deterministic bounded quaternion-error PD primitive.
- Full native multi-motor RocketPy staging and full pitch-program multistage Dymos ascent
  optimization.
- Native Tudat variational-equation covariance propagation remains deferred; current Tudat
  covariance uses the suite finite-difference transition product.
- Full standards-grade precession-nutation reductions beyond the current compact
  `iau_2006_2000a_simplified` geodetic-station option, native standards-grade DSN ODF/TNF parsing
  beyond the normalized CSV and suite-owned ASTRODSN1 binary tracking bridges plus truth-tagged
  station-bias calibration product, and
  operational CCSDS support beyond current KVN TDM measurement
  families, suite multi-leg radiometric TDM extension, OEM ephemeris interchange, and AEM
  quaternion attitude export.
- Richer JAX high-fidelity force models and full differentiable OD estimator workflows.

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
`EstimateResult` model through the runtime abstraction. Public CLI exposure is available through
`astro estimate-measurements --estimator orekit-native`, and the real Java/Orekit estimator has an
opt-in live validation gate. Short-arc singular covariance extraction is handled explicitly by
returning a zero covariance fallback with `covariance_status = "unavailable"` and the Orekit error
recorded in metadata.

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
- Orekit covariance-history propagation records the selected transition propagator and force-model
  list, so high-fidelity finite-difference covariance products identify J2, drag, SRP, and
  Sun/Moon third-body dynamics used by the perturbed-state transitions.
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
- `astro estimate-measurements --estimator orekit-native` exposes the native Orekit estimator
  bridge explicitly while preserving the suite SciPy estimator as the default.
- `tests/astro_backends/test_orekit_estimation.py::test_live_orekit_native_od_executes_batch_estimator`
  is an opt-in `ASTRO_RUN_OREKIT_LIVE=1` gate that executes the real Java/Orekit
  `BatchLSEstimator` against generated geodetic range/range-rate measurements.
- Native Orekit covariance extraction records `covariance_status = "available"` when the physical
  covariance matrix is returned and `covariance_status = "unavailable"` with `covariance_fallback =
  "zero_6x6"` when Orekit reports a singular matrix.
- The shared measurement surface supports range, range-rate, one-way Doppler in Hz, iterative
  linearized two-way and three-way range/range-rate with explicit participant-path metadata,
  inertial right ascension, declination, and local-horizon azimuth/elevation records; wrapped angle
  residuals handle 0/360 degree crossings.
- Two-way/three-way radiometric records carry iterative vacuum light-time diagnostics over a
  linearized spacecraft state, including uplink/downlink light time, transmit/reflection/receive
  offsets, iteration count, tolerance, and media-correction metadata. Scenarios can configure
  constant uplink/downlink media range delays with a source label, or opt into a configured
  `weather_frequency` model with surface pressure, temperature, relative humidity, zenith TEC,
  carrier frequency, and per-leg elevation mapping. `astro dsn-calibration` now summarizes those
  generated radiometric media records into a DSN-style calibration product with per-record leg
  delays, elevation diagnostics, model/source provenance, and aggregate delay statistics. The same
  command can also build the calibration product from loaded JSON/CSV/TDM measurement files, and
  the suite TDM extension preserves the `ASTRO_*` media-correction metadata required for that
  summary. `astro import-dsn-tracking` ingests normalized ODF/TNF-style DSN tracking CSV rows into
  suite measurement JSON with source-format, tracking-format, participant-path, and transmitter
  provenance, while `astro import-dsn-binary-tracking` ingests the suite-owned ASTRODSN1 fixed-record
  binary bridge. `astro station-calibration` estimates per-station/per-measurement-type bias
  products from truth-tagged measurement records. Native standards-grade ODF/TNF parsing and full
  DSN station calibration remain deeper standards work.
- `examples/scenarios/leo_doppler.yaml` provides a checked-in local one-way Doppler synthesis,
  JSON/CSV product, and local residual-prediction fixture. `leo_radiometric_links.yaml` provides a
  checked-in iterative two-way/three-way radiometric synthesis fixture. `leo_radiometric_media.yaml`
  provides a checked-in configured constant media-delay fixture, and
  `leo_radiometric_weather_frequency.yaml` provides a checked-in weather/frequency media fixture.
  Explicit Hz Doppler, two-way, and three-way suite records round-trip through TDM with
  `ASTRO_MEASUREMENT_TYPE` metadata extensions so legacy TDM files are not reinterpreted.
- Ground stations support fixed `position_eci_km` definitions and WGS-84 geodetic
  `latitude_deg`/`longitude_deg`/`altitude_km` definitions, with geodetic stations rotated into
  inertial measurement geometry at each measurement epoch using a deterministic UTC sidereal-time
  model, scenario-provided fixed Earth-orientation corrections, or linearly interpolated tabulated
  Earth-orientation samples for UT1-UTC and polar motion. Scenarios can opt into compact
  `iau_2006_2000a_simplified` precession/nutation for geodetic station reduction, and
  `astro import-earth-orientation --format iers-finals` converts IERS finals/finals2000A-style rows
  into suite `EarthOrientationConfig` JSON for tabulated samples.
- OD metadata records the selected propagation backend, estimator settings, residual statistics,
  convergence diagnostics, validation trajectory backend, and Orekit wrapper/version/data/propagator
  provenance when Orekit propagation metadata is available.
- Cross-check tests compare local two-body OD and Orekit-backed two-body OD on the controlled
  two-station scenario.
- Missing Java/`orekit-jpype` still produces actionable `UnsupportedBackendError` diagnostics.

Future native Orekit estimator scope:

- Add checked-in calibrated native Orekit OD fixtures with tighter accuracy expectations, not just
  execution and product-shape validation.
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
finite burns, local thrust-vector finite burns with mass depletion, CSV export, CCSDS OEM
ephemeris export/import, CCSDS AEM quaternion attitude export, finite-difference local covariance propagation, opt-in local two-body
and J2 variational covariance propagation with analytic two-body and finite-difference J2
acceleration Jacobians, optional white-acceleration process noise, explicit per-sample and
accumulated state-transition matrices, per-sample process-noise covariance matrices, Orekit
finite-difference covariance propagation through the selected Orekit force model, seeded
initial-state Monte Carlo, time-aligned conjunction screening with covariance-aware encounter-plane
probability methods, including a numerical 2D Gaussian hard-body disk integral, conservative
conjunction screening assessment reports, and commanded-attitude
trajectory samples for maneuvered local propagation, plus diagonal rigid-body torque and bounded
quaternion-error PD attitude-control propagation products. The attitude-coupled finite-burn modes rotate thrust along instantaneous velocity or
local radial directions and record body-to-inertial unit quaternion samples for the commanded body
+X axis. Local orbital propagation annotates periapsis/apoapsis `TrajectoryEvent` records for
deterministic mission-analysis products, using radial-velocity root location for no-maneuver local
trajectories and sample-safe extrema annotation for maneuvered trajectories. The
`examples/scenarios/leo_eccentric_two_body.yaml` scenario exercises an interior apoapsis root
through the public CLI; drag/SRP/third-body
variational-equation covariance dynamics, externally validated production conjunction services, and
validated production actuator/sensor ACS models remain deferred.

Definition of done:

- Orbital `Trajectory` supports event and maneuver records without breaking launch products.
- Ephemeris export/import supports JSON plus CSV export, CCSDS OEM KVN interchange, and CCSDS AEM
  KVN quaternion attitude export selected for the suite's current maturity. OEM import requires
  scenario context because OEM does not encode the suite force model.
- Maneuver products cover impulsive delta-v, local finite burns with constant inertial
  acceleration, local thrust-vector finite burns with mass-flow depletion, and velocity-aligned,
  radial-outward, and radial-inward thrust directions for commanded attitude-coupled burn modes.
- Maneuvered local trajectories include `AttitudeState` samples with body-to-inertial unit
  quaternions and target-direction metadata for the commanded body +X axis.
- `astro propagate-attitude` supports scheduled open-loop body torques and a bounded
  quaternion-error PD closed-loop control primitive with per-sample control-torque provenance.
- Monte Carlo hooks produce repeatable seeded ensembles for local and Orekit propagation.
- Covariance-history products are schema-supported. Local propagation can populate them from a
  scenario initial covariance using finite-difference state transitions, for two-body scenarios
  without maneuvers using opt-in variational state transitions integrated with the analytic
  two-body acceleration Jacobian, or for J2 scenarios without maneuvers using opt-in variational
  state transitions integrated with a finite-difference J2 acceleration Jacobian. These local paths
  support optional white-acceleration process noise. Orekit
  propagation can populate the same suite product by rebuilding and propagating perturbed Orekit
  states through the selected Orekit force model. These paths include per-sample state-transition
  matrices, accumulated state-transition matrices, and process-noise covariance matrices.

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

Status: first external-backend boundary implemented, first live RocketPy direct runner added,
RocketPy multistage suite-stage composition added, and first live stage-aware Dymos/OpenMDAO phase
transcription added. RocketPy and Dymos/OpenMDAO optional runtime gates, smoke commands, launch
backend dispatch, typed RocketPy launch-scenario configuration, checked-in RocketPy-configured
launch examples, configured solid RocketPy direct simulation, multistage suite stage-event/sample
annotation around one configured RocketPy flight, stage-schedule completeness metadata, a neutral
`optimize-launch` command, compatible optional dependency pins, optional import timeout diagnostics,
Dymos stage-aware vertical-ascent phase transcription, Dymos suite stage-plan metadata,
pitch-program control-point metadata, optimized pitch-program schedule metadata, tuned point
indices, and Dymos adapter optimization diagnostics are implemented. Full native multi-motor
RocketPy staging and full pitch-program multistage Dymos ascent optimization remain gated on
validated backend runners.

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
- `astro optimize-launch --backend dymos` runs a stage-aware Dymos/OpenMDAO vertical-ascent phase
  transcription that spans the configured burn schedule, then returns the existing suite
  launch-tuning product with explicit Dymos phase diagnostics.
- Dymos adapter results preserve suite tuning products and add optimizer status, convergence flag,
  iteration count, candidate count, path-constraint summary, best score, target insertion
  residuals, Dymos version, OpenMDAO version, phase duration, final altitude, final velocity,
  original and optimized pitch-program control-point schedules, tuned pitch point indices, explicit
  pitch-program optimization scope metadata, and optimizer message.
- Dymos adapter provenance includes the suite stage plan, multistage flag, total burn duration, and
  whether the Dymos phase duration covers the full stage schedule.

Definition of done:

- `astro launch --backend rocketpy` runs a direct simulation adapter when `astro[launch]` is
  installed and a scenario supplies RocketPy-specific vehicle/motor/flight configuration.
- RocketPy adapter returns the existing `LaunchTrajectory` schema, not RocketPy-native objects.
- RocketPy multistage suite composition reports reached stage ignition, burnout, separation, sample
  stage names, and stage-schedule completeness without claiming native RocketPy multi-motor staging.
- `astro optimize-launch --backend dymos` solves a stage-aware ascent optimization example through
  a bounded vertical-ascent Dymos phase model.
- Dymos/OpenMDAO adapter reports path constraints, pitch-program control points, tuned pitch point
  indices, optimized pitch-program control points, optimizer status, convergence diagnostics, and
  target insertion residuals.
- Dymos/OpenMDAO adapter reports the suite multistage plan and configured-burn coverage without
  claiming the current vertical-phase model is a full pitch-program multistage ascent optimizer.
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
and first JAX force-model expansion. TudatPy and JAX runtime gates, smoke commands, native Tudat
two-body, J2, configured high-order Earth spherical harmonic, drag, SRP, and Sun/Moon point-mass
third-body cross-check propagation runners, built-in JAX two-body and J2 research propagation
runners, suite finite-difference covariance-history propagation through selected Tudat force
models, a Tudat-vs-reference calibrated comparison product, `orekit_high_fidelity`
screening through the JAX J2 baseline, configured degree/order high-order gravity screening through
the same J2 baseline with explicit provenance, JAX research approximations for atmospheric drag, solar
radiation pressure, and analytic circular Sun/Moon third-body gravity, opt-in JAX final-state
transition sensitivities, JAX range/range-rate, inertial RA/Dec, and local-horizon az/el OD
residual Jacobian products, a first JAX research Gauss-Newton OD estimate workflow, and a
Nyx/ANISE evaluation gate are
implemented. The JAX research estimator now uses backtracking Gauss-Newton step acceptance for
range/range-rate and topocentric azimuth/elevation OD products. Native Tudat variational equations
and broader calibrated live comparison campaigns beyond the current
two-body/J2/high-order-gravity/drag/SRP/third-body runner and Tudat-vs-reference comparison product,
JAX ephemeris-backed third-body force models, and operational-grade differentiable OD estimator
workflows remain gated on validated runner implementations.

Implemented slice:

- `astro tudat-smoke` and `astro jax-smoke` return structured availability diagnostics.
- `astro propagate --backend tudat` runs native Tudat two-body Earth point-mass, J2 degree/order 2
  spherical-harmonic, configured high-order Earth spherical harmonic gravity, atmospheric drag,
  cannonball SRP, and Sun/Moon point-mass third-body cross-checks when TudatPy is installed, using
  Tudat environment/body setup, fixed-step RK4 integration, Cowell translational propagation
  settings, and suite `Trajectory` product mapping.
- `astro propagate --backend tudat` populates suite covariance-history products when an initial
  covariance is supplied by finite-differencing one-step Tudat state propagations through the
  selected Tudat force model. This is an adapter-level cross-check product, not native Tudat
  variational-equation propagation.
- `astro compare-tudat-reference` runs Tudat and a reference backend on the same scenario, then
  writes max/RMS/final position and velocity deltas, tolerance pass/fail status, and Tudat
  runner/force-model provenance.
- `astro compare-tudat-campaign` runs the same calibrated Tudat-vs-reference comparison across
  multiple scenarios and writes a suite campaign product with scenario pass/fail counts, worst-case
  position/velocity deltas, per-scenario comparison records, and campaign provenance.
- `astro research-propagate --backend local` runs seeded local ensembles.
- `astro research-propagate --backend jax` runs vectorized two-body and J2 RK4 seeded ensembles and
  returns suite `MonteCarloResult` products.
- `astro research-propagate --backend jax` accepts `orekit_high_fidelity` as a research screening
  baseline backed by J2, including configured degree/order high-order gravity metadata, plus
  explicit atmospheric-drag and solar-radiation-pressure force flags with metadata marking the
  product as screening-only rather than an operational ephemeris.
- `astro research-propagate --backend jax` accepts `third_body_gravity: true` through an analytic
  circular Sun/Moon point-mass approximation and records
  `third_body_ephemeris_model = "analytic_circular_sun_moon_screening"` so the product remains
  distinguishable from operational ephemeris-backed Orekit/Tudat propagation.
- `astro research-propagate --backend jax --include-sensitivities` records a nominal final-state
  transition matrix in `MonteCarloResult.metadata` using JAX autodiff.
- `astro research-od-sensitivity --backend jax` records normalized range/range-rate, inertial
  right-ascension/declination, and local-horizon azimuth/elevation residuals plus a residual
  Jacobian with respect to the initial Cartesian state in an `OdSensitivityResult`.
- `astro research-estimate --backend jax` runs a research backtracking Gauss-Newton correction loop
  over the same normalized residual/Jacobian model and returns the suite `EstimateResult` product
  with explicit research-backend metadata, including accepted step scales for angular OD damping.
- `docs/research/nyx-evaluation.md` records the current Nyx/ANISE decision as evaluation-only.

Definition of done:

- `astro propagate --backend tudat` supports checked-in two-body, J2, configured high-order
  gravity, drag, SRP, and third-body cross-check scenarios and records Tudat provenance through the
  native runners. Initial-covariance scenarios produce finite-difference covariance-history products
  through the selected Tudat force model. `astro compare-tudat-reference` and
  `astro compare-tudat-campaign` write calibrated reference-delta products for single-scenario and
  multi-scenario live Tudat validation. Native Tudat variational equations remain gated on validated
  body and acceleration-model construction.
- `astro research-propagate --backend jax` runs seeded two-body and J2 batch propagation plus
  screening-only `orekit_high_fidelity`, configured degree/order high-order gravity provenance,
  drag, SRP, and analytic Sun/Moon third-body force flags without replacing operational Orekit
  semantics; `astro research-od-sensitivity --backend jax` and
  `astro research-estimate --backend jax` provide the first differentiable OD residual/Jacobian and
  research correction-loop primitives for range/range-rate, inertial RA/Dec, and local-horizon
  az/el sensitivity and topocentric az/el estimation. Ephemeris-backed third-body force models and
  operational-grade differentiable OD estimators still require validated JAX runners.
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
