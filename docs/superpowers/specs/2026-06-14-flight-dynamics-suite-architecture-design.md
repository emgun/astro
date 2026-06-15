# Flight Dynamics Suite Architecture Design

Date: 2026-06-14
Status: Draft for user review

## North Star

Build an operationally credible flight dynamics suite for mission analysis, orbit propagation, orbit determination, and launch/ascent analysis. The suite should expose a coherent Python API and CLI while using mature astrodynamics engines behind explicit adapters.

The product should own the mission vocabulary, validation rules, scenarios, products, and workflows. External engines should provide specialized computation without dictating the entire architecture.

## Scope

This architecture covers four first-class domains:

1. Orbital simulation: state propagation, force models, events, ephemerides, and uncertainty.
2. Flight dynamics: maneuvers, mission timelines, operational products, Monte Carlo, and analysis workflows.
3. Orbit determination: measurements, estimation, residuals, covariance, and quality checks.
4. Launch/ascent: vehicle models, atmosphere, staging, guidance, trajectory optimization, and target insertion.

The first implementation milestone should be orbital simulation plus flight dynamics plus synthetic-measurement orbit determination. Launch is designed as a first-class module from the beginning, but its implementation should follow after the orbital/OD spine is validated.

## Design Principles

- Stable domain model first: core concepts should not leak backend-specific classes.
- Mature engines over reimplementation: use trusted libraries for frames, time, force models, OD, and optimization.
- Optional heavy dependencies: engine adapters should be installed through extras instead of forcing every user to install every backend.
- Validation as a product feature: every major workflow needs deterministic reference cases and tolerances.
- Explicit units and frames: state, time, frame, and coordinate conventions must be declared at data boundaries.
- Modular workflows: launch, OD, propagation, and mission analysis should share common scenario and trajectory concepts while remaining independently testable.

## Core Domain Model

### Scenario

A scenario defines the mission context. It includes bodies, time span, environment models, spacecraft or vehicles, analysis products, and backend preferences.

Required concepts:

- Scenario ID and description.
- Time system and epoch handling.
- Central body and celestial bodies.
- Environment references such as gravity field, atmosphere, ephemerides, and Earth orientation data.
- Actors: spacecraft, launch vehicles, ground stations, sensors.
- Workflows: propagate, estimate, optimize launch, generate products.

### State

State represents the physical state of a spacecraft, launch vehicle, stage, or estimated object.

Supported state families:

- Cartesian state.
- Keplerian elements.
- Equinoctial or modified equinoctial elements.
- TLE-derived state.
- Ephemeris-backed sampled state.
- Launch/ascent state with translational, mass, throttle, attitude, and staging variables.

All states must carry epoch, frame, central body, units, and metadata about generation method.

### Spacecraft

Spacecraft describes objects after orbital insertion.

Required concepts:

- Dry/wet mass.
- Cross-sectional area.
- Drag coefficient.
- Reflectivity coefficient.
- Attitude mode or attitude provider reference.
- Propulsion capability and maneuver model.
- Optional covariance and estimation parameters.

### Launch Vehicle

Launch vehicle describes powered ascent systems.

Required concepts:

- Stages.
- Engines.
- Thrust curves or throttle schedules.
- Specific impulse.
- Propellant mass and dry mass.
- Aerodynamic coefficients.
- Guidance law or optimizer control variables.
- Separation events.
- Target orbit or insertion conditions.

### Force Model

Force model describes non-control dynamics for orbit or launch propagation.

Orbital force models:

- Two-body gravity.
- Higher-order geopotential.
- Third-body gravity.
- Atmospheric drag.
- Solar radiation pressure.
- Relativistic terms.
- Empirical accelerations.
- Maneuvers as discrete or finite burns.

Launch/ascent force models:

- Gravity.
- Atmosphere.
- Aerodynamic drag and lift.
- Thrust.
- Mass depletion.
- Winds if available.
- Earth rotation and launch-site inertial effects.

### Measurement

Measurement represents observations used for OD or validation.

Supported initial families:

- Range.
- Range rate / Doppler.
- Right ascension and declination.
- Azimuth and elevation.
- GNSS-like position fixes.
- Synthetic measurements generated from truth trajectories.

Later families:

- CCSDS TDM.
- DSN TNF/ODF style radiometric observations.
- Optical survey tracks.
- Radar tracks.

### Estimator

Estimator is an abstract OD workflow, not a single algorithm.

Initial estimator types:

- Batch least squares.
- Sequential Kalman-style estimation.
- Smoother, once sequential OD is stable.

Estimator outputs:

- Estimated state.
- Estimated parameters.
- Covariance.
- Residuals.
- Fit statistics.
- Iteration diagnostics.
- Reproducibility metadata.

### Trajectory

Trajectory is the common product of orbit propagation, launch optimization, and estimation.

Required contents:

- Time-indexed states.
- Events.
- Maneuvers or stage events.
- Derived quantities.
- Backend provenance.
- Validation metadata.
- Optional covariance history.

## Module Boundaries

### `astro_core`

Dependency-light domain layer.

Responsibilities:

- Scenario schema.
- Units, frame labels, and time labels.
- Domain models.
- Serialization and validation.
- Shared errors.

Non-responsibilities:

- Heavy propagation.
- Java/Python engine interop.
- Numerical optimization.

### `astro_dynamics`

Orbit propagation and flight dynamics workflow layer.

Responsibilities:

- Propagator interfaces.
- Force model configuration.
- Maneuver and event models.
- Ephemeris product generation.
- Monte Carlo hooks.

### `astro_od`

Orbit determination workflow layer.

Responsibilities:

- Measurement models.
- Synthetic measurement generation.
- Estimator interfaces.
- Residual and covariance products.
- OD validation reports.

### `astro_launch`

Launch and ascent workflow layer.

Responsibilities:

- Launch vehicle and stage models.
- Launch site modeling.
- Ascent trajectory phases.
- Powered-flight propagation interfaces.
- Ascent optimization interfaces.
- Orbit insertion targeting.

### `astro_backends`

Adapter layer for external engines.

Responsibilities:

- Convert suite domain models to backend-specific objects.
- Execute backend computations.
- Convert backend results back to suite products.
- Record backend version and configuration.

Adapters should be narrow and testable. No external backend object should become part of the public core model.

### `astro_cli`

Command-line workflows.

Responsibilities:

- Validate scenario files.
- Run propagation.
- Run synthetic OD.
- Generate products.
- Print diagnostics.

### `astro_examples`

Runnable examples and reference scenarios.

Responsibilities:

- Demonstrate API usage.
- Provide validation cases.
- Provide scenario templates.

## Backend Strategy

### Orekit

Primary operational backend for orbital simulation, flight dynamics, and OD.

Why:

- Mature treatment of frames and time.
- Strong propagation and force model support.
- Maneuver support.
- CCSDS support.
- Batch and sequential OD capabilities.

Python adapter choice:

- The Orekit adapter should use the official Orekit Python wrapper path rather than a custom Java bridge.
- For new-project implementation, evaluate `orekit_jpype` first because it is thinner, pip-installable, closer to the Java API, and is recommended by the wrapper maintainer for new projects.
- Keep the legacy JCC-based `orekit` wrapper as a compatibility fallback, especially if a workflow needs subclassing behavior that the JPype wrapper does not support cleanly.
- The adapter boundary must hide wrapper-specific details so the suite can switch between `orekit_jpype`, the legacy wrapper, or a direct Java service later without changing user-facing models.

Tradeoff:

- Java runtime, wrapper selection, VM initialization, and Orekit data setup add packaging complexity.

### Tudat / TudatPy

Secondary high-fidelity and research backend.

Why:

- Strong research workflows.
- High-fidelity propagation and estimation.
- Good cross-check backend for validation.
- Useful for planetary and deep-space expansion.

Tradeoff:

- Different model vocabulary and dependency footprint from Orekit.

### Dymos / OpenMDAO

Primary launch/ascent optimization backend.

Why:

- Mature trajectory optimization framework.
- Multiphase collocation.
- Analytic derivative workflows.
- Suitable for ascent, reentry, and constrained trajectories.

Tradeoff:

- More optimization framework complexity than direct propagation.

### RocketPy

Launch/ascent simulation and dispersion backend for lower-atmosphere rocket studies.

Why:

- Focused rocket trajectory simulation.
- Atmospheric and stochastic launch modeling.
- Useful for early launch examples and education.

Tradeoff:

- Not sufficient alone for orbital launch vehicle optimization.

### Nyx / ANISE

Future high-performance backend for mission design, OD, Monte Carlo, and SPICE-like geometry.

Why:

- Modern Rust foundation.
- Thread-safe and performance-oriented.
- Strong conceptual fit for scalable astrodynamics.

Tradeoff:

- Younger ecosystem than Orekit.

### JAX Research Backend

Future differentiable and accelerated backend.

Why:

- Useful for gradients, sensitivity, ML-assisted residual models, and batch propagation.
- Enables research workflows without compromising the operational baseline.

Tradeoff:

- Must not become the source of truth for frames, time, or operational OD before validation is strong.

## Package and Dependency Model

The base package should install only lightweight dependencies.

Proposed extras:

- `astro[orekit]`: Orekit adapter dependencies using the selected official Python wrapper path.
- `astro[orekit-jcc]`: optional legacy JCC wrapper support if needed for compatibility.
- `astro[tudat]`: TudatPy adapter dependencies.
- `astro[launch]`: Dymos/OpenMDAO and RocketPy dependencies.
- `astro[jax]`: differentiable research backend.
- `astro[dev]`: tests, docs, linting, and examples.

This keeps the default install small and prevents one heavy backend from blocking unrelated use cases.

## Data Flow

Common workflow:

1. User defines a scenario through Python or a scenario file.
2. `astro_core` validates units, frames, epochs, and required fields.
3. A workflow layer selects the requested operation.
4. `astro_backends` converts domain objects into backend-specific objects.
5. Backend executes propagation, OD, or launch optimization.
6. Results return as `Trajectory`, `EstimateResult`, or `LaunchResult` products.
7. Validation metadata and provenance are attached.
8. CLI/API writes products or returns Python objects.

## Error Handling

Errors should be domain-specific and actionable.

Required error classes:

- Invalid scenario.
- Unsupported frame.
- Unsupported time scale.
- Missing ephemeris or environment data.
- Backend unavailable.
- Backend conversion failure.
- Numerical convergence failure.
- Validation tolerance failure.

Error messages should include the scenario field, backend, and suggested fix when possible.

## Validation Philosophy

The validation suite is a core deliverable, not an afterthought.

Required validation tiers:

1. Pure math/unit tests for conversions and invariants.
2. Deterministic reference propagation cases.
3. Cross-backend comparisons.
4. Synthetic OD recovery cases.
5. Launch/ascent physics sanity checks.
6. End-to-end CLI scenario tests.

The suite should track tolerances explicitly. When tolerance changes are needed, the reason should be documented in the test or reference case metadata.

## Roadmap

### Milestone 1: Core Spine

- Project scaffold.
- Domain models.
- Scenario validation.
- Two-body and J2 local baseline propagator.
- CLI scenario validation.
- Reference case harness.

### Milestone 2: Orbital/OD MVP

- Orekit adapter boundary.
- LEO/MEO/GEO propagation workflows.
- Synthetic range/range-rate and inertial right-ascension/declination measurements.
- Batch least-squares OD recovery.
- Ephemeris output.
- Deterministic examples.

### Milestone 3: Launch/Ascent Module

- Launch vehicle schema.
- Launch site and atmosphere schema.
- RocketPy-style direct simulation adapter.
- Dymos/OpenMDAO ascent optimization adapter.
- Target orbit insertion product.
- Launch validation cases.

### Milestone 4: High-Fidelity and Research

- Tudat adapter.
- CCSDS/TDM ingestion.
- Optical and radiometric measurement families.
- Nyx evaluation.
- JAX differentiable propagation experiments.
- Monte Carlo and batch acceleration.

## External References

- Orekit: https://www.orekit.org/
- Orekit Python wrapper: https://gitlab.orekit.org/orekit-labs/python-wrapper
- Orekit JPype wrapper: https://gitlab.orekit.org/orekit/orekit_jpype
- NASA GMAT: https://etd.gsfc.nasa.gov/capabilities/capabilities-listing/general-mission-analysis-tool-gmat/
- Tudat: https://docs.tudat.space/
- Basilisk: https://avslab.github.io/basilisk/
- Dymos: https://openmdao.org/dymos/docs/
- RocketPy: https://docs.rocketpy.org/
- SPICE / NAIF: https://naif.jpl.nasa.gov/naif/
- PyKEP: https://esa.github.io/pykep/
- Nyx Space: https://nyxspace.com/

## Out of Scope for the First Implementation Plan

- Full spacecraft attitude dynamics.
- Hardware-in-the-loop simulation.
- Operational ground-system integration.
- Real tracking-data ingestion.
- Full launch vehicle performance certification.
- Human-rated mission constraints.
- Production-grade collision avoidance.

These are important future directions, but including them in the first implementation plan would dilute the validation spine.
