# Orbital Simulation, Flight Dynamics, and Orbit Determination MVP Design

Date: 2026-06-14
Status: Draft for user review

## Purpose

Deliver the first credible vertical slice of the flight dynamics suite: define scenarios, propagate orbital states, generate products, synthesize measurements, and recover an orbit through a controlled OD workflow.

This MVP is intentionally narrower than the whole suite. It builds the validated spine that launch and research modules can reuse later.

## Success Criteria

The MVP is successful when a user can:

1. Define a spacecraft scenario in Python and in a scenario file.
2. Validate frames, time scale, units, body, spacecraft, and force model choices.
3. Propagate a LEO, MEO, or GEO object.
4. Generate an ephemeris-like trajectory product.
5. Generate synthetic range and range-rate measurements from a truth trajectory.
6. Estimate an initial state from noisy synthetic measurements.
7. Inspect residuals, covariance, and fit diagnostics.
8. Run deterministic tests that prove the workflow is repeatable.

## MVP Non-Goals

- Real tracking-data ingestion.
- Full CCSDS product support.
- Full launch/ascent implementation.
- Attitude dynamics.
- Operational scheduling.
- High-performance GPU propagation.
- End-user graphical interface.

These are deliberately excluded from the first MVP to protect correctness and validation.

## User Workflows

### Propagation Workflow

Input:

- Scenario.
- Initial orbit state.
- Spacecraft properties.
- Force model selection.
- Time span and step policy.
- Backend selection.

Output:

- `Trajectory` product.
- Event list.
- Backend provenance.
- Validation diagnostics.

Initial supported force model levels:

1. Two-body.
2. J2 perturbation.
3. Optional higher-order backend force model through Orekit once the adapter is stable.

### Synthetic Measurement Workflow

Input:

- Truth trajectory.
- Ground station definitions.
- Measurement schedule.
- Noise model.
- Measurement types.

Output:

- Measurement set.
- Truth linkage metadata.
- Noise metadata.

Initial measurement types:

- Range.
- Range rate.

### Orbit Determination Workflow

Input:

- Measurement set.
- A priori initial state.
- Estimated parameter selection.
- Estimator settings.
- Backend selection.

Output:

- Estimated state.
- Residuals.
- Covariance.
- Fit statistics.
- Iteration diagnostics.

Initial estimator:

- Batch least squares through the first backend that supports the full workflow.

## Domain Objects

### `OrbitState`

Fields:

- Epoch.
- Frame.
- Central body.
- Representation type.
- Position and velocity for Cartesian states.
- Element values for Keplerian states.
- Units.

Validation:

- Epoch must include time scale.
- Frame must be explicit.
- Central body must be known.
- Representation values must be finite.

### `Spacecraft`

Fields:

- Name.
- Mass.
- Cross-sectional area.
- Drag coefficient.
- Reflectivity coefficient.
- Optional maneuver capability.

Validation:

- Mass and area must be positive.
- Coefficients must be finite and within configured sanity ranges.

### `ForceModelConfig`

Fields:

- Gravity model.
- Drag model.
- Solar radiation pressure.
- Third bodies.
- Maneuvers.

MVP variants:

- `two_body`.
- `j2`.
- `orekit_high_fidelity` after adapter availability.

### `GroundStation`

Fields:

- Name.
- Body-fixed position or geodetic coordinates.
- Elevation mask.
- Measurement availability window.

Validation:

- Coordinates must specify frame or geodetic convention.
- Elevation mask must be bounded.

### `Measurement`

Fields:

- Type.
- Epoch.
- Observer.
- Observed object.
- Value.
- Sigma.
- Units.
- Metadata.

Validation:

- Type-specific dimensions must match units.
- Sigma must be positive.

### `Trajectory`

Fields:

- States.
- Events.
- Step metadata.
- Force model metadata.
- Backend metadata.
- Optional covariance samples.

Validation:

- Epochs must be monotonic.
- All states must share compatible frames or include explicit frame conversions.

## Backend Plan

### Local Baseline Backend

The local baseline backend should implement two-body and J2 propagation directly in Python.

Purpose:

- Provide deterministic smoke tests.
- Keep early development unblocked by external engine installation.
- Provide sanity checks for backend adapters.

Tradeoff:

- It must remain intentionally small. It should not grow into a replacement for Orekit or Tudat.

### Orekit Adapter

The Orekit adapter should be the first operational adapter.

Responsibilities:

- Use the official Orekit Python wrapper path as the integration surface.
- Prefer `orekit_jpype` for the first new-project smoke test because it is thinner, pip-installable, and close to the Java API.
- Keep the legacy JCC-based `orekit` wrapper as a fallback if JPype limitations block a required workflow.
- Convert suite `OrbitState`, `Spacecraft`, `ForceModelConfig`, and `GroundStation` into Orekit objects.
- Run propagation.
- Generate synthetic measurements if supported by the chosen OD path.
- Run batch least-squares OD once the measurement interface is stable.
- Convert outputs back to suite products.

First smoke checks:

- Wrapper import.
- Java runtime and VM initialization.
- Orekit version check.
- Orekit data loading.
- Frame and time-scale access.
- Simple two-body propagation through the wrapper.

Risk:

- Python/Java packaging, wrapper choice, VM initialization, and data setup can be frustrating. The first adapter milestone should isolate installation and data-loading checks before deep OD work.

### Tudat Adapter

Tudat should come after Orekit.

Purpose:

- Cross-check propagation cases.
- Support high-fidelity research workflows.
- Prepare for real tracking-data OD.

## Scenario File Shape

The scenario format should be explicit and boring. YAML or JSON is acceptable, with a strong preference for a schema-validated structure that can be represented in Python models.

Top-level sections:

- `scenario`.
- `environment`.
- `spacecraft`.
- `initial_state`.
- `force_model`.
- `propagation`.
- `measurements`.
- `estimation`.
- `outputs`.

The schema should reject missing units, implicit frames, and ambiguous time scales.

## CLI Shape

Initial commands:

- `astro validate scenario.yaml`
- `astro propagate scenario.yaml --backend local`
- `astro synth-measurements scenario.yaml --backend local`
- `astro estimate scenario.yaml --backend orekit`

The CLI should emit human-readable diagnostics and structured output files.

## Testing Plan

### Unit Tests

- Unit conversions.
- State validation.
- Scenario validation.
- Force model validation.
- Measurement validation.

### Reference Cases

1. Two-body circular LEO propagation.
2. Two-body eccentric orbit propagation.
3. J2 nodal precession sanity case.
4. GEO near-stationary propagation sanity case.
5. Synthetic range measurement generation.
6. Synthetic range-rate measurement generation.
7. Batch OD recovery from noisy synthetic measurements.

### Backend Adapter Tests

- Backend availability checks.
- Conversion round-trip checks.
- Propagation tolerance comparisons against local reference cases.
- Provenance metadata checks.

## Error Handling

The MVP should fail early for:

- Missing units.
- Missing frame.
- Missing time scale.
- Unsupported backend.
- Unsupported force model for backend.
- Measurement type not supported by estimator.
- Estimator non-convergence.
- Reference tolerance failure.

## First Implementation Slice

Recommended first slice:

1. Create project scaffold and package layout.
2. Implement `astro_core` domain models.
3. Implement scenario validation.
4. Implement local two-body propagation.
5. Add deterministic reference tests.
6. Add CLI validation and propagation commands.
7. Add J2.
8. Add synthetic range/range-rate measurements.
9. Add OD interface and first backend boundary.

This gives fast feedback before heavy external dependencies enter the critical path.

## Tradeoffs

Starting with a local baseline risks duplicated work, but it is useful if it stays small and test-focused. Starting directly with Orekit would reach operational features faster, but backend installation and interop could block core model design. The chosen path builds a minimal local reference first, then moves operational work into Orekit.

## Open Decisions For Implementation Planning

- Python version target.
- Package manager.
- Scenario file format.
- Whether `orekit_jpype` is sufficient for the first OD workflow or the legacy JCC wrapper is needed for a specific subclassing or extension point.
- Whether the first OD workflow is implemented through Orekit immediately or after local measurement generation is complete.
- Reference tolerance values for each propagation case.
