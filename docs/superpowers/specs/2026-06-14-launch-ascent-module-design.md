# Launch and Ascent Module Design

Date: 2026-06-14
Status: Draft for user review

## Purpose

Design the launch/ascent module as a first-class part of the flight dynamics suite while keeping it independent from the first orbital/OD MVP implementation plan.

Launch is not just another force model. It combines atmospheric flight, vehicle performance, mass depletion, staging, guidance, path constraints, and target orbit insertion. It should reuse the suite's scenario, trajectory, event, backend, validation, and product concepts, but it needs its own module and validation cases.

## Success Criteria

The launch module is successful when a user can:

1. Define a launch site, launch vehicle, target orbit, and ascent scenario.
2. Simulate a staged powered ascent through atmosphere.
3. Model thrust, mass depletion, drag, staging, and simple guidance.
4. Produce a trajectory with stage events, dynamic pressure, acceleration, altitude, velocity, and insertion conditions.
5. Optimize an ascent profile against a target orbit.
6. Validate launch results against deterministic sanity cases.
7. Hand the insertion state to the orbital propagation and OD modules.

## Module Boundaries

### In Scope

- Launch site model.
- Launch vehicle model.
- Stage and engine model.
- Atmosphere and aerodynamic model configuration.
- Powered-flight propagation interface.
- Staging and separation events.
- Guidance model interface.
- Trajectory optimization interface.
- Target orbit and insertion product.
- Launch validation cases.

### Out of Scope Initially

- Detailed structural loads.
- Full range safety system.
- Closed-loop flight software.
- Plume/aero interaction.
- Human-rating constraints.
- Detailed vehicle design optimization.
- Reentry and recovery beyond simple event markers.

## Domain Objects

### `LaunchScenario`

Fields:

- Scenario ID.
- Launch site.
- Launch azimuth or targeting policy.
- Launch window.
- Vehicle.
- Atmosphere configuration.
- Guidance configuration.
- Target orbit.
- Backend selection.
- Output products.

The launch scenario should be embeddable inside the common `Scenario` model.

### `LaunchSite`

Fields:

- Name.
- Latitude.
- Longitude.
- Altitude.
- Body.
- Local frame convention.
- Optional launch rail or pad orientation.
- Optional range constraints.

Validation:

- Geodetic coordinates must be explicit.
- Body must be known.
- Altitude units must be explicit.

### `LaunchVehicle`

Fields:

- Name.
- Stages.
- Payload mass.
- Fairing model if relevant.
- Aerodynamic coefficient model.
- Guidance model reference.

Validation:

- At least one stage.
- Positive total mass.
- Stage order must be explicit.

### `Stage`

Fields:

- Name.
- Dry mass.
- Propellant mass.
- Engine list.
- Burn duration or shutdown condition.
- Separation condition.
- Aerodynamic reference area.

Validation:

- Dry mass and propellant mass must be nonnegative.
- Burn duration or shutdown condition must be present.
- Stage sequence must be valid.

### `Engine`

Fields:

- Name.
- Thrust model.
- Specific impulse model.
- Mixture or propellant metadata.
- Throttle bounds.

Thrust and specific impulse can initially be constants or piecewise curves.

### `AtmosphereConfig`

Fields:

- Atmosphere model name.
- Wind model.
- Density model.
- Temperature model if backend supports it.

Initial support can be simple standard atmosphere, then backend-specific higher fidelity models.

### `GuidanceConfig`

Initial guidance families:

- Fixed pitch program.
- Gravity turn.
- Optimized control profile.

Later guidance families:

- Closed-loop guidance law.
- Powered explicit guidance.
- Model-predictive guidance experiments.

### `TargetOrbit`

Fields:

- Target representation.
- Altitude or semi-major axis.
- Inclination.
- Eccentricity target.
- RAAN or launch-window policy.
- Tolerances.

Output should include target miss metrics, not only final state.

### `LaunchTrajectory`

Fields:

- Time-indexed states.
- Mass history.
- Stage history.
- Throttle history.
- Dynamic pressure.
- Acceleration.
- Flight path angle.
- Events.
- Insertion state.
- Target miss metrics.
- Backend provenance.

The insertion state must be convertible to the common `OrbitState` used by the orbital module.

## Backend Plan

### RocketPy Adapter

Role:

- Direct launch simulation and dispersion-oriented workflows.
- Useful for atmospheric rocket dynamics and early examples.

Strength:

- Focused on rocket trajectory modeling.
- Useful stochastic and atmospheric workflows.

Limit:

- Not the main backend for orbital launch optimization.

### Dymos / OpenMDAO Adapter

Role:

- Primary ascent optimization backend.

Strength:

- Multiphase trajectory optimization.
- Collocation methods.
- Path constraints.
- Analytic derivative support.
- Strong fit for staging and target insertion.

Limit:

- Requires careful transcription, scaling, and problem setup.

### Local Launch Baseline

Role:

- Minimal deterministic sanity checks.

Scope:

- 1D vertical ascent toy problem.
- Simple mass depletion.
- Constant thrust.
- Constant or simple atmosphere.

Limit:

- It is not a production launch simulator. It exists to verify data flow and test harnesses.

## Launch Data Flow

1. User defines a launch scenario.
2. `astro_core` validates common scenario fields.
3. `astro_launch` validates vehicle, site, stage, atmosphere, and target orbit.
4. Adapter converts launch objects into backend-specific model.
5. Backend runs simulation or optimization.
6. Adapter returns `LaunchTrajectory`.
7. Insertion state is converted into `OrbitState`.
8. Optional orbital propagation continues from insertion.

## Validation Plan

### Sanity Cases

1. Constant-thrust vertical ascent with no drag.
2. Constant-thrust vertical ascent with simple drag.
3. Two-stage ascent event sequencing.
4. Gravity-turn qualitative trajectory.
5. Target orbit insertion handoff to orbital propagator.
6. Dymos small ascent optimization example.

### Invariants

- Mass never increases except by explicit configured event.
- Propellant cannot go negative.
- Stage separation events are monotonic.
- Throttle stays within bounds.
- Dynamic pressure is nonnegative.
- Insertion state includes complete frame, epoch, and units.

## Error Handling

Launch-specific errors:

- Invalid launch site.
- Invalid stage sequence.
- Invalid engine model.
- Propellant depletion before expected event.
- Target orbit infeasible for configured vehicle.
- Optimizer convergence failure.
- Path constraint violation.
- Backend does not support requested guidance or atmosphere model.

## Integration With Orbital Module

Launch output should hand off cleanly to orbital workflows.

The interface is:

- `LaunchTrajectory.insertion_state -> OrbitState`
- `LaunchTrajectory.events -> Trajectory.events`
- `TargetOrbit.miss_metrics -> validation report`

This keeps launch and orbital propagation coupled through products, not through internal implementation details.

## Tradeoffs

Launch deserves its own spec because it has a separate physics and optimization surface. Combining it into the orbital/OD MVP would make the first implementation too broad and weakly validated. Separating it too aggressively would risk incompatible data models. The chosen design keeps launch first-class in the umbrella architecture and gives it a dedicated module spec, while delaying implementation until the orbital spine is stable.

## Recommended Implementation Timing

Implement launch after the orbital/OD MVP reaches these gates:

- Scenario schema is stable.
- `Trajectory` and `OrbitState` products are stable.
- CLI can validate and run a scenario.
- Reference test harness exists.
- Backend adapter pattern is proven with at least one orbital backend.

Once those gates pass, start launch with the local 1D baseline and scenario validation, then add RocketPy for direct simulation, then Dymos/OpenMDAO for ascent optimization.

## Open Decisions For Launch Planning

- First launch backend order: RocketPy before Dymos, or Dymos before RocketPy.
- Initial launch vehicle example.
- Whether to target orbital launch insertion or suborbital rocket simulation first.
- Atmosphere model baseline.
- Target orbit parameterization and tolerance scheme.
