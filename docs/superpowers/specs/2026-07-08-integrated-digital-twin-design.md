# Integrated Digital Twin Design

Date: 2026-07-08
Status: Draft for user review

## North Star

Build a suite-owned spacecraft digital twin that turns an Astro orbit scenario and spacecraft design
into a time-indexed mission evidence product: orbital state, sunlight/eclipse, power, thermal,
ADCS/pointing, ground access, link budget, mass budget, and design margins.

The product should be useful for early mission design, trade studies, and verifiable workflow
demonstrations. It should not claim flight qualification, detailed thermal certification, RF
certification, or operational autonomy. External engines may improve individual physics later, but
public outputs stay in Astro Suite models with explicit provenance and claim boundaries.

## Selected Approach

Use an integrated `astro_twin` product layer rather than isolated subsystem utilities.

The twin owns mission-level orchestration and report products. It reuses existing Astro propagation,
attitude, OD, launch, and workflow conventions where they already exist, then adds deterministic
screening models for subsystem interactions. A single command should run a full v1 twin and produce
one JSON result plus a concise margin summary.

## Scope

Version 1 targets one Earth-orbiting spacecraft over a finite analysis window.

Included:

- Orbit and geometry timeline from an existing `Scenario` and local propagation.
- Sunlight/eclipse flag using deterministic Earth-shadow geometry.
- Power timeline with solar generation, mode loads, battery state of charge, and power margins.
- Lumped thermal screening for bus, battery, and payload nodes.
- ADCS/pointing diagnostics for mission modes and simple slew/torque margins.
- Ground-station or target access windows with elevation masks.
- RF link budget per access window with free-space path loss, C/N0, Eb/N0, and margin.
- Static and dynamic mass/design margin tracking.
- JSON result and text summary products.
- Golden example workflow for a LEO Earth-observation spacecraft with one ground station downlink.

Deferred:

- Constellation aggregation across multiple spacecraft.
- Detailed multi-node thermal networks and orbital beta-angle thermal campaigns.
- Reaction-wheel momentum management, desaturation, flexible body dynamics, star-tracker blinding,
  and flight-qualified ADCS modeling.
- Rain-fade models, atmospheric attenuation certification, antenna pattern files, and regulatory RF
  analysis.
- Power degradation, battery aging, solar array temperature effects, and detailed EPS switching.
- Closed-loop payload operations planning or autonomous schedule optimization.

## Architecture

```text
DigitalTwinScenario
  -> load base orbit Scenario
  -> propagate local Trajectory
  -> build TwinTimeline
  -> run subsystem models
  -> aggregate DesignMarginReport
  -> write DigitalTwinResult
```

### Module Boundaries

`astro_twin.models`
: Pydantic schemas for twin scenarios, subsystem configs, timeline samples, subsystem samples,
  margins, and final result products.

`astro_twin.io`
: YAML/JSON loaders and result writers. Keeps validation errors consistent with `astro_core.io`.

`astro_twin.geometry`
: Converts a trajectory into deterministic time-indexed geometry samples: altitude, sunlight,
  ground-station elevation, range, and access flags.

`astro_twin.power`
: Computes solar generation, loads, battery state of charge, depth-of-discharge, and power margins.

`astro_twin.thermal`
: Computes lumped-node thermal screening temperatures and thermal limit margins.

`astro_twin.adcs`
: Computes mission-mode pointing/slew/torque diagnostics and ADCS resource margins. Reuses the
  existing attitude module when a full rigid-body attitude propagation is requested later.

`astro_twin.coverage`
: Extracts ground-station and target access windows from geometry samples.

`astro_twin.link_budget`
: Computes per-link and per-contact RF link margins and data volume.

`astro_twin.margins`
: Aggregates mass, power, thermal, ADCS, coverage, and link budget checks into one design margin
  report.

`astro_twin.runner`
: Orchestrates the full workflow and returns `DigitalTwinResult`.

`astro_cli.main`
: Adds `astro run-twin <scenario.yaml> --output <result.json> --summary-output <summary.txt>`.

## Data Flow

1. A twin scenario references an existing orbit scenario.
2. The runner loads the orbit scenario and propagates it with the existing local backend.
3. Geometry samples are derived from trajectory samples.
4. Subsystem models consume the same timeline and mission mode schedule.
5. Coverage produces access windows; link budgets consume access windows.
6. Margins aggregate the limiting values from every subsystem.
7. The CLI writes a suite-owned JSON result and optional text summary.

## Product Boundary

The v1 twin is a deterministic screening product. Reports must describe margins as design-screening
evidence, not operational readiness, flight certification, or production mission assurance.

Every result includes:

- `workflow = "integrated_digital_twin_v1"`
- Astro Suite version metadata when available.
- Input scenario path.
- Backend provenance for orbit propagation.
- Explicit model names for power, thermal, ADCS, coverage, and link budget.
- A warning list for simplified assumptions.

## Validation Strategy

Required local gates:

- Unit tests for every subsystem formula and margin classification.
- Scenario loader tests for invalid units, missing mission modes, negative capacities, and invalid
  threshold ordering.
- End-to-end CLI test using `examples/twin/leo_observer.yaml`.
- Golden result assertions for sample count, min battery state of charge, max thermal node
  temperature, contact count, worst link margin, and limiting design margin.
- Packaging tests ensuring `astro_twin` is included in wheel and mypy packages.
- `python -m ruff check .`, `python -m mypy`, `python -m pytest -q`, `git diff --check`, and
  `python -m build`.

## Version 1 Acceptance

The first implementation is complete when:

- `astro run-twin examples/twin/leo_observer.yaml --output /tmp/astro-twin.json
  --summary-output /tmp/astro-twin.txt` succeeds without optional backend dependencies.
- The JSON product contains orbit, power, thermal, ADCS, coverage, link budget, mass budget, and
  design margin sections.
- The text summary identifies the limiting margin and major subsystem minima/maxima.
- The validation matrix documents the command and claim boundary.
- No public product leaks external engine-native objects.

## Later Roadmap

1. Constellation aggregation: multi-spacecraft coverage, fleet contact schedules, and per-spacecraft
   margin outliers.
2. Higher-fidelity thermal: configurable multi-node conductance/radiation networks.
3. ADCS depth: wheel momentum, desaturation, sensor exclusion angles, and pointing timeline coupling.
4. RF depth: antenna patterns, atmospheric losses, rain zones, Doppler, and CCSDS link products.
5. Design optimization: sweep solar array, battery, radiator, antenna, and mass allocations against
   margin targets.
