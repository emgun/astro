# Subsystem Fidelity Pack Design

## Goal

Improve Astro Suite's deterministic digital twin subsystem evidence while preserving the suite-owned design-screening boundary.

## Scope

This pack extends the existing single-spacecraft twin and all constellation member twins because constellation runs embed `DigitalTwinResult` products. The implementation stays inside `src/astro_twin` and does not introduce optional backend dependencies.

## Product Surface

- Power: support named scheduled load additions, battery charge/discharge efficiency, battery energy evidence, and scheduled-load evidence per sample.
- Thermal: support simple orbital flux terms and mission-mode heat scaling while preserving the existing lumped-node model.
- ADCS: add deterministic slew-rate and actuator-utilization screening evidence.
- Mass/design margins: add itemized mass budget rollups with contingency and a consistency margin against configured spacecraft mass.

## Claim Boundary

The pack raises screening fidelity for early architecture trades. It is not an electrical power system simulator, thermal certification model, flight-qualified GNC simulation, mass-properties authority, or operational design approval.

## Acceptance

- Existing simple twin scenarios remain valid.
- The checked `examples/twin/leo_observer.yaml` exercises the new fields.
- `astro run-twin` writes the richer subsystem evidence and margin names.
- Constellation member twins inherit the richer `DigitalTwinResult` products without a separate constellation-specific subsystem implementation.
- Required local tests, lint, typing, packaging, and build gates pass without optional backends.
