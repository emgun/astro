# Constellation Coverage Maps Design

## Goal

Add deterministic constellation target-grid coverage evidence to the existing suite-owned digital twin workflow.

## Scope

Coverage Maps v1 belongs to `astro run-constellation-twin`. A constellation scenario may define one or more target grids, each with a nadir-pointed sensor cone and latitude/longitude targets. The runner evaluates each member twin's sampled orbit geometry against those targets and writes fleet-level coverage summaries into the `ConstellationTwinResult`.

## Product Boundary

The product remains design-screening evidence. It uses spherical Earth geometry, uniform Earth rotation, sampled local propagation, target elevation, optional range, and off-nadir sensor cone checks. It does not claim operational coverage authority, onboard pointing feasibility, scheduling, collision avoidance, spectrum coordination, certified sensor performance, or provider-backed mission operations.

## Result Shape

Each coverage map summary records:

- target count and covered target count
- mean and minimum target coverage fraction
- maximum target gap
- maximum simultaneous spacecraft
- per-target coverage fraction, total covered duration, longest gap, mean gap, and simultaneous-spacecraft count

The fleet margin report includes coverage-map minimum-fraction and optional maximum-gap margins.

## Acceptance

- Scenario models validate coverage map, sensor, and target definitions.
- Duplicate coverage map names and duplicate target names are rejected.
- Aggregation handles covered and uncovered targets deterministically.
- `astro run-constellation-twin examples/twin/constellation_leo_observers.yaml` writes coverage-map JSON and text summary evidence.
- Tests, lint, typing, and package checks pass without optional backend runtimes.
