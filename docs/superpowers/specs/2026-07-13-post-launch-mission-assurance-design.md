# Post-Launch Mission Assurance Design

**Status:** Implemented, verified, and integrated through PR #17
**Date:** 2026-07-13

## Goal

Make post-launch orbit acquisition the first closed-loop Astro Suite mission-assurance product. A
single workflow starts from a launch insertion, creates a dispersed simulation truth case, generates
tracking, estimates the orbit, designs one bounded corrective impulse, replays the correction against
both the estimate and truth, updates the digital twin, and returns one evidence-bound decision.

## Product Boundary

`astro_assurance` owns orchestration, continuity checks, margins, manifests, artifact bundles, and
claim boundaries. It reuses launch, local propagation, measurement generation, suite OD, impulsive
maneuvers, and the digital twin. It does not duplicate their physics or return backend-native objects.

The first release is deterministic local design screening. Synthetic truth and truth-error metrics
are simulation-only diagnostics. The corrective maneuver is a bounded single-impulse targeting
screen, not an operational flight-dynamics command, finite-burn design, navigation certification,
or autonomous spacecraft control.

## Workflow

```text
launch insertion
  -> nominal acquisition trajectory
  -> configured truth dispersion
  -> synthetic tracking
  -> batch orbit determination from nominal prior
  -> bounded impulsive correction targeting
  -> estimated-state correction replay
  -> truth-state correction replay
  -> corrected digital twin
  -> continuity, margin, manifest, and decision products
```

## Scenario Contract

`PostLaunchAssuranceScenario` references:

- a launch scenario and backend;
- an orbit tracking template that owns stations, measurements, force model, cadence, and duration;
- a digital-twin template;
- a six-component truth insertion dispersion;
- correction epoch and verification epoch aligned to propagation samples;
- correction search bounds and position/velocity objective scales;
- OD, correction, recovery, propellant, and twin requirement thresholds.

All referenced paths and loaded-product digests are recorded in the result manifest. The tracking
template's initial state and spacecraft mass are replaced by launch handoff values; its force model,
stations, measurements, and propagation settings remain authoritative for this workflow.

## Result Contract

`MissionAssuranceCase` contains suite-owned launch, scenario, trajectory, measurement, estimate,
maneuver, corrected trajectory, and digital-twin products plus:

- continuity checks for epoch, state, and product handoffs;
- decision margins with stable units and pass/warn/fail status;
- separate estimate-predicted and simulation-truth recovery errors;
- a phase manifest with source digests and evaluator provenance;
- `passed`, derived only from continuity and required margins;
- warnings that preserve the deterministic screening and simulation-truth boundaries.

## Correction Strategy

The local targeting screen propagates the estimated state to the correction epoch, applies a trial
inertial impulse, and minimizes scaled terminal position and velocity error against the nominal
trajectory. SciPy least-squares is bounded per component. The same resulting maneuver is replayed
from simulation truth. Failure to converge, an out-of-bound solution, or a failed recovery margin is
visible evidence rather than a silently accepted maneuver.

## Acceptance

- The checked reference begins from a nonzero launch insertion dispersion.
- OD converges with a full-rank Jacobian from generated measurements.
- Every retained synthetic observation meets its station elevation mask.
- The correction reduces truth terminal position error relative to the uncorrected truth case.
- The result distinguishes estimated and truth recovery metrics.
- The corrected truth trajectory drives the returned digital twin.
- Any failed corrected-twin design margin fails the top-level assurance decision.
- The loaded assurance definition and three referenced templates remain digest-stable through the
  run; the fixed artifact schema rejects omissions and tampering.
- Artifact loading, summaries, CLI execution, and failure gates are tested.
- Required local, typing, packaging, and full regression gates pass.
