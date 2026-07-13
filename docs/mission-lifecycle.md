# Mission Lifecycle Workflow

Astro Suite's mission lifecycle workflow connects the existing launch, orbital propagation,
integrated digital twin, deorbit, and reentry products into one checked phase chain:

```text
launch -> orbit handoff -> operations twin -> deorbit burn/coast -> reentry
```

The workflow does not replace the phase models. It orchestrates them and verifies the state, epoch,
and mass boundaries between their suite-owned products.

## Run The Reference Mission

```bash
astro run-mission-lifecycle examples/lifecycle/leo_round_trip.yaml \
  --output /tmp/astro-mission-lifecycle.json \
  --summary-output /tmp/astro-mission-lifecycle.txt \
  --artifacts-dir /tmp/astro-mission-lifecycle-artifacts
```

The checked reference uses deterministic local models throughout. Its launch fixture satisfies its
declared altitude, velocity, and radial-velocity insertion tolerances before the workflow is allowed
to proceed. The operations trajectory is passed directly into the digital twin, so subsystem
screening uses the same orbit product that feeds the deorbit phase.

## Product Contract

`MissionLifecycleResult` embeds:

- the launch trajectory and generated orbit scenario;
- the operations trajectory and integrated digital-twin result;
- the retrograde deorbit maneuver, post-burn scenario, and coast-to-interface trajectory;
- the generated reentry scenario and reentry result;
- continuity checks, cross-phase margins, and an ordered phase manifest.

With `--artifacts-dir`, the command also writes independently parseable phase files and
`manifest.json`. Artifact names are stable within the `mission_lifecycle_v1` workflow.

## Typed Campaign Overrides

Lifecycle uncertainty campaigns may override solar-array area and efficiency, battery capacity,
and named thermal-node emissivity or internal-heat fraction through
`MissionLifecycleInputOverrides`. Thermal fields are stored as one typed override per node and are
resolved only against names present in the referenced twin template. Missing or duplicate node
names fail before campaign execution, and the complete resolved `DigitalTwinScenario` is validated
before the twin runs.

CLI campaign definitions record both the referenced twin-template digest and the fully resolved
twin-scenario digest after lifecycle overrides. A changed template or preconfigured override
therefore changes the campaign definition digest and prevents evidence resume under stale nested
physics.

These overrides are campaign inputs, not edits to the referenced scenario files. They preserve the
base scenario and sampled-value provenance in each case.

## Fail-Closed Gates

Execution stops before downstream phases when:

- launch insertion exceeds any declared target tolerance;
- the digital-twin wet mass differs from the launch payload mass;
- the rocket-equation deorbit burn violates the configured propellant reserve;
- the deorbit coast does not encounter the descending entry interface;
- the sampled interface altitude exceeds its configured tolerance; or
- a requested phase backend is unavailable.

The result reports launch insertion margins, the twin limiting margin, deorbit reserve and
interface margins, and every reentry requirement margin. Warnings remain visible without being
promoted to failures.

## Claim Boundary

This is deterministic design-screening evidence, not launch certification, an operational mission
digital twin, flight-qualified GNC, certified propulsion sizing, atmospheric uncertainty authority,
TPS certification, or landing prediction. The reference deorbit is an impulsive two-body burn and
coast. Reentry retains the assumptions and limitations documented in
[Reentry Modeling And Simulation](reentry.md), and the subsystem evidence retains the boundaries in
[Digital Twin](digital-twin.md).

Optional launch or reentry backends remain adapters. Public lifecycle outputs are Astro Suite
models with explicit backend provenance; native engine objects do not cross the product boundary.
