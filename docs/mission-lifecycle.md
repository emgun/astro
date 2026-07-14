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
astro verify-mission-lifecycle-result /tmp/astro-mission-lifecycle.json \
  examples/lifecycle/leo_round_trip.yaml
astro review-mission-lifecycle /tmp/astro-mission-lifecycle.json \
  examples/lifecycle/leo_round_trip.yaml \
  --output /tmp/astro-mission-lifecycle-review.json \
  --summary-output /tmp/astro-mission-lifecycle-review.txt
astro verify-mission-lifecycle-review /tmp/astro-mission-lifecycle-review.json
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

## Deterministic Assurance Review

Lifecycle review v1 supports local launch and reentry backends. It binds the exact result and
scenario bytes, re-runs the authoritative five-phase lifecycle, and requires canonical product
payload equality before deriving findings. It also binds the exact launch, twin, and reentry
scenario-file digests and rejects changes during verification. The verifier executes against
temporary staged copies of those captured bytes, so concurrent replacement of the original files
cannot change the evidence being reproduced. The review covers integrity, manifest order,
continuity, typed margins, model caveats, and claim boundaries. Structured failed continuity checks
and `warn`/`fail` margins produce stable blocker or warning findings and one non-executing triage
action per unresolved finding.

The checked reference also reports one evidence-quality warning: its promoted digital-twin limiting
margin uses the generic unit label `native`. The lifecycle itself passes, but the review requires an
explicit physical unit before that margin is used in unit-specific design decisions.

Free-text lifecycle, twin, and reentry warnings remain informational evidence-boundary findings.
The reviewer does not infer anomaly severity or root cause from prose or rank unlike margins by raw
magnitude. Typed status and the lifecycle runner's limiting-margin identity remain authoritative.
`verify-mission-lifecycle-review` repeats the complete chain and rejects stored review tampering.

This v1 proves exact output reproducibility against the captured top-level and referenced scenario
files under the current local runtime. The existing lifecycle artifact manifest has no file
digests, so the review does not claim
cryptographic integrity for a separately copied artifact bundle. It does not establish probability,
causality, flight qualification, subsystem certification, operational diagnosis, or remediation
authority.

## Claim Boundary

This is deterministic design-screening evidence, not launch certification, an operational mission
digital twin, flight-qualified GNC, certified propulsion sizing, atmospheric uncertainty authority,
TPS certification, or landing prediction. The reference deorbit is an impulsive two-body burn and
coast. Reentry retains the assumptions and limitations documented in
[Reentry Modeling And Simulation](reentry.md), and the subsystem evidence retains the boundaries in
[Digital Twin](digital-twin.md).

Optional launch or reentry backends remain adapters. Public lifecycle outputs are Astro Suite
models with explicit backend provenance; native engine objects do not cross the product boundary.
