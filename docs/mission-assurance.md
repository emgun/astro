# Post-Launch Mission Assurance

Astro Suite's post-launch mission-assurance workflow connects launch insertion, radiometric
tracking, orbit determination, candidate correction design, truth replay, and the integrated
digital twin into one suite-owned result.

## Run The Reference

```bash
astro run-mission-assurance examples/assurance/post_launch_orbit_acquisition.yaml \
  --output /tmp/astro-mission-assurance.json \
  --summary-output /tmp/astro-mission-assurance.txt \
  --artifacts-dir /tmp/astro-mission-assurance-artifacts

astro verify-mission-assurance /tmp/astro-mission-assurance-artifacts
```

The command exits `0` when continuity and required margins pass, `1` when the workflow completes
with failed requirements, and `2` for invalid input or execution failure. A completed failed case
still writes its result and evidence bundle.

## Workflow

The reference performs these phases in order:

1. Run the configured launch and hand its insertion state and mass into the orbit template.
2. Apply a configured six-component insertion dispersion to a separate simulation-truth state.
3. Propagate nominal and truth trajectories and generate deterministic synthetic tracking.
4. Estimate the initial orbit from the nominal prior with suite-owned batch OD.
5. Design one bounded inertial impulsive correction from the estimated trajectory.
6. Replay the same candidate against both the estimate and simulation truth.
7. Drive the digital twin with the corrected truth trajectory.
8. Return continuity checks, margins, product digests, warnings, and a derived decision.

The checked example uses a one-hour, six-site geodetic acquisition network. It generates 1,452
deterministic range and range-rate candidates and retains 130 records that meet each station's
elevation mask. Its correction objective balances terminal position and velocity because one
three-component impulse cannot independently satisfy all six terminal state components.

## Products And Evidence

`MissionAssuranceCase` embeds every suite-owned product needed to audit the decision: launch and
orbit products, measurements, OD estimate, candidate maneuver, estimated and truth replays, the
corrected digital twin, continuity checks, requirement margins, and a phase-ordered digest
manifest. The artifact directory writes 16 digest-bound products plus the manifest. Input
references bind the assurance, launch, tracking, and twin source files by absolute path and
file-byte digest.
`astro verify-mission-assurance` recomputes every canonical artifact digest, input digest, and the
exact expected file set. Bundle publication stages all files, writes the completion manifest last,
uses an exclusive lock to serialize Astro Suite writers, checks the destination again immediately
before atomic rename, and refuses stale or concurrently created destinations. Uncoordinated
external filesystem mutation after the final check is outside this cooperative-writer guarantee.
Digests provide integrity and completeness checking; they are not signatures and do not establish
publisher identity or adversarial authenticity.

Estimate-predicted recovery and simulation-truth recovery remain separate. Requirement margins
name their evidence scope as `simulation_truth`, `design_screening`, or `decision_available`.
`passed` is derived from continuity and margin status; callers cannot set it independently. Any
failed embedded digital-twin margin contributes a failed top-level margin count.

## Claim Boundary

This workflow is deterministic local simulation and design-screening evidence. Synthetic
measurements do not establish RF contact, spacecraft identity, real navigation performance, or
ground-system readiness. The correction is a candidate for manual review, not a flight command,
finite-burn design, autonomous control action, or operational maneuver authority. Digital-twin
resource results retain the limitations documented in [Digital Twin](digital-twin.md).

The first version intentionally uses the local propagator for correction replay. Optional backends
do not yet share a uniform maneuver contract, so backend expansion requires a separately validated
adapter campaign rather than silent substitution.

## Next Validation Layer

The deterministic reference is the baseline for a future mission-assurance uncertainty campaign.
That campaign should disperse launch, tracking, OD, execution, and subsystem inputs through the
existing `astro_uq` contract while preserving failed cases and evidence scopes. Operational or
probabilistic claims require external tracking fixtures, calibrated noise and bias models, maneuver
execution uncertainty, and reviewed acceptance criteria; Monte Carlo volume alone does not supply
that authority.
