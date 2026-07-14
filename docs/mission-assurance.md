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
6. Replay the commanded candidate against the estimate and a separately modeled executed impulse
   against simulation truth.
7. Drive the digital twin with the corrected truth trajectory.
8. Return continuity checks, margins, product digests, warnings, and a derived decision.

The checked example uses a one-hour, six-site geodetic acquisition network. It generates 1,452
deterministic range and range-rate candidates; 130 meet each station's elevation mask, and OD uses
the 64 available by the 30-minute correction decision. The 66 later visible observations remain
truth-verification evidence and cannot influence targeting. Its correction objective balances
terminal position and velocity because one
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

## Uncertainty Campaign

The first integrated campaign is available through the generic UQ commands:

```bash
astro validate-campaign examples/campaigns/leo_mission_assurance_robustness.yaml
astro run-campaign examples/campaigns/leo_mission_assurance_robustness.yaml \
  --output-dir /tmp/astro-assurance-uq --workers 2
astro summarize-campaign /tmp/astro-assurance-uq
```

The checked eight-case LHS campaign varies insertion position and velocity, tracking range and
range-rate sigma, correction execution scale, solar-array efficiency, and bus emissivity. The
input contract labels insertion and execution dispersion as aleatory and sensor/subsystem values as
epistemic. Model variants carry an explicit `model_form` label, but the reference declares none:
no alternate force or measurement model has yet passed a paired validation campaign.

The tracking generator intentionally retains a fixed seed, so cases use common random numbers for
parameter screening rather than independent noise realizations. Its bounds are illustrative and
the reported requirement fractions use all completed cases as the denominator. They are
conditional design-space frequencies, not calibrated mission-success probabilities. Operational
or probabilistic claims still require independent and calibrated tracking noise/bias evidence,
separate truth and estimator noise assumptions, maneuver timing and pointing errors, paired
force-model mismatch, external tracking fixtures, and reviewed acceptance criteria. Monte Carlo
volume alone does not supply that authority.

## Paired Model Validation

The paired validation protocol separates model-form mismatch from continuous realization inputs:

```bash
astro validate-assurance-validation \
  examples/assurance/paired_force_model_validation.yaml
astro run-assurance-validation \
  examples/assurance/paired_force_model_validation.yaml \
  --output /tmp/astro-paired-assurance.json \
  --summary-output /tmp/astro-paired-assurance.txt
astro verify-assurance-validation /tmp/astro-paired-assurance.json
astro review-assurance-validation /tmp/astro-paired-assurance.json \
  --output /tmp/astro-paired-assurance-review.json \
  --summary-output /tmp/astro-paired-assurance-review.txt
```

Each explicit coordinate carries its own tracking-noise seed, truth sigma and bias, estimator sigma
and bias, insertion dispersion, correction magnitude/timing/two-axis pointing errors, and selected
power/thermal inputs. The same coordinate and seed run through two profiles: matched two-body truth
and estimation, then J2 truth with two-body estimation and targeting. Every successful profile
embeds its complete `MissionAssuranceCase`; paired deltas use the convention mismatched minus
matched. Failed physics profiles remain visible and are never removed from requested-case counts.
Each successful profile reports both the embedded case decision under the diagnostic solver envelope
and the stricter protocol pass after reapplying the base scenario's maneuver-authority limits. The
verifier rechecks the protocol and assurance sources, every nested assurance input, each embedded
case digest, and every manifest product digest against the embedded product. It also loads the
protocol's calibration manifest, verifies its file digest, and requires exact parameter coverage,
matching units, and containment of every configured value within its declared envelope.

### Calibration Evidence

`examples/assurance/paired_force_model_calibration.yaml` gives all 26 configured numeric dimensions
an executable evidence record. Each bound declares its authority, source identifiers, rationale,
limitations, unit, and allowed envelope. Promotion derives from the weakest bound and cannot be set
independently:

- `illustrative` means at least one bound is illustrative.
- `reference_informed` means every bound has project or external-reference support, but at least one
  lacks mission-test or flight calibration.
- `mission_calibrated` requires every bound to be backed by mission-test or flight data.

The current manifest remains `illustrative`. NASA Near Space Network and LRO tracking references
inform range and range-rate noise envelopes; a Vega C user manual informs insertion scale; NASA TESS
and JPL Cassini navigation reports inform maneuver-magnitude and pointing categories. These sources
do not calibrate this synthetic LEO network, spacecraft, propulsion system, or thermal/power model.
The configured tracking biases and four-second maneuver timing error remain illustrative. Promotion
therefore requires mission-specific station residuals, insertion covariance with frame and
correlation semantics, propulsion execution residuals including timing, and reviewed subsystem
acceptance evidence. A larger case count cannot promote the evidence class.

### Deterministic Review

`review-assurance-validation` verifies the complete paired result before deriving a suite-owned
decision-support artifact. Findings cover integrity, calibration authority, pair completeness,
pass reversals, selected signed metric shifts, and the source claim boundary. Unlike units are not
ranked by raw magnitude. Identical source bytes
produce identical review bytes. Tampered evidence or path collisions prevent publication.

The review disposition is `additional_evidence_required` when calibration or completeness blocks
claim promotion; otherwise it is `design_review_ready`. Neither disposition authorizes navigation,
probability, autonomous action, or a flight command. A later AI explainer may reference finding ids,
but cannot create, remove, downgrade, or modify deterministic findings.

The checked eight-coordinate, one-hour protocol uses 30 minutes of causal pre-decision tracking and
30 minutes of post-decision truth verification. It completed all eight pairs. All eight matched
profiles passed the preserved base authority and all mismatched profiles failed, producing eight
pass-to-fail reversals. Relative to matched cases, model mismatch added a median `13.246 km` of OD
position error, `36.785 km` of truth-recovery position error, and `43.861` of normalized residual
RMS. Mismatched candidates required `3.375` to `7.729 m/s` beyond the original total authority;
their executed impulses exceeded the original component limit by `7.795` to `11.446 m/s`.

The protocol uses a `50 m/s` component and `80 m/s` total diagnostic targeting envelope so both
profiles can produce comparable trajectories. Profile pass disposition separately enforces the
original assurance scenario's `20 m/s` component and `25 m/s` total authority. The wider diagnostic
envelope is not an authorized maneuver bound. Results report profile counts, physics-complete pair
counts, paired metric deltas, and reversal counts. They contain no pooled success probability.
Coordinates and the overall manifest remain illustrative until launch, sensor, propulsion, and
subsystem evidence calibrates every bound; the product is simulation design-space validation, not
navigation certification or flight authority.
