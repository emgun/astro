# Astro Suite Current State

Date: 2026-07-15 00:00 PDT

## Canonical Workspace

Use `/Users/emerygunselman/Code/astro` for this roadmap thread.

The sibling checkout at `/Users/emerygunselman/Documents/astro` is stale for this work. It remains on
`codex/orbit-fd-od-mvp` at `ecccfa6`, has `main` at `dc5e253`, and contains an untracked
`docs/research/2026-06-20-verifiable-ai-space-workflows.md` file. Do not edit or merge from that
checkout unless the user explicitly asks to sync the old workspace.

Integrated `main` state after the post-launch mission-assurance merge:

- Pre-lifecycle release baseline: annotated tag `v0.1.0-rc.1` at `1182493`.
- Lifecycle-inclusive release candidate: annotated tag `v0.1.0-rc.2` at `6cc7d2f`.
- Final release: annotated tag `v0.1.0` at the reviewed candidate commit `6cc7d2f`.
- Release candidate: annotated tag `v0.2.0-rc.1` at reviewed merge commit `f1baa02`, covering the
  integrated reentry, lifecycle, uncertainty, digital-twin campaign, and mission-assurance
  increment after `v0.1.0`. Local, GitHub CI, and detached merged-commit gates pass, including `958
  tests with 11 optional-backend skips`, package builds and inspection, and the public paired-
  assurance workflow from an isolated wheel installation.
- Branch: `main`; Mission Lifecycle Workflow merge commit `d37b9a6` and state commit `6cc7d2f`.
- Required local release gates: passed on merged `main` after PR #9.
- Optional backend smoke refresh on 2026-07-09: Orekit, RocketPy, Dymos/OpenMDAO, and JAX
  were available on the current machine. TudatPy was not installed in the base environment, and
  the previously recorded isolated Tudat environment at `/tmp/astro-tudat-live-env` no longer
  existed.
- Bounded live validation refresh on `codex/orekit-validation-refresh`: Orekit high-fidelity
  covariance, drag, SRP, and Sun/Moon third-body trajectory products executed with the explicit
  Homebrew OpenJDK and `~/.orekit/orekit-data.zip` environment. This is machine-scoped optional
  backend evidence, not a production covariance certification. PR #7 merged this scope into `main`
  at merge commit `b668e1f`.

Integrated reentry implementation:

- PR #8 merged `codex/reentry-suite` into `main` at `3c73f8b`.
- Public products and commands: suite-owned `ReentryScenario`, `ReentryResult`, and
  `ReentryOptimizationResult`; `astro simulate-reentry`, `astro optimize-reentry`, and
  `astro handoff-reentry`.
- Required local gates on 2026-07-10: `676 passed, 11 skipped`; `ruff`, strict `mypy` across 92
  source files, `git diff --check`, 4 packaging tests, and `python -m build` passed.
- GitHub CI passed before merge; the required local gates were repeated on integrated `main`.

## North Star

Astro Suite is a Python flight-dynamics suite with suite-owned product boundaries for orbital
simulation, flight dynamics, orbit determination, launch/ascent, reentry, integrated digital twins,
optional operational backends, and research workflows. External engines are adapters; public
outputs stay in Astro Suite models with explicit provenance and claim boundaries.

## Current Roadmap Decision

The broader product direction is now **evidence-bound mission assurance**: connect Astro Suite's
existing physics, estimation, uncertainty, and digital-twin products into closed-loop decisions
with explicit continuity, provenance, claim boundaries, and review authority. The first flagship is
post-launch orbit acquisition and recovery: launch insertion, simulated tracking, OD, a bounded
candidate correction, estimate/truth replay, and a corrected digital twin in one result. PR #17
merged the implementation to `main` at `0d73d5b`; it remains deterministic local design screening
and does not claim RF acquisition, operational navigation, or flight-command authority.

The first uncertainty layer over the deterministic assurance reference is implemented through the
generic campaign engine. A paired assurance-validation protocol now separates independent tracking
realizations, estimator-noise mismatch, execution timing/pointing, and force-model mismatch. AI should
enter through typed planning, evidence review, anomaly triage, and bounded decision support before
any autonomous execution. Surrogate training remains stopped for current cheap local evaluators
unless a mission-assurance teacher or other evaluator passes the existing cost and fidelity gate.

Branch verification on 2026-07-13 passed the visibility-filtered reference with 130 above-mask
synthetic measurements from 1,452 candidates. Causal hardening limits OD to the 64 measurements
available by the correction decision and reserves 66 later records for truth verification. The
updated case converges with a `0.004060974 km/s` candidate impulse and reduces truth position error
from `8.558551 km` to `0.528331 km`. Failed embedded twin margins
now fail the assurance case, correction propellant depletion updates the returned twin, and the
17-file bundle binds four inputs plus 16 artifacts with a public verifier and atomic publication.
Focused assurance and packaging tests pass `21` tests; the full suite passes `930 tests with 11
optional-backend skips`; Ruff, strict MyPy across 127 source files, `git diff --check`, the fresh
public CLI run and verifier, wheel inspection, and sdist/wheel builds pass. GitHub CI passed in
`1m25s` before PR #17 merged the scope at `0d73d5b`.

Mission Assurance Uncertainty Campaign v1 adds typed UQ bindings for insertion components,
tracking range/range-rate sigma, correction execution scale, solar-array efficiency, battery
capacity, and named thermal-node properties. The truth replay now distinguishes the commanded
candidate from the executed impulse, depletes executed propellant, and gates executed delta-v.
Model alternatives carry an explicit `model_form` classification, while the checked reference
declares no alternate model. A fresh two-worker 8-case LHS run completed 8/8 cases with all ten
configured requirements passing inside illustrative bounds. The causal refresh retains 64
pre-decision measurements per case. Recovery position error is `0.4151` to `0.6741 km` at q05/q95,
recovery velocity error is `0.002841` to `0.004101 km/s`, and propellant reserve is `49.1017` to
`49.5196 kg`. The fixed tracking seed provides common random
numbers, not independent measurement-noise draws; these are all-completed-case design-space
frequencies, not calibrated success probabilities or operational authority. Independent review
found and the implementation fixed missing top-level assurance-source binding and repository-CWD
path dependence. The final local gate passes `935 tests with 11 optional-backend skips`, Ruff,
strict MyPy across 128 source files, `git diff --check`, package builds, and the public campaign from
outside the repository.

Paired Mission-Assurance Validation v1 adds a suite-owned protocol/result surface and public
validate, run, and verify commands. Eight explicit coordinates use unique tracking seeds across
cases and common random numbers within each matched/mismatched pair. The one-hour arc reserves the
first 30 minutes for causal OD before the correction decision and the second 30 minutes for truth
verification. All eight pairs complete.
Matched two-body profiles pass 8/8; truth-J2/two-body-estimator profiles pass 0/8, yielding eight
pass-to-fail reversals. Median mismatched-minus-matched shifts are `+13.246 km` for OD position
error, `+36.785 km` for truth-recovery position error, and `+43.861` for normalized residual RMS.
Mismatched candidate targeting requires `3.375` to `7.729 m/s` beyond the original total delta-v
authority, and executed components exceed the original component limit by `7.795` to `11.446 m/s`.
The protocol applies a wider 50/80 m/s diagnostic solver envelope to both profiles but separately
preserves the base 20/25 m/s authority in profile pass gates. This is an explicit comparison device,
not expanded maneuver authority. Profile counts remain unpooled and coordinates remain illustrative.
GitHub CI passed in `2m10s`; PR #19 merged the reviewed scope to `main` at `d7710f5`.

Paired Assurance Calibration Evidence binds a separate manifest to the paired protocol and result.
All 26 configured numeric dimensions have typed envelopes, units, rationale, limitations, and source
authority. NASA/JPL tracking and maneuver references and a Vega C insertion reference inform some
dimensions, but they do not calibrate Astro Suite's synthetic LEO network or spacecraft. Tracking
biases and the configured maneuver timing error remain illustrative, so weakest-bound derivation
keeps the manifest status `illustrative`. Promotion now has an executable gate: mission-specific
station residuals, insertion covariance semantics, propulsion execution residuals, and reviewed
subsystem evidence must cover every configured dimension. More samples or more model profiles do
not change that evidence class.
The checked public campaign ran from outside the repository, retained the reviewed 8/8 matched and
0/8 mismatched dispositions, and passed result verification with an explicit `illustrative` status.
Focused validation passes 21 tests; the full suite passes `957 tests with 11 optional-backend skips`;
Ruff, strict MyPy across 133 source files, `git diff --check`, 8 package/import tests, and sdist/wheel
builds pass. Independent subagent review was attempted but could not start because its usage quota
was exhausted; local adversarial review added and tested end-of-run calibration drift rejection.
GitHub CI passed in `2m10s`; PR #20 merged the scope to `main` at `f84dee7`.

AI-Native Assurance Review v1 adds a deterministic decision-support product over verified paired
assurance evidence. The public review command binds source path and digest, preserves every signed
metric shift, emits typed integrity/calibration/completeness/model-form/metric/claim findings, and
returns `additional_evidence_required` for the illustrative checked calibration. The assistant
compiles only a fixed verify-then-review plan with source/output continuity and explicit approval
for writes. The checked public artifact contains 10 findings: one calibration blocker, two warnings,
and seven informational findings. Focused review/assistant tests pass 12 tests; the full suite passes
`970 tests with 11 optional-backend skips`; Ruff, strict MyPy across 137 source files, package tests,
builds, and `git diff --check` pass. GitHub CI passed in `2m30s`; PR #22 merged the scope to
`main` at `0d1f32b`.

Project-specific steward learning: never rank heterogeneous assurance metrics by raw magnitude.
Preserve all signed values and use an explicit decision-metric priority so units and engineering
meaning remain visible.

Assurance Review Comparison v1 re-verifies two deterministic reviews and their original paired
evidence, binds the exact verified review and source digests, and compares findings and metrics by
stable identifiers. Direction uses an ordered risk vector of disposition, calibration authority,
blockers, and warnings instead of a mixed-unit scalar score. The checked identical-review public
artifact reports `unchanged`, zero finding changes, zero metric changes, and three unresolved
evidence recommendations under a non-operational claim boundary. Focused comparison/assistant
tests pass 20 tests; release-scale gates pass `984 tests with 11 optional-backend skips`, Ruff,
strict MyPy across 138 source files, 9 package/import tests, sdist/wheel builds, and
`git diff --check`. Independent review drove comparison re-verification, exact-byte digest binding,
hard-link collision rejection, semantic metric finding ids, and complete assistant path checks.
GitHub CI passed in `3m31s`; PR #23 merged the scope to `main` at `1564549`.

Project-specific steward learning: compare digests must cover the same bytes that were verified,
and added or removed metrics are real transitions even when no numeric delta exists. Evidence
recommendations remain deterministic copies of unresolved findings, never executable actions.

Lifecycle Assurance Review v1 adds exact local output-reproducibility verification and bounded
anomaly triage over `MissionLifecycleResult`. The review binds the result, lifecycle scenario, and
referenced launch, twin, and reentry scenario digests; requires canonical product equality against
a fresh local run; and derives stable integrity, manifest, continuity, margin, evidence-boundary,
and claim-boundary findings. The checked lifecycle physically passes, while review disposition is
`additional_review_required` because the promoted digital-twin limiting margin still uses the
non-specific unit `native`. The artifact contains 13 findings, one warning, and one non-executing
triage action to publish an explicit physical unit. Seven model caveats remain informational rather
than inferred anomalies. Focused lifecycle-review tests pass 16 tests; affected mission, assurance,
and assistant tests pass 183 tests; the full suite passes `1000 tests with 11 optional-backend
skips`; Ruff, strict MyPy across 142 source files, 9 package/import tests, sdist/wheel builds, public
verify/review/re-verify commands, and `git diff --check` pass. Independent review drove structured
lifecycle-failure handling, protected assistant plan ids, exact-byte parsing, five-file digest
binding with staged captured-input execution, numeric-id normalization, and private
verified-evidence-only derivation.
GitHub CI passed in `2m58s`; PR #24 merged the scope to `main` at `beb0b81`.

Project-specific steward learning: distinguish physics pass state from review readiness. A passing
lifecycle can still require bounded evidence-quality action, and free-text model caveats must not be
promoted into anomaly severity or causal diagnosis.

Lifecycle Margin Units v1 transitions integrated and constellation twin result workflows to v2 and
makes `DesignMargin.unit` a required source-level contract. Dimensionless values use `1`; physical margins
carry `kg`, `K`, `deg`, `N*m`, `deg/s`, `dB`, or `s`, and constellation member aggregation preserves
the originating unit. Lifecycle promotion now copies that structured field instead of emitting
`native`. The checked lifecycle still passes with the same `5.25 kg` limiting mass-budget margin,
while its deterministic review moves from `additional_review_required` with 13 findings, one
warning, and one triage action to `design_review_ready` with 12 findings and no warnings or triage.
Affected twin, mission, assurance, UQ, and CLI tests pass 243 tests; the full suite passes `1000
tests with 11 optional-backend skips`; Ruff, strict MyPy across 142 source files, five package/import
tests, sdist/wheel builds, and `git diff --check` pass. The public run, result verifier, review, and
review verifier pass with artifacts under `/tmp/astro-lifecycle-margin-units/`. Independent review
found and drove explicit v2 workflow identifiers, current UQ documentation, and metric-to-unit
mapping assertions; re-review found no P1/P2 issues.
Previously serialized v1 twin results without a margin `unit` do not satisfy the v2 schema and must
be regenerated rather than assigned an inferred fallback unit.
GitHub CI passed in `3m7s`; PR #25 merged the scope to `main` at `45e4d11`.

Project-specific steward learning: unit authority belongs where a numeric engineering quantity is
created. Downstream lifecycle and assurance products should preserve structured units, never infer
them from metric names or substitute a heterogeneous placeholder.

Mission Calibration Evidence Pack v1 adds optional typed evidence products to the existing
digest-bound assurance calibration manifest. Station residual evidence binds observable, unit,
station, band/mode/integration context, arc, selection/outlier policies, and summary statistics;
propulsion execution evidence binds command and achieved epochs and vectors in EME2000 plus timing,
magnitude, pointing, class, and reconstruction semantics; insertion covariance binds epoch, time
scale, body, EME2000 frame, Cartesian state order/units, population/confidence semantics, and a
symmetric positive-semidefinite 6x6 matrix with derived correlations. Mission-test and flight
authority now fail closed unless calibrated bounds cite matching-source, applicable typed evidence
and a declared derivation. `astro inspect-assurance-calibration` reports evidence counts, authority
coverage, blockers, digest, and claim boundary without mutating or promoting evidence. The checked
synthetic fixture contains all three families but correctly remains illustrative with zero calibrated
bounds. Focused evidence/validation gates pass 36 tests; the full suite passes `1015 tests with 11
optional-backend skips`, including host multiprocessing cases; Ruff, strict MyPy
across 142 source files, five package/import tests, sdist/wheel builds, public artifact parsing, and
`git diff --check` pass. Independent adversarial review found and drove exact evidence-derived
envelopes, observable/unit mapping, scale-aware covariance validation, derived propulsion pointing,
protocol/context binding, explicit coordinate allowlists, and protocol-completeness disclosure;
final re-review found no P1/P2 issues.
GitHub CI passed in `2m35s`; PR #26 merged the scope to `main` at `aa98d1f`.

Project-specific steward learning: evidence acquisition and authority promotion must remain separate
operations. Typed residuals make prerequisites reviewable, but synthetic completeness, sample count,
or a source-kind label cannot promote a calibration bound without applicable evidence and derivation.

Assurance Model-Form Matrix v1 adds a separate four-cell truth/estimator factorial without changing
paired validation v1. Each of eight explicit realizations runs two-body/two-body, two-body/J2,
J2/two-body, and J2/J2 with one shared seed per realization, and reports two same-truth estimator
contrasts plus a difference-in-differences interaction. The checked run completes all 32 cells and
all 24 contrasts: both matched profiles pass 8/8 and both cross-model profiles pass 0/8. Under
two-body truth, switching the estimator to J2 increases median OD position error by `11.369 km`;
under J2 truth, switching the estimator to J2 reduces it by `13.247 km`. The interaction median is
`-24.616 km`. Counts and contrasts remain unpooled, calibration status remains `illustrative`, and
matched J2 behavior is internal consistency rather than physical-truth or flight authority.
Independent review found and the implementation fixed post-execution input-drift exposure, missing
result-level calibration-protocol binding, and thin adversarial verification coverage. Focused
matrix tests pass 13 tests; the full suite passes `1028 tests with 11 optional-backend skips`; Ruff,
strict MyPy across 146 source files, seven packaging tests, sdist/wheel builds, `git diff --check`,
and the checked public run and exact verifier pass.
GitHub CI passed in `3m0s`; PR #27 merged the scope to `main` at `6c2e606`.

Project-specific steward learning: a derived validation product that reuses another protocol's
calibration must bind both identities, and long-running evidence generation must recheck every
mutable source after execution before publishing its result.

Production Covariance Validation Criteria v1 defines a suite-owned acceptance contract over
candidate/reference covariance trajectories and optional raw empirical consistency samples. Exact
epoch alignment, state agreement, symmetry, positive definiteness, conditioning, normalized
Frobenius covariance error, generalized eigenvalue ratios, and aggregate/pointwise NEES are
recomputed from bound evidence. Passing numerical agreement is kept separate from implementation
independence, high-fidelity force coverage, empirical truth independence, and sample sufficiency.
The checked duplicate-local fixture passes 11/11 numerical epochs but correctly returns
`additional_evidence_required` with four blockers: no independent implementation, no bound
trajectory covariance semantics, no drag/SRP/third-body coverage, and no empirical consistency
evidence. The result explicitly makes no
flight or operational certification claim.
Independent adversarial review found and the implementation fixed missing trace/STM criteria,
duplicate empirical-sample sufficiency, covariance definiteness ambiguity, unbound trajectory
semantics, parse-before-digest races, partial multi-output publication, underpowered-campaign
misclassification, and unaudited independence declarations. The final review found no remaining
P1/P2 issues. Focused covariance tests pass 11 tests; affected assurance/CLI tests pass 196 tests;
the full suite passes `1039 tests with 11 optional-backend skips`; Ruff, strict MyPy across 150
source files, seven packaging tests, sdist/wheel builds, `git diff --check`, and the checked public
assessment and exact verifier pass.

Project-specific steward learning: covariance agreement and covariance consistency are independent
evidence classes. Promotion criteria must bind raw errors, the covariance used for each error,
realization independence, implementation-independence review, force coverage, units, and exact input
bytes; repeating one observation or comparing shared code cannot manufacture authority.

The supporting **AI-Native Uncertainty Campaigns and Validated Surrogates** architecture and staged
roadmap are recorded in `docs/superpowers/specs/2026-07-12-ai-native-uncertainty-surrogate-roadmap.md`;
the executable plan is
`docs/superpowers/plans/2026-07-12-ai-native-uncertainty-surrogate-implementation.md`.

The first delivery is suite-level uncertainty campaign infrastructure with a Robust Mission
Lifecycle Campaign as the flagship. Monte Carlo, LHS, Sobol, sweeps, and model ensembles are
sampling strategies over one campaign contract. Surrogates are separately validated evaluators,
not samplers or replacements for authoritative physics. A benchmark must select or kill the first
learned target before neural dependencies or training work enter the critical path.

Astro Uncertainty Campaigns v1 is integrated on `main` through PR #10 at merge commit `6ca5d78`.
The suite now has
typed uncertainty/distribution/correlation contracts; deterministic pseudorandom, LHS, Sobol,
sweep, and model-ensemble sampling; allow-listed parameter and metric registries; explicit failure
outcomes; weighted statistics; fixed-count, CI, and metric-stability stopping; deterministic
retention; atomic resumable campaign artifacts; bounded process execution; and authoritative orbit,
digital-twin, reentry, and mission-lifecycle evaluators. Public commands are `astro
validate-campaign`, `astro run-campaign`, and `astro summarize-campaign`; execution also supports
worker bounds, max-case caps, dry runs, and integrity-checked resume.

The checked `leo-lifecycle-robustness-v1` campaign completed 8/8 LHS cases with effective sample
size 8.0 and requirement fractions 1.0 for lifecycle success and propellant reserve. Its reviewed
epistemic inputs span launch thrust, shared spacecraft wet mass, twin power efficiency, deorbit
design values, reentry atmosphere density, and reentry drag. These are design-space results, not
operational probabilities. The benchmark found no
evaluator-dominated cost and component timings had not previously been instrumented, so surrogate
training, neural dependencies, and model promotion are stopped. Safe immutable surrogate schemas
and integrity-checked NPZ/JSON artifacts exist to support a future reopened gate without granting
model authority.

A follow-up machine-scoped timing gate now separates profiling from deterministic campaign
statistics and records setup, evaluation, metric extraction, retained-result serialization,
unattributed overhead, and total instrumented time with source-digest and machine/runtime
provenance. Seven-case checked profiles found median evaluator times
of 0.000553 seconds for local J2 orbit and 0.001431 seconds for the integrated digital twin. Although
their evaluator shares were 0.9737 and 0.9918, both fail the preregistered 0.050-second absolute-cost
gate. Surrogate acceleration therefore remains stopped for these local evaluators. Final local
verification passed 888 tests with 11 skipped, Ruff, strict MyPy across 120 source files, and
sdist/wheel builds.

Lifecycle Sensitivity and Margin Attribution v1 adds a separate digest-bound Spearman/PRCC product
over completed campaign evidence. A checked 64-case LHS run over the seven reviewed cross-phase
inputs completed 64/64 cases with effective sample size 64, 56 residual degrees of freedom, full
rank, condition number 1.6532, and no ties. Deorbit delta-v and specific impulse dominate
propellant-use association; delta-v, specific impulse, and wet mass dominate reserve-margin
association; reentry drag coefficient and deorbit delta-v dominate peak heat-rate and
dynamic-pressure association. The largest observed absolute PRCC for entry-interface margin is
0.2015; no practical-effect threshold is asserted. These are non-causal design-space associations,
not Sobol indices or operational risk contributions. The integrated scope passes 901 tests with 11
skipped, Ruff, strict MyPy across 121 source files, and sdist/wheel builds.

Lifecycle Subsystem Margin Targets add unit-stable campaign extractors for battery SOC, the minimum
hot/cold thermal envelope, ADCS pointing/torque/slew/utilization, worst observed contact link margin,
propellant fraction, and itemized mass-budget rollup, plus separate contact availability, count, and
duration metrics. A fresh 64-case run completed 64/64 with all ten subsystem requirements passing.
Only observed link and propellant-fraction margins varied under the existing seven inputs: link
margin's largest observed absolute PRCC values were launch thrust `-0.9965` and wet mass `+0.8913`,
but its full target range was only `0.0035 dB` and no practical-effect threshold was evaluated. Wet
mass had PRCC `+1.0000` with propellant-fraction margin. Constant battery, thermal, ADCS, rollup,
and contact targets were excluded from attribution rather than assigned artificial ranks. These
remain local
configured-design-space associations, not causal, operational-reliability, coverage-authority, or
certification claims.
The completed implementation passes 903 tests with 11 skipped, Ruff, strict MyPy across 121 source
files, and sdist/wheel builds.

Lifecycle Power/Thermal Input Pack adds typed solar-array area, battery-capacity, and named
thermal-node overrides alongside the existing efficiency input. A separate checked 64-case LHS
campaign over five inputs completed 64/64 cases with all four declared requirements passing,
effective sample size 64, 58 residual degrees of freedom, rank 5, and condition number 1.3968. Bus
hot-margin PRCC was `-0.9779` for internal-heat fraction and `+0.9723` for emissivity. Battery-SOC
margin had 26 unique values, tie fraction `0.5938`, and largest observed PRCC `+0.7404` for array
area; smaller coefficients are not promoted because the target plateaus at initial SOC and array
area interacts multiplicatively with efficiency. The checked window is only 600 seconds and
sunlit, with no load shedding or battery-thermal feedback, so this is preliminary sizing evidence,
not full-orbit energy balance, EPS reliability, or thermal certification.
The integrated implementation passes 910 tests with 11 skipped, Ruff, strict MyPy across 121
source files, and sdist/wheel builds.

Full-Orbit Power/Thermal Campaign extends the standalone digital-twin product with interval battery
energy change, unmet load/energy, and curtailed surplus-energy accounting plus UQ metrics for
minimum and terminal battery energy, net battery-energy change, eclipse duration, sunlit fraction,
and named thermal margins. Digital-twin campaigns now bind the twin template and referenced orbit
scenario digests, so resume rejects nested-input drift. The checked 64-case, five-input LHS run
completed 64/64 local evaluations over a 6,000-second orbit with a sampled 2,100-second eclipse.
Requirement fractions were `0.421875` for minimum SOC, `0.734375` for zero unmet energy, `0.65625`
for bus hot margin, and `1.0` for bus cold margin. Battery capacity had the largest absolute PRCC
for SOC and unmet-energy margins; emissivity (`+0.9806`) and internal-heat fraction (`-0.9764`)
dominated hot margin. Unmet-energy targets were highly tied (`0.71875`), so this remains passive
shortage and lumped-thermal design screening, not active load shedding, EPS reliability, thermal
qualification, or operational probability evidence.
The completed implementation passes 914 tests with 11 optional-backend skips, Ruff, strict MyPy
across 121 source files, packaging tests, and sdist/wheel builds.

Independent review hardened the campaign boundary before integration: non-authoritative evaluator
labels now fail closed, resume verifies the definition, samples, cases, statistics, and regenerated
deterministic sample plan, and a non-resume run refuses to overwrite existing evidence. Requirement
fractions use all completed cases as their denominator, so failed evaluations cannot silently
disappear. OS-backed campaign locks release on process death, write-ahead append transactions recover
index/manifest crash windows, metric extraction failures remain typed case outcomes under every
retention policy, and CI stopping fails closed for unequal weights. A deliberately failed launch
records the stopping phase without fabricating downstream artifacts. Final local verification on
2026-07-13 passed 879 tests with 11 skipped, Ruff, strict
MyPy across 119 source files, packaging tests, sdist/wheel builds, and a fresh two-worker public-CLI
run of the checked lifecycle campaign. GitHub CI passed the merged PR after an identity-safe Tudat
SPICE-kernel cache correction removed an order-dependent fake-module collision exposed by the
expanded suite.

The previous suite-owned roadmap pass, Verifiable OD Workflow Pack, integrated digital twin,
constellation digital twin, constellation coverage-map stack, and subsystem fidelity pack are
integrated on `main` and pushed to `origin/main`. The current digital twin surface includes
deterministic subsystem fidelity evidence for scheduled power loads, battery efficiency/energy,
thermal heat balance, ADCS slew/utilization margins, and itemized mass-budget rollups.

The Reentry Suite adds the next suite-owned product boundary: deterministic
ballistic, prescribed-bank lifting, and target-tracking guided entry with local guidance
optimization and orbit-state handoff. It is integrated on `main` and locally release-ready within
the documented deterministic screening claim boundary.

Implemented and verified in the current pass:

- Local deterministic propagation, covariance, events, maneuvers, conjunction, attitude, OD,
  measurement IO, DSN-style bridge/calibration products, launch/ascent, launch tuning/reporting, and
  research Monte Carlo workflows.
- Orekit, RocketPy, Dymos/OpenMDAO, TudatPy, and JAX runtime gates and adapter boundaries.
- Required local release checklist and packaging gate.
- Optional live backend campaign ledger with clear environment and claim boundaries.
- Verifiable OD Workflow Pack slices: `examples/workflows/local_od/manifest.yaml`,
  `examples/workflows/local_od/golden_prompts.yaml`, `astro ask --report-output`,
  `astro ask --report-summary-output`,
  workflow-report metrics for measurement count, TDM line count, estimate convergence, iteration
  count, RMS, Jacobian rank, residual count, and max residual, plus golden prompt regression checks
  across every supported local OD scenario.
- OD workflow pack branch release-readiness pass: focused assistant tests, real CLI execution with
  trace/JSON/text report artifacts, `ruff`, `mypy`, full `pytest`, packaging tests,
  `git diff --check`, and `python -m build` passed locally on `codex/od-workflow-pack`.
- Integrated Digital Twin planning artifacts: `docs/superpowers/specs/2026-07-08-integrated-digital-twin-design.md`
  and `docs/superpowers/plans/2026-07-08-integrated-digital-twin-implementation.md`.
- Integrated Digital Twin implementation surface on `codex/digital-twin-plan`:
  `src/astro_twin` for suite-owned models, IO, geometry, power, thermal, ADCS, coverage,
  link-budget, margin aggregation, and runner orchestration; `astro run-twin` in
  `src/astro_cli/main.py`; `examples/twin/leo_observer.yaml`; `tests/astro_twin`; and
  `docs/digital-twin.md`.
- Integrated Digital Twin verification on `codex/digital-twin-plan`:
  `astro run-twin examples/twin/leo_observer.yaml --output /tmp/astro-twin-result.json --summary-output /tmp/astro-twin-summary.txt`
  wrote JSON and text artifacts with 11 samples, 1 access window, worst link margin 30.280 dB,
  limiting mass-margin warning 0.037, and explicit design-screening warnings;
  `python -m pytest tests/astro_twin -q` passed with 15 tests; `python -m ruff check .` passed;
  `python -m mypy` passed with no issues in 78 source files; `python -m pytest -q` passed with
  606 tests and 11 optional-backend skips; `git diff --check` was clean;
  `python -m pytest tests/test_packaging.py -q` passed with 3 tests; and `python -m build`
  produced `astro_suite-0.1.0.tar.gz` and `astro_suite-0.1.0-py3-none-any.whl`.
- Constellation Digital Twin planning artifacts:
  `docs/superpowers/specs/2026-07-08-constellation-digital-twin-design.md` and
  `docs/superpowers/plans/2026-07-08-constellation-digital-twin-implementation.md`.
- Constellation Digital Twin implementation surface on `codex/constellation-twin-design`:
  `src/astro_twin/constellation_models.py`, `src/astro_twin/constellation_io.py`,
  `src/astro_twin/constellation.py`, `astro run-constellation-twin` in `src/astro_cli/main.py`,
  `examples/twin/constellation_leo_observers.yaml`,
  `examples/twin/leo_observer_plane_a.yaml`, `examples/twin/leo_observer_plane_b.yaml`,
  `examples/scenarios/leo_two_body_phase_minus_4deg.yaml`,
  `tests/astro_twin/test_constellation_models.py`,
  `tests/astro_twin/test_constellation_io.py`,
  `tests/astro_twin/test_constellation_aggregation.py`,
  `tests/astro_twin/test_constellation_runner.py`, and constellation CLI tests in
  `tests/astro_cli/test_cli.py`.
- Constellation Digital Twin verification on `codex/constellation-twin-design`:
  `astro run-constellation-twin examples/twin/constellation_leo_observers.yaml --output /tmp/astro-constellation-twin.json --summary-output /tmp/astro-constellation-twin.txt`
  wrote JSON and text artifacts for `leo-observers` with 2 members, a 0.0 to 600.0 second analysis
  window, 1 `equator-eci` fleet access summary, 300.0 seconds total fleet access, 0.5 coverage
  fraction, 300.0 second longest gap, max simultaneous spacecraft 2, 1080.000 Mbit total data
  volume, member data volumes of 480.0 Mbit for `plane-a` and 600.0 Mbit for `plane-b`, and a
  limiting `fleet_longest_gap_s_equator-eci` margin with status `warn` and margin 0.000. The
  product keeps explicit design-screening warnings and does not claim operational constellation
  coverage authority.
- Constellation Digital Twin final gates on `codex/constellation-twin-design`:
  `python -m pytest tests/astro_twin/test_constellation_models.py tests/astro_twin/test_constellation_io.py tests/astro_twin/test_constellation_aggregation.py tests/astro_twin/test_constellation_runner.py tests/astro_cli/test_cli.py::test_run_constellation_twin_command_writes_json_and_summary -q`
  passed with `24 passed in 0.86s`; `python -m ruff check .` passed; `python -m mypy` passed with no
  issues in 81 source files; `python -m pytest -q` passed with
  `632 passed, 11 skipped in 6.87s`; `git diff --check` was clean;
  `python -m pytest tests/test_packaging.py -q` passed with `3 passed in 0.03s`; and
  `python -m build` produced `astro_suite-0.1.0.tar.gz` and
  `astro_suite-0.1.0-py3-none-any.whl`.
- Constellation Coverage Maps v1 planning artifacts:
  `docs/superpowers/specs/2026-07-09-constellation-coverage-maps-design.md` and
  `docs/superpowers/plans/2026-07-09-constellation-coverage-maps-implementation.md`.
- Constellation Coverage Maps v1 implementation surface on `codex/coverage-maps-v1`:
  `ConstellationCoverageSensorConfig`, `ConstellationCoverageTargetConfig`,
  `ConstellationCoverageMapConfig`, `CoverageMapSummary`, and `CoverageMapTargetSummary` in
  `src/astro_twin/constellation_models.py`; `aggregate_coverage_maps` and coverage-map fleet
  margins in `src/astro_twin/constellation.py`; summary formatting in
  `src/astro_twin/constellation_io.py`; the checked two-target `equatorial-targets` map in
  `examples/twin/constellation_leo_observers.yaml`; and focused model, aggregation, runner, IO, and
  CLI tests.
- Constellation Coverage Maps v1 verification on `codex/coverage-maps-v1`:
  `astro run-constellation-twin examples/twin/constellation_leo_observers.yaml --output /tmp/astro-constellation-coverage-map.json --summary-output /tmp/astro-constellation-coverage-map.txt`
  wrote JSON and text artifacts for `leo-observers` with 1 coverage map, 2 configured targets, 2
  covered targets, 0.250 mean coverage fraction, 0.200 minimum target coverage fraction, 480.0
  second maximum target gap, max simultaneous spacecraft 2, target fractions of 0.200 for
  `prime-meridian` and 0.300 for `east-equator`, and limiting
  `coverage_map_min_fraction_equatorial-targets` margin with status `warn` and margin 0.000. The
  focused constellation and CLI slice passed with `29 passed in 0.86s`; `python -m ruff check .`
  passed; `python -m mypy` passed with no issues in 81 source files; the twin test slice passed with
  `43 passed in 0.32s`; `python -m pytest -q` passed with `637 passed, 11 skipped in 7.09s`;
  `git diff --check` was clean; the packaging test slice passed with `3 passed in 0.03s`; and
  `python -m build` produced
  `astro_suite-0.1.0.tar.gz` and `astro_suite-0.1.0-py3-none-any.whl`. The product keeps explicit
  design-screening warnings and does not claim operational coverage authority or certified sensor
  performance.
- Subsystem Fidelity Pack planning artifacts:
  `docs/superpowers/specs/2026-07-09-subsystem-fidelity-pack-design.md` and
  `docs/superpowers/plans/2026-07-09-subsystem-fidelity-pack-implementation.md`.
- Subsystem Fidelity Pack implementation surface merged from `codex/subsystem-fidelity-pack`:
  `PowerLoadSchedule`, `MassBudgetItemConfig`, and `MassBudgetSummary` in
  `src/astro_twin/models.py`; scheduled-load and battery-efficiency logic in
  `src/astro_twin/power.py`; albedo/planet-IR/mode-heat balance logic in
  `src/astro_twin/thermal.py`; slew-rate and actuator-utilization evidence in
  `src/astro_twin/adcs.py`; `build_mass_budget_summary` in `src/astro_twin/mass.py`; subsystem
  margins in `src/astro_twin/margins.py`; runner wiring in `src/astro_twin/runner.py`; richer
  assumptions in `examples/twin/leo_observer.yaml`; and focused model, power, thermal, ADCS, mass,
  margin, runner, and CLI tests.
- Subsystem Fidelity Pack verification on `codex/subsystem-fidelity-pack`:
  `astro run-twin examples/twin/leo_observer.yaml --output /tmp/astro-subsystem-fidelity-twin.json --summary-output /tmp/astro-subsystem-fidelity-twin.txt`
  wrote JSON and text artifacts with 11 samples, minimum battery SOC 0.850, first scheduled-load
  sample at 120.0 seconds with 35.0 W scheduled load, 1042.686 Wh battery energy, bus heat-balance
  evidence, ADCS slew-rate margin 0.150 deg/s, actuator utilization 0.375, itemized mass base
  128.0 kg, contingency 12.6 kg, total 140.6 kg, dry-plus-payload reference 145.0 kg, and limiting
  `mass_budget_rollup_margin_kg` status `warn` with margin 4.400 kg. `python -m pytest
  tests/astro_twin -q` passed with `48 passed in 0.60s`; the twin CLI test passed with
  `1 passed in 0.86s`; `python -m ruff check .` passed; `python -m mypy` passed with no issues in
  82 source files; `python -m pytest -q` passed with `642 passed, 11 skipped in 7.00s`;
  `git diff --check` was clean; the packaging test slice passed with `3 passed in 0.02s`; and
  `python -m build` produced `astro_suite-0.1.0.tar.gz` and
  `astro_suite-0.1.0-py3-none-any.whl`. The product keeps explicit design-screening warnings and
  does not claim EPS certification, thermal certification, mass-properties authority, or
  flight-qualified GNC simulation. PR #6 merged this scope into `main` at merge commit `729a7d4`.
- External validation refresh on `codex/orekit-validation-refresh`:
  `JAVA_HOME=/opt/homebrew/opt/openjdk/libexec/openjdk.jdk/Contents/Home PATH="/opt/homebrew/opt/openjdk/bin:$PATH" astro orekit-smoke`
  reported Orekit JPype 13.1.5.0 available; `astro rocketpy-smoke`, `astro dymos-smoke`, and
  `astro jax-smoke` also reported available runtimes; `astro tudat-smoke` reported TudatPy not
  installed, and `conda run -p /tmp/astro-tudat-live-env ...` reported the historical Tudat
  environment path no longer exists. The Orekit validation refresh passed
  `ASTRO_RUN_OREKIT_LIVE=1 python -m pytest tests/astro_backends/test_orekit_propagation.py::test_live_orekit_covariance_history_returns_suite_product tests/astro_backends/test_orekit_propagation.py::test_live_orekit_high_fidelity_covariance_records_force_models -q`
  with `2 passed in 9.22s` and wrote `/tmp/astro-orekit-validation-high-fidelity-covariance-20260709.json`,
  `/tmp/astro-orekit-validation-drag-20260709.json`, `/tmp/astro-orekit-validation-srp-20260709.json`,
  and `/tmp/astro-orekit-validation-third-body-20260709.json`. The high-fidelity covariance product
  contains 11 samples, 11 covariance samples, `orekit_finite_difference_state_transition`, white
  acceleration process noise, and transition force models for J2, drag, SRP, Sun, and Moon. PR #7
  merged this validation refresh into `main` at merge commit `b668e1f`.
- Reentry Suite implementation merged from `codex/reentry-suite` through PR #8:
  `src/astro_reentry` owns scenario/result schemas, exponential atmosphere and vacuum controls,
  Sutton-Graves-style convective heating, spherical-Earth 3-DOF RK4 propagation, ballistic,
  constant-bank, velocity-scheduled, and target-tracking guidance, peak/event extraction,
  requirement margins, bounded Powell bank-schedule optimization, EME2000-to-Earth-fixed
  trajectory handoff, and local backend dispatch. `src/astro_cli/main.py` exposes
  `simulate-reentry`, `optimize-reentry`, and `handoff-reentry`; checked examples cover a ballistic
  capsule, prescribed-bank lifting body, and guided lifting body. Pre-merge review hardening rejects
  ignored guidance fields and initially terminal states, records resolved trajectory sample indices,
  and prevents optimization products from accepting a regressing candidate.
- Reentry Suite numerical and public-command evidence:
  the ballistic case terminates at 314.213 s with peak dynamic pressure 54,334.589 Pa, peak
  aerodynamic deceleration 18.133 g, peak convective heat rate 2,327,838.015 W/m^2, and total heat
  load 99,670,993.043 J/m^2. The prescribed lifting case terminates at 480.539 s with one bank
  reversal; the guided case terminates at 464.320 s with 18.431 km target miss. Bounded guidance
  optimization converges in 13 iterations and reduces target miss to 6.231 km. The ballistic
  internal-step convergence check agrees on terminal time within 0.001 s across 1.0, 0.5, and 0.25
  second steps and preserves monotonic heat load.
- Reentry Suite final local gates repeated on merged `main` after PR #8:
  `python -m pytest -q` passed with `676 passed, 11 skipped in 12.90s`; the focused reentry and
  CLI slice passed with `33 passed in 7.39s`; `python -m ruff check .`
  passed; `python -m mypy` passed with no issues in 92 source files; `git diff --check` was clean;
  `python -m pytest tests/test_packaging.py -q` passed with `4 passed in 0.03s`; and
  `python -m build` produced `astro_suite-0.1.0.tar.gz` and
  `astro_suite-0.1.0-py3-none-any.whl`, with `astro_reentry` present in the wheel.
- Mission Lifecycle Workflow branch evidence on `codex/mission-lifecycle`: the checked
  `leo-round-trip` command wrote one lifecycle result, a text summary, and nine phase artifacts.
  Launch insertion passed its three declared tolerances; eight state/epoch/mass continuity checks
  passed; the deorbit burn consumed 32.861 kg and retained 17.139 kg; entry handoff occurred at
  119.794 km; and reentry terminated at zero altitude with all cross-phase margins passing. Focused
  lifecycle tests passed with `8 passed`; full local verification passed with
  `686 passed, 11 skipped`, Ruff, strict MyPy across 97 source files, `git diff --check`, wheel
  content inspection, and both distribution builds. GitHub CI passed and PR #9 merged the workflow
  to `main` at `d37b9a6`.

Post-MVP / external-campaign items:

- Production-grade covariance certification through external drag, SRP, and third-body dynamics.
- Flight-qualified actuator/sensor ACS modeling beyond deterministic screening products.
- Operational constellation coverage authority, scheduling, or certified sensor performance beyond
  deterministic target-grid screening.
- Native RocketPy multi-motor/staged-separation execution if a validated upstream API becomes
  available.
- Full high-fidelity multistage Dymos ascent design optimization beyond the current suite product
  boundary.
- Official standards-grade DSN ODF/TNF decoding, official station calibration solving, and deeper
  astrometry/CCSDS authority.
- Operational-grade differentiable OD services beyond the current JAX research products.
- External correlation or higher-authority reentry campaigns for rotating-Earth dynamics,
  atmosphere/wind uncertainty, 6-DOF aerodynamics and GNC, rarefied/real-gas/radiative flow, CFD,
  ablation/material response, terrain/descent systems, and operational landing dispersion.

## Active Work Registry

| ID | Status | Lane | Owner | Scope | Acceptance |
| --- | --- | --- | --- | --- | --- |
| roadmap-finish-state | done | decide/verify | steward | `docs/current-state.md`, release and live-ledger docs | State file records canonical workspace, release readiness, optional smoke evidence, and remaining post-MVP boundaries. |
| verifiable-od-workflow-pack | done | productize/verify | steward | `src/astro_assistant`, `examples/workflows/local_od`, assistant docs and validation matrix | Local OD workflow has a manifest, golden prompt fixtures, JSON and text report outputs, focused tests, real CLI execution evidence, and is pushed on `main`. |
| integrated-digital-twin | done | productize/verify | steward | `src/astro_twin`, `astro run-twin`, `examples/twin`, `tests/astro_twin`, `docs/digital-twin.md` | Single-spacecraft v1 writes suite-owned orbit geometry, power, thermal, ADCS, coverage, link budget, mass, and design-margin evidence from one scenario; required local gates pass with explicit design-screening claim boundaries and the stack is merged on `main`. |
| constellation-digital-twin | done | productize/verify | steward | `src/astro_twin/constellation*`, `astro run-constellation-twin`, `examples/twin/constellation_leo_observers.yaml`, constellation tests, `docs/digital-twin.md` | Constellation v1 embeds member `DigitalTwinResult`s and writes suite-owned fleet access, revisit, link, data-volume, and margin evidence for a checked two-member reference; required local gates pass with explicit design-screening and non-operational claim boundaries and the stack is merged on `main`. |
| constellation-coverage-maps | done | productize/verify | steward | `src/astro_twin/constellation*`, `examples/twin/constellation_leo_observers.yaml`, `docs/digital-twin.md`, constellation tests | Coverage Maps v1 writes deterministic target-grid sensor coverage summaries and fleet margins from member geometry samples; required local gates pass with explicit design-screening and non-operational claim boundaries and the stack is merged on `main`. |
| subsystem-fidelity-pack | done | productize/verify | steward | `src/astro_twin`, `examples/twin/leo_observer.yaml`, `docs/digital-twin.md`, subsystem tests | Adds scheduled power loads, battery energy/efficiency, thermal heat-balance evidence, ADCS slew/utilization margins, and itemized mass-budget rollups; required local gates pass with explicit design-screening and non-certification claim boundaries and the scope is merged on `main`. |
| external-validation-refresh | done | verify | steward | `docs/validation/live-backend-campaigns.md`, optional runtime smoke checks, Orekit live covariance gate | Refreshes machine-scoped optional validation evidence: Orekit high-fidelity covariance and drag/SRP/third-body products pass on this machine; Tudat-specific current release claims are not refreshed because the historical isolated Tudat environment is absent; PR #7 is merged on `main`. |
| reentry-suite | done | productize/verify | steward | `src/astro_reentry`, reentry CLI commands, `examples/reentry`, `tests/astro_reentry`, `docs/reentry.md` | Ballistic, prescribed-bank lifting, target-tracking guidance, aerothermal/load/margin products, local optimization, trajectory handoff, documentation, convergence checks, full local gates, packaging, review hardening, GitHub CI, and merged-main verification pass; PR #8 is merged at `3c73f8b`. |
| mission-lifecycle | done | productize/verify | steward | `src/astro_mission`, `astro run-mission-lifecycle`, checked phase fixtures, lifecycle tests and docs | Connects launch, operations propagation, the integrated twin, deorbit, and reentry through suite-owned products; fails closed on continuity or phase gates; writes an ordered artifact manifest; passes focused, full, package, public-command, and GitHub CI gates; PR #9 is merged at `d37b9a6`. |
| ai-native-uncertainty-surrogates | done | productize/verify | steward | `astro_uq`, `astro_surrogates`, campaign CLI, assistant campaign plans, checked lifecycle campaign, benchmark and validation docs | UQ campaigns v1, safe surrogate artifact contracts, deterministic assistant campaign plans, and separate machine-scoped timing profiles are implemented. Checked local J2 orbit and integrated-twin profiles have complete phase accounting but fail the preregistered 0.050-second median evaluator-cost gate. The hardened timing scope passes 888 tests, lint, type checking, and package builds; surrogate training/promotion remains deliberately stopped. |
| lifecycle-sensitivity-attribution | done | productize/verify | steward | `astro_uq.sensitivity`, `astro analyze-campaign-sensitivity`, checked 64-case lifecycle campaign, sensitivity docs and tests | Produces digest-bound Spearman and PRCC association evidence for numeric metrics and signed requirement margins with sample-size, residual-DF, tie, weight, failure, and conditioning gates. The checked lifecycle campaign completes 64/64 cases and records bounded design-space attribution without causal, Sobol-index, operational-probability, or certification claims. |
| lifecycle-subsystem-margin-targets | done | productize/verify | steward | lifecycle UQ adapter, checked lifecycle campaigns, subsystem requirement tests and docs | Exposes unit-stable battery, thermal-envelope, ADCS, worst-observed-link, propellant-fraction, and mass-budget margins plus contact metrics through lifecycle campaigns. The checked 64-case run passes every subsystem requirement and attributes only the two nonconstant targets, preserving practical-effect, no-contact, constant-target, and coverage claim boundaries. |
| lifecycle-power-thermal-inputs | done | productize/verify | steward | typed lifecycle overrides, dynamic named-node bindings and metrics, checked five-input campaign, tests and docs | Screens solar-array, battery-capacity, and bus thermal design inputs against battery-SOC and separate hot/cold margins. The checked campaign records the battery tie warning and short-sunlit-window boundary instead of claiming full-orbit EPS or thermal authority. |
| post-launch-mission-assurance | done | productize/verify | steward | `src/astro_assurance`, `astro run-mission-assurance`, `examples/assurance`, assurance tests and docs | Connects launch insertion, visibility-filtered simulation tracking, OD, a bounded manual-review correction, estimate/truth replay, and the corrected digital twin through one suite-owned evidence case. Independent review, focused and release-scale local gates, public digest verification, GitHub CI, and merged-main integration pass; PR #17 merged at `0d73d5b`. |
| mission-assurance-uncertainty | done | productize/verify | steward | `src/astro_uq/adapters/assurance.py`, `examples/campaigns/leo_mission_assurance_robustness.yaml`, assurance/UQ tests and docs | Adds a bounded first campaign over insertion, tracking sigma, execution magnitude, and subsystem margins. The checked 8-case run, full release gates, independent review, and GitHub CI pass; PR #18 merged at `c10268e`. |
| paired-assurance-validation | done | productize/verify | steward | `src/astro_assurance/validation_*`, paired validation CLI, explicit protocol fixture, tests and docs | Separates matched two-body and truth-J2/two-body-estimator profiles over shared coordinates. Independent review found and the implementation fixed noncausal OD, forged-derived-evidence acceptance, swallowed source drift, duplicate coordinate seeds, omitted executed-component authority, coordinate/profile substitution, and output-path collision. The corrected checked result completes 8/8 pairs with eight regressions; `951 passed, 11 skipped`, lint, typing, packaging, builds, outside-repository public run/verifier, GitHub CI, and integration pass. PR #19 merged at `d7710f5`. |
| assurance-calibration-evidence | done | calibrate/verify | steward | `examples/assurance/paired_force_model_calibration.yaml`, calibration contracts, paired runner/verifier, assurance docs and tests | Every configured numeric dimension has one digest-bound evidence envelope and promotion derives from the weakest authority. Current status remains illustrative. Public campaign verification, all local release gates, GitHub CI, and integration pass; PR #20 merged at `f84dee7`. |
| v0.2.0-rc.1 | done | release/verify | steward | package metadata, campaign compatibility, release notes, merged-main and detached-worktree gates | Package metadata, built-wheel public workflow, local gates, GitHub CI, merged-commit detached verification, and annotated tag agree at `f1baa02`. |
| assurance-review-operator | done | productize/verify | steward | deterministic paired-assurance review, public CLI, typed assistant plan, tests and docs | Public artifact, local gates, GitHub CI, and integration pass. Findings remain deterministic and non-operational; PR #22 merged at `0d1f32b`. |
| assurance-review-comparison | done | productize/verify | steward | review re-verification, deterministic cross-run comparison, public CLI, typed assistant plan, tests and docs | Public compare/verify artifacts, independent review, local release gates, GitHub CI, and integration pass. Comparison avoids heterogeneous scalarization, claim promotion, and recommendation execution; PR #23 merged at `1564549`. |
| lifecycle-assurance-review | done | productize/verify | steward | lifecycle result re-verification, deterministic findings and triage, public CLI, typed assistant plan, tests and docs | Exact staged-input re-execution, five-file digest binding, stable findings, one bounded unit-quality action, public artifacts, independent review, 1,000-test local gates, GitHub CI, and integration pass; PR #24 merged at `beb0b81`. |
| lifecycle-margin-units | done | productize/verify | steward | `DesignMargin` unit contract, twin and constellation producers, lifecycle promotion, checked review | Required source-level units are preserved into lifecycle evidence; checked review is warning-free without physics changes; affected, full-release, public, independent-review, GitHub CI, and integration gates pass; PR #25 merged at `45e4d11`. |
| mission-calibration-evidence | done | calibrate/verify | steward | typed station residuals, propulsion execution residuals, insertion covariance, inspection CLI, checked synthetic fixture | Evidence schemas and fail-closed promotion prerequisites are implemented; focused/full/public/package, independent-review, GitHub CI, and integration gates pass; PR #26 merged at `aa98d1f`. Actual mission evidence acquisition remains external work. |
| assurance-model-form-matrix | done | productize/verify | steward | four-cell truth/estimator protocol, contrasts, public CLI, exact verifier, checked fixture | Four profiles remain separate and unpooled; local source/calibration, exact reexecution, focused/full, public, package, independent-review, GitHub CI, and integration gates pass without changing paired v1. PR #27 merged at `6c2e606`. |
| production-covariance-criteria | in_progress | define/verify | steward | typed covariance comparison and empirical consistency criteria, public assess/verify CLI, checked local blocker fixture | Numerical agreement, independence, force coverage, and raw NEES evidence remain separate; focused/full/public/package, independent review, GitHub CI, and integration must pass without promoting screening evidence to certification. |

## Next Best Paths

1. Ingest reviewed mission-specific station residuals, propulsion execution residuals including
   timing, and insertion covariance into the typed evidence contract; replace illustrative bounds
   only where applicability and derivation review support promotion. Add subsystem test evidence
   before considering a mission-calibrated protocol claim.
2. Acquire an independent high-fidelity covariance comparison covering drag, SRP, and third-body
   dynamics plus a raw empirical truth-error campaign before attempting to satisfy the new
   production covariance criteria. Criteria satisfaction still does not confer flight certification.
3. Add a digest-bearing lifecycle artifact manifest only if separately copied bundle provenance is
   needed; do not infer bundle integrity from result re-execution.
4. Add optional evidence-bound explanations over verified finding ids only after deterministic
   triage is sufficient. Keep autonomous execution closed, and keep surrogate acceleration closed
   unless a measured assurance teacher or other evaluator passes the cost and fidelity gate.
5. Recreate an isolated TudatPy environment only if a future claim specifically needs a fresh
   Tudat-vs-local comparison or native variational covariance refresh.
6. Choose any higher-authority reentry campaign deliberately and keep it separate from the local
   screening product: atmosphere uncertainty, 6-DOF/GNC, aerothermal/material response, or external
   Dymos/Tudat/Orekit correlation.
