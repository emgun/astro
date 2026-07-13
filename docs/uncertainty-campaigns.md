# Uncertainty Campaigns

Astro uncertainty campaigns apply one validated uncertainty definition to a suite-owned workflow.
Sampling, workflow evaluation, metric extraction, requirements, statistics, and evidence retention
remain separate contracts.

The checked lifecycle campaign is:

```bash
astro validate-campaign examples/campaigns/leo_lifecycle_robustness.yaml
astro run-campaign examples/campaigns/leo_lifecycle_robustness.yaml \
  --output-dir /tmp/astro-lifecycle-robustness --workers 2
astro summarize-campaign /tmp/astro-lifecycle-robustness
astro profile-campaign /tmp/astro-lifecycle-robustness \
  --output /tmp/astro-lifecycle-profile.json
```

Its eight Latin-hypercube cases vary reviewed epistemic launch-thrust, shared-wet-mass, twin-power,
deorbit, reentry-atmosphere, and reentry-aerodynamics inputs through typed owning-phase overrides.
The resulting pass fraction is a design-space summary, not an operational frequency or certified
mission-success probability.

Campaign artifacts include the resolved definition and digest, sampled physical values, one typed
outcome per requested case, aggregate statistics, evaluator timing, and a concise text summary.
Invalid realizations, workflow failures, numerical failures, OOD decisions, and policy rejections
remain explicit outcomes and are not silently removed from the evidence.

Execution is crash-resumable and supports bounded worker processes. Workers initialize independent
suite runtimes; the parent writes cases in deterministic sample order and checkpoints each stopping
batch. Fixed-count, confidence-interval, and metric-stability rules are explicit campaign contracts.
Every case outcome record is retained. Successful result artifacts can be retained for all cases,
no cases, requirement-boundary cases, or a deterministic audit sample; failed evaluations retain
their typed case evidence but have no result object to serialize. CI half-width stopping currently
requires equal sample weights and fails closed for weighted ensembles. `--max-cases` bounds an
execution and `--dry-run` resolves the scenario and registries without running physics or writing
evidence.

`astro profile-campaign` derives a machine-scoped timing product from integrity-checked case
records. It reports setup, authoritative evaluation, metric extraction, retained-result
serialization, unattributed overhead, total instrumented time, median absolute deviation, and
evaluator share of instrumented time. The profile binds the source case digest, software/runtime
compatibility, and machine identity. It does not modify deterministic campaign statistics. Cases written before extraction
instrumentation remain visibly incomplete and cannot produce evaluator-share claims.

`astro analyze-campaign-sensitivity` derives digest-bound Spearman and PRCC association evidence
for selected numeric metrics and signed requirement margins. It requires a completed, equal-weight
campaign that passes the configured sample-size gates and rejects failed cases, missing targets, boolean margins, and
singular ranked designs. See `docs/campaign-sensitivity.md` for the validity gates and the checked
64-case lifecycle attribution result.

## Surrogate Boundary

The baseline benchmark does not currently justify a learned evaluator. Checked campaign profiling
found that local J2 orbit evaluation and the integrated digital twin account for more than 97% of
their instrumented phase time, but their median evaluations are only about 0.000553 and 0.001431 seconds.
Both fail the preregistered 0.050-second absolute-cost gate. Surrogate training therefore remains
stopped until a materially more expensive bounded evaluator is measured and the cost cannot be
removed more safely through batching, vectorization, caching, or analytical changes.

A future promoted surrogate may screen a bounded campaign, but decision-critical, boundary, failed,
and out-of-domain cases must be replayed through authoritative physics. Training never grants
promotion, and a model trained against one teacher cannot claim another teacher's fidelity.
