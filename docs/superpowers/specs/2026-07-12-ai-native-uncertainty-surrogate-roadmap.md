# AI-Native Uncertainty And Surrogate Roadmap

**Status:** Implemented through the benchmark-gated surrogate stop condition
**Date:** 2026-07-12
**North star:** Make uncertainty a suite-level mission abstraction and make AI a typed,
evidence-bound orchestrator. Deterministic Astro workflows and explicitly validated external
adapters remain authoritative. Learned models may accelerate bounded analyses but may not silently
expand a claim.

## Executive Decision

Astro should not add Monte Carlo separately to launch, orbit, thermal, power, ADCS, constellation,
reentry, and lifecycle modules. It should add one reusable uncertainty campaign layer that can run
any suite-owned workflow with interchangeable sampling, evaluation, aggregation, stopping, and
fidelity policies.

A surrogate is an evaluator, not a sampler:

```text
UncertaintyModel -> SamplePlan -> Evaluator -> ObservationSet -> Statistics/Evidence
                                   |
                                   +-- authoritative Astro workflow
                                   +-- optional external adapter
                                   +-- validated surrogate
                                   +-- progressive-fidelity policy
```

The first flagship product is a **Robust Mission Lifecycle Campaign** over the existing
launch-to-reentry workflow. The first learned-model candidate is a bounded residual surrogate for
an expensive in-orbit or coupled-twin evaluation, selected only after a benchmark shows where
campaign time is actually spent.

## Why This Path

The current `astro_dynamics.monte_carlo` surface is useful but narrow: it applies independent
Gaussian position and velocity perturbations and stores complete propagated cases. The suite now
has enough integrated products that copying this pattern into each module would create different
seed rules, distributions, statistics, failure semantics, and artifact formats.

The existing JAX backend already provides batched research propagation and differentiable
sensitivities. It is a useful implementation substrate, but its force-model boundary is explicitly
screening-oriented. It is not itself a training system, model registry, or promotion authority.

The assistant already demonstrates the desired AI boundary: intent becomes a typed plan, an
allow-listed executor invokes deterministic tools, validators inspect artifacts, and a trace records
the outcome. The uncertainty and surrogate layers should extend that pattern rather than introduce
a second autonomous execution model.

## Product Principles

1. **Uncertainty is domain infrastructure.** Monte Carlo, Latin hypercube, Sobol, sweeps, model
   ensembles, importance sampling, and active learning are strategies over one campaign contract.
2. **A realization is a complete mission input.** Workflows receive a validated scenario
   realization and do not know how it was sampled.
3. **Physics authority is explicit.** Every result identifies its evaluator, model version,
   assumptions, source artifacts, and claim boundary.
4. **Surrogates accelerate; they do not certify.** Promotion grants use within a validated domain,
   not equivalence outside it.
5. **Requirements drive validation.** Position error alone is insufficient when event timing,
   margin sign, or mission success classification is the decision surface.
6. **Failures are data.** Invalid samples, workflow exceptions, numerical failures, and requirement
   failures are distinct typed outcomes.
7. **Reproducibility survives parallelism.** Sample identity and random streams derive from a root
   seed and stable sample index, not worker scheduling.
8. **AI proposes and explains; gates decide.** An agent can design campaigns, nominate challengers,
   and recommend promotion. Deterministic policy and evidence decide execution and promotion.
9. **Default installation remains local and deterministic.** Advanced training dependencies stay in
   an optional extra. Loading an artifact must never execute arbitrary serialized code.
10. **No hidden operational claims.** Optional backend evidence, trained-model evidence, and
    machine-specific performance remain scoped to their recorded environment and domain.

## Core Domain Model

### Uncertainty definitions

- `UncertainParameter`: stable parameter id, unit, allowed workflow targets, distribution, and
  epistemic/aleatory classification.
- `DistributionSpec`: constant, uniform, normal, lognormal, triangular, bounded empirical, or
  discrete categorical distribution.
- `CorrelationModel`: independent, correlation matrix, or named joint transform. Invalid or
  non-positive-semidefinite matrices fail before sampling.
- `ModelVariant`: a discrete, provenance-bearing alternative such as an atmosphere or force model.
- `UncertaintyModel`: ordered parameters, correlations, constraints, and model variants.

Parameter application uses an allow-listed target registry owned by each workflow adapter. The
campaign engine must not mutate arbitrary object paths supplied by an untrusted prompt.

### Sampling and realization

- `SamplerSpec`: pseudorandom, Latin hypercube, Sobol, grid/sweep, or discrete ensemble in the first
  release; weighted importance sampling follows after weighted statistics are proven.
- `SamplePlan`: sampler, root seed, requested count, skip/scramble controls, and deterministic
  sample ids.
- `ParameterRealization`: physical values, normalized coordinates, sample weight, source sampler,
  and model choices.
- `ScenarioRealization`: base scenario digest, applied parameter bindings, resolved scenario digest,
  and validation result.

### Evaluation

- `EvaluatorSpec`: evaluator id, kind, workflow, implementation version, supported domain, and
  fallback policy.
- `AuthoritativeEvaluator`: invokes a suite-owned workflow or explicitly selected external adapter.
- `SurrogateEvaluator`: loads a safe artifact, checks domain/OOD policy, evaluates, and returns
  approximate suite-owned predictions.
- `ProgressiveFidelityEvaluator`: routes screening cases, boundary cases, tails, failures, and OOD
  cases across a declared evaluator ladder.
- `EvaluationOutcome`: success, invalid realization, execution failure, numerical failure, OOD,
  or policy rejection, with timings and artifact references.

### Observation and decision

- `MetricSpec`: extracts a typed scalar, category, event time, or time-series summary from a workflow
  result.
- `RequirementSpec`: comparison, tolerance, target interval, or boolean rule over a metric.
- `CaseObservation`: metric values, requirement outcomes, evaluator provenance, and referenced
  artifacts.
- `CampaignStatistics`: weighted/unweighted summaries, quantiles, confidence intervals, failure
  counts, effective sample size, and convergence history.
- `CampaignResult`: immutable campaign manifest, case index, aggregate evidence, warnings, and claim
  boundary. Large trajectories remain separate case artifacts instead of being embedded repeatedly.

## Proposed Package Boundaries

```text
src/astro_uq/
  models.py              # stable campaign, uncertainty, outcome, and evidence schemas
  distributions.py       # distribution validation and inverse transforms
  correlations.py        # joint transforms and PSD checks
  samplers.py             # random, LHS, Sobol, sweep, ensemble
  parameters.py           # allow-listed workflow parameter bindings
  evaluators.py           # evaluator protocol and authoritative evaluator
  progressive.py          # fidelity routing and replay policy
  metrics.py              # metric extraction and requirement evaluation
  statistics.py           # weighted summaries and confidence intervals
  stopping.py             # fixed-count and convergence stopping rules
  runner.py               # resumable campaign orchestration
  io.py                   # manifests, case index, summaries, safe resume
  adapters/               # workflow-specific bindings and metric registries

src/astro_surrogates/
  models.py               # dataset, training, artifact, domain, validation, promotion schemas
  datasets.py             # episode generation, manifests, splits, integrity checks
  features.py             # unit-aware deterministic feature/target transforms
  baselines.py            # mean, linear, ridge, polynomial/residual baselines
  training.py             # framework-neutral training orchestration
  validation.py           # rollout, event, margin, OOD, speed, and invariants gates
  runtime.py              # safe model loading and inference
  registry.py             # local immutable registry and promotion records
  active_learning.py      # candidate ranking; no autonomous promotion
  jax_models.py           # optional residual MLP after the baseline gate
```

`astro_uq` is a required package using NumPy/SciPy. `astro_surrogates` initially supports a required
NumPy/SciPy baseline. Neural training remains behind an optional `surrogate` extra. A compatibility
spike must choose and pin the smallest JAX optimizer stack; do not add a general ML platform.

## Public Workflow

The first campaign command should be:

```bash
astro run-campaign examples/campaigns/leo_lifecycle_robustness.yaml \
  --output-dir /tmp/astro-lifecycle-campaign
```

The output directory contains:

```text
campaign.json             # resolved definition, versions, digests, state, claim boundary
samples.jsonl             # immutable realization records
cases.jsonl               # outcome and observation index
statistics.json           # aggregate estimates and convergence history
summary.txt               # human-readable decision summary
artifacts/<sample-id>/     # authoritative workflow artifacts when retained
```

Supporting commands are staged rather than added at once:

```text
astro validate-campaign
astro run-campaign
astro summarize-campaign
astro build-surrogate-dataset
astro train-surrogate
astro validate-surrogate
astro promote-surrogate
astro inspect-surrogate
```

Promotion and campaign execution are separate actions. `train-surrogate` never makes a model
eligible for production campaign routing.

## Campaign Configuration Shape

```yaml
campaign_id: leo-lifecycle-robustness-v1
workflow:
  kind: mission_lifecycle
  scenario: examples/lifecycle/leo_round_trip.yaml
uncertainty:
  parameters:
    - parameter_id: insertion_velocity_x
      target: mission.launch.insertion.velocity_km_s[0]
      unit: km/s
      kind: aleatory
      distribution: {kind: normal, mean: 0.0, sigma: 0.002}
    - parameter_id: spacecraft_dry_mass
      target: mission.spacecraft.dry_mass_kg
      unit: kg
      kind: epistemic
      distribution: {kind: triangular, low: 135.0, mode: 145.0, high: 160.0}
sampler:
  kind: latin_hypercube
  samples: 128
  seed: 7122026
evaluator:
  kind: authoritative
  backend: local
metrics:
  - {metric_id: final_mass_margin_kg, extractor: lifecycle.final_mass_margin_kg}
  - {metric_id: reentry_peak_heat_rate, extractor: lifecycle.reentry.peak_heat_rate_w_m2}
requirements:
  - {requirement_id: propellant_reserve, metric: final_mass_margin_kg, operator: ge, value: 0.0}
stopping:
  kind: fixed_count
retention:
  keep: failures_and_boundaries
```

The exact target and extractor strings are registry identifiers, not free-form JSONPath.

## Workflow Adoption Order

1. **Orbit propagation reference adapter:** cheapest proving surface and compatibility bridge for the
   existing `research-propagate` behavior.
2. **Digital twin adapter:** power, thermal, ADCS, link, coverage, mass, and design-margin metrics.
3. **Reentry adapter:** atmosphere, aerodynamic, guidance, load, heating, target-miss, and terminal
   event uncertainty.
4. **Mission lifecycle adapter:** cross-phase uncertainty propagation and robust mission success.
5. **Constellation adapter:** fleet coverage, revisit, data-volume, and member-failure uncertainty.
6. **Launch and OD adapters:** added only with domain-specific metrics and correct stochastic
   semantics; OD measurement-noise campaigns must not be confused with truth-state uncertainty.

## Statistical Correctness Gates

Before a new sampling strategy is public, tests must establish:

- identical seed and definition produce identical sample values and ids;
- LHS stratifies every marginal once per stratum;
- Sobol points match SciPy reference behavior and document power-of-two balance guidance;
- correlated transforms recover target moments within preregistered tolerance;
- weighted means, variance, quantiles, and effective sample size match independent fixtures;
- confidence intervals state their estimator and assumptions;
- failed evaluations are never silently dropped from a success-probability denominator;
- resume does not duplicate cases or change sample identity;
- parallel and serial execution produce equivalent ordered evidence;
- model-ensemble results preserve variant identity rather than averaging incompatible models.

## Surrogate Strategy

### Selection gate

Profile representative orbit, twin, constellation, reentry, and lifecycle campaigns before choosing
the first learned target. Measure evaluator wall time, serialization time, artifact size, and metric
extraction time. A surrogate branch proceeds only if one bounded evaluator dominates campaign cost
and a simple analytical or vectorized implementation cannot remove the bottleneck more safely.

Expected first candidate: a residual correction from suite local/JAX propagation to a bounded
high-fidelity in-orbit teacher:

```text
x_teacher(t + dt) = Phi_baseline(x, p, u, dt) + delta_theta(x, p, u, dt)
```

The teacher, baseline, force models, time horizon, orbital regime, and parameter ranges are part of
the model domain. A model trained against local physics cannot claim Orekit or Tudat fidelity.

### Model ladder

1. Zero-correction and mean-correction baselines.
2. Linear/ridge and polynomial residual models.
3. Gaussian-process challenger when dataset size and uncertainty estimates justify it.
4. JAX residual MLP with one-step and rollout losses.
5. Query-at-time operator model for variable output times.
6. Graph or neural-operator models only for field/network problems such as multi-node thermal or
   interacting constellation models.

Replacing cheap two-body/J2 propagation with a neural network is not a roadmap objective.

### Dataset contract

`SimulationDatasetSpec` records teacher and baseline evaluators, scenario family, parameter domain,
feature/target schemas, units, frames, time coordinates, split policy, seed, retention policy, and
software/environment versions. `SimulationEpisode` records a complete trajectory or workflow
realization.

Splits occur by scenario, orbit, spacecraft configuration, and complete episode. Random timestep
splits are forbidden because adjacent states leak trajectory identity. Required split classes are:

- training;
- in-domain validation;
- held-out in-domain test;
- boundary challenge;
- out-of-domain challenge.

### Promotion gate

A `SurrogatePromotionDecision` can be `rejected`, `experimental`, or `campaign_screening`. Initial
promotion requires all of the following:

- immutable dataset and model artifact digests;
- deterministic retraining within documented tolerance;
- improvement over zero/linear baselines on held-out episodes;
- preregistered one-step and rollout error thresholds;
- event-time, terminal-state, metric, and requirement-classification thresholds;
- domain and OOD tests with explicit fallback behavior;
- relevant invariant checks;
- no unresolved false negatives in designated failure challenge cases;
- measured end-to-end batch speedup, including loading and feature transforms;
- independent validation command producing the same decision.

Promotion never grants operational or certification authority. Domain mismatch, low confidence, or
OOD status triggers authoritative fallback.

## Progressive Fidelity Policy

A surrogate-assisted campaign should use the surrogate for broad screening and replay these cases
authoritatively:

- all predicted failures;
- all samples within a configured distance of a requirement boundary;
- all OOD or low-confidence samples;
- deterministic quantile representatives and random audit samples;
- any sample selected for a release, design, or operational claim.

The campaign records the selection policy and computes disagreement and missed-failure estimates.
Probability estimates must account for screening and reweighting; the surrogate may not silently
remove samples from the population.

## AI-Native Operating Model

The assistant may compile natural-language intent into a typed `CampaignPlan` or
`SurrogateExperimentPlan`. The allow-listed tool registry exposes validation, dry-run, execution,
artifact inspection, and promotion-review operations.

An AI research loop may:

- propose uncertainty parameters and challenge domains;
- compare samplers and evaluator ladders;
- nominate baseline and challenger models;
- select active-learning candidates from high-error, high-uncertainty, OOD-adjacent, and
  requirement-boundary regions;
- summarize evidence and recommend continue, pivot, or kill.

It may not:

- invent unregistered parameter bindings or metric extractors;
- execute arbitrary code from a model artifact;
- promote its own model;
- suppress failed or unfavorable cases;
- broaden claim language beyond the validation record;
- invoke paid or sensitive providers without approval.

Every AI-directed run writes the typed plan, tool calls, policy decisions, input/output digests,
model/provider identity when applicable, and final deterministic gate result.

## Roadmap Phases And Exit Gates

### Phase 0: Benchmark and contracts

Define the flagship decision, benchmark current workflow costs, freeze schemas, and preregister
statistical and claim-boundary tests. Exit when a reviewed architecture decision record identifies
the first workflow and first surrogate candidate or rejects surrogate work as premature.

### Phase 1: UQ kernel

Implement validated uncertainty models, deterministic realizations, random/LHS/Sobol/sweep/ensemble
samplers, workflow parameter registries, and artifact IO. Exit when sampling fixtures, seed replay,
resume, and serial/parallel equivalence pass.

### Phase 2: Evidence and stopping

Implement metric extraction, requirements, weighted statistics, confidence intervals, failure
accounting, retention, and fixed/convergence stopping. Exit when independent statistical fixtures
and malformed/failure campaign tests pass.

### Phase 3: Workflow products

Ship orbit, twin, reentry, and lifecycle adapters plus `astro run-campaign`. Exit when the checked
Robust Mission Lifecycle Campaign produces reproducible case and aggregate artifacts with explicit
claim boundaries.

### Phase 4: Surrogate lifecycle

Ship dataset generation, baseline models, validation, safe artifacts, registry, and promotion.
Proceed to a JAX residual model only after the baseline and bottleneck gates. Exit when a promoted
screening model demonstrates bounded accuracy, failure recall, fallback, and end-to-end speedup.

### Phase 5: AI-native experiment loop

Add typed campaign and surrogate plans to the assistant registry, deterministic validators,
approval policy, traces, and active-learning recommendation artifacts. Exit when golden prompts can
plan and dry-run the flagship campaign without arbitrary execution and an executed local fixture is
fully replayable.

### Phase 6: Advanced reliability and robust design

Add weighted importance sampling, global sensitivity, surrogate-assisted optimization, Bayesian
experimental design, and constellation/coupled-twin challengers one decision at a time. Each method
requires a baseline, independent fixture, and a concrete mission decision it improves.

## Release Strategy

- Treat Phases 0-3 as the next coherent product increment.
- Keep Phase 4 separately promotable because learned artifacts and optional dependencies add a new
  evidence boundary.
- Keep AI provider integration optional; deterministic planners and golden fixtures remain required.
- Do not add unfinished Phase 5-6 items to the release checklist until their public surfaces exist.
- Update `docs/validation-matrix.md` with required local gates as each phase lands; optional live
  teacher campaigns remain in `docs/validation/live-backend-campaigns.md`.

## Decision Gates

Stop or pivot when:

- campaign abstraction cannot represent an existing workflow without workflow-specific leakage;
- weighted statistics or resume semantics are not independently verified;
- campaign overhead exceeds ten percent for representative local workloads without a clear reason;
- a surrogate does not beat a simple baseline or deliver meaningful end-to-end speedup;
- OOD/fallback behavior cannot be made deterministic;
- AI orchestration duplicates existing typed assistant behavior instead of extending it;
- a proposed method lacks a named mission decision, metric, and falsifiable acceptance gate.

## Recommended First Delivery

Implement Phases 0-3 as **Astro Uncertainty Campaigns v1**. Use the existing lifecycle example as the
flagship, but begin implementation with the orbit adapter to prove deterministic realization and
sampling semantics. In parallel with the campaign product, run the benchmark that selects the first
surrogate teacher/target. Do not make neural-model dependencies part of the v1 critical path.
