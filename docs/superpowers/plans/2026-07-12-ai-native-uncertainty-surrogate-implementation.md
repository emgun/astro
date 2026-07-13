# AI-Native Uncertainty And Surrogate Implementation Plan

> **For agentic workers:** Execute task-by-task. Keep one task or tightly coupled task group per
> commit, run the stated focused gate before each commit, and run the full release gate at phase
> boundaries. Do not begin neural training work until the benchmark and baseline promotion gates
> authorize it.

**Goal:** Deliver a reusable uncertainty campaign product for Astro workflows, demonstrate it with
a robust launch-to-reentry lifecycle campaign, then add a validated surrogate lifecycle and typed
AI orchestration without weakening suite-owned physics or claim boundaries.

**Architecture:** `astro_uq` owns uncertainty, sampling, evaluation, statistics, stopping, campaign
execution, and workflow adapters. `astro_surrogates` owns datasets, training artifacts, validation,
promotion, safe inference, and active-learning recommendations. `astro_assistant` compiles intent
into allow-listed plans but does not compute physics or promote models.

**Design:** `docs/superpowers/specs/2026-07-12-ai-native-uncertainty-surrogate-roadmap.md`

**Tech stack:** Python 3.12, Pydantic v2, NumPy, SciPy QMC/statistics, Typer, PyYAML, pytest; optional
JAX training only after its compatibility spike.

---

## Execution Outcome - 2026-07-13

The plan reached its evidence-backed stop condition on `codex/uncertainty-campaigns`:

- Phases 0-3 are implemented, including bounded process execution, adaptive stopping, retention,
  interruption/resume integrity, the legacy orbit compatibility path, typed cross-phase lifecycle
  inputs, literal twin/reentry campaign gates, the public CLI, and the checked lifecycle campaign.
- Phase 4 Task 16 is implemented as safe immutable surrogate schemas and integrity-checked NPZ/JSON
  artifacts. Tasks 17-23 were not authorized because the benchmark found no evaluator-dominated
  target; no training dependency, learned model, promotion, or surrogate evaluator was added.
- Phase 5 typed campaign plans and deterministic assistant policy are implemented. Active-learning
  selection remains inapplicable until a future benchmark reopens the surrogate gate.
- Phase 6 remains an explicit advanced-methods backlog, not hidden implementation work.

The unchecked task criteria below preserve the original design and conditional gates. They must not
be read as an instruction to bypass the benchmark kill decision or as a claim that conditional
surrogate work ran.

## Scope And Delivery Rules

Branch target: create `codex/uncertainty-campaigns` from current `main`.

Required v1 delivery:

- uncertainty, distribution, correlation, realization, sampler, evaluator, observation, statistics,
  stopping, and campaign schemas;
- pseudorandom, Latin hypercube, Sobol, sweep, and discrete-ensemble strategies;
- authoritative orbit, digital-twin, reentry, and lifecycle adapters;
- deterministic/resumable local campaign execution and bounded parallel execution;
- campaign validation, execution, summary, and artifact commands;
- one checked Robust Mission Lifecycle Campaign;
- campaign documentation and required local gates.

Separately promotable delivery:

- simulation dataset and split contracts;
- NumPy/SciPy residual baselines;
- safe surrogate artifact and local registry;
- rollout, event, requirement, OOD, invariant, and speed validation;
- progressive-fidelity screening and authoritative replay;
- optional JAX residual MLP if the benchmark gate passes;
- typed assistant planning and active-learning recommendation artifacts.

Explicit non-goals:

- no operational probability, flight certification, or standards-authority claims;
- no arbitrary Python import, pickle, JSONPath, shell, or user-supplied extractor execution;
- no neural replacement for cheap two-body/J2 propagation;
- no hidden dropping of failed cases;
- no distributed scheduler, cloud training platform, hosted model registry, or required AI provider;
- no FNO/DeepONet implementation until a field/operator problem and dataset justify it.

## Phase 0: Benchmark And Contract Freeze

### Task 1: Record benchmark protocol and claim boundaries

**Files:**
- Create: `docs/benchmarks/uncertainty-campaign-baseline.md`
- Create: `examples/campaigns/benchmark-manifest.yaml`
- Test: `tests/benchmarks/test_campaign_benchmark_manifest.py`

- [ ] Define representative orbit, twin, constellation, reentry, and lifecycle commands using
  checked examples.
- [ ] Define measured fields: warm-up count, repeated wall time, evaluator time, serialization time,
  metric extraction time, peak RSS, artifact bytes, and machine/runtime metadata.
- [ ] Define the surrogate selection rule: the candidate evaluator must dominate campaign cost,
  have a bounded input/output contract, and resist a simpler vectorization/caching fix.
- [ ] Record that optional Orekit/Tudat teacher results are machine-scoped evidence.
- [ ] Add a schema test so every benchmark entry names its workflow, command, repetitions, expected
  artifact, and claim boundary.
- [ ] Run the local benchmark protocol and commit the summary. Do not add generated bulk artifacts.

**Gate:** The document identifies one first surrogate candidate, one challenger, and a kill decision
if neither earns acceleration work.

### Task 2: Freeze public contracts with schema tests

**Files:**
- Create: `src/astro_uq/models.py`
- Create: `tests/astro_uq/test_models.py`
- Create: `src/astro_uq/__init__.py`
- Modify: `pyproject.toml`

- [ ] Add enums for uncertainty kind, distribution kind, sampler kind, evaluator kind, outcome
  status, metric value kind, requirement operator, retention policy, and campaign state.
- [ ] Add `UncertainParameter`, `DistributionSpec`, `CorrelationModel`, `ModelVariant`,
  `UncertaintyModel`, `SamplerSpec`, `SamplePlan`, `ParameterRealization`, `ScenarioRealization`,
  `EvaluatorSpec`, `EvaluationOutcome`, `MetricSpec`, `RequirementSpec`, `CaseObservation`,
  `CampaignStatistics`, `CampaignDefinition`, and `CampaignResult`.
- [ ] Reject booleans as numeric inputs, non-finite values, duplicate ids, invalid bounds, ambiguous
  units, invalid requirement operands, and unknown schema versions.
- [ ] Use tuples for immutable ordered definitions and explicit `schema_version` fields on persisted
  top-level products.
- [ ] Add package exports, wheel inclusion, and strict MyPy package inclusion.

**Gate:** `python -m pytest tests/astro_uq/test_models.py -q` and strict MyPy for `astro_uq` pass.

## Phase 1: UQ Kernel

### Task 3: Implement distributions and joint transforms

**Files:**
- Create: `src/astro_uq/distributions.py`
- Create: `src/astro_uq/correlations.py`
- Test: `tests/astro_uq/test_distributions.py`
- Test: `tests/astro_uq/test_correlations.py`

- [ ] Implement inverse-CDF transforms for constant, uniform, normal, lognormal, triangular,
  bounded empirical, and categorical definitions.
- [ ] Keep normalized coordinates in every realization for replay and sensitivity analysis.
- [ ] Implement independent and Gaussian-copula correlation transforms.
- [ ] Validate symmetry, unit diagonal, dimensions, parameter ordering, and PSD tolerance before any
  sample executes.
- [ ] Compare transforms against SciPy fixtures and seeded moment checks.

**Gate:** Invalid distributions/correlations fail before workflow execution; independent fixtures
match analytical moments within preregistered test tolerances.

### Task 4: Implement deterministic samplers

**Files:**
- Create: `src/astro_uq/samplers.py`
- Test: `tests/astro_uq/test_samplers.py`

- [ ] Define a `Sampler` protocol returning stable indexed normalized points and weights.
- [ ] Implement pseudorandom, SciPy Latin hypercube, scrambled/unscrambled Sobol, Cartesian sweep,
  and discrete model ensemble samplers.
- [ ] Derive sample ids and child random streams from campaign digest, root seed, and sample index.
- [ ] Record SciPy engine configuration, Sobol skip/scramble settings, and power-of-two warnings.
- [ ] Test golden seeded values, LHS marginal stratification, Sobol reference points, sweep ordering,
  and serial/parallel identity.

**Gate:** Repeated generation is byte-stable after canonical JSON serialization.

### Task 5: Add allow-listed parameter registries

**Files:**
- Create: `src/astro_uq/parameters.py`
- Create: `src/astro_uq/adapters/base.py`
- Test: `tests/astro_uq/test_parameters.py`

- [ ] Define `ParameterBinding` with id, workflow kind, unit, value type, bounds, getter, and pure
  scenario-update function.
- [ ] Require adapters to register bindings explicitly; configuration refers to stable binding ids.
- [ ] Apply all values to a copy, validate the resolved Pydantic model, and calculate base/resolved
  digests.
- [ ] Reject duplicate writes, incompatible units/types, unsupported parameter/workflow pairs, and
  constraint violations.
- [ ] Test that the base scenario is unchanged after realization.

**Gate:** A campaign cannot mutate an unregistered field or construct an invalid scenario.

### Task 6: Add campaign artifact IO and resume state

**Files:**
- Create: `src/astro_uq/io.py`
- Test: `tests/astro_uq/test_io.py`

- [ ] Implement atomic writes for `campaign.json`, `samples.jsonl`, `cases.jsonl`,
  `statistics.json`, and `summary.txt`.
- [ ] Canonicalize JSON before hashing and store relative artifact references.
- [ ] Add a lock/ownership record, campaign definition digest, completed sample ids, and explicit
  interrupted/failed/completed states.
- [ ] Resume only when the definition digest and software compatibility policy match.
- [ ] Recover safely from a truncated final JSONL record and reject interior corruption.
- [ ] Never deserialize pickle or import model-defined code.

**Gate:** Killing and resuming a fixture produces the same final case index as an uninterrupted run.

## Phase 2: Evaluation And Evidence

### Task 7: Implement evaluator protocol and authoritative execution

**Files:**
- Create: `src/astro_uq/evaluators.py`
- Test: `tests/astro_uq/test_evaluators.py`

- [ ] Define a generic typed evaluator protocol over resolved scenarios and suite-owned results.
- [ ] Record setup, evaluation, serialization, and total timings separately.
- [ ] Map exceptions to typed outcomes without swallowing traceback summaries or sample context.
- [ ] Distinguish invalid realization, policy rejection, numerical failure, and execution failure.
- [ ] Add fake evaluators for deterministic unit tests.

**Gate:** Every requested sample produces exactly one outcome record, including failures.

### Task 8: Implement metrics and requirements

**Files:**
- Create: `src/astro_uq/metrics.py`
- Test: `tests/astro_uq/test_metrics.py`

- [ ] Define an allow-listed metric extractor registry parallel to parameter bindings.
- [ ] Support scalar numeric, boolean, category, event time, and declared time-series summary values.
- [ ] Implement `lt`, `le`, `gt`, `ge`, `between`, `within_tolerance`, and boolean requirements.
- [ ] Record missing/not-applicable separately from failed.
- [ ] Reject arbitrary expressions and mismatched units.

**Gate:** Requirement outcomes can be recomputed from retained observations without rerunning physics.

### Task 9: Implement statistics and convergence

**Files:**
- Create: `src/astro_uq/statistics.py`
- Create: `src/astro_uq/stopping.py`
- Test: `tests/astro_uq/test_statistics.py`
- Test: `tests/astro_uq/test_stopping.py`

- [ ] Implement weighted/unweighted count, mean, variance, standard error, quantiles, effective sample
  size, categorical frequency, requirement pass probability, and outcome-status counts.
- [ ] Use explicit methods for binomial intervals and weighted uncertainty estimates; record method
  names and assumptions in the result.
- [ ] Implement fixed count, CI half-width, and metric-stability stopping rules with minimum/maximum
  counts and batch boundaries.
- [ ] Treat invalid/execution outcomes according to a declared denominator policy and always report
  raw counts.
- [ ] Validate against analytical distributions and independent static fixtures.

**Gate:** Weighted fixtures match independently calculated expected values; convergence cannot stop
before minimum count or when effective sample size is insufficient.

### Task 10: Implement campaign runner and retention

**Files:**
- Create: `src/astro_uq/runner.py`
- Test: `tests/astro_uq/test_runner.py`

- [ ] Orchestrate sample generation, realization, evaluation, metric extraction, statistics, stopping,
  retention, and atomic checkpointing.
- [ ] Implement serial execution first, then bounded process execution with deterministic merge
  ordering and per-worker initialization.
- [ ] Add retention policies: all, none, failures, failures-and-boundaries, and deterministic audit
  sample.
- [ ] Bound in-flight work so convergence stopping does not launch an unbounded tail.
- [ ] Add cancellation handling that leaves a resumable campaign state.

**Gate:** Serial and two-worker fixture campaigns produce equivalent samples, observations,
statistics, and retained-artifact decisions.

## Phase 3: Workflow Adapters And Flagship

### Task 11: Migrate orbit Monte Carlo behind the campaign API

**Files:**
- Create: `src/astro_uq/adapters/orbit.py`
- Modify: `src/astro_dynamics/monte_carlo.py`
- Test: `tests/astro_uq/test_orbit_adapter.py`
- Test: `tests/astro_dynamics/test_monte_carlo.py`

- [ ] Register initial Cartesian state, spacecraft mass/area/Cd/Cr, and supported force-model
  bindings with units and bounds.
- [ ] Register final state, extrema, event, and trajectory-duration metrics.
- [ ] Preserve `run_initial_state_monte_carlo` as a compatibility wrapper that delegates to the new
  engine and converts back to `MonteCarloResult`.
- [ ] Add a golden parity test for current seeded local behavior.
- [ ] Prevent compatibility conversion when a campaign uses unsupported correlations, weights, or
  model variants.

**Gate:** Existing Monte Carlo tests and the new campaign orbit fixture pass without public breakage.

### Task 12: Add digital twin and reentry adapters

**Files:**
- Create: `src/astro_uq/adapters/twin.py`
- Create: `src/astro_uq/adapters/reentry.py`
- Test: `tests/astro_uq/test_twin_adapter.py`
- Test: `tests/astro_uq/test_reentry_adapter.py`

- [ ] Register bounded power, thermal, ADCS, link, mass, atmosphere, aerodynamics, and guidance
  parameters that correspond to real Pydantic fields.
- [ ] Register SOC, temperature, pointing, actuator, link, coverage, mass, heating, load, target-miss,
  and terminal-event metrics.
- [ ] Include design-screening claim boundaries from source products in every observation.
- [ ] Test requirement boundary signs and failed-workflow accounting.

**Gate:** Small LHS campaigns run deterministically against checked twin and reentry examples.

### Task 13: Add lifecycle adapter and uncertainty propagation policy

**Files:**
- Create: `src/astro_uq/adapters/lifecycle.py`
- Modify: `src/astro_mission/models.py` only if a typed uncertainty-compatible input is missing
- Test: `tests/astro_uq/test_lifecycle_adapter.py`

- [ ] Register cross-phase parameters at their owning phase rather than patching generated artifacts.
- [ ] Preserve state, epoch, and mass continuity checks for every realization.
- [ ] Register phase success, final reserve, twin limiting margin, deorbit consumption, entry interface,
  peak reentry loads, target miss, and overall mission success metrics.
- [ ] Record which phase stopped a failed realization and do not execute downstream phases after a
  fail-closed gate.

**Gate:** A deliberately failing insertion case is counted as a lifecycle requirement failure with
no downstream artifact fabrication.

### Task 14: Add campaign CLI surface

**Files:**
- Modify: `src/astro_cli/main.py`
- Create: `src/astro_uq/cli.py`
- Test: `tests/astro_cli/test_campaign_cli.py`

- [ ] Add `astro validate-campaign`, `astro run-campaign`, and `astro summarize-campaign`.
- [ ] Support output directory, resume, worker count, max cases, and dry-run options.
- [ ] Make `validate-campaign` resolve registries and scenarios without executing workflows.
- [ ] Print concise JSON errors with campaign/sample ids and nonzero exit status.
- [ ] Keep command handlers thin; orchestration belongs in `astro_uq`.

**Gate:** CLI tests cover validation, successful run, malformed definition, execution failure, resume,
and summary regeneration.

### Task 15: Ship Robust Mission Lifecycle Campaign v1

**Files:**
- Create: `examples/campaigns/leo_lifecycle_robustness.yaml`
- Create: `docs/uncertainty-campaigns.md`
- Modify: `docs/mission-lifecycle.md`
- Modify: `docs/validation-matrix.md`
- Test: `tests/astro_uq/test_reference_campaign.py`

- [ ] Select a small reviewed set of insertion, mass, power, atmosphere, and aerodynamic uncertainties
  with defensible units and bounds.
- [ ] Use LHS for the checked design-space campaign; keep test count small and document a larger
  decision campaign separately.
- [ ] Produce mission success, phase failure, reserve, limiting-margin, peak-load, and target-miss
  evidence.
- [ ] Document aleatory versus epistemic interpretation and prohibit certification language.
- [ ] Add a checked expected-summary fixture tolerant only to explicitly justified floating-point
  variation.

**Gate:**

```bash
astro validate-campaign examples/campaigns/leo_lifecycle_robustness.yaml
astro run-campaign examples/campaigns/leo_lifecycle_robustness.yaml \
  --output-dir /tmp/astro-lifecycle-robustness
astro summarize-campaign /tmp/astro-lifecycle-robustness
python -m pytest tests/astro_uq tests/astro_cli/test_campaign_cli.py -q
```

## Phase 4: Surrogate Lifecycle

### Task 16: Implement surrogate schemas and safe artifact format

**Files:**
- Create: `src/astro_surrogates/__init__.py`
- Create: `src/astro_surrogates/models.py`
- Create: `src/astro_surrogates/io.py`
- Modify: `pyproject.toml`
- Test: `tests/astro_surrogates/test_models.py`
- Test: `tests/astro_surrogates/test_io.py`

- [ ] Add `SimulationDatasetSpec`, `SimulationEpisodeRef`, `DatasetManifest`, `DatasetSplit`,
  `SurrogateDomain`, `SurrogateTrainingRun`, `SurrogateModelArtifact`,
  `SurrogateValidationReport`, and `SurrogatePromotionDecision`.
- [ ] Store arrays in NPZ with JSON manifests and SHA-256 digests; prohibit pickle and executable
  loaders.
- [ ] Record teacher/baseline evaluator identities, units, frames, time coordinates, source scenario
  digests, software versions, and environment metadata.
- [ ] Add wheel and strict MyPy inclusion.

**Gate:** Tampered arrays/manifests are rejected and no artifact loading path executes code.

### Task 17: Build leakage-resistant datasets

**Files:**
- Create: `src/astro_surrogates/datasets.py`
- Create: `src/astro_surrogates/features.py`
- Test: `tests/astro_surrogates/test_datasets.py`
- Test: `tests/astro_surrogates/test_features.py`

- [ ] Generate complete simulation episodes through `astro_uq` evaluators.
- [ ] Split by scenario/orbit/configuration/episode before timestep expansion.
- [ ] Provide train, validation, held-out, boundary challenge, and OOD challenge splits.
- [ ] Fit normalization on training data only and store transforms with unit/frame metadata.
- [ ] Detect duplicate source digests and cross-split episode leakage.

**Gate:** A fixture with adjacent timesteps from one episode cannot be split across train and test.

### Task 18: Implement mandatory baselines

**Files:**
- Create: `src/astro_surrogates/baselines.py`
- Create: `src/astro_surrogates/training.py`
- Test: `tests/astro_surrogates/test_baselines.py`
- Test: `tests/astro_surrogates/test_training.py`

- [ ] Implement zero-correction, mean-correction, ridge, and bounded polynomial residual models with
  NumPy/SciPy.
- [ ] Keep feature transforms and output reconstruction deterministic and artifact-addressed.
- [ ] Record training metrics, wall time, seed, hyperparameters, data digests, and numerical warnings.
- [ ] Require every challenger to compare against zero and linear baselines.

**Gate:** Training is reproducible within declared tolerance and the selected baseline wins on an
independent nonlinear fixture before domain data is attempted.

### Task 19: Implement multidimensional validation and promotion

**Files:**
- Create: `src/astro_surrogates/validation.py`
- Create: `src/astro_surrogates/registry.py`
- Test: `tests/astro_surrogates/test_validation.py`
- Test: `tests/astro_surrogates/test_registry.py`

- [ ] Evaluate one-step, rollout, terminal-state, event-time, mission-metric, requirement-classification,
  invariant, OOD, and end-to-end speed metrics.
- [ ] Define threshold sets in versioned validation specs, not hard-coded model code.
- [ ] Require failure-challenge recall and report false-positive/false-negative confusion matrices.
- [ ] Implement immutable local registration and a separate signed-by-record promotion decision.
- [ ] Support only `rejected`, `experimental`, and `campaign_screening` states in v1.
- [ ] Prevent a training command from writing promotion state.

**Gate:** A deliberately fast but inaccurate model and an accurate but OOD-unsafe model are both
rejected with machine-readable reasons.

### Task 20: Run the surrogate target decision and baseline campaign

**Files:**
- Create: `examples/surrogates/<selected-target>/dataset.yaml`
- Create: `examples/surrogates/<selected-target>/validation.yaml`
- Create: `docs/validation/surrogate-campaigns.md`
- Test: `tests/astro_surrogates/test_reference_surrogate.py`

- [ ] Use Task 1 benchmark evidence to select the orbit-residual candidate, the coupled-twin
  challenger, or kill learned-model work for this increment.
- [ ] Freeze a bounded domain and teacher claim before generating data.
- [ ] Train mandatory baselines and record all rejected candidates, not only the winner.
- [ ] Run held-out, boundary, OOD, invariant, and speed gates.
- [ ] Promote at most to `campaign_screening`; retain authoritative replay for claims.

**Gate:** Proceed only if the model beats the simple baseline, passes designated failure challenges,
and achieves at least the preregistered end-to-end speedup. Otherwise record a kill/pivot decision.

### Task 21: Spike optional JAX residual model

**Files:**
- Create: `docs/research/jax-surrogate-stack-spike.md`
- Create only after approval: `src/astro_surrogates/jax_models.py`
- Modify only after approval: `pyproject.toml`
- Test: `tests/astro_surrogates/test_jax_models.py`

- [ ] Compare pure JAX plus a minimal optimizer dependency against a heavier model framework.
- [ ] Check Python 3.12, current `research` extra, CPU-only installation, checkpoint safety, wheel
  impact, deterministic seeding, and maintenance cost.
- [ ] Implement a residual MLP only if the baseline dataset and promotion gate justify it.
- [ ] Train with one-step plus multi-step rollout loss and compare to the best mandatory baseline.
- [ ] Keep neural-operator work out of this task.

**Gate:** Do not merge a dependency or model that lacks a measured accuracy/speed advantage on the
selected domain.

### Task 22: Add surrogate evaluator and progressive fidelity

**Files:**
- Create: `src/astro_surrogates/runtime.py`
- Create: `src/astro_uq/progressive.py`
- Modify: `src/astro_uq/evaluators.py`
- Test: `tests/astro_uq/test_progressive.py`
- Test: `tests/astro_surrogates/test_runtime.py`

- [ ] Check model digest, promotion state, domain, feature schema, and OOD policy before inference.
- [ ] Route predicted failures, boundary cases, OOD cases, deterministic audit samples, and selected
  quantiles to authoritative replay.
- [ ] Record surrogate and authoritative results when replay occurs, plus disagreement metrics.
- [ ] Never use an unpromoted model unless the campaign explicitly opts into experimental evidence.
- [ ] Estimate missed-failure risk from replay; stop the campaign if a configured disagreement gate
  fails.

**Gate:** Injected OOD and false-negative cases deterministically fall back or stop; they cannot be
silently counted as surrogate successes.

### Task 23: Add surrogate CLI and documentation

**Files:**
- Create: `src/astro_surrogates/cli.py`
- Modify: `src/astro_cli/main.py`
- Create: `docs/surrogate-models.md`
- Modify: `docs/validation-matrix.md`
- Test: `tests/astro_cli/test_surrogate_cli.py`

- [ ] Add build-dataset, train, validate, inspect, and promote commands with explicit input/output
  paths and dry-run where applicable.
- [ ] Require a validation report digest for promotion.
- [ ] Print model domain, teacher, status, validation decision, and claim boundary in inspection.
- [ ] Document experimental versus screening use and authoritative fallback.

**Gate:** The full checked baseline lifecycle runs from dataset definition through rejected or
screening-promoted decision using only local artifacts.

## Phase 5: AI-Native Orchestration

### Task 24: Add typed campaign and surrogate plans

**Files:**
- Modify: `src/astro_assistant/models.py`
- Modify: `src/astro_assistant/registry.py`
- Modify: `src/astro_assistant/policy.py`
- Modify: `src/astro_assistant/validators.py`
- Test: `tests/astro_assistant/test_uq_plans.py`

- [ ] Add plan types for campaign validation/run/summary and surrogate dataset/train/validate/review.
- [ ] Generate command specs only from allow-listed scenario, campaign, dataset, and model ids.
- [ ] Classify training and promotion review separately; promotion remains a deterministic Astro
  command requiring explicit approval.
- [ ] Validate produced artifacts and digests before a later step consumes them.
- [ ] Preserve existing OD assistant behavior and trace schema compatibility.

**Gate:** Adversarial prompts cannot inject paths, tools, extractors, arbitrary flags, or promotion.

### Task 25: Add deterministic planners and golden prompts

**Files:**
- Modify: `src/astro_assistant/planner.py`
- Create: `examples/workflows/uncertainty/manifest.yaml`
- Create: `examples/workflows/uncertainty/golden_prompts.yaml`
- Test: `tests/astro_assistant/test_uq_workflow_pack.py`

- [ ] Add golden plans for lifecycle robustness, surrogate validation review, and campaign summary.
- [ ] Add unsupported-intent responses for operational probability, autonomous promotion, and
  unregistered parameters.
- [ ] Ensure dry-run reads no stale generated result artifacts.
- [ ] Report assumption and approval questions as structured plan fields.

**Gate:** Every supported golden prompt resolves to the intended checked definition and stable dry-run
trace; unsupported prompts fail closed with a useful reason.

### Task 26: Add active-learning recommendation artifacts

**Files:**
- Create: `src/astro_surrogates/active_learning.py`
- Modify: `src/astro_surrogates/models.py`
- Test: `tests/astro_surrogates/test_active_learning.py`

- [ ] Rank candidates by held-out error, model uncertainty when available, OOD proximity,
  requirement-boundary distance, and diversity.
- [ ] Emit a reviewable `AcquisitionRecommendation`, not an automatically executed dataset mutation.
- [ ] Deduplicate against existing dataset episode digests and cap each acquisition batch.
- [ ] Record why each candidate was selected and which evidence would change the branch decision.

**Gate:** Candidate selection is deterministic for a fixed artifact set and cannot promote or train a
model by itself.

## Phase 6: Advanced Methods Backlog

These are gated scopes, not implied work for the first release:

- [ ] Importance sampling only after weighted estimators and likelihood-ratio fixtures are reviewed.
- [ ] Global sensitivity using Sobol indices only after independent analytical fixtures pass.
- [ ] Robust optimization only after campaign metrics and feasibility classifications are stable.
- [ ] Bayesian experimental design only for a named information-gain decision.
- [ ] Multi-node thermal graph surrogate only after a higher-fidelity thermal teacher exists.
- [ ] Query-at-time operator model only when variable-time trajectory queries create a measured need.
- [ ] FNO/PINO only for field-valued dynamics where the grid/operator structure is real.
- [ ] Constellation surrogate only after member permutation/equivariance and fleet failure semantics
  are specified.

## Verification Matrix

Run after each phase:

```bash
python -m ruff check .
python -m mypy
python -m pytest -q
git diff --check
python -m pytest tests/test_packaging.py -q
python -m build
```

Additional required review before merging Phases 3-5:

- parse every generated JSON/JSONL artifact;
- rerun the reference command from a clean temporary output directory;
- compare serial and parallel campaign summaries;
- inspect failure denominator and retained-case policy manually;
- review claim language against `docs/current-state.md`, `docs/validation-matrix.md`, and source
  product boundaries;
- independently reproduce surrogate validation and promotion decisions;
- inspect wheel contents for `astro_uq` and `astro_surrogates`;
- run GitHub CI before merge.

Optional teacher campaigns are recorded separately in
`docs/validation/live-backend-campaigns.md` and do not become required local gates.

## Commit Sequence

Use small, reviewable commits in this order:

1. `docs: define uncertainty campaign benchmark`
2. `feat: add uncertainty campaign schemas`
3. `feat: add uncertainty distributions and samplers`
4. `feat: add campaign parameter registries and artifacts`
5. `feat: add campaign evaluation and statistics`
6. `feat: add resumable campaign runner`
7. `feat: add orbit uncertainty adapter`
8. `feat: add twin and reentry uncertainty adapters`
9. `feat: add lifecycle robustness campaign`
10. `docs: publish uncertainty campaign workflow`
11. `feat: add surrogate dataset and artifact contracts`
12. `feat: add surrogate baselines and validation`
13. `feat: add promoted surrogate campaign evaluator`
14. `feat: add assistant uncertainty workflow plans`
15. `feat: add active learning recommendations`

## Definition Of Done

The roadmap is implemented when:

- one uncertainty definition can drive orbit, twin, reentry, and lifecycle workflows without each
  module implementing its own sampler;
- random, LHS, Sobol, sweep, and ensemble campaigns are reproducible and statistically tested;
- campaigns resume safely, retain failures, and produce suite-owned evidence artifacts;
- the checked lifecycle campaign answers a concrete robustness question with explicit confidence
  and claim boundaries;
- a measured benchmark selects or kills the first surrogate target;
- any selected surrogate has immutable data/model provenance, independent validation, OOD fallback,
  failure-challenge evidence, and a separate promotion decision;
- surrogate-assisted results replay decision-critical cases authoritatively;
- the assistant can plan and trace allow-listed campaign/model workflows without computing physics
  or self-promoting models;
- focused, full, typing, lint, packaging, build, artifact, and CI gates pass;
- `docs/current-state.md` names the next real gate rather than carrying this plan as a hidden backlog.
