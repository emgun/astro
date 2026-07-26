# AI-Native Mission Operator Design

## Decision

Build a provider-neutral adaptive operator around Astro Suite's typed physics and evidence tools.
The operator chooses the next action from the evolving evidence state; a deterministic kernel
validates the action against the current authority grant, invokes the typed tool, and appends the
result to a replayable journal.

No component is declared the owner of physical truth. Scenarios are declarations, simulations are
model-conditioned evidence, measurements are observations with provenance and uncertainty, and
operator conclusions are inferences. A digest proves byte identity, not physical truth, freshness,
calibration, or decision authority.

The core rule is:

> Verified evidence constrains decisions; active, scoped, revocable grants determine authority.

## Why A Separate Package

`astro_assistant` remains the fixed local-OD workflow planner and executor. Its compile-once command
plans are useful, but stretching that contract into an adaptive operator would mix two different
products. `astro_operator` owns the repeated decide, authorize, invoke, observe, and reduce loop. It
calls existing Astro domain functions in-process and does not duplicate physics.

## Authority Model

Authority is progressive rather than categorically withheld:

1. `research`: evaluate candidates, request evidence, and conclude.
2. `decision_support`: also construct typed command proposals.
3. `supervised_autonomy`: execute scoped commands after per-action approval.
4. `delegated_autonomy`: execute allowed command types within a declared mission envelope.
5. `mission_autonomy`: plan and execute within a broad but still versioned and revocable mission
   grant.

The level is a legible maturity label. The enforceable schema `1.0` contract is the grant: mission
scope, allowed action kinds, allowed command types, approval requirements, action and evaluation
budgets, grant version, and revoked state. Every adaptive action is checked at call time and again
during persisted-run verification. Approvals bind a stable approval id to the canonical digest of
the exact action rather than trusting a reasoner-chosen id alone.

Legacy schemas `1.0` and `1.1` can propose commands but cannot commit them. Schema `1.2` implements
the higher-authority path for explicitly registered simulation-only tools: exact proposal binding,
private SQLite write-ahead preparation, persistent idempotency, single-use approval consumption,
authority and world-state revalidation, typed parameter/resource envelopes, and digest-bound
terminal records. A checked supervised-autonomy burn exercises this path end to end. Delegated and
mission-autonomy grants use the same controls and cannot bypass envelopes, qualification, freshness,
revocation, or idempotency. Real-effect tools remain unregistered and unqualified.

Future real-effect handlers must add expiry, remote-effect reconciliation, operational resource
models, and qualification evidence appropriate to their risk. They inherit rather than replace the
implemented idempotency, prepare/commit, freshness, revocation, and envelope controls.

## Provider Boundary

`MissionReasoner.decide(OperatorState) -> ReasonerDecision` is the only intelligence-provider
contract. A decision contains one typed `OperatorAction` plus a provider-neutral invocation record
with adapter, provider, model, request, usage, metadata, and canonical input/output digests. A
provider can use a hosted model, local model, learned policy, search procedure, or deterministic
replay. It cannot submit shell commands, arbitrary paths, direct state mutations, or its own
authority grant. The engine verifies the invocation digests before applying authority policy and
journals provenance separately from the action so provider metadata cannot change action identity
or approval binding.

Adapters normalize configuration/authentication, unavailable/rate-limit, invalid-response, and
cancellation failures into provider-neutral reasoner errors. Provider SDK response objects,
credentials, retry settings, and exception types do not cross into the kernel. Retries, when added,
remain adapter-owned and must retain attempt provenance.

The public `ConditionalReplayReasoner` is explicitly a deterministic branching harness. Its typed
conditions inspect current step count, evidence ids, and the last candidate observation. The
checked flow finishes if the lighter design passes and evaluates a higher-reserve recovery candidate
only after the typed failure observation. `ScriptedReasoner` remains a fixed unit-test fixture.
Neither is presented as an AI model; a real provider adapter can replace them without changing the
kernel, evaluator, evidence journal, or authority policy.

## State And Evidence

Each action has a stable id, concise rationale, and citations to evidence already present when the
decision is made. Each evidence reference records:

- a stable id and artifact kind;
- an epistemic kind: declared, observed, estimated, simulated, or inferred;
- a claim scope;
- an artifact path and SHA-256 digest.

The journal is append-only in meaning. Verification reconstructs the evidence inventory in event
order, rejects citations to future or unknown evidence, checks contiguous steps and unique ids, and
requires the final selection and conclusion to match the final action. Local artifacts are then
checked against their recorded digests.

Schema `1.2` adds a versioned evidence-tool registry and typed acquisition results. Assertions bind
subject, predicate, JSON value, epistemic kind, scope, source evidence, producer identity, optional
valid time, and canonical digest. The pure reducer retains every assertion and emits explicit
conflict sets for incompatible values; it never silently overwrites an assertion with a newer one.
Conclusion claims cite assertion IDs. A categorical supported claim fails closed when it cites an
unresolved conflict, while qualified or disputed claims must state their qualification.

Schema `1.3` adds deterministic claim predicates. Supported claims may bind exact scalar
comparisons, exact values, freshness relative to a cited digest-bound time assertion, and
applicability through exact subject, scope, predicate, and value matching. Live completion and
offline verification call the same pure evaluator; the reasoner cannot turn a failed predicate
into a supported claim by changing prose. Empty predicate fields are omitted from legacy action
digests so existing schema `1.2` journals retain their prior identities.

The provider projection exposes only allow-listed assertion and conflict fields. Artifact paths,
arbitrary evidence metadata, credentials, and provider internals remain outside the reasoner input.

## Simulation Command Commit

The first executable tool is `simulated_burn` version `1.0`. It has no external or physical side
effects. A command-capable CLI run requires a mission/grant-scoped `--command-store` reused across
publication attempts. The coordinator durably reserves an idempotency key, approval, and
grant-envelope capacity before dispatch; rechecks the current grant and exact world-state digest
before dispatch and before commit; and records committed, failed, or indeterminate terminal
outcomes. A prepared or
indeterminate execution is never automatically re-executed. Offline verification reconstructs the
proposal, execution, authority receipt, world-state receipt, record digests, and result digest
without invoking the tool.

## First Working Slice

The first slice is an adaptive mission lifecycle trade study:

- A checked objective declares a base lifecycle scenario, design-variable targets and bounds,
  metric goals, and a research authority grant.
- The reasoner proposes assignments using logical variable ids.
- The kernel rejects unknown, non-finite, duplicate, or out-of-envelope proposals before physics
  runs.
- `LifecycleCandidateEvaluator` translates allow-listed scalar variables into typed
  `MissionLifecycleInputOverrides` and calls the existing launch-to-reentry runner.
- Passing evaluations emit every lifecycle margin and digest-bound scenario/result artifacts.
- Domain evaluation failures become typed, digest-bound observations so the reasoner can adapt
  rather than losing the run history.
- Finish can select only a previously evaluated candidate.

The checked replay establishes the baseline, discovers that reducing wet mass violates the deorbit
propellant-reserve gate, evaluates a bounded higher-reserve candidate, and selects that passing
design. This is adaptive control flow over real suite physics, not an extra deterministic ranking
layer.

Public commands:

```bash
astro run-mission-operator examples/operator/leo_lifecycle_trade_study.yaml \
  --reasoner-replay examples/operator/leo_lifecycle_trade_study_replay.yaml \
  --output-dir /tmp/astro-mission-operator
astro verify-mission-operator /tmp/astro-mission-operator

astro run-mission-operator examples/operator/supervised_simulated_burn.yaml \
  --reasoner-replay examples/operator/supervised_simulated_burn_replay.yaml \
  --command-store /tmp/astro-supervised-command-ledger.sqlite3 \
  --output-dir /tmp/astro-supervised-simulated-burn
astro verify-mission-operator /tmp/astro-supervised-simulated-burn
```

Publication uses an invocation-owned unique partial sibling directory and renames it only after the
run journal is complete. Concurrent invocations cannot delete each other's partial output. The
output contains the operator journal plus per-candidate scenario, result, or failure artifacts.

## Exit Gate

The slice exits when:

- adaptive success and domain-failure paths are checked end to end;
- research, revoked, approval, envelope, and budget policies fail closed, while legacy command
  shapes remain non-executable and schema `1.2` permits only the checked simulation transaction;
- persisted journal structure and local evidence digests verify and tampering fails;
- the checked public example runs from the source tree and installed wheel;
- focused, full, lint, strict typing, packaging, and independent-review gates pass.

## Long-Term North Star: Self-Improving Mission Orchestrator

The mission operator is one mode of a broader AI-native mission engineering system. The long-term
product should orchestrate the full mission lifecycle: translate intent into requirements, explore
mission and spacecraft designs, commission analyses at appropriate fidelity, select and verify a
baseline, operate against that baseline, learn from outcomes, and safely replan when evidence
changes.

The target loop is:

```text
mission intent -> requirements and design variables -> adaptive analysis plan
      -> physics, estimation, test, and procedure tools -> evidence world state
      -> trade decision and versioned baseline -> verification and operations
      -> observed outcomes and residuals -> knowledge graph and learning datasets
      -> improved orchestration policies and qualified surrogates -> next mission decision
```

This is not a claim that one model should absorb every responsibility. The system assigns planning,
analysis, approval, and execution authority through explicit grants and tool contracts. Learned
components may own meaningful planning or control decisions when the applicable grant and
qualification evidence permit them. All components retain enough provenance, evaluation evidence,
and version identity for their outputs and effects to be inspected, reproduced where possible, and
rolled back.

### Mission Design Orchestration

A mission-design run begins from a typed `MissionIntent`: objectives, hard constraints,
preferences, budgets, initial assumptions, and allowed decisions. The orchestrator converts that
intent into a dependency-aware engineering plan instead of invoking every tool in a fixed sequence.

The design layer should add these provider-neutral contracts:

- `MissionIntent`: mission objectives, constraints, preferences, analysis budgets, and authority.
- `RequirementGraph`: requirements, parents, verification methods, margins, evidence status, and
  applicability.
- `CapabilityCatalog`: versioned tools with typed inputs and outputs, fidelity, cost, latency,
  applicability, qualification, and dependency metadata.
- `DesignCandidate`: an immutable assignment of architecture and design variables with lineage.
- `AnalysisPlan`: a dependency DAG with expected information gain, acceptance conditions, budgets,
  and stop rules.
- `DesignDecision`: a selected candidate, rejected alternatives, cited evidence, unresolved
  uncertainty, sensitivity, and rationale.
- `MissionBaseline`: a versioned configuration that assurance and operations can reference.
- `VerificationPlan`: required simulations, campaigns, tests, measurements, reviews, and promotion
  gates for the selected baseline.

The first integrated design director should compose existing Astro capabilities for launch and
insertion, orbit and trajectory design, coverage and communications, mass, power, thermal, ADCS,
propellant, disposal, uncertainty, assurance, and reentry. External engines remain typed
capabilities in the same catalog rather than special provider-specific paths.

The planner should use adaptive fidelity. It can screen many candidates with inexpensive local
models, eliminate infeasible regions, run uncertainty and sensitivity analysis on survivors, and
escalate only decision-relevant cases to higher-cost or independently implemented tools. It should
stop when additional analysis is unlikely to change the decision under the declared budgets and
acceptance criteria.

### Mission Knowledge Graph

The knowledge layer is an evidence graph, not an unqualified fact store. It connects:

```text
mission -> requirement -> design variable -> candidate -> analysis -> evidence
evidence -> assertion -> claim -> decision -> baseline -> observed outcome
tool/version -> produced evidence -> applicability domain -> qualification evidence
```

Nodes and edges retain source identity, version, valid time, applicability, uncertainty, and
digest-bound provenance. Conflicting assertions coexist until a deterministic rule or qualified
decision resolves their relevance. Documents, papers, procedures, and reports may be indexed with
embeddings for retrieval, but extracted claims enter the graph with their source and scope rather
than becoming global facts.

The graph should support decision-oriented questions such as:

- Which assumptions or requirements eliminated earlier designs?
- Which evidence and tool versions justified this baseline?
- Where did predicted and observed mission behavior diverge?
- Which models are qualified in the current design regime?
- Which unresolved uncertainty is most likely to change the selected design?
- Which prior mission episodes are applicable to the current objective?

The append-only operator journal remains the event source for a run. The graph is a versioned,
rebuildable read model across runs and missions; graph mutations do not replace the evidence bytes
or silently rewrite historical decisions.

### Learning Loops

The system should improve at three separate levels, with independent datasets and promotion gates:

1. **Knowledge learning** extracts reusable, scoped relationships from evidence, decisions, and
   outcomes while preserving provenance and contradiction.
2. **Orchestration-policy learning** improves candidate generation, tool selection, fidelity
   escalation, information-value ranking, and stop decisions from recorded mission episodes.
3. **Physics-surrogate learning** approximates expensive domain tools inside explicit applicability
   and uncertainty envelopes.

Learned planner or routing policies start as challengers in offline replay, then shadow checked
workflows before they can influence live decisions. Promotion depends on independently checked
decision outcomes, constraint violations, abstention behavior, cost, and robustness rather than
persuasive explanations alone. Each promoted policy is versioned and reversible.

Operational observations close the loop. For example, propulsion execution residuals, OD
residuals, atmospheric-density mismatch, power degradation, thermal discrepancies, and link-budget
errors can update scoped models and identify affected baseline claims. New evidence creates a new
state and a new decision; it does not retroactively change the evidence available to an older
decision.

### Neural Surrogate Lifecycle

A neural surrogate is a registered capability with more than a model file. Its contract binds:

- target quantities, units, input schema, and source solver or experiment;
- training dataset, selection policy, split, and digest;
- architecture, weights, code, runtime, and random-seed provenance;
- applicability domain and out-of-distribution detector;
- predictive uncertainty or calibration method;
- held-out and challenger evaluation artifacts;
- allowed decision roles and maturity status;
- expiry, supersession, rollback, and retraining triggers.

The maturity ladder is `experimental -> shadow -> screening -> decision_support ->
qualified_for_declared_use`. Promotion is specific to the exact surrogate version, dataset,
metrics, domain, and decision role. A model promoted for design-space screening is not thereby
qualified for final verification or operational control.

Active learning connects design orchestration to surrogate improvement:

1. Evaluate broad candidate regions with the current surrogate.
2. Identify high uncertainty, out-of-domain cases, and requirement or Pareto boundaries.
3. Select high-fidelity runs by expected decision information rather than uniform coverage alone.
4. Add verified results to a new immutable dataset version.
5. Train a challenger surrogate and evaluate it on locked held-out and stress campaigns.
6. Promote, retain in shadow, or reject the challenger under preregistered gates.
7. Resume the mission trade using only the role and domain actually earned.

This produces compounding value: mission studies generate targeted training evidence, while better
surrogates make future studies faster and enable larger uncertainty campaigns. Periodic independent
high-fidelity checks remain necessary to detect drift and blind spots.

### Unifying Flagship

The eventual flagship should demonstrate one continuous evidence chain:

> Design a feasible LEO mission, identify its limiting uncertainty, commission targeted
> high-fidelity simulations, improve and qualify a screening surrogate, select and verify a
> mission baseline, encounter a simulated post-launch deviation, update the evidence graph, and
> safely replan or abstain under the active authority grant.

The flagship should be scored on requirement satisfaction, evidence traceability, decision quality,
uncertainty reduction, tool cost, surrogate calibration and domain compliance, safe abstention,
recovery behavior, and exact replayability. Provider fluency is secondary to these outcomes.

## Architecture Status And Next Maturity Steps

The reference AI-native architecture is implemented through schema `1.4`: provider-neutral
reasoning, versioned evidence acquisition, assertion-preserving world state, deterministic checked
claims, progressive grants, typed baseline context, and durable simulation-only command
prepare/commit. Schemas `1.0` through `1.3` remain verifiable.

The Mission Design Director is implemented as an outer, digest-bound
bundle without changing the released operator journal schemas. It compiles a typed mission intent,
acyclic requirement graph, exact capability catalog, and bounded conditional analysis plan; reuses the adaptive
lifecycle evaluator across launch, orbit, digital-twin subsystems, deorbit, and reentry; reduces the
selected observation into exact-unit requirement assessments; and emits a versioned baseline only
when every hard requirement passes. Its exact-inventory verifier reconstructs the decision from the
verified operator journal without rerunning physics or invoking a provider.

Director schema `1.1` registers the existing lifecycle uncertainty campaign as a costed capability
that depends on the lifecycle screen. A typed rule declares the requirement, units, maximum
absolute margin within which new evidence could plausibly change the decision, and rationale. The
checked selected design has a `0.908437 km` entry-interface margin, so a declared `1.0 km` band
recommends the uncertainty campaign; narrowing the band to `0.5 km` defers it. The relevance score
is normalized distance to that declared boundary. It is deliberately not called expected
information value because the screening observation does not carry a calibrated decision-change
probability. A recommendation binds the exact candidate and eligible baseline. If any hard
requirement remains unresolved, the baseline is absent and the analysis is deterministically
deferred rather than exposed as an execution handoff. This slice recommends and reserves analysis;
it does not execute the campaign.

Public commands:

```bash
astro run-mission-design-director examples/design/leo_mission_design_director.yaml \
  --reasoner-replay examples/operator/leo_lifecycle_trade_study_replay.yaml \
  --output-dir /tmp/astro-mission-design-director
astro verify-mission-design-director /tmp/astro-mission-design-director

astro run-mission-design-conditional-campaign \
  /tmp/astro-mission-design-director \
  examples/design/leo_mission_design_conditional_campaign.yaml \
  --output-dir /tmp/astro-mission-design-conditional-campaign
astro verify-mission-design-conditional-campaign \
  /tmp/astro-mission-design-conditional-campaign

astro run-mission-operator examples/operator/post_launch_recovery_review.yaml \
  --reasoner-replay examples/operator/post_launch_recovery_review_replay.yaml \
  --mission-design-context /tmp/astro-mission-design-director \
  --output-dir /tmp/astro-post-launch-recovery-review
astro verify-mission-operator /tmp/astro-post-launch-recovery-review
```

The perception-to-decision slice now registers concrete, source-ID-addressed tools for simulated
telemetry, orbit-estimate snapshots, and declared procedures. The checked post-launch workflow
captures all three sources into the run, preserves their simulated, estimated, and declared
epistemic kinds, binds the estimate to the exact telemetry-source digest, and recomputes every
tool assertion from captured bytes during offline verification before checking uncertainty,
convergence, simulated-time freshness, procedure-validity, configuration, mode, and manual-review
gates. Its conclusion is explicitly as-of the captured simulated decision time and indicates
readiness for manual review, not maneuver authorization. Catalog entries cannot be absolute,
traverse their source root, or pass through symlinks. The tools consume typed snapshots in this
first slice; future live
telemetry and estimator adapters can produce the same contracts without changing the kernel.

The first mission knowledge graph slice is implemented as a versioned, rebuildable read model over
complete captured Director and operator bundles. Its exact-inventory verifier invokes each native
source verifier and reconstructs the graph before accepting the stored convenience artifact.
Typed nodes and edges retain the complete source records for requirements, candidates,
assessments, evidence, assertions, conflicts, claims, decisions, baselines, and tool versions. A
bounded public query traces the exact justification chain behind a baseline.

The baseline handoff and first conditional-planning reducer are now implemented. Operator schema
`1.4` carries the mission, Director run digest, baseline identity/version/digest, and operational
configuration. Graph schema `1.1` resolves that context against exactly one eligible Director
baseline and emits an explicit `operates_against` edge; it never infers the relationship from
similar names or prose. A provider-neutral query reduces the verified edge, run authority, target
claim, exact predicate outcomes, applicability, conflicts, and named manual-review gate into
`continue`, `hold`, or `abstain`. The implemented scope is manual-review readiness only:
`continue` advances an evidence package into review and does not approve or execute operations.

The first conditional campaign execution slice is also implemented. It copies and natively verifies
the complete Director source bundle, captures the exact selected candidate plus launch, twin,
orbit, and reentry inputs, executes the registered fixed-count lifecycle campaign with resumable
case evidence, and independently reconstructs its deterministic samples and statistics. The
checked campaign completes 8/8 successful cases; both hard-requirement configured-design-space
fractions are `1.0`, so the reducer returns `retain` for the bound baseline. `Retain` is limited to
the declared design space, `revise` requests a new Director decision without selecting a
replacement, and `abstain` covers incomplete or failed campaign evidence. The relocatable
exact-inventory verifier does not rerun physics or invoke a provider.

The remaining path expands the kernel into the full north star:

1. **Verification episodes in knowledge:** add the conditional campaign and its disposition as
   typed graph evidence, then route `revise` into a new bounded Director decision without rewriting
   the historical baseline.
2. **Knowledge and outcomes:** add verified operational outcome and residual contracts, then extend
   the graph and learning datasets without rewriting historical decisions.
3. **Adaptive verification:** add calibrated decision-change probabilities where evidence supports
   them, then choose uncertainty, sensitivity, independent-backend, and high-fidelity work by
   expected information value rather than boundary proximity alone.
4. **Surrogate lifecycle and active learning:** integrate the existing surrogate and campaign
   foundations with dataset lineage, applicability, uncertainty, shadow evaluation, promotion,
   rollback, and targeted high-fidelity acquisition.
5. **Closed-loop mission orchestration:** exercise simulated design-to-operations replanning,
   including conflict, staleness, crash recovery, concurrency, resource depletion, expiry, and
   revocation campaigns at progressively broader grants.
6. **Operational qualification:** define remote-effect reconciliation, qualification evidence, and
   operational resource models before registering any real-effect handler.
7. **Learned orchestration and provider comparison:** evaluate learned planners and providers only
   against the richer mission-design and closed-loop tasks, with transport availability separated
   from capability and cost.
