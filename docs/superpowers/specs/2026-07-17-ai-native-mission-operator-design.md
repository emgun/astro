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

## Architecture Status And Next Maturity Steps

The reference AI-native architecture is implemented through schema `1.2`: provider-neutral
reasoning, versioned evidence acquisition, assertion-preserving world state, exact-conclusion-bound
assertion claims, progressive grants, and durable simulation-only command prepare/commit.

The remaining work is qualification and breadth, not another missing kernel layer:

1. Add concrete local evidence tools for mission telemetry, estimation, and procedure sources, then
   exercise conflicting and stale evidence through public workflows.
2. Expand checked claim kinds beyond conflict-aware citation to deterministic threshold, temporal,
   and applicability predicates.
3. Exercise delegated and mission-autonomy grants only in simulation, with longer crash/recovery,
   concurrency, resource-depletion, expiry, and revocation campaigns.
4. Define reconciliation and qualification evidence for a real-effect handler before registering
   any such tool. External or flight-adjacent execution remains out of scope until that separate
   gate passes.
5. Resume provider comparison only against these richer tasks; do not optimize models against the
   earlier lifecycle-only benchmark.
