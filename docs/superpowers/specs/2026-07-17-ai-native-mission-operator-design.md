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

Schema `1.0` can propose commands and represent higher authority levels, but the kernel refuses
command commit. Supervised, delegated, and mission autonomy are staged architecture until a command
tool adds write-ahead prepare/commit events, persistent idempotency, approval consumption,
fresh-state revalidation, and typed parameter/resource envelopes. This is the current maturity
gate, not a permanent prohibition on AI command authority.

Future operational handlers must add expiry, resource envelopes, idempotency keys, prepare/commit,
fresh-state revalidation, and qualification evidence appropriate to their risk. Those are command
tool requirements, not reasons to prohibit command-capable architecture.

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
```

Publication uses an invocation-owned unique partial sibling directory and renames it only after the
run journal is complete. Concurrent invocations cannot delete each other's partial output. The
output contains the operator journal plus per-candidate scenario, result, or failure artifacts.

## Exit Gate

The slice exits when:

- adaptive success and domain-failure paths are checked end to end;
- research, revoked, approval, envelope, and budget policies fail closed, while higher command
  grants remain valid but command commit is explicitly rejected by schema `1.0`;
- persisted journal structure and local evidence digests verify and tampering fails;
- the checked public example runs from the source tree and installed wheel;
- focused, full, lint, strict typing, packaging, and independent-review gates pass.

## Next Architecture Steps

1. Add the first real provider adapter behind `MissionReasoner`, including prompt-template, schema,
   tool, raw-response, and attempt provenance; do not retain hidden chain-of-thought.
2. Add an evidence-acquisition tool registry and deterministic world-state reducer that preserves
   conflicting assertions rather than overwriting them.
3. Add conclusion claims that cite typed assertions, not only artifact ids.
4. Add prepare/commit command tools and exercise supervised execution in simulation before moving
   through delegated or mission autonomy.
