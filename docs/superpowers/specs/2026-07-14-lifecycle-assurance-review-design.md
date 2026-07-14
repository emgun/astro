# Lifecycle Assurance Review Design

## Decision

Add deterministic lifecycle evidence review and bounded anomaly triage over a verified
`MissionLifecycleResult`. Do not add a second lifecycle runner, provider-authored findings,
probabilistic diagnosis, autonomous remediation, or operational authority.

## Ownership

- `astro_mission` remains authoritative for lifecycle execution, continuity, margins, and embedded
  launch, orbit, twin, deorbit, and reentry products.
- `astro_assurance` re-verifies lifecycle evidence, derives typed findings and triage actions, and
  owns review artifact IO and public commands.
- `astro_assistant` compiles only a fixed verify-then-review plan. Deterministic verification remains
  the execution authority.

## Integrity Contract

V1 accepts a lifecycle result path and its lifecycle scenario path. It supports local launch and
reentry backends only. Verification loads the scenario and stored result, re-runs the lifecycle,
requires canonical product payload equality, checks that both inputs remain unchanged during
verification, and
binds their exact file digests plus the referenced launch, twin, and reentry scenario-file digests
in the review. Invalid, stale, substituted, or tampered evidence
produces no review artifact.
Fresh execution uses temporary staged copies of the captured referenced bytes, preventing a review
from binding one file version while executing another.

## Review Contract

The review records:

- result and scenario paths and byte digests;
- scenario and workflow identities;
- lifecycle pass state, continuity state, margin state, and phase manifest summary;
- deterministic findings with stable semantic ids, severity, category, evidence, implication, and
  required action;
- non-executing triage actions copied from unresolved blocker and warning findings;
- a review disposition and explicit non-operational claim boundary.

Findings cover integrity, each failed continuity check, each warning or failed margin, embedded
phase warnings, manifest completeness, and the lifecycle claim boundary. Passing evidence remains
visible through aggregate informational findings. No heterogeneous margin is ranked by raw
magnitude; the lifecycle runner's typed limiting margin and status remain authoritative.
The non-specific margin unit `native` is an evidence-quality warning until replaced by an explicit
physical unit.

## Disposition And Triage

- `design_review_ready`: verified evidence has no blocker or warning findings.
- `additional_review_required`: verified evidence contains a failed continuity check, warning or
  failed margin, incomplete phase manifest, or phase warning requiring engineering review.

Triage actions are deterministic decision support. They cannot change evidence, execute a command,
infer a root cause, assign probability, certify a subsystem, or promote a claim.

## Exit Gate

The slice exits when public verify and review commands reject tampering and path aliases, the
checked local lifecycle produces a byte-stable review, the assistant compiles a fixed
approval-gated plan, focused and release-scale gates pass, and independent review finds no open
high-severity issue.

The separately published lifecycle artifact bundle remains outside this integrity claim because its
manifest does not contain artifact digests. Bundle provenance requires a later digest-manifest
extension rather than an implied claim from re-execution.
