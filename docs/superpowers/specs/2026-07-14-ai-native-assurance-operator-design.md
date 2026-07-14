# AI-Native Assurance Operator Design

## Decision

Build the operator as a typed decision-support layer over verified Astro Suite evidence. Do not add
a second physics orchestrator, an autonomous maneuver agent, or an LLM-authored pass/fail path.

The first product is a deterministic paired-assurance review. It verifies the source artifact,
extracts outcome reversals, calibration limits, claim boundaries, and decision-relevant metric
shifts, and writes a suite-owned review artifact. An optional model may later explain or prioritize
that artifact, but the deterministic review remains the authority.

## Ownership

- `astro_assurance` owns review models, deterministic derivation, source verification, artifact IO,
  and the public review command.
- `astro_assistant` owns allow-listed plans and policy for verify/review workflows.
- External models may produce commentary linked to review finding ids. They may not create, delete,
  downgrade, or change deterministic findings.
- Existing physics, OD, UQ, calibration, and paired-validation modules remain authoritative for
  their own products.

## Review Contract

The v1 review records:

- reviewed result path and byte digest;
- protocol and calibration identity;
- integrity-verification status;
- matched and mismatched completion/pass counts and pass reversals;
- all paired metric shifts plus a fixed, unit-aware decision-metric selection using existing
  aggregate evidence;
- calibration promotion status and claim boundary;
- typed findings with severity, category, evidence, implication, and required next action;
- a deterministic disposition and an explicit non-operational claim boundary.

The command fails without writing output when source verification fails. Review output is atomic and
cannot share a path with the source artifact.

## Deterministic Findings

V1 emits findings for:

1. Calibration status below `mission_calibrated`, which blocks claim promotion.
2. Pass regressions or improvements between paired profiles, which require model-form review.
3. Incomplete or execution-failed profile pairs, which block complete comparison.
4. A fixed set of decision-relevant signed metric shifts, reported without ranking unlike units and
   without causal explanations.
5. Unpooled-count and flight-authority boundaries, which remain mandatory in every review.

No frequency is labeled a probability. No source reference is promoted beyond its manifest
authority. No recommendation is a command.

## Dispositions

- `design_review_ready`: verified evidence is complete enough for bounded engineering review.
- `additional_evidence_required`: integrity passes, but calibration or completeness blocks the
  requested evidence promotion.

Invalid or tampered inputs produce no review artifact rather than an `invalid` disposition.

## AI-Native Extension

A later provider-backed explainer may consume only the typed review plus explicitly selected source
artifacts. Its output must record provider/model identity, prompt-template digest, input digests,
finding-id coverage, unsupported-claim diagnostics, and deterministic post-validation. Provider use
is optional and never part of the required release gate.

## Exit Gate

The first slice exits when a public CLI command can verify and review the checked paired-assurance
result, the output is byte-stable for identical input bytes, tampering prevents publication, the
assistant registry can compile an allow-listed verify/review plan, and focused/full release gates
pass.
