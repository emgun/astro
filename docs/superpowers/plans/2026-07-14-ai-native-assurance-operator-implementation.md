# AI-Native Assurance Operator Implementation Plan

**Goal:** Add deterministic evidence review and typed assistant orchestration over paired
mission-assurance validation without granting autonomous or operational authority.

**Design:** `docs/superpowers/specs/2026-07-14-ai-native-assurance-operator-design.md`

## Phase 1: Review Product

- Add review enums and models under `astro_assurance`.
- Derive deterministic findings from a verified `PairedAssuranceValidationResult`.
- Preserve every signed metric shift and select findings from an explicit decision-metric priority;
  never rank unlike units by raw magnitude.
- Bind the source file path and byte digest.
- Add atomic JSON write/load and concise text formatting.

Gate: model, derivation, byte-stability, and malformed-input tests pass.

## Phase 2: Public Command

- Add `astro review-assurance-validation RESULT --output REVIEW [--summary-output SUMMARY]`.
- Verify the source before review derivation.
- Refuse source/output and output/summary path collisions.
- Publish no output after verification or derivation failure.

Gate: checked public result produces a verified review from outside the repository; tampering and
path-collision fixtures fail without partial output.

## Phase 3: Typed Assistant Plan

- Add verify/review tools and artifact kinds to `astro_assistant`.
- Add strict input contracts and fixed command builders.
- Require approval for the artifact-writing review step.
- Add deterministic verification for step order, source continuity, and output continuity.

Gate: path/flag injection fails closed and existing OD/campaign plans remain unchanged.

## Phase 4: Documentation And Integration

- Document findings, dispositions, claim boundaries, and provider separation.
- Run focused tests, public artifact generation, full tests, Ruff, strict MyPy, packaging, and
  builds.
- Obtain independent review when subagent quota is available; otherwise use local adversarial
  review plus GitHub CI and record the limitation.

## Deferred

- Provider-backed explanation and finding prioritization.
- Cross-campaign trend and anomaly review.
- Mission-assurance case and lifecycle review beyond paired validation.
- Active evidence-acquisition recommendations tied to real station or propulsion residuals.
- Any autonomous execution, model promotion, or operational command path.
