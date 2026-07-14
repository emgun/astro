# Lifecycle Assurance Review Implementation Plan

**Goal:** Add verified deterministic review and bounded anomaly triage over mission lifecycle
evidence without granting causal, probabilistic, autonomous, or operational authority.

**Design:** `docs/superpowers/specs/2026-07-14-lifecycle-assurance-review-design.md`

## Phase 1: Contracts And Verification

- Add lifecycle review enums and models under `astro_assurance`.
- Re-run local lifecycle evidence and require exact stored-result equality.
- Bind exact scenario and result bytes and reject mid-verification changes.
- Bind and guard the referenced launch, twin, and reentry scenario bytes.
- Fail closed for optional launch or reentry backends.

## Phase 2: Findings And Triage

- Derive stable findings for integrity, continuity, margins, phase warnings, manifest completeness,
  and claim boundary.
- Preserve typed status and units without scalarizing unlike margins.
- Copy unresolved blocker and warning actions into non-executing triage records.

## Phase 3: Public And Assistant Surfaces

- Add `astro verify-mission-lifecycle-result RESULT SCENARIO`.
- Add `astro review-mission-lifecycle RESULT SCENARIO --output REVIEW` with optional summary.
- Publish atomically and reject path or file-identity aliases.
- Add a fixed verify-then-review assistant plan with approval required for writes.

## Phase 4: Verification And Integration

- Add tampering, stale source, optional backend, alias, derivation, CLI, assistant, and atomic IO
  tests.
- Run the checked public lifecycle and review from outside the repository.
- Run focused tests, Ruff, strict MyPy, package/import tests, full tests, and builds.
- Obtain independent read-only review, resolve findings, then merge through GitHub CI.
