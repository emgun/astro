# Assurance Model-Form Matrix v1

## Objective

Add a separately versioned 2x2 truth/estimator force-model validation product while preserving the
reviewed paired-assurance v1 schema and behavior.

## Matrix

The required cells are two-body/two-body, two-body/J2, J2/two-body, and J2/J2. Every realization
uses the same coordinate and tracking-noise seed across all four cells. Seeds remain unique between
realizations.

V1 reports two primary same-truth contrasts:

- J2 estimator minus two-body estimator under two-body truth (over-modeling sensitivity).
- J2 estimator minus two-body estimator under J2 truth (under-modeling recovery).

A difference-in-differences interaction may be derived from those two contrasts. Counts and pass
dispositions remain per-profile and per-contrast; they are never pooled into model-success or
operational probabilities.

## Product Boundary

The matrix is a new workflow and does not alter `paired_mission_assurance_validation_v1`. It reuses
the same explicit realizations, calibration coverage, local assurance execution, source digests, and
embedded-case integrity checks. A matched J2 result demonstrates internal consistency under that
configured profile, not that J2 is physical truth or sufficient flight dynamics.

The matrix records `calibration_protocol_id` separately from its own `protocol_id`. This makes reuse
of the paired protocol's calibration envelope explicit without rebinding or duplicating evidence.

## Public Surface

- `astro validate-model-form-matrix PROTOCOL`
- `astro run-model-form-matrix PROTOCOL --output RESULT`
- `astro verify-model-form-matrix RESULT`

## Acceptance

The protocol requires the exact ordered four-cell matrix. The verifier rebinds protocol, assurance,
and calibration bytes; verifies every embedded case and force role; recomputes metrics, contrasts,
interaction, transitions, and summaries; and rejects incomplete contrasts or derived-field forgery.
The result preserves calibration and claim blockers and never infers preferred truth, causality,
operational risk, or probability. A separate deterministic review product is deferred until matrix
results create a concrete review decision that the verified result itself cannot represent.
