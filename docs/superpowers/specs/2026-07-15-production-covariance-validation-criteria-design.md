# Production Covariance Validation Criteria v1

## Objective

Define and execute a suite-owned, fail-closed assessment for covariance validation evidence without
promoting local covariance wiring, optional-backend smoke runs, or backend agreement into production
certification.

## Evidence Classes

The assessment keeps two evidence classes separate:

1. **Propagation comparison:** epoch-aligned candidate and reference covariance histories are checked
   for symmetry, positive definiteness, trace ratio, relative Frobenius covariance error, and
   accumulated state-transition error. Each comparison declares whether the implementations are
   independent and which high-fidelity force features are exercised.
2. **Empirical consistency:** raw six-dimensional state errors and predicted 6x6 covariance matrices
   are used to recompute normalized estimation error squared (NEES). The gate uses a preregistered
   confidence level, a chi-square interval for the campaign mean, and a minimum individual-sample
   coverage fraction. Summaries without raw samples are not accepted.

## Promotion Contract

`production_validation_criteria_satisfied` requires all of the following:

- every configured comparison passes all numerical gates;
- the candidate/reference comparison has a separately typed and digest-bound independence review
  naming the reviewer, implementations, evidence reviewed, conclusion, and limitations;
- every required force feature is covered by a passing independent comparison;
- the bound empirical campaign declares independent realizations, contains no duplicate
  observations, meets the configured sample count, and passes its mean and
  coverage gates;
- protocol and every evidence file remain digest-bound and unchanged through execution.

Any failed numerical gate yields `criteria_failed`. Missing independence, force coverage, sample
count, or empirical evidence yields `additional_evidence_required`. Passing criteria means only that
the preregistered validation criteria were satisfied for the bound evidence. It is not regulatory,
mission, navigation, or flight certification.

## Public Surface

- `astro validate-covariance-validation PROTOCOL`
- `astro assess-covariance-validation PROTOCOL --output RESULT`
- `astro verify-covariance-validation RESULT`

The verifier rebinds every input digest and requires exact deterministic local reassessment.

## Checked Reference

The repository fixture compares byte-identical local two-body finite-difference covariance
histories. It exercises the assessment mechanics but intentionally lacks bound trajectory semantics,
independent high-fidelity
force coverage and empirical consistency evidence, so its expected disposition is
`additional_evidence_required`.
