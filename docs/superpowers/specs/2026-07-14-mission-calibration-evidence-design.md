# Mission Calibration Evidence Pack v1

## Objective

Add suite-owned, digest-bound evidence contracts for station residuals, propulsion execution
residuals, and insertion covariance. The contracts make mission-calibration prerequisites
machine-checkable without treating illustrative or synthetic evidence as mission authority.

## Product Boundary

The existing assurance calibration manifest remains the authority-bearing product. It gains an
optional discriminated `evidence_products` collection. Existing illustrative manifests remain
valid. Mission-test- or flight-calibrated bounds fail closed unless they cite applicable typed
evidence and an explicit derivation convention.

The implementation does not import provider data, transform covariance frames, mutate calibration
bounds, promote a manifest automatically, estimate operational probabilities, or certify a mission.

## Evidence Contracts

- Station residual evidence binds station, observable, unit, band/mode/integration context, arc,
  selection and outlier policy, sample count, bias, sample standard deviation, and RMS.
- Propulsion execution evidence binds command and achieved epochs and delta-v vectors in one frame,
  timing residual, magnitude scale, two convention-specific pointing residuals, reconstruction
  method, and propulsion class.
- Insertion covariance evidence binds epoch, central body, frame, time scale, fixed Cartesian state
  ordering and units, population and confidence semantics, and a symmetric positive-semidefinite
  6x6 covariance. Correlations are derived, never independently authored.

## Promotion Rules

Mission-test and flight authority require evidence whose source kind matches the claimed authority.
Tracking bounds may cite only matching station-observable evidence; maneuver execution bounds may
cite only propulsion evidence; Cartesian insertion bounds may cite only insertion covariance.
Every promoted bound declares a derivation and must exactly equal its deterministic evidence-derived
envelope: station mean or standard-deviation extrema, propulsion execution-residual extrema, or a
symmetric covariance sigma envelope. Protocol validation also binds assurance/tracking scenario,
station, epoch, frame, body, and launcher context. Engineering acceptance still requires review.

## Public Surface

`astro inspect-assurance-calibration CALIBRATION [--protocol PROTOCOL]` validates the manifest and
reports evidence counts, authority coverage, unresolved promotion blockers, protocol completeness,
and the unchanged non-operational claim boundary. Without a protocol, promotion eligibility remains
explicitly unchecked.

## Acceptance

- All three evidence schemas reject ambiguous units, timing, frames, covariance shape/symmetry/PSD,
  source authority, and duplicate identifiers.
- Existing illustrative calibration remains valid.
- A checked synthetic evidence pack remains illustrative and reports blockers rather than promotion.
- Focused and full release gates, public CLI execution, artifact parsing, and independent review pass.
