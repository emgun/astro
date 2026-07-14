from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from astro_assurance.review_models import (
    AssuranceMetricShift,
    AssuranceReviewCategory,
    AssuranceReviewDisposition,
    AssuranceReviewEvidence,
    AssuranceReviewFinding,
    AssuranceReviewSeverity,
    AssuranceValidationReview,
)
from astro_assurance.validation_io import verify_paired_assurance_validation_result
from astro_assurance.validation_models import (
    AssuranceCalibrationPromotionStatus,
    PairedAssuranceValidationResult,
)

_DECISION_METRIC_PRIORITY = (
    "truth_recovery_position_error_km",
    "od_position_error_km",
    "normalized_residual_rms",
    "original_candidate_total_delta_v_margin_km_s",
    "propellant_reserve_kg",
)


def review_assurance_validation(result_path: Path | str) -> AssuranceValidationReview:
    path = Path(result_path).resolve()
    result = verify_paired_assurance_validation_result(path)
    source_digest = sha256(path.read_bytes()).hexdigest()
    return derive_assurance_validation_review(result, str(path), source_digest)


def derive_assurance_validation_review(
    result: PairedAssuranceValidationResult,
    source_path: str,
    source_digest: str,
) -> AssuranceValidationReview:
    summary = result.summary
    metric_shifts = tuple(
        AssuranceMetricShift(
            metric=metric,
            count=value.count,
            minimum=value.minimum,
            median=value.median,
            maximum=value.maximum,
        )
        for metric, value in sorted(summary.paired_metric_deltas.items())
    )
    findings = [
        _finding(
            "integrity_verified",
            AssuranceReviewSeverity.INFO,
            AssuranceReviewCategory.INTEGRITY,
            (
                "The paired assurance artifact and its bound sources passed deterministic "
                "verification."
            ),
            ("source_digest", source_digest),
            "The review may use the verified embedded evidence.",
            "Preserve the source artifact and digest with any downstream review.",
        )
    ]
    calibration_blocked = (
        result.calibration_promotion_status
        is not AssuranceCalibrationPromotionStatus.MISSION_CALIBRATED
    )
    findings.append(
        _finding(
            "calibration_authority",
            (
                AssuranceReviewSeverity.BLOCKER
                if calibration_blocked
                else AssuranceReviewSeverity.INFO
            ),
            AssuranceReviewCategory.CALIBRATION,
            f"Calibration promotion status is {result.calibration_promotion_status.value}.",
            ("calibration_id", result.calibration_id),
            (
                "Operational or probabilistic claim promotion is blocked."
                if calibration_blocked
                else "The declared parameter envelopes meet the mission-calibrated authority gate."
            ),
            (
                "Acquire the mission-specific evidence required by the calibration manifest."
                if calibration_blocked
                else "Retain calibration provenance in subsequent reviews."
            ),
        )
    )
    incomplete = summary.paired_complete != summary.requested_pairs
    findings.append(
        _finding(
            "pair_completeness",
            AssuranceReviewSeverity.BLOCKER if incomplete else AssuranceReviewSeverity.INFO,
            AssuranceReviewCategory.COMPLETENESS,
            f"Paired profiles completed {summary.paired_complete}/{summary.requested_pairs} cases.",
            ("paired_complete", f"{summary.paired_complete}/{summary.requested_pairs}"),
            (
                "Incomplete pairs prevent a complete matched comparison."
                if incomplete
                else "All requested paired comparisons are available."
            ),
            (
                "Resolve execution failures before comparison."
                if incomplete
                else "No completeness action is required."
            ),
        )
    )
    reversals = summary.pass_regressions + summary.pass_improvements
    findings.append(
        _finding(
            "model_form_disposition",
            AssuranceReviewSeverity.WARNING if reversals else AssuranceReviewSeverity.INFO,
            AssuranceReviewCategory.MODEL_FORM,
            (
                f"Paired profiles contain {summary.pass_regressions} regressions and "
                f"{summary.pass_improvements} improvements."
            ),
            ("pass_reversals", str(reversals)),
            (
                "Model-form choice changes at least one paired pass disposition."
                if reversals
                else "Model-form choice did not change paired pass disposition."
            ),
            (
                "Review the signed metric shifts and force-model applicability; do not pool "
                "profile counts."
                if reversals
                else "Retain separate profile reporting."
            ),
        )
    )
    shifts_by_metric = {shift.metric: shift for shift in metric_shifts}
    selected_shifts = [
        shifts_by_metric[metric]
        for metric in _DECISION_METRIC_PRIORITY
        if metric in shifts_by_metric
    ]
    for index, shift in enumerate(selected_shifts, start=1):
        findings.append(
            _finding(
                f"metric_shift_{index:02d}",
                AssuranceReviewSeverity.INFO,
                AssuranceReviewCategory.METRIC_SHIFT,
                f"{shift.metric} has a median mismatched-minus-matched shift of {shift.median}.",
                ("metric", shift.metric),
                "This is a paired design-space observation, not a causal or probabilistic effect.",
                "Use the signed range and source cases when assessing engineering significance.",
            )
        )
    findings.append(
        _finding(
            "claim_boundary",
            AssuranceReviewSeverity.WARNING,
            AssuranceReviewCategory.CLAIM_BOUNDARY,
            "The source remains simulation design-space evidence without operational authority.",
            ("source_claim_boundary", result.claim_boundary),
            "The review cannot authorize navigation, probability, or flight-command claims.",
            "Carry this boundary into every human or model-generated explanation.",
        )
    )
    disposition = (
        AssuranceReviewDisposition.ADDITIONAL_EVIDENCE_REQUIRED
        if calibration_blocked or incomplete
        else AssuranceReviewDisposition.DESIGN_REVIEW_READY
    )
    return AssuranceValidationReview(
        review_id=f"{result.protocol_id}-review-v1",
        source_path=source_path,
        source_digest=source_digest,
        protocol_id=result.protocol_id,
        calibration_id=result.calibration_id,
        calibration_promotion_status=result.calibration_promotion_status,
        disposition=disposition,
        metric_shifts=metric_shifts,
        findings=tuple(findings),
        source_claim_boundary=result.claim_boundary,
    )


def _finding(
    finding_id: str,
    severity: AssuranceReviewSeverity,
    category: AssuranceReviewCategory,
    statement: str,
    evidence: tuple[str, str],
    implication: str,
    required_action: str,
) -> AssuranceReviewFinding:
    return AssuranceReviewFinding(
        finding_id=finding_id,
        severity=severity,
        category=category,
        statement=statement,
        evidence=(AssuranceReviewEvidence(key=evidence[0], value=evidence[1]),),
        implication=implication,
        required_action=required_action,
    )
