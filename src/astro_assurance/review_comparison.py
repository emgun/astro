from __future__ import annotations

import os
from hashlib import sha256
from pathlib import Path

from astro_assurance.review import verify_assurance_validation_review
from astro_assurance.review_models import (
    AssuranceEvidenceRecommendation,
    AssuranceFindingChange,
    AssuranceFindingChangeKind,
    AssuranceMetricShiftChange,
    AssuranceReviewComparison,
    AssuranceReviewSeverity,
    AssuranceReviewTrend,
    AssuranceValidationReview,
)
from astro_assurance.validation_models import AssuranceCalibrationPromotionStatus
from astro_core.errors import InvalidScenarioError

_CALIBRATION_RISK = {
    AssuranceCalibrationPromotionStatus.MISSION_CALIBRATED: 0,
    AssuranceCalibrationPromotionStatus.REFERENCE_INFORMED: 1,
    AssuranceCalibrationPromotionStatus.ILLUSTRATIVE: 2,
}


def compare_assurance_validation_reviews(
    baseline_path: Path | str,
    candidate_path: Path | str,
) -> AssuranceReviewComparison:
    baseline_file = Path(baseline_path).resolve()
    candidate_file = Path(candidate_path).resolve()
    if baseline_file == candidate_file or os.path.samefile(baseline_file, candidate_file):
        raise InvalidScenarioError("baseline and candidate review paths must differ")
    baseline_bytes = baseline_file.read_bytes()
    candidate_bytes = candidate_file.read_bytes()
    baseline = verify_assurance_validation_review(baseline_file)
    candidate = verify_assurance_validation_review(candidate_file)
    reviews_changed = (
        baseline_file.read_bytes() != baseline_bytes
        or candidate_file.read_bytes() != candidate_bytes
    )
    if reviews_changed:
        raise InvalidScenarioError("assurance review changed during verification")
    return derive_assurance_review_comparison(
        baseline,
        candidate,
        baseline_review_path=str(baseline_file),
        baseline_review_digest=sha256(baseline_bytes).hexdigest(),
        candidate_review_path=str(candidate_file),
        candidate_review_digest=sha256(candidate_bytes).hexdigest(),
    )


def verify_assurance_review_comparison(
    comparison_path: Path | str,
) -> AssuranceReviewComparison:
    from astro_assurance.review_io import load_assurance_review_comparison

    comparison = load_assurance_review_comparison(comparison_path)
    expected = compare_assurance_validation_reviews(
        comparison.baseline_review_path,
        comparison.candidate_review_path,
    )
    if comparison != expected:
        raise InvalidScenarioError(
            "assurance review comparison does not match its verified review evidence"
        )
    return comparison


def derive_assurance_review_comparison(
    baseline: AssuranceValidationReview,
    candidate: AssuranceValidationReview,
    *,
    baseline_review_path: str,
    baseline_review_digest: str,
    candidate_review_path: str,
    candidate_review_digest: str,
) -> AssuranceReviewComparison:
    if baseline.protocol_id != candidate.protocol_id:
        raise InvalidScenarioError("assurance reviews must use the same protocol id")
    finding_changes = _finding_changes(baseline, candidate)
    metric_changes = _metric_changes(baseline, candidate)
    baseline_risk = _risk_vector(baseline)
    candidate_risk = _risk_vector(candidate)
    if candidate_risk < baseline_risk:
        trend = AssuranceReviewTrend.IMPROVED
    elif candidate_risk > baseline_risk:
        trend = AssuranceReviewTrend.REGRESSED
    elif finding_changes or metric_changes:
        trend = AssuranceReviewTrend.MIXED
    else:
        trend = AssuranceReviewTrend.UNCHANGED
    recommendations = tuple(
        AssuranceEvidenceRecommendation(
            recommendation_id=f"address_{finding.finding_id}",
            priority=finding.severity,
            source_finding_id=finding.finding_id,
            action=finding.required_action,
        )
        for finding in candidate.findings
        if finding.severity in {AssuranceReviewSeverity.BLOCKER, AssuranceReviewSeverity.WARNING}
    )
    return AssuranceReviewComparison(
        comparison_id=f"{candidate.protocol_id}-comparison-v1",
        protocol_id=candidate.protocol_id,
        baseline_review_path=baseline_review_path,
        baseline_review_digest=baseline_review_digest,
        baseline_source_digest=baseline.source_digest,
        candidate_review_path=candidate_review_path,
        candidate_review_digest=candidate_review_digest,
        candidate_source_digest=candidate.source_digest,
        baseline_risk_vector=baseline_risk,
        candidate_risk_vector=candidate_risk,
        trend=trend,
        finding_changes=finding_changes,
        metric_changes=metric_changes,
        recommendations=recommendations,
    )


def _risk_vector(review: AssuranceValidationReview) -> tuple[int, int, int, int]:
    blockers = sum(
        finding.severity is AssuranceReviewSeverity.BLOCKER for finding in review.findings
    )
    warnings = sum(
        finding.severity is AssuranceReviewSeverity.WARNING for finding in review.findings
    )
    disposition_risk = int(review.disposition.value == "additional_evidence_required")
    return (
        disposition_risk,
        _CALIBRATION_RISK[review.calibration_promotion_status],
        blockers,
        warnings,
    )


def _finding_changes(
    baseline: AssuranceValidationReview,
    candidate: AssuranceValidationReview,
) -> tuple[AssuranceFindingChange, ...]:
    baseline_findings = {finding.finding_id: finding for finding in baseline.findings}
    candidate_findings = {finding.finding_id: finding for finding in candidate.findings}
    changes = []
    for finding_id in sorted(set(baseline_findings) | set(candidate_findings)):
        before = baseline_findings.get(finding_id)
        after = candidate_findings.get(finding_id)
        if before is None and after is not None:
            kind = AssuranceFindingChangeKind.ADDED
        elif before is not None and after is None:
            kind = AssuranceFindingChangeKind.RESOLVED
        elif before is not None and after is not None and before.severity is not after.severity:
            kind = AssuranceFindingChangeKind.SEVERITY_CHANGED
        elif before != after:
            kind = AssuranceFindingChangeKind.CONTENT_CHANGED
        else:
            continue
        changes.append(
            AssuranceFindingChange(
                finding_id=finding_id,
                kind=kind,
                baseline_severity=None if before is None else before.severity,
                candidate_severity=None if after is None else after.severity,
            )
        )
    return tuple(changes)


def _metric_changes(
    baseline: AssuranceValidationReview,
    candidate: AssuranceValidationReview,
) -> tuple[AssuranceMetricShiftChange, ...]:
    before = {shift.metric: shift for shift in baseline.metric_shifts}
    after = {shift.metric: shift for shift in candidate.metric_shifts}
    changes = []
    for metric in sorted(set(before) | set(after)):
        baseline_median = None if metric not in before else float(before[metric].median)
        candidate_median = None if metric not in after else float(after[metric].median)
        delta = (
            None
            if baseline_median is None or candidate_median is None
            else candidate_median - baseline_median
        )
        if baseline_median is not None and candidate_median is not None and delta == 0.0:
            continue
        changes.append(
            AssuranceMetricShiftChange(
                metric=metric,
                baseline_median=baseline_median,
                candidate_median=candidate_median,
                delta_candidate_minus_baseline=delta,
            )
        )
    return tuple(changes)
