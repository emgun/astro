from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, FiniteFloat

from astro_assurance.validation_models import AssuranceCalibrationPromotionStatus
from astro_core.models import AstroModel


class AssuranceReviewSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    BLOCKER = "blocker"


class AssuranceReviewCategory(StrEnum):
    INTEGRITY = "integrity"
    CALIBRATION = "calibration"
    COMPLETENESS = "completeness"
    MODEL_FORM = "model_form"
    METRIC_SHIFT = "metric_shift"
    CLAIM_BOUNDARY = "claim_boundary"


class AssuranceReviewDisposition(StrEnum):
    DESIGN_REVIEW_READY = "design_review_ready"
    ADDITIONAL_EVIDENCE_REQUIRED = "additional_evidence_required"


class AssuranceReviewEvidence(AstroModel):
    key: str = Field(min_length=1)
    value: str = Field(min_length=1)


class AssuranceReviewFinding(AstroModel):
    finding_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    severity: AssuranceReviewSeverity
    category: AssuranceReviewCategory
    statement: str = Field(min_length=1)
    evidence: tuple[AssuranceReviewEvidence, ...] = Field(min_length=1)
    implication: str = Field(min_length=1)
    required_action: str = Field(min_length=1)


class AssuranceMetricShift(AstroModel):
    metric: str = Field(min_length=1)
    count: int = Field(gt=0)
    minimum: FiniteFloat
    median: FiniteFloat
    maximum: FiniteFloat


class AssuranceValidationReview(AstroModel):
    review_id: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    workflow: Literal["paired_assurance_deterministic_review_v1"] = (
        "paired_assurance_deterministic_review_v1"
    )
    source_path: str = Field(min_length=1)
    source_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_id: str = Field(min_length=1)
    calibration_id: str = Field(min_length=1)
    calibration_promotion_status: AssuranceCalibrationPromotionStatus
    integrity_verified: Literal[True] = True
    disposition: AssuranceReviewDisposition
    metric_shifts: tuple[AssuranceMetricShift, ...]
    findings: tuple[AssuranceReviewFinding, ...] = Field(min_length=1)
    source_claim_boundary: str = Field(min_length=1)
    claim_boundary: Literal[
        "deterministic_decision_support_not_autonomous_or_operational_authority"
    ] = "deterministic_decision_support_not_autonomous_or_operational_authority"


class AssuranceFindingChangeKind(StrEnum):
    ADDED = "added"
    RESOLVED = "resolved"
    SEVERITY_CHANGED = "severity_changed"
    CONTENT_CHANGED = "content_changed"


class AssuranceReviewTrend(StrEnum):
    UNCHANGED = "unchanged"
    IMPROVED = "improved"
    REGRESSED = "regressed"
    MIXED = "mixed"


class AssuranceFindingChange(AstroModel):
    finding_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    kind: AssuranceFindingChangeKind
    baseline_severity: AssuranceReviewSeverity | None = None
    candidate_severity: AssuranceReviewSeverity | None = None


class AssuranceMetricShiftChange(AstroModel):
    metric: str = Field(min_length=1)
    baseline_median: FiniteFloat | None = None
    candidate_median: FiniteFloat | None = None
    delta_candidate_minus_baseline: FiniteFloat | None = None


class AssuranceEvidenceRecommendation(AstroModel):
    recommendation_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    priority: AssuranceReviewSeverity
    source_finding_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    action: str = Field(min_length=1)
    authority_boundary: Literal["recommendation_only_no_execution_or_claim_promotion"] = (
        "recommendation_only_no_execution_or_claim_promotion"
    )


class AssuranceReviewComparison(AstroModel):
    comparison_id: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    workflow: Literal["paired_assurance_review_comparison_v1"] = (
        "paired_assurance_review_comparison_v1"
    )
    protocol_id: str = Field(min_length=1)
    baseline_review_path: str = Field(min_length=1)
    baseline_review_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline_source_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_review_path: str = Field(min_length=1)
    candidate_review_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_source_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline_risk_vector: tuple[int, int, int, int]
    candidate_risk_vector: tuple[int, int, int, int]
    trend: AssuranceReviewTrend
    finding_changes: tuple[AssuranceFindingChange, ...]
    metric_changes: tuple[AssuranceMetricShiftChange, ...]
    recommendations: tuple[AssuranceEvidenceRecommendation, ...]
    claim_boundary: Literal[
        "cross_run_decision_support_not_probability_causality_or_operational_authority"
    ] = "cross_run_decision_support_not_probability_causality_or_operational_authority"
