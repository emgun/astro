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
