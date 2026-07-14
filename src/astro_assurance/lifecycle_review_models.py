from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from astro_assurance.review_models import AssuranceReviewEvidence, AssuranceReviewSeverity
from astro_core.models import AstroModel
from astro_mission.models import LifecycleStatus

LifecycleReviewInputRole = Literal[
    "launch_scenario", "twin_scenario", "reentry_scenario"
]


class LifecycleReviewDisposition(StrEnum):
    DESIGN_REVIEW_READY = "design_review_ready"
    ADDITIONAL_REVIEW_REQUIRED = "additional_review_required"


class LifecycleReviewCategory(StrEnum):
    INTEGRITY = "integrity"
    MANIFEST = "manifest"
    CONTINUITY = "continuity"
    MARGIN = "margin"
    EVIDENCE_BOUNDARY = "evidence_boundary"
    CLAIM_BOUNDARY = "claim_boundary"


class LifecycleReviewFinding(AstroModel):
    finding_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    severity: AssuranceReviewSeverity
    category: LifecycleReviewCategory
    statement: str = Field(min_length=1)
    evidence: tuple[AssuranceReviewEvidence, ...] = Field(min_length=1)
    implication: str = Field(min_length=1)
    required_action: str = Field(min_length=1)


class LifecycleTriageAction(AstroModel):
    action_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    priority: AssuranceReviewSeverity
    source_finding_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    action: str = Field(min_length=1)
    authority_boundary: Literal["triage_only_no_execution_or_claim_promotion"] = (
        "triage_only_no_execution_or_claim_promotion"
    )


class LifecycleReviewInputReference(AstroModel):
    role: LifecycleReviewInputRole
    path: str = Field(min_length=1)
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class MissionLifecycleReview(AstroModel):
    review_id: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    workflow: Literal["mission_lifecycle_review_v1"] = "mission_lifecycle_review_v1"
    result_path: str = Field(min_length=1)
    result_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    scenario_path: str = Field(min_length=1)
    scenario_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    referenced_inputs: tuple[LifecycleReviewInputReference, ...] = Field(
        min_length=3, max_length=3
    )
    scenario_id: str = Field(min_length=1)
    source_workflow: Literal["mission_lifecycle_v1"] = "mission_lifecycle_v1"
    integrity_verified: Literal[True] = True
    lifecycle_passed: bool
    continuity_all_passed: bool
    margin_status: LifecycleStatus
    phase_order: tuple[str, ...] = Field(min_length=1)
    findings: tuple[LifecycleReviewFinding, ...] = Field(min_length=1)
    triage_actions: tuple[LifecycleTriageAction, ...]
    disposition: LifecycleReviewDisposition
    claim_boundary: Literal[
        "deterministic_lifecycle_review_not_causal_probabilistic_or_operational_authority"
    ] = "deterministic_lifecycle_review_not_causal_probabilistic_or_operational_authority"

    @model_validator(mode="after")
    def findings_and_triage_must_be_consistent(self) -> MissionLifecycleReview:
        finding_ids = [finding.finding_id for finding in self.findings]
        if len(set(finding_ids)) != len(finding_ids):
            raise ValueError("lifecycle review finding ids must be unique")
        roles = [reference.role for reference in self.referenced_inputs]
        expected_roles = {"launch_scenario", "twin_scenario", "reentry_scenario"}
        if len(set(roles)) != len(roles) or set(roles) != expected_roles:
            raise ValueError("lifecycle review must bind each referenced scenario exactly once")
        unresolved = {
            finding.finding_id
            for finding in self.findings
            if finding.severity
            in {AssuranceReviewSeverity.BLOCKER, AssuranceReviewSeverity.WARNING}
        }
        action_sources = [action.source_finding_id for action in self.triage_actions]
        if len(set(action_sources)) != len(action_sources) or set(action_sources) != unresolved:
            raise ValueError("lifecycle triage must cover each unresolved finding exactly once")
        expected = (
            LifecycleReviewDisposition.ADDITIONAL_REVIEW_REQUIRED
            if unresolved
            else LifecycleReviewDisposition.DESIGN_REVIEW_READY
        )
        if self.disposition is not expected:
            raise ValueError("lifecycle review disposition must match unresolved findings")
        return self
