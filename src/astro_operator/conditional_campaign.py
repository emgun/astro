"""Typed reduction of conditional mission-design campaign evidence."""

from __future__ import annotations

import json
from enum import StrEnum
from hashlib import sha256
from typing import Any, Literal

from pydantic import Field, FiniteFloat, StrictInt, field_validator, model_validator

from astro_core.models import AstroModel
from astro_operator.director import (
    ConditionalAnalysisDecision,
    ConditionalAnalysisDisposition,
    MissionDesignRun,
)
from astro_uq.models import (
    CampaignDefinition,
    CampaignState,
    CampaignStatistics,
    CaseObservation,
    FixedCountStopping,
    MetricValueKind,
    OutcomeStatus,
    ParameterRealization,
    RequirementOperator,
)


def canonical_digest(value: object) -> str:
    payload = value.model_dump(mode="json") if isinstance(value, AstroModel) else value
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class ConditionalCampaignDisposition(StrEnum):
    RETAIN = "retain"
    REVISE = "revise"
    ABSTAIN = "abstain"


class ConditionalCampaignReason(StrEnum):
    ALL_DECLARED_DESIGN_SPACE_GATES_PASSED = "all_declared_design_space_gates_passed"
    DECLARED_DESIGN_SPACE_GATE_FAILED = "declared_design_space_gate_failed"
    CAMPAIGN_EXECUTION_INCOMPLETE = "campaign_execution_incomplete"
    CAMPAIGN_CASE_FAILURE = "campaign_case_failure"


class CampaignAcceptanceGate(AstroModel):
    director_requirement_id: str = Field(min_length=1)
    campaign_requirement_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    campaign_metric_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    minimum_pass_fraction: FiniteFloat = Field(ge=0.0, le=1.0)

    @field_validator("minimum_pass_fraction", mode="before")
    @classmethod
    def fraction_must_be_numeric(cls, value: Any) -> Any:
        if isinstance(value, bool | str):
            raise ValueError("minimum pass fraction must be numeric")
        return value


class ConditionalCampaignExecutionSpec(AstroModel):
    schema_version: Literal["1.0"] = "1.0"
    workflow: Literal["conditional_mission_design_campaign_v1"] = (
        "conditional_mission_design_campaign_v1"
    )
    execution_id: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    rule_id: str = Field(min_length=1)
    capability_id: Literal["astro.mission_lifecycle_uncertainty"] = (
        "astro.mission_lifecycle_uncertainty"
    )
    campaign_template_path: str = Field(min_length=1)
    acceptance_gates: tuple[CampaignAcceptanceGate, ...] = Field(min_length=1)
    claim_boundary: Literal[
        "configured_design_space_verification_not_operational_probability_qualification_or_authority"
    ] = (
        "configured_design_space_verification_not_operational_probability_"
        "qualification_or_authority"
    )

    @model_validator(mode="after")
    def gates_must_be_unique(self) -> ConditionalCampaignExecutionSpec:
        director_ids = [gate.director_requirement_id for gate in self.acceptance_gates]
        campaign_ids = [gate.campaign_requirement_id for gate in self.acceptance_gates]
        metric_ids = [gate.campaign_metric_id for gate in self.acceptance_gates]
        if len(set(director_ids)) != len(director_ids):
            raise ValueError("Director acceptance-gate requirement IDs must be unique")
        if len(set(campaign_ids)) != len(campaign_ids):
            raise ValueError("campaign acceptance-gate requirement IDs must be unique")
        if len(set(metric_ids)) != len(metric_ids):
            raise ValueError("campaign acceptance-gate metric IDs must be unique")
        return self


class ConditionalCampaignBinding(AstroModel):
    schema_version: Literal["1.0"] = "1.0"
    execution_id: str = Field(min_length=1)
    execution_spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    director_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    director_run_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline_id: str = Field(min_length=1)
    baseline_version: StrictInt = Field(ge=1)
    baseline_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_id: str = Field(min_length=1)
    rule_id: str = Field(min_length=1)
    conditional_decision_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    capability_id: Literal["astro.mission_lifecycle_uncertainty"]
    campaign_template_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_candidate_scenario_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    campaign_definition_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def binding_digest_must_match(self) -> ConditionalCampaignBinding:
        payload = self.model_dump(mode="json", exclude={"binding_sha256"})
        if self.binding_sha256 != canonical_digest(payload):
            raise ValueError("conditional campaign binding digest mismatch")
        return self


class CampaignGateAssessment(AstroModel):
    director_requirement_id: str = Field(min_length=1)
    campaign_requirement_id: str = Field(min_length=1)
    campaign_metric_id: str = Field(min_length=1)
    passed_samples: StrictInt = Field(ge=0)
    completed_samples: StrictInt = Field(ge=0)
    observed_pass_fraction: FiniteFloat = Field(ge=0.0, le=1.0)
    minimum_pass_fraction: FiniteFloat = Field(ge=0.0, le=1.0)
    passed: bool
    assessment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ConditionalCampaignOutcome(AstroModel):
    schema_version: Literal["1.0"] = "1.0"
    execution_id: str = Field(min_length=1)
    binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    director_run_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline_id: str = Field(min_length=1)
    baseline_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_id: str = Field(min_length=1)
    conditional_decision_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    campaign_definition_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    campaign_samples_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    campaign_cases_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    campaign_statistics_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    campaign_state: CampaignState
    requested_samples: StrictInt = Field(gt=0)
    completed_samples: StrictInt = Field(ge=0)
    outcome_counts: dict[str, StrictInt]
    gate_assessments: tuple[CampaignGateAssessment, ...] = Field(min_length=1)
    disposition: ConditionalCampaignDisposition
    reason: ConditionalCampaignReason
    claim_boundary: str = Field(min_length=1)
    outcome_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def outcome_digest_must_match(self) -> ConditionalCampaignOutcome:
        payload = self.model_dump(mode="json", exclude={"outcome_sha256"})
        if self.outcome_sha256 != canonical_digest(payload):
            raise ValueError("conditional campaign outcome digest mismatch")
        return self


def select_conditional_decision(
    director: MissionDesignRun,
    spec: ConditionalCampaignExecutionSpec,
) -> ConditionalAnalysisDecision:
    if director.schema_version != "1.1":
        raise ValueError("conditional campaign execution requires Director schema 1.1")
    if director.decision.disposition != "selected" or director.baseline is None:
        raise ValueError("conditional campaign execution requires an eligible baseline")
    matches = [
        item
        for item in director.verification_plan.conditional_analyses
        if item.rule_id == spec.rule_id and item.capability_id == spec.capability_id
    ]
    if len(matches) != 1:
        raise ValueError("execution spec must select exactly one conditional analysis")
    decision = matches[0]
    if decision.disposition != ConditionalAnalysisDisposition.RECOMMENDED:
        raise ValueError("conditional campaign execution requires a recommendation")
    if (
        decision.candidate_id != director.baseline.candidate_id
        or decision.baseline_id != director.baseline.baseline_id
    ):
        raise ValueError("conditional analysis does not bind the eligible baseline")
    return decision


def validate_campaign_contract(
    director: MissionDesignRun,
    spec: ConditionalCampaignExecutionSpec,
    definition: CampaignDefinition,
) -> None:
    if definition.workflow.kind != "mission_lifecycle":
        raise ValueError("conditional campaign must use mission_lifecycle")
    if not isinstance(definition.stopping, FixedCountStopping):
        raise ValueError("first conditional campaign slice requires fixed-count stopping")
    if definition.evaluator.kind.value != "authoritative":
        raise ValueError("conditional campaign must use the authoritative evaluator")
    if definition.evaluator.claim_boundary != spec.claim_boundary:
        raise ValueError("campaign claim boundary does not match the execution spec")
    hard_ids = {
        item.requirement_id for item in director.requirement_graph.requirements if item.hard
    }
    if {item.director_requirement_id for item in spec.acceptance_gates} != hard_ids:
        raise ValueError("acceptance gates must cover every hard Director requirement")
    requirements = {item.requirement_id: item for item in definition.requirements}
    metrics = {item.metric_id: item for item in definition.metrics}
    director_requirements = {
        item.requirement_id: item for item in director.requirement_graph.requirements
    }
    for gate in spec.acceptance_gates:
        requirement = requirements.get(gate.campaign_requirement_id)
        metric = metrics.get(gate.campaign_metric_id)
        director_requirement = director_requirements[gate.director_requirement_id]
        if requirement is None or metric is None:
            raise ValueError("acceptance gate references missing campaign contracts")
        if requirement.metric_id != metric.metric_id:
            raise ValueError("campaign requirement and metric mapping do not match")
        if requirement.operator != RequirementOperator.GE:
            raise ValueError("conditional campaign margin gates must use ge")
        if requirement.value != 0.0:
            raise ValueError("conditional campaign margins must use a zero threshold")
        if metric.value_kind != MetricValueKind.NUMERIC:
            raise ValueError("conditional campaign gate metric must be numeric")
        if metric.unit != director_requirement.unit:
            raise ValueError("campaign gate unit does not match the Director requirement")


def build_conditional_campaign_outcome(
    *,
    spec: ConditionalCampaignExecutionSpec,
    binding: ConditionalCampaignBinding,
    definition: CampaignDefinition,
    campaign_state: CampaignState,
    samples: tuple[ParameterRealization, ...],
    cases: tuple[CaseObservation, ...],
    statistics: CampaignStatistics,
    samples_sha256: str,
    cases_sha256: str,
    statistics_sha256: str,
) -> ConditionalCampaignOutcome:
    assessments: list[CampaignGateAssessment] = []
    for gate in spec.acceptance_gates:
        outcomes = []
        for case in cases:
            matches = [
                requirement
                for requirement in case.requirements
                if requirement.requirement_id == gate.campaign_requirement_id
            ]
            expected_count = 1 if case.outcome_status == OutcomeStatus.SUCCESS else 0
            if len(matches) != expected_count:
                raise ValueError("successful campaign cases must contain each acceptance gate")
            outcomes.extend(matches)
        passed_samples = sum(item.passed is True for item in outcomes)
        fraction = 0.0 if not cases else passed_samples / len(cases)
        payload = {
            **gate.model_dump(mode="json"),
            "passed_samples": passed_samples,
            "completed_samples": len(cases),
            "observed_pass_fraction": fraction,
            "passed": fraction >= gate.minimum_pass_fraction,
        }
        assessments.append(
            CampaignGateAssessment.model_validate(
                {**payload, "assessment_sha256": canonical_digest(payload)}
            )
        )
    non_success = any(item.outcome_status != OutcomeStatus.SUCCESS for item in cases)
    incomplete = (
        campaign_state != CampaignState.COMPLETED
        or len(samples) != definition.sampler.samples
        or len(cases) != definition.sampler.samples
        or statistics.completed_samples != definition.sampler.samples
    )
    if incomplete:
        disposition = ConditionalCampaignDisposition.ABSTAIN
        reason = ConditionalCampaignReason.CAMPAIGN_EXECUTION_INCOMPLETE
    elif non_success:
        disposition = ConditionalCampaignDisposition.ABSTAIN
        reason = ConditionalCampaignReason.CAMPAIGN_CASE_FAILURE
    elif all(item.passed for item in assessments):
        disposition = ConditionalCampaignDisposition.RETAIN
        reason = ConditionalCampaignReason.ALL_DECLARED_DESIGN_SPACE_GATES_PASSED
    else:
        disposition = ConditionalCampaignDisposition.REVISE
        reason = ConditionalCampaignReason.DECLARED_DESIGN_SPACE_GATE_FAILED
    payload = {
        "schema_version": "1.0",
        "execution_id": spec.execution_id,
        "binding_sha256": binding.binding_sha256,
        "director_run_sha256": binding.director_run_sha256,
        "baseline_id": binding.baseline_id,
        "baseline_sha256": binding.baseline_sha256,
        "candidate_id": binding.candidate_id,
        "conditional_decision_sha256": binding.conditional_decision_sha256,
        "campaign_definition_digest": binding.campaign_definition_digest,
        "campaign_samples_sha256": samples_sha256,
        "campaign_cases_sha256": cases_sha256,
        "campaign_statistics_sha256": statistics_sha256,
        "campaign_state": campaign_state,
        "requested_samples": definition.sampler.samples,
        "completed_samples": len(cases),
        "outcome_counts": statistics.outcome_counts,
        "gate_assessments": [item.model_dump(mode="json") for item in assessments],
        "disposition": disposition,
        "reason": reason,
        "claim_boundary": spec.claim_boundary,
    }
    return ConditionalCampaignOutcome.model_validate(
        {**payload, "outcome_sha256": canonical_digest(payload)}
    )


__all__ = [
    "CampaignAcceptanceGate",
    "CampaignGateAssessment",
    "ConditionalCampaignBinding",
    "ConditionalCampaignDisposition",
    "ConditionalCampaignExecutionSpec",
    "ConditionalCampaignOutcome",
    "ConditionalCampaignReason",
    "build_conditional_campaign_outcome",
    "canonical_digest",
    "select_conditional_decision",
    "validate_campaign_contract",
]
