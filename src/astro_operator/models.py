from __future__ import annotations

import json
from datetime import datetime
from enum import StrEnum
from hashlib import sha256
from typing import Any, Literal

from pydantic import Field, FiniteFloat, JsonValue, StrictInt, field_validator, model_validator

from astro_core.models import AstroModel


class AuthorityLevel(StrEnum):
    RESEARCH = "research"
    DECISION_SUPPORT = "decision_support"
    SUPERVISED_AUTONOMY = "supervised_autonomy"
    DELEGATED_AUTONOMY = "delegated_autonomy"
    MISSION_AUTONOMY = "mission_autonomy"


class OperatorActionKind(StrEnum):
    EVALUATE_CANDIDATE = "evaluate_candidate"
    REQUEST_EVIDENCE = "request_evidence"
    PROPOSE_COMMAND = "propose_command"
    EXECUTE_COMMAND = "execute_command"
    FINISH = "finish"


class EpistemicKind(StrEnum):
    DECLARED = "declared"
    OBSERVED = "observed"
    ESTIMATED = "estimated"
    SIMULATED = "simulated"
    INFERRED = "inferred"


class AcquisitionStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ClaimDisposition(StrEnum):
    SUPPORTED = "supported"
    QUALIFIED = "qualified"
    DISPUTED = "disputed"


class NumericComparisonOperator(StrEnum):
    LESS_THAN = "lt"
    LESS_THAN_OR_EQUAL = "le"
    GREATER_THAN = "gt"
    GREATER_THAN_OR_EQUAL = "ge"
    EQUAL = "eq"


_MINIMUM_AUTHORITY = {
    OperatorActionKind.EVALUATE_CANDIDATE: AuthorityLevel.RESEARCH,
    OperatorActionKind.REQUEST_EVIDENCE: AuthorityLevel.RESEARCH,
    OperatorActionKind.PROPOSE_COMMAND: AuthorityLevel.DECISION_SUPPORT,
    OperatorActionKind.EXECUTE_COMMAND: AuthorityLevel.SUPERVISED_AUTONOMY,
    OperatorActionKind.FINISH: AuthorityLevel.RESEARCH,
}

_AUTHORITY_RANK = {
    AuthorityLevel.RESEARCH: 1,
    AuthorityLevel.DECISION_SUPPORT: 2,
    AuthorityLevel.SUPERVISED_AUTONOMY: 3,
    AuthorityLevel.DELEGATED_AUTONOMY: 4,
    AuthorityLevel.MISSION_AUTONOMY: 5,
}


class ActionApproval(AstroModel):
    approval_id: str = Field(min_length=1)
    grant_version: int = Field(ge=1)
    action_id: str = Field(min_length=1)
    action_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("grant_version", mode="before")
    @classmethod
    def grant_version_must_be_strict(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("approval grant_version must be an integer")
        return value


class AllowedEvidenceTool(AstroModel):
    tool_id: str = Field(min_length=1)
    tool_version: str = Field(min_length=1)
    request_kinds: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def request_kinds_must_be_unique(self) -> AllowedEvidenceTool:
        if len(set(self.request_kinds)) != len(self.request_kinds):
            raise ValueError("allowed evidence request kinds must be unique")
        return self


class CommandParameterLimit(AstroModel):
    parameter: str = Field(min_length=1)
    minimum: FiniteFloat
    maximum: FiniteFloat
    unit: str = Field(min_length=1)

    @field_validator("minimum", "maximum", mode="before")
    @classmethod
    def limits_must_be_numeric(cls, value: Any) -> Any:
        if isinstance(value, bool | str):
            raise ValueError("command parameter limits must be numeric scalars")
        return value

    @model_validator(mode="after")
    def limits_must_be_ordered(self) -> CommandParameterLimit:
        if self.minimum > self.maximum:
            raise ValueError("command parameter minimum must not exceed maximum")
        return self


class CommandEnvelope(AstroModel):
    command_type: str = Field(min_length=1)
    tool_version: str = Field(min_length=1)
    tool_qualification_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    simulation_only: Literal[True] = True
    allowed_asset_ids: tuple[str, ...] = Field(min_length=1)
    parameter_limits: tuple[CommandParameterLimit, ...] = Field(min_length=1)
    max_commits: int = Field(ge=1, le=10_000)

    @field_validator("max_commits", mode="before")
    @classmethod
    def max_commits_must_be_strict(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("command envelope max_commits must be an integer")
        return value

    @model_validator(mode="after")
    def envelope_entries_must_be_unique(self) -> CommandEnvelope:
        if len(set(self.allowed_asset_ids)) != len(self.allowed_asset_ids):
            raise ValueError("command envelope asset IDs must be unique")
        parameters = [item.parameter for item in self.parameter_limits]
        if len(set(parameters)) != len(parameters):
            raise ValueError("command envelope parameter limits must be unique")
        return self


class AuthorityGrant(AstroModel):
    grant_id: str = Field(min_length=1)
    grant_version: int = Field(default=1, ge=1)
    level: AuthorityLevel
    mission_scope: str = Field(min_length=1)
    allowed_actions: tuple[OperatorActionKind, ...] = Field(min_length=1)
    allowed_command_types: tuple[str, ...] = ()
    command_envelopes: tuple[CommandEnvelope, ...] = ()
    approval_required_for: tuple[OperatorActionKind, ...] = ()
    approvals: tuple[ActionApproval, ...] = ()
    allowed_evidence_tools: tuple[AllowedEvidenceTool, ...] = ()
    max_steps: int = Field(ge=1, le=10_000)
    max_candidate_evaluations: int = Field(ge=0, le=10_000)
    max_evidence_acquisitions: int = Field(default=0, ge=0, le=10_000)
    valid_from: datetime | None = None
    expires_at: datetime | None = None
    revoked: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_authority_contract(self) -> AuthorityGrant:
        for value in (self.valid_from, self.expires_at):
            if value is not None and (value.tzinfo is None or value.utcoffset() is None):
                raise ValueError("authority validity timestamps must include timezone information")
        if (
            self.valid_from is not None
            and self.expires_at is not None
            and self.expires_at <= self.valid_from
        ):
            raise ValueError("authority expiry must follow its validity start")
        if len(set(self.allowed_actions)) != len(self.allowed_actions):
            raise ValueError("allowed actions must be unique")
        if len(set(self.allowed_command_types)) != len(self.allowed_command_types):
            raise ValueError("allowed command types must be unique")
        envelope_types = [item.command_type for item in self.command_envelopes]
        if len(set(envelope_types)) != len(envelope_types):
            raise ValueError("command envelopes must be unique by command type")
        if not set(envelope_types).issubset(self.allowed_command_types):
            raise ValueError("command envelopes must cover only allowed command types")
        tool_keys = [
            (item.tool_id, item.tool_version) for item in self.allowed_evidence_tools
        ]
        if len(set(tool_keys)) != len(tool_keys):
            raise ValueError("allowed evidence tools must be unique by ID and version")
        if len(set(self.approval_required_for)) != len(self.approval_required_for):
            raise ValueError("approval requirements must be unique")
        if not set(self.approval_required_for).issubset(self.allowed_actions):
            raise ValueError("approval requirements must be allowed actions")
        approval_ids = [item.approval_id for item in self.approvals]
        if len(set(approval_ids)) != len(approval_ids):
            raise ValueError("approval IDs must be unique")
        if any(item.grant_version != self.grant_version for item in self.approvals):
            raise ValueError("approval grant versions must match the authority grant")
        for action in self.allowed_actions:
            if _AUTHORITY_RANK[self.level] < _AUTHORITY_RANK[_MINIMUM_AUTHORITY[action]]:
                raise ValueError(
                    f"authority level {self.level.name.lower()} cannot grant {action.value}"
                )
        command_actions = {
            OperatorActionKind.PROPOSE_COMMAND,
            OperatorActionKind.EXECUTE_COMMAND,
        }
        if command_actions.intersection(self.allowed_actions) and not self.allowed_command_types:
            raise ValueError("command actions require at least one allowed command type")
        if (
            OperatorActionKind.EXECUTE_COMMAND in self.allowed_actions
            and self.level == AuthorityLevel.SUPERVISED_AUTONOMY
            and OperatorActionKind.EXECUTE_COMMAND not in self.approval_required_for
        ):
            raise ValueError("supervised command execution must require approval")
        if (
            OperatorActionKind.EXECUTE_COMMAND in self.allowed_actions
            and OperatorActionKind.PROPOSE_COMMAND not in self.allowed_actions
        ):
            raise ValueError("command execution requires command proposal authority")
        if OperatorActionKind.REQUEST_EVIDENCE in self.allowed_actions:
            if not self.allowed_evidence_tools:
                raise ValueError("evidence requests require at least one allowed evidence tool")
            if self.max_evidence_acquisitions == 0:
                raise ValueError("evidence requests require a positive acquisition budget")
        return self

    @field_validator(
        "grant_version",
        "max_steps",
        "max_candidate_evaluations",
        "max_evidence_acquisitions",
        mode="before",
    )
    @classmethod
    def integer_fields_must_be_strict(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("authority integer fields must be integers")
        return value


class DesignVariable(AstroModel):
    variable_id: str = Field(min_length=1)
    target: str = Field(min_length=1)
    lower_bound: FiniteFloat
    upper_bound: FiniteFloat
    unit: str = Field(min_length=1)

    @field_validator("lower_bound", "upper_bound", mode="before")
    @classmethod
    def bounds_must_be_numeric(cls, value: Any) -> Any:
        if isinstance(value, bool | str):
            raise ValueError("design variable bounds must be numeric scalars")
        return value

    @model_validator(mode="after")
    def bounds_must_be_ordered(self) -> DesignVariable:
        if self.lower_bound > self.upper_bound:
            raise ValueError("design variable lower_bound must not exceed upper_bound")
        return self


class MetricGoal(AstroModel):
    metric_id: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    unit: str = Field(min_length=1)


class EvidenceReference(AstroModel):
    evidence_id: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    epistemic_kind: EpistemicKind
    claim_scope: str = Field(min_length=1)
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceToolSpec(AstroModel):
    tool_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    request_kind: str = Field(min_length=1)
    parameter_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_assertion_kinds: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def assertion_kinds_must_be_unique(self) -> EvidenceToolSpec:
        if len(set(self.output_assertion_kinds)) != len(self.output_assertion_kinds):
            raise ValueError("output assertion kinds must be unique")
        return self


class EvidenceAssertion(AstroModel):
    assertion_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    subject: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    value: JsonValue
    epistemic_kind: EpistemicKind
    scope: str = Field(min_length=1)
    source_evidence_ids: tuple[str, ...] = Field(min_length=1)
    producer_tool_id: str = Field(min_length=1)
    producer_tool_version: str = Field(min_length=1)
    valid_at: datetime | None = None
    assertion_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @field_validator("valid_at")
    @classmethod
    def valid_at_must_be_aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("assertion valid_at must include timezone information")
        return value

    @model_validator(mode="after")
    def source_ids_must_be_unique(self) -> EvidenceAssertion:
        if len(set(self.source_evidence_ids)) != len(self.source_evidence_ids):
            raise ValueError("assertion source evidence IDs must be unique")
        return self


class AssertionConflict(AstroModel):
    conflict_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    subject: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    scope: str = Field(min_length=1)
    valid_at: datetime | None = None
    assertion_ids: tuple[str, ...] = Field(min_length=2)


class WorldState(AstroModel):
    assertions: tuple[EvidenceAssertion, ...] = ()
    conflicts: tuple[AssertionConflict, ...] = ()
    state_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class MissionObjective(AstroModel):
    objective_id: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    design_variables: tuple[DesignVariable, ...] = Field(min_length=1)
    metric_goals: tuple[MetricGoal, ...] = Field(min_length=1)
    base_evidence: tuple[EvidenceReference, ...] = ()
    base_assertions: tuple[EvidenceAssertion, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def identifiers_must_be_unique(self) -> MissionObjective:
        variable_ids = [item.variable_id for item in self.design_variables]
        targets = [item.target for item in self.design_variables]
        metric_ids = [item.metric_id for item in self.metric_goals]
        evidence_ids = [item.evidence_id for item in self.base_evidence]
        assertion_ids = [item.assertion_id for item in self.base_assertions]
        if len(set(variable_ids)) != len(variable_ids):
            raise ValueError("design variable IDs must be unique")
        if len(set(targets)) != len(targets):
            raise ValueError("design variable targets must be unique")
        if len(set(metric_ids)) != len(metric_ids):
            raise ValueError("metric goal IDs must be unique")
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("base evidence IDs must be unique")
        if len(set(assertion_ids)) != len(assertion_ids):
            raise ValueError("base assertion IDs must be unique")
        known_evidence = set(evidence_ids)
        if any(
            not set(assertion.source_evidence_ids).issubset(known_evidence)
            for assertion in self.base_assertions
        ):
            raise ValueError("base assertions must cite base evidence")
        return self


class CandidateProposal(AstroModel):
    candidate_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
    assignments: dict[str, FiniteFloat]
    description: str = ""

    @field_validator("assignments", mode="before")
    @classmethod
    def assignments_must_be_numeric(cls, value: Any) -> Any:
        if isinstance(value, dict):
            for assignment in value.values():
                if isinstance(assignment, bool | str):
                    raise ValueError("candidate assignments must be numeric scalars")
        return value


class ObservedMetric(AstroModel):
    metric_id: str = Field(min_length=1)
    value: FiniteFloat
    unit: str = Field(min_length=1)
    status: str | None = None

    @field_validator("value", mode="before")
    @classmethod
    def value_must_be_numeric(cls, value: Any) -> Any:
        if isinstance(value, bool | str):
            raise ValueError("observed metric value must be a numeric scalar")
        return value


class CandidateObservation(AstroModel):
    candidate: CandidateProposal
    evaluation_status: str = Field(min_length=1)
    passed: bool
    metrics: tuple[ObservedMetric, ...]
    evidence: tuple[EvidenceReference, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def observation_ids_must_be_unique(self) -> CandidateObservation:
        metric_ids = [item.metric_id for item in self.metrics]
        evidence_ids = [item.evidence_id for item in self.evidence]
        if len(set(metric_ids)) != len(metric_ids):
            raise ValueError("observation metric IDs must be unique")
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("observation evidence IDs must be unique")
        return self


class EvidenceRequest(AstroModel):
    request_id: str = Field(min_length=1)
    tool_id: str = Field(min_length=1)
    tool_version: str = Field(min_length=1)
    request_kind: str = Field(min_length=1)
    parameters: dict[str, JsonValue] = Field(default_factory=dict)


class EvidenceAcquisitionResult(AstroModel):
    request: EvidenceRequest
    tool: EvidenceToolSpec
    status: AcquisitionStatus
    evidence: tuple[EvidenceReference, ...] = ()
    assertions: tuple[EvidenceAssertion, ...] = ()
    message: str = ""

    @model_validator(mode="after")
    def result_must_match_request_and_bind_assertions(self) -> EvidenceAcquisitionResult:
        if (
            self.tool.tool_id != self.request.tool_id
            or self.tool.version != self.request.tool_version
            or self.tool.request_kind != self.request.request_kind
        ):
            raise ValueError("acquisition tool identity must match its request")
        evidence_ids = [item.evidence_id for item in self.evidence]
        assertion_ids = [item.assertion_id for item in self.assertions]
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("acquisition evidence IDs must be unique")
        if len(set(assertion_ids)) != len(assertion_ids):
            raise ValueError("acquisition assertion IDs must be unique")
        for assertion in self.assertions:
            if (
                assertion.producer_tool_id != self.tool.tool_id
                or assertion.producer_tool_version != self.tool.version
            ):
                raise ValueError("assertion producer must match acquisition tool")
            if assertion.predicate not in self.tool.output_assertion_kinds:
                raise ValueError("assertion predicate is outside the tool output contract")
        if self.status == AcquisitionStatus.SUCCEEDED and not (
            self.evidence or self.assertions
        ):
            raise ValueError("successful acquisition must produce evidence or assertions")
        if self.status == AcquisitionStatus.FAILED and self.assertions:
            raise ValueError("failed acquisition cannot produce assertions")
        return self


class NumericThresholdPredicate(AstroModel):
    kind: Literal["numeric_threshold"] = "numeric_threshold"
    predicate_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    assertion_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    assertion_predicate: str = Field(min_length=1)
    operator: NumericComparisonOperator
    threshold_value: FiniteFloat | None = None
    threshold_assertion_id: str | None = Field(
        default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
    )
    threshold_assertion_predicate: str | None = Field(default=None, min_length=1)

    @field_validator("threshold_value", mode="before")
    @classmethod
    def threshold_must_be_numeric(cls, value: Any) -> Any:
        if isinstance(value, bool | str):
            raise ValueError("numeric threshold must be a numeric scalar")
        return value

    @model_validator(mode="after")
    def threshold_source_must_be_unambiguous(self) -> NumericThresholdPredicate:
        if (self.threshold_value is None) == (self.threshold_assertion_id is None):
            raise ValueError(
                "numeric predicate requires exactly one literal or assertion threshold"
            )
        if (self.threshold_assertion_id is None) != (
            self.threshold_assertion_predicate is None
        ):
            raise ValueError(
                "assertion threshold requires its exact expected assertion predicate"
            )
        return self


class FreshnessPredicate(AstroModel):
    kind: Literal["freshness"] = "freshness"
    predicate_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    assertion_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    assertion_predicate: str = Field(min_length=1)
    reference_assertion_id: str = Field(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
    )
    reference_assertion_predicate: str = Field(min_length=1)
    max_age_s: FiniteFloat | None = Field(default=None, ge=0.0)
    max_age_assertion_id: str | None = Field(
        default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
    )
    max_age_assertion_predicate: str = Field(default="maximum_age_s", min_length=1)

    @field_validator("max_age_s", mode="before")
    @classmethod
    def max_age_must_be_numeric(cls, value: Any) -> Any:
        if isinstance(value, bool | str):
            raise ValueError("freshness maximum age must be a numeric scalar")
        return value

    @model_validator(mode="after")
    def age_source_must_be_unambiguous(self) -> FreshnessPredicate:
        if (self.max_age_s is None) == (self.max_age_assertion_id is None):
            raise ValueError(
                "freshness predicate requires exactly one literal or assertion maximum age"
            )
        return self


class ApplicabilityPredicate(AstroModel):
    kind: Literal["applicability"] = "applicability"
    predicate_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    actual_assertion_id: str = Field(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
    )
    actual_assertion_predicate: str = Field(min_length=1)
    required_assertion_id: str = Field(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
    )
    required_assertion_predicate: str = Field(min_length=1)
    expected_subject: str = Field(min_length=1)
    expected_scope: str = Field(min_length=1)


class ExactValuePredicate(AstroModel):
    kind: Literal["exact_value"] = "exact_value"
    predicate_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    assertion_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    assertion_predicate: str = Field(min_length=1)
    expected_value: JsonValue


ClaimPredicate = (
    NumericThresholdPredicate
    | FreshnessPredicate
    | ApplicabilityPredicate
    | ExactValuePredicate
)


class ConclusionClaim(AstroModel):
    claim_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    statement: str = Field(min_length=1)
    conclusion_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    disposition: ClaimDisposition
    assertion_ids: tuple[str, ...] = Field(min_length=1)
    predicates: tuple[ClaimPredicate, ...] = ()
    qualification: str | None = None

    @model_validator(mode="after")
    def claim_must_be_consistent(self) -> ConclusionClaim:
        if len(set(self.assertion_ids)) != len(self.assertion_ids):
            raise ValueError("claim assertion IDs must be unique")
        predicate_assertion_ids: set[str] = set()
        predicate_ids: list[str] = []
        for predicate in self.predicates:
            predicate_ids.append(predicate.predicate_id)
            if isinstance(predicate, NumericThresholdPredicate):
                predicate_assertion_ids.add(predicate.assertion_id)
                if predicate.threshold_assertion_id is not None:
                    predicate_assertion_ids.add(predicate.threshold_assertion_id)
            elif isinstance(predicate, FreshnessPredicate):
                predicate_assertion_ids.update(
                    (predicate.assertion_id, predicate.reference_assertion_id)
                )
                if predicate.max_age_assertion_id is not None:
                    predicate_assertion_ids.add(predicate.max_age_assertion_id)
            elif isinstance(predicate, ApplicabilityPredicate):
                predicate_assertion_ids.update(
                    (predicate.actual_assertion_id, predicate.required_assertion_id)
                )
            else:
                predicate_assertion_ids.add(predicate.assertion_id)
        if len(set(predicate_ids)) != len(predicate_ids):
            raise ValueError("claim predicate IDs must be unique")
        if not predicate_assertion_ids.issubset(self.assertion_ids):
            raise ValueError("claim predicates must reference cited assertion IDs")
        if self.disposition != ClaimDisposition.SUPPORTED and not self.qualification:
            raise ValueError("qualified and disputed claims require a qualification")
        return self


class CommandRequest(AstroModel):
    command_id: str = Field(min_length=1)
    command_type: str = Field(min_length=1)
    asset_id: str | None = None
    parameters: dict[str, JsonValue] = Field(default_factory=dict)


class CommandExecutionRequest(AstroModel):
    proposal_action_id: str = Field(min_length=1)
    command_id: str = Field(min_length=1)
    idempotency_key: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    expected_world_state_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    approval_id: str | None = None


class CommandResult(AstroModel):
    command: CommandRequest
    status: str = Field(min_length=1)
    evidence: tuple[EvidenceReference, ...] = ()
    assertions: tuple[EvidenceAssertion, ...] = ()
    message: str = ""

    @model_validator(mode="after")
    def command_result_ids_must_be_unique(self) -> CommandResult:
        if len({item.evidence_id for item in self.evidence}) != len(self.evidence):
            raise ValueError("command result evidence IDs must be unique")
        if len({item.assertion_id for item in self.assertions}) != len(self.assertions):
            raise ValueError("command result assertion IDs must be unique")
        return self


class CommandTerminalStatus(StrEnum):
    COMMITTED = "committed"
    FAILED = "failed"
    INDETERMINATE = "indeterminate"


class CommandPreparedRecord(AstroModel):
    execution_id: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    proposal_action_id: str = Field(min_length=1)
    proposal_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_action_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    command_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tool_version: str = Field(min_length=1)
    tool_qualification_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    simulation_only: bool
    grant_id: str = Field(min_length=1)
    grant_version: int = Field(ge=1)
    authority_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    world_state_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    approval_id: str | None = None
    prepared_at: datetime
    record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("prepared_at")
    @classmethod
    def prepared_at_must_be_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("command preparation timestamp must include timezone information")
        return value


class CommandTerminalRecord(AstroModel):
    execution_id: str = Field(min_length=1)
    status: CommandTerminalStatus
    result_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    completed_at: datetime
    message: str = ""
    record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("completed_at")
    @classmethod
    def completed_at_must_be_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("command completion timestamp must include timezone information")
        return value

    @model_validator(mode="after")
    def committed_terminal_requires_result(self) -> CommandTerminalRecord:
        if self.status == CommandTerminalStatus.COMMITTED and self.result_sha256 is None:
            raise ValueError("committed command terminal record requires a result digest")
        return self


class CommandExecutionRecord(AstroModel):
    prepared: CommandPreparedRecord
    terminal: CommandTerminalRecord

    @model_validator(mode="after")
    def execution_ids_must_match(self) -> CommandExecutionRecord:
        if self.prepared.execution_id != self.terminal.execution_id:
            raise ValueError("command preparation and terminal execution IDs must match")
        if self.terminal.completed_at < self.prepared.prepared_at:
            raise ValueError("command terminal event cannot precede preparation")
        return self


class OperatorAction(AstroModel):
    action_id: str = Field(min_length=1)
    kind: OperatorActionKind
    rationale: str = Field(min_length=1)
    evidence_ids: tuple[str, ...] = ()
    candidate: CandidateProposal | None = None
    evidence_request: EvidenceRequest | None = None
    command: CommandRequest | None = None
    command_execution: CommandExecutionRequest | None = None
    selected_candidate_id: str | None = None
    conclusion: str | None = None
    conclusion_claims: tuple[ConclusionClaim, ...] = ()

    @model_validator(mode="after")
    def payload_must_match_action(self) -> OperatorAction:
        required = {
            OperatorActionKind.EVALUATE_CANDIDATE: self.candidate,
            OperatorActionKind.REQUEST_EVIDENCE: self.evidence_request,
            OperatorActionKind.PROPOSE_COMMAND: self.command,
        }
        for kind, payload in required.items():
            if self.kind == kind and payload is None:
                raise ValueError(f"{kind.value} requires its typed payload")
        if self.kind == OperatorActionKind.EXECUTE_COMMAND and (
            self.command is None and self.command_execution is None
        ):
            raise ValueError("execute_command requires a command execution request")
        if self.kind == OperatorActionKind.EXECUTE_COMMAND and (
            self.command is not None and self.command_execution is not None
        ):
            raise ValueError("execute_command cannot contain two command payloads")
        if self.kind == OperatorActionKind.FINISH and not self.conclusion:
            raise ValueError("finish requires a conclusion")
        if self.kind != OperatorActionKind.EVALUATE_CANDIDATE and self.candidate is not None:
            raise ValueError("candidate payload is only valid for evaluate_candidate")
        if self.kind != OperatorActionKind.REQUEST_EVIDENCE and self.evidence_request is not None:
            raise ValueError("evidence request payload is only valid for request_evidence")
        if self.kind not in {
            OperatorActionKind.PROPOSE_COMMAND,
            OperatorActionKind.EXECUTE_COMMAND,
        } and self.command is not None:
            raise ValueError("command payload is only valid for command actions")
        if self.kind != OperatorActionKind.EXECUTE_COMMAND and self.command_execution is not None:
            raise ValueError("command execution payload is only valid for execute_command")
        if self.kind != OperatorActionKind.FINISH and (
            self.selected_candidate_id is not None
            or self.conclusion is not None
            or self.conclusion_claims
        ):
            raise ValueError("selection and conclusion are only valid for finish")
        claim_ids = [item.claim_id for item in self.conclusion_claims]
        if len(set(claim_ids)) != len(claim_ids):
            raise ValueError("conclusion claim IDs must be unique")
        if self.kind == OperatorActionKind.FINISH and self.conclusion_claims:
            assert self.conclusion is not None
            conclusion_sha256 = sha256(self.conclusion.encode("utf-8")).hexdigest()
            if any(
                item.conclusion_sha256 != conclusion_sha256
                for item in self.conclusion_claims
            ):
                raise ValueError("conclusion claims must bind the exact conclusion text")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("action evidence IDs must be unique")
        return self


class ReasonerInvocation(AstroModel):
    """Provider-neutral provenance for one reasoner decision."""

    adapter: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    record_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    request_id: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    usage: dict[str, StrictInt] = Field(default_factory=dict)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("started_at", "completed_at")
    @classmethod
    def timestamps_must_be_aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("reasoner timestamps must include timezone information")
        return value

    @model_validator(mode="after")
    def invocation_must_be_consistent(self) -> ReasonerInvocation:
        if (
            self.started_at is not None
            and self.completed_at is not None
            and self.completed_at < self.started_at
        ):
            raise ValueError("reasoner completion cannot precede its start")
        if any(
            not key or isinstance(value, bool) or value < 0
            for key, value in self.usage.items()
        ):
            raise ValueError("reasoner usage values must be non-negative integers")
        if len(json.dumps(self.metadata, separators=(",", ":")).encode("utf-8")) > 16_384:
            raise ValueError("reasoner metadata must not exceed 16384 encoded bytes")
        return self


class ReasonerDecision(AstroModel):
    """An untrusted typed action and the invocation that produced it."""

    action: OperatorAction
    invocation: ReasonerInvocation


class ReasonerAttemptProvenance(AstroModel):
    """Content-free provenance retained for a successful or failed provider attempt."""

    adapter: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    attempt: int = Field(ge=1)
    started_at: datetime
    completed_at: datetime
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tool_definitions_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_response_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def timestamps_must_be_aware_and_ordered(self) -> ReasonerAttemptProvenance:
        for value in (self.started_at, self.completed_at):
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError("reasoner attempt timestamps must include timezone information")
        if self.completed_at < self.started_at:
            raise ValueError("reasoner attempt completion cannot precede its start")
        return self


class OperatorStep(AstroModel):
    sequence: int = Field(ge=1)
    action: OperatorAction
    reasoner_invocation: ReasonerInvocation | None = None
    observation: CandidateObservation | None = None
    acquired_evidence: tuple[EvidenceReference, ...] = ()
    acquisition_result: EvidenceAcquisitionResult | None = None
    command_result: CommandResult | None = None
    command_execution_record: CommandExecutionRecord | None = None

    @model_validator(mode="after")
    def outputs_must_match_action(self) -> OperatorStep:
        if self.action.kind == OperatorActionKind.EVALUATE_CANDIDATE:
            if self.observation is None:
                raise ValueError("evaluate_candidate step requires an observation")
            if self.acquired_evidence or self.command_result is not None:
                raise ValueError("evaluate_candidate step has invalid outputs")
        elif self.action.kind == OperatorActionKind.REQUEST_EVIDENCE:
            if self.observation is not None or self.command_result is not None:
                raise ValueError("request_evidence step has invalid outputs")
            if self.acquisition_result is not None:
                if self.acquired_evidence:
                    raise ValueError("typed acquisition cannot duplicate acquired evidence")
                if self.action.evidence_request != self.acquisition_result.request:
                    raise ValueError("acquisition result does not match its action")
        elif self.action.kind == OperatorActionKind.EXECUTE_COMMAND:
            if self.command_result is None:
                raise ValueError("execute_command step requires a command result")
            if (
                self.action.command_execution is not None
                and self.command_execution_record is None
            ):
                raise ValueError("execute_command step requires a command execution record")
            if self.observation is not None or self.acquired_evidence:
                raise ValueError("execute_command step has invalid outputs")
        elif (
            self.observation is not None
            or self.acquired_evidence
            or self.command_result is not None
            or self.command_execution_record is not None
        ):
            raise ValueError("action step has outputs that do not match its kind")
        return self


class OperatorState(AstroModel):
    objective: MissionObjective
    authority: AuthorityGrant
    steps: tuple[OperatorStep, ...]
    known_evidence: tuple[EvidenceReference, ...]
    world_state: WorldState | None = None
    remaining_steps: int = Field(ge=0)
    remaining_candidate_evaluations: int = Field(ge=0)
    remaining_evidence_acquisitions: int = Field(default=0, ge=0)


class OperatorRunStatus(StrEnum):
    COMPLETED = "completed"
    BUDGET_EXHAUSTED = "budget_exhausted"


class OperatorRun(AstroModel):
    schema_version: Literal["1.0", "1.1", "1.2", "1.3"] = "1.0"
    objective: MissionObjective
    authority: AuthorityGrant
    status: OperatorRunStatus
    steps: tuple[OperatorStep, ...]
    known_evidence: tuple[EvidenceReference, ...]
    world_state: WorldState | None = None
    selected_candidate_id: str | None = None
    conclusion: str

    @model_validator(mode="after")
    def journal_must_be_self_consistent(self) -> OperatorRun:
        if self.schema_version == "1.1" and any(
            step.reasoner_invocation is None for step in self.steps
        ):
            raise ValueError("operator schema 1.1 requires reasoner provenance for every step")
        if self.schema_version in {"1.2", "1.3"} and any(
            step.reasoner_invocation is None for step in self.steps
        ):
            raise ValueError(
                f"operator schema {self.schema_version} requires reasoner provenance "
                "for every step"
            )
        if self.schema_version == "1.0" and any(
            step.reasoner_invocation is not None for step in self.steps
        ):
            raise ValueError("operator schema 1.0 does not contain reasoner provenance")
        if self.schema_version not in {"1.2", "1.3"}:
            if self.world_state is not None or self.objective.base_assertions:
                raise ValueError("legacy operator schemas do not contain typed world state")
            if any(
                step.acquisition_result is not None
                or step.action.command_execution is not None
                or step.command_execution_record is not None
                or step.action.conclusion_claims
                or (
                    step.command_result is not None
                    and bool(step.command_result.assertions)
                )
                for step in self.steps
            ):
                raise ValueError("legacy operator schemas do not contain schema 1.2 events")
        if self.schema_version != "1.3" and any(
            claim.predicates
            for step in self.steps
            for claim in step.action.conclusion_claims
        ):
            raise ValueError("claim predicates require operator schema 1.3")
        if self.schema_version == "1.3":
            claims = tuple(
                claim
                for step in self.steps
                for claim in step.action.conclusion_claims
            )
            if not claims or any(
                claim.disposition == ClaimDisposition.SUPPORTED
                and not claim.predicates
                for claim in claims
            ):
                raise ValueError(
                    "operator schema 1.3 requires predicates on every supported claim"
                )
        if [step.sequence for step in self.steps] != list(range(1, len(self.steps) + 1)):
            raise ValueError("operator step sequences must be contiguous and one-based")
        action_ids = [step.action.action_id for step in self.steps]
        if len(set(action_ids)) != len(action_ids):
            raise ValueError("operator action IDs must be unique")
        known = list(self.objective.base_evidence)
        known_ids = {item.evidence_id for item in known}
        evaluated: set[str] = set()
        for step in self.steps:
            if not set(step.action.evidence_ids).issubset(known_ids):
                raise ValueError("operator action cites evidence not yet present in the journal")
            additions: tuple[EvidenceReference, ...] = step.acquired_evidence
            if step.observation is not None:
                if step.action.candidate != step.observation.candidate:
                    raise ValueError("operator observation does not match its action")
                evaluated.add(step.observation.candidate.candidate_id)
                additions += step.observation.evidence
            if step.command_result is not None:
                additions += step.command_result.evidence
            if step.acquisition_result is not None:
                additions += step.acquisition_result.evidence
            for evidence in additions:
                if evidence.evidence_id in known_ids:
                    raise ValueError("operator evidence IDs must be unique")
                known.append(evidence)
                known_ids.add(evidence.evidence_id)
        if tuple(known) != self.known_evidence:
            raise ValueError("operator evidence inventory must match the event journal")
        if self.selected_candidate_id is not None and self.selected_candidate_id not in evaluated:
            raise ValueError("selected candidate must have an observation")
        if self.status == OperatorRunStatus.COMPLETED:
            if not self.steps or self.steps[-1].action.kind != OperatorActionKind.FINISH:
                raise ValueError("completed operator run must end with finish")
            final = self.steps[-1].action
            if final.selected_candidate_id != self.selected_candidate_id:
                raise ValueError("run selection must match the finish action")
            if final.conclusion != self.conclusion:
                raise ValueError("run conclusion must match the finish action")
            if self.schema_version in {"1.2", "1.3"} and self.world_state is None:
                raise ValueError(
                    f"operator schema {self.schema_version} requires a world state"
                )
        elif any(step.action.kind == OperatorActionKind.FINISH for step in self.steps):
            raise ValueError("budget-exhausted operator run cannot contain finish")
        if any(
            step.action.kind == OperatorActionKind.FINISH for step in self.steps[:-1]
        ):
            raise ValueError("finish can appear only as the final operator step")
        return self
