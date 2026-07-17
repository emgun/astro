from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import Field, FiniteFloat, field_validator, model_validator

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


class AuthorityGrant(AstroModel):
    grant_id: str = Field(min_length=1)
    grant_version: int = Field(default=1, ge=1)
    level: AuthorityLevel
    mission_scope: str = Field(min_length=1)
    allowed_actions: tuple[OperatorActionKind, ...] = Field(min_length=1)
    allowed_command_types: tuple[str, ...] = ()
    approval_required_for: tuple[OperatorActionKind, ...] = ()
    approvals: tuple[ActionApproval, ...] = ()
    max_steps: int = Field(ge=1, le=10_000)
    max_candidate_evaluations: int = Field(ge=0, le=10_000)
    revoked: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_authority_contract(self) -> AuthorityGrant:
        if len(set(self.allowed_actions)) != len(self.allowed_actions):
            raise ValueError("allowed actions must be unique")
        if len(set(self.allowed_command_types)) != len(self.allowed_command_types):
            raise ValueError("allowed command types must be unique")
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
        return self

    @field_validator("grant_version", "max_steps", "max_candidate_evaluations", mode="before")
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


class MissionObjective(AstroModel):
    objective_id: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    design_variables: tuple[DesignVariable, ...] = Field(min_length=1)
    metric_goals: tuple[MetricGoal, ...] = Field(min_length=1)
    base_evidence: tuple[EvidenceReference, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def identifiers_must_be_unique(self) -> MissionObjective:
        variable_ids = [item.variable_id for item in self.design_variables]
        targets = [item.target for item in self.design_variables]
        metric_ids = [item.metric_id for item in self.metric_goals]
        evidence_ids = [item.evidence_id for item in self.base_evidence]
        if len(set(variable_ids)) != len(variable_ids):
            raise ValueError("design variable IDs must be unique")
        if len(set(targets)) != len(targets):
            raise ValueError("design variable targets must be unique")
        if len(set(metric_ids)) != len(metric_ids):
            raise ValueError("metric goal IDs must be unique")
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("base evidence IDs must be unique")
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
    evidence_kind: str = Field(min_length=1)
    query: str = Field(min_length=1)


class CommandRequest(AstroModel):
    command_id: str = Field(min_length=1)
    command_type: str = Field(min_length=1)
    parameters: dict[str, Any] = Field(default_factory=dict)


class CommandResult(AstroModel):
    command: CommandRequest
    status: str = Field(min_length=1)
    evidence: tuple[EvidenceReference, ...] = ()
    message: str = ""


class OperatorAction(AstroModel):
    action_id: str = Field(min_length=1)
    kind: OperatorActionKind
    rationale: str = Field(min_length=1)
    evidence_ids: tuple[str, ...] = ()
    candidate: CandidateProposal | None = None
    evidence_request: EvidenceRequest | None = None
    command: CommandRequest | None = None
    selected_candidate_id: str | None = None
    conclusion: str | None = None

    @model_validator(mode="after")
    def payload_must_match_action(self) -> OperatorAction:
        required = {
            OperatorActionKind.EVALUATE_CANDIDATE: self.candidate,
            OperatorActionKind.REQUEST_EVIDENCE: self.evidence_request,
            OperatorActionKind.PROPOSE_COMMAND: self.command,
            OperatorActionKind.EXECUTE_COMMAND: self.command,
        }
        for kind, payload in required.items():
            if self.kind == kind and payload is None:
                raise ValueError(f"{kind.value} requires its typed payload")
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
        if self.kind != OperatorActionKind.FINISH and (
            self.selected_candidate_id is not None or self.conclusion is not None
        ):
            raise ValueError("selection and conclusion are only valid for finish")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("action evidence IDs must be unique")
        return self


class OperatorStep(AstroModel):
    sequence: int = Field(ge=1)
    action: OperatorAction
    observation: CandidateObservation | None = None
    acquired_evidence: tuple[EvidenceReference, ...] = ()
    command_result: CommandResult | None = None

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
        elif self.action.kind == OperatorActionKind.EXECUTE_COMMAND:
            if self.command_result is None:
                raise ValueError("execute_command step requires a command result")
            if self.observation is not None or self.acquired_evidence:
                raise ValueError("execute_command step has invalid outputs")
        elif (
            self.observation is not None
            or self.acquired_evidence
            or self.command_result is not None
        ):
            raise ValueError("action step has outputs that do not match its kind")
        return self


class OperatorState(AstroModel):
    objective: MissionObjective
    authority: AuthorityGrant
    steps: tuple[OperatorStep, ...]
    known_evidence: tuple[EvidenceReference, ...]
    remaining_steps: int = Field(ge=0)
    remaining_candidate_evaluations: int = Field(ge=0)


class OperatorRunStatus(StrEnum):
    COMPLETED = "completed"
    BUDGET_EXHAUSTED = "budget_exhausted"


class OperatorRun(AstroModel):
    schema_version: Literal["1.0"] = "1.0"
    objective: MissionObjective
    authority: AuthorityGrant
    status: OperatorRunStatus
    steps: tuple[OperatorStep, ...]
    known_evidence: tuple[EvidenceReference, ...]
    selected_candidate_id: str | None = None
    conclusion: str

    @model_validator(mode="after")
    def journal_must_be_self_consistent(self) -> OperatorRun:
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
        elif any(step.action.kind == OperatorActionKind.FINISH for step in self.steps):
            raise ValueError("budget-exhausted operator run cannot contain finish")
        if any(
            step.action.kind == OperatorActionKind.FINISH for step in self.steps[:-1]
        ):
            raise ValueError("finish can appear only as the final operator step")
        return self
