"""Provider-neutral mission design direction over verified operator evidence."""

from __future__ import annotations

import json
from enum import StrEnum
from hashlib import sha256
from typing import Any, Literal

from pydantic import Field, FiniteFloat, StrictInt, field_validator, model_validator

from astro_core.models import AstroModel
from astro_mission.models import MissionLifecycleScenario
from astro_operator.models import (
    AuthorityGrant,
    CandidateObservation,
    MissionObjective,
    OperatorActionKind,
    OperatorRun,
    OperatorRunStatus,
)
from astro_operator.reasoner import model_digest


def _canonical_digest(value: object) -> str:
    payload = value.model_dump(mode="json") if isinstance(value, AstroModel) else value
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class RequirementComparator(StrEnum):
    GREATER_THAN_OR_EQUAL = "ge"
    LESS_THAN_OR_EQUAL = "le"


class FidelityClass(StrEnum):
    SCREENING = "screening"


class DesignDecisionDisposition(StrEnum):
    SELECTED = "selected"
    ABSTAINED = "abstained"


class MissionIntent(AstroModel):
    intent_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    summary: str = Field(min_length=1)
    decision_question: str = Field(min_length=1)
    objectives: tuple[str, ...] = Field(min_length=1)
    allowed_capability_ids: tuple[str, ...] = Field(min_length=1)
    authority_grant_id: str = Field(min_length=1)
    max_analysis_cost_units: StrictInt = Field(ge=1)
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def entries_must_be_unique(self) -> MissionIntent:
        if len(set(self.objectives)) != len(self.objectives):
            raise ValueError("mission intent objectives must be unique")
        if len(set(self.allowed_capability_ids)) != len(self.allowed_capability_ids):
            raise ValueError("allowed mission-design capabilities must be unique")
        return self


class RequirementNode(AstroModel):
    requirement_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    statement: str = Field(min_length=1)
    parent_ids: tuple[str, ...] = ()
    metric_id: str = Field(min_length=1)
    comparator: RequirementComparator
    threshold: FiniteFloat
    unit: str = Field(min_length=1)
    verification_capability_id: str = Field(min_length=1)
    hard: bool = True

    @field_validator("threshold", mode="before")
    @classmethod
    def threshold_must_be_numeric(cls, value: Any) -> Any:
        if isinstance(value, bool | str):
            raise ValueError("requirement threshold must be numeric")
        return value


class RequirementGraph(AstroModel):
    graph_id: str = Field(min_length=1)
    requirements: tuple[RequirementNode, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def graph_must_be_well_formed(self) -> RequirementGraph:
        ids = [item.requirement_id for item in self.requirements]
        metrics = [item.metric_id for item in self.requirements]
        if len(set(ids)) != len(ids):
            raise ValueError("requirement IDs must be unique")
        if len(set(metrics)) != len(metrics):
            raise ValueError("first-slice requirements must map one-to-one to metrics")
        known = set(ids)
        if any(not set(item.parent_ids).issubset(known) for item in self.requirements):
            raise ValueError("requirement parents must exist in the graph")
        _topological_order({item.requirement_id: item.parent_ids for item in self.requirements})
        return self


class CapabilitySpec(AstroModel):
    capability_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    input_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    fidelity: FidelityClass
    deterministic: bool
    simulation_only: Literal[True] = True
    cost_units: StrictInt = Field(ge=1)
    output_metric_ids: tuple[str, ...] = Field(min_length=1)
    applicability: str = Field(min_length=1)
    qualification_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    depends_on: tuple[str, ...] = ()

    @model_validator(mode="after")
    def capability_entries_must_be_unique(self) -> CapabilitySpec:
        if len(set(self.output_metric_ids)) != len(self.output_metric_ids):
            raise ValueError("capability output metrics must be unique")
        if len(set(self.depends_on)) != len(self.depends_on):
            raise ValueError("capability dependencies must be unique")
        return self


class CapabilityCatalog(AstroModel):
    catalog_id: str = Field(min_length=1)
    capabilities: tuple[CapabilitySpec, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def catalog_must_be_well_formed(self) -> CapabilityCatalog:
        ids = [item.capability_id for item in self.capabilities]
        if len(set(ids)) != len(ids):
            raise ValueError("capability IDs must be unique")
        known = set(ids)
        if any(not set(item.depends_on).issubset(known) for item in self.capabilities):
            raise ValueError("capability dependencies must exist in the catalog")
        _topological_order({item.capability_id: item.depends_on for item in self.capabilities})
        return self


class MissionDesignDirectorSpec(AstroModel):
    schema_version: Literal["1.0"] = "1.0"
    base_scenario_path: str = Field(min_length=1)
    intent: MissionIntent
    requirement_graph: RequirementGraph
    capability_catalog: CapabilityCatalog
    objective: MissionObjective
    authority: AuthorityGrant

    @model_validator(mode="after")
    def contracts_must_align(self) -> MissionDesignDirectorSpec:
        capabilities = {
            item.capability_id: item for item in self.capability_catalog.capabilities
        }
        if set(capabilities) != set(self.intent.allowed_capability_ids):
            raise ValueError("mission intent capability allow-list must match the catalog")
        if self.intent.authority_grant_id != self.authority.grant_id:
            raise ValueError("mission intent must bind the exact authority grant")
        goals = {item.metric_id: item for item in self.objective.metric_goals}
        for requirement in self.requirement_graph.requirements:
            goal = goals.get(requirement.metric_id)
            if goal is None:
                raise ValueError(
                    f"requirement metric {requirement.metric_id!r} is not an objective goal"
                )
            if goal.unit != requirement.unit:
                raise ValueError(
                    f"requirement metric {requirement.metric_id!r} has inconsistent units"
                )
            capability = capabilities.get(requirement.verification_capability_id)
            if capability is None or requirement.metric_id not in capability.output_metric_ids:
                raise ValueError(
                    f"requirement {requirement.requirement_id!r} lacks a producing capability"
                )
        if set(goals) != {item.metric_id for item in self.requirement_graph.requirements}:
            raise ValueError("objective goals must map exactly to first-slice requirements")
        if self.authority.level.value not in {"research", "decision_support"}:
            raise ValueError("first-slice design direction permits research authority only")
        if set(self.authority.allowed_actions) != {
            OperatorActionKind.EVALUATE_CANDIDATE,
            OperatorActionKind.FINISH,
        }:
            raise ValueError("first-slice design authority must allow evaluate and finish only")
        return self


class AnalysisNode(AstroModel):
    node_id: str = Field(min_length=1)
    capability_id: str = Field(min_length=1)
    capability_version: str = Field(min_length=1)
    depends_on: tuple[str, ...] = ()
    requirement_ids: tuple[str, ...] = ()
    cost_units: StrictInt = Field(ge=1)


class AnalysisPlan(AstroModel):
    plan_id: str = Field(min_length=1)
    intent_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    requirement_graph_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    capability_catalog_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    nodes: tuple[AnalysisNode, ...] = Field(min_length=1)
    total_cost_units: StrictInt = Field(ge=1)
    max_cost_units: StrictInt = Field(ge=1)
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RequirementAssessment(AstroModel):
    assessment_id: str = Field(min_length=1)
    requirement_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    metric_id: str = Field(min_length=1)
    observed_value: FiniteFloat
    threshold: FiniteFloat
    comparator: RequirementComparator
    unit: str = Field(min_length=1)
    passed: bool
    hard: bool
    evidence_ids: tuple[str, ...] = Field(min_length=1)
    operator_action_id: str = Field(min_length=1)
    assessment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class DesignDecision(AstroModel):
    disposition: DesignDecisionDisposition
    selected_candidate_id: str | None = None
    rejected_candidate_ids: tuple[str, ...] = ()
    assessment_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    unresolved_requirement_ids: tuple[str, ...] = ()
    rationale: str = Field(min_length=1)
    operator_run_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class MissionBaseline(AstroModel):
    baseline_id: str = Field(min_length=1)
    version: StrictInt = Field(default=1, ge=1)
    candidate_id: str = Field(min_length=1)
    assignments: dict[str, FiniteFloat]
    decision_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    operator_run_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    claim_boundary: str = Field(min_length=1)
    baseline_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class VerificationCheck(AstroModel):
    requirement_id: str = Field(min_length=1)
    capability_id: str = Field(min_length=1)
    assessment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["passed", "failed"]


class VerificationPlan(AstroModel):
    baseline_id: str | None = None
    checks: tuple[VerificationCheck, ...] = Field(min_length=1)
    remaining_hard_requirement_ids: tuple[str, ...] = ()
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class MissionDesignRun(AstroModel):
    schema_version: Literal["1.0"] = "1.0"
    workflow: Literal["mission_design_director_v1"] = "mission_design_director_v1"
    spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    operator_run_path: Literal["operator/operator-run.json"] = "operator/operator-run.json"
    operator_run_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    intent: MissionIntent
    requirement_graph: RequirementGraph
    capability_catalog: CapabilityCatalog
    analysis_plan: AnalysisPlan
    consumed_analysis_cost_units: StrictInt = Field(ge=1)
    assessments: tuple[RequirementAssessment, ...] = Field(min_length=1)
    decision: DesignDecision
    baseline: MissionBaseline | None = None
    verification_plan: VerificationPlan
    claim_boundary: str = Field(min_length=1)
    run_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def lifecycle_screen_capability() -> CapabilitySpec:
    """Return the exact built-in composite lifecycle screening capability contract."""

    return CapabilitySpec(
        capability_id="astro.mission_lifecycle_screen",
        version="1.0",
        input_schema_sha256=_canonical_digest(MissionLifecycleScenario.model_json_schema()),
        output_schema_sha256=_canonical_digest(CandidateObservation.model_json_schema()),
        fidelity=FidelityClass.SCREENING,
        deterministic=True,
        cost_units=1,
        output_metric_ids=(
            "margin:deorbit:propellant_reserve",
            "margin:deorbit:entry_interface_altitude_error",
        ),
        applicability=(
            "Local integrated launch, orbit, digital-twin, deorbit, and reentry "
            "design screening for the declared lifecycle scenario."
        ),
        qualification_sha256=sha256(
            b"astro.mission_lifecycle_screen@1.0:checked-local-composite-screening"
        ).hexdigest(),
    )


def build_analysis_plan(spec: MissionDesignDirectorSpec) -> AnalysisPlan:
    """Compile a deterministic, budgeted capability DAG from the design contracts."""

    expected = lifecycle_screen_capability()
    if spec.capability_catalog.capabilities != (expected,):
        raise ValueError(
            "first-slice capability catalog must contain only the registered lifecycle capability"
        )
    catalog_by_id = {item.capability_id: item for item in spec.capability_catalog.capabilities}
    registered = catalog_by_id.get(expected.capability_id)
    if registered != expected:
        raise ValueError("capability catalog does not match the registered lifecycle capability")
    order = _topological_order(
        {item.capability_id: item.depends_on for item in spec.capability_catalog.capabilities}
    )
    requirements_by_capability: dict[str, list[str]] = {}
    for requirement in spec.requirement_graph.requirements:
        requirements_by_capability.setdefault(
            requirement.verification_capability_id, []
        ).append(requirement.requirement_id)
    nodes = tuple(
        AnalysisNode(
            node_id=f"analyze:{capability_id}",
            capability_id=capability_id,
            capability_version=catalog_by_id[capability_id].version,
            depends_on=tuple(
                f"analyze:{dependency}"
                for dependency in catalog_by_id[capability_id].depends_on
            ),
            requirement_ids=tuple(sorted(requirements_by_capability.get(capability_id, []))),
            cost_units=catalog_by_id[capability_id].cost_units,
        )
        for capability_id in order
    )
    total = sum(node.cost_units for node in nodes) * spec.authority.max_candidate_evaluations
    if total > spec.intent.max_analysis_cost_units:
        raise ValueError("analysis plan exceeds the mission intent cost budget")
    payload = {
        "plan_id": f"{spec.intent.intent_id}:plan",
        "intent_sha256": model_digest(spec.intent),
        "requirement_graph_sha256": model_digest(spec.requirement_graph),
        "capability_catalog_sha256": model_digest(spec.capability_catalog),
        "nodes": [node.model_dump(mode="json") for node in nodes],
        "total_cost_units": total,
        "max_cost_units": spec.intent.max_analysis_cost_units,
    }
    return AnalysisPlan.model_validate(
        {**payload, "plan_sha256": _canonical_digest(payload)}
    )


def build_mission_design_run(
    *,
    spec: MissionDesignDirectorSpec,
    operator_run: OperatorRun,
    spec_sha256: str,
    operator_run_sha256: str,
) -> MissionDesignRun:
    """Reduce a verified operator journal into a deterministic design decision bundle."""

    if operator_run.status != OperatorRunStatus.COMPLETED:
        raise ValueError("mission design direction requires a completed operator run")
    if operator_run.authority != spec.authority:
        raise ValueError("operator authority does not match the design specification")
    if operator_run.objective != spec.objective.model_copy(
        update={"base_evidence": operator_run.objective.base_evidence}
    ):
        raise ValueError("operator objective does not match the design specification")
    selected_id = operator_run.selected_candidate_id
    if selected_id is None:
        raise ValueError("mission design direction requires an explicit candidate selection")
    selected_action_id: str | None = None
    selected_observation: CandidateObservation | None = None
    evaluated: list[str] = []
    for step in operator_run.steps:
        if step.observation is None:
            continue
        evaluated.append(step.observation.candidate.candidate_id)
        if step.observation.candidate.candidate_id == selected_id:
            selected_observation = step.observation
            selected_action_id = step.action.action_id
    if selected_observation is None or selected_action_id is None:
        raise ValueError("selected design candidate lacks an observation")
    if selected_observation.evaluation_status != "evaluated":
        raise ValueError("selected design candidate lacks completed analysis evidence")
    metrics = {item.metric_id: item for item in selected_observation.metrics}
    evidence_ids = tuple(item.evidence_id for item in selected_observation.evidence)
    assessments: list[RequirementAssessment] = []
    for requirement in spec.requirement_graph.requirements:
        metric = metrics.get(requirement.metric_id)
        if metric is None:
            raise ValueError(f"selected candidate lacks metric {requirement.metric_id!r}")
        if metric.unit != requirement.unit:
            raise ValueError(f"selected candidate metric {metric.metric_id!r} has wrong unit")
        passed = (
            metric.value >= requirement.threshold
            if requirement.comparator == RequirementComparator.GREATER_THAN_OR_EQUAL
            else metric.value <= requirement.threshold
        )
        assessment_payload = {
            "assessment_id": f"assessment:{selected_id}:{requirement.requirement_id}",
            "requirement_id": requirement.requirement_id,
            "candidate_id": selected_id,
            "metric_id": requirement.metric_id,
            "observed_value": metric.value,
            "threshold": requirement.threshold,
            "comparator": requirement.comparator,
            "unit": requirement.unit,
            "passed": passed,
            "hard": requirement.hard,
            "evidence_ids": evidence_ids,
            "operator_action_id": selected_action_id,
        }
        assessments.append(
            RequirementAssessment.model_validate(
                {
                    **assessment_payload,
                    "assessment_sha256": _canonical_digest(assessment_payload),
                }
            )
        )
    unresolved = tuple(
        item.requirement_id for item in assessments if item.hard and not item.passed
    )
    disposition = (
        DesignDecisionDisposition.ABSTAINED
        if unresolved or not selected_observation.passed
        else DesignDecisionDisposition.SELECTED
    )
    decision_payload = {
        "disposition": disposition,
        "selected_candidate_id": selected_id if disposition == "selected" else None,
        "rejected_candidate_ids": tuple(item for item in evaluated if item != selected_id),
        "assessment_ids": tuple(item.assessment_id for item in assessments),
        "evidence_ids": evidence_ids,
        "unresolved_requirement_ids": unresolved,
        "rationale": operator_run.conclusion,
        "operator_run_sha256": operator_run_sha256,
    }
    decision = DesignDecision.model_validate(
        {**decision_payload, "decision_sha256": _canonical_digest(decision_payload)}
    )
    baseline: MissionBaseline | None = None
    if disposition == DesignDecisionDisposition.SELECTED:
        baseline_payload = {
            "baseline_id": f"{spec.intent.intent_id}:baseline",
            "version": 1,
            "candidate_id": selected_id,
            "assignments": selected_observation.candidate.assignments,
            "decision_sha256": decision.decision_sha256,
            "operator_run_sha256": operator_run_sha256,
            "claim_boundary": spec.intent.claim_boundary,
        }
        baseline = MissionBaseline.model_validate(
            {**baseline_payload, "baseline_sha256": _canonical_digest(baseline_payload)}
        )
    checks = tuple(
        VerificationCheck(
            requirement_id=assessment.requirement_id,
            capability_id=next(
                item.verification_capability_id
                for item in spec.requirement_graph.requirements
                if item.requirement_id == assessment.requirement_id
            ),
            assessment_sha256=assessment.assessment_sha256,
            status="passed" if assessment.passed else "failed",
        )
        for assessment in assessments
    )
    verification_payload = {
        "baseline_id": baseline.baseline_id if baseline is not None else None,
        "checks": [item.model_dump(mode="json") for item in checks],
        "remaining_hard_requirement_ids": unresolved,
    }
    verification_plan = VerificationPlan.model_validate(
        {**verification_payload, "plan_sha256": _canonical_digest(verification_payload)}
    )
    plan = build_analysis_plan(spec)
    consumed_cost = sum(
        1 for step in operator_run.steps if step.observation is not None
    ) * sum(node.cost_units for node in plan.nodes)
    if consumed_cost > plan.total_cost_units:
        raise ValueError("operator run consumed more analysis cost than the reserved plan")
    run_payload = {
        "schema_version": "1.0",
        "workflow": "mission_design_director_v1",
        "spec_sha256": spec_sha256,
        "operator_run_path": "operator/operator-run.json",
        "operator_run_sha256": operator_run_sha256,
        "intent": spec.intent.model_dump(mode="json"),
        "requirement_graph": spec.requirement_graph.model_dump(mode="json"),
        "capability_catalog": spec.capability_catalog.model_dump(mode="json"),
        "analysis_plan": plan.model_dump(mode="json"),
        "consumed_analysis_cost_units": consumed_cost,
        "assessments": [item.model_dump(mode="json") for item in assessments],
        "decision": decision.model_dump(mode="json"),
        "baseline": baseline.model_dump(mode="json") if baseline is not None else None,
        "verification_plan": verification_plan.model_dump(mode="json"),
        "claim_boundary": spec.intent.claim_boundary,
    }
    return MissionDesignRun.model_validate(
        {**run_payload, "run_sha256": _canonical_digest(run_payload)}
    )


def _topological_order(dependencies: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    ordered: list[str] = []
    temporary: set[str] = set()
    permanent: set[str] = set()

    def visit(node: str) -> None:
        if node in permanent:
            return
        if node in temporary:
            raise ValueError("dependency graph must be acyclic")
        temporary.add(node)
        for dependency in sorted(dependencies[node]):
            visit(dependency)
        temporary.remove(node)
        permanent.add(node)
        ordered.append(node)

    for node in sorted(dependencies):
        visit(node)
    return tuple(ordered)


__all__ = [
    "AnalysisNode",
    "AnalysisPlan",
    "CapabilityCatalog",
    "CapabilitySpec",
    "DesignDecision",
    "DesignDecisionDisposition",
    "FidelityClass",
    "MissionBaseline",
    "MissionDesignDirectorSpec",
    "MissionDesignRun",
    "MissionIntent",
    "RequirementAssessment",
    "RequirementComparator",
    "RequirementGraph",
    "RequirementNode",
    "VerificationCheck",
    "VerificationPlan",
    "build_analysis_plan",
    "build_mission_design_run",
    "lifecycle_screen_capability",
]
