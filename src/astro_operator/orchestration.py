"""Checked cross-run orchestration dispositions over a verified mission graph."""

from __future__ import annotations

import json
from enum import StrEnum
from hashlib import sha256
from typing import Literal

from pydantic import Field, model_validator

from astro_core.models import AstroModel
from astro_operator.knowledge import (
    KnowledgeNode,
    KnowledgeNodeKind,
    MissionKnowledgeGraph,
)
from astro_operator.models import (
    ApplicabilityPredicate,
    ClaimDisposition,
    ConclusionClaim,
    EvidenceAssertion,
    ExactValuePredicate,
    FreshnessPredicate,
    MissionBaselineContext,
    NumericThresholdPredicate,
    OperatorActionKind,
)
from astro_operator.world_state import evaluate_claim_predicate


class OrchestrationDisposition(StrEnum):
    CONTINUE = "continue"
    HOLD = "hold"
    ABSTAIN = "abstain"


class OrchestrationReason(StrEnum):
    ALL_MANUAL_REVIEW_GATES_SATISFIED = "all_manual_review_gates_satisfied"
    BASELINE_BINDING_MISMATCH = "baseline_binding_mismatch"
    TARGET_OPERATOR_MISSING_OR_AMBIGUOUS = "target_operator_missing_or_ambiguous"
    OPERATOR_RUN_INCOMPLETE = "operator_run_incomplete"
    OPERATIONAL_AUTHORITY_PRESENT = "operational_authority_present"
    TARGET_CLAIM_MISSING_OR_AMBIGUOUS = "target_claim_missing_or_ambiguous"
    CLAIM_QUALIFIED = "claim_qualified"
    CLAIM_DISPUTED = "claim_disputed"
    PREDICATE_CHECK_ERROR = "predicate_check_error"
    APPLICABILITY_PREDICATE_FAILED = "applicability_predicate_failed"
    APPLICABILITY_ASSERTION_CONFLICT = "applicability_assertion_conflict"
    READINESS_PREDICATE_FAILED = "readiness_predicate_failed"
    READINESS_ASSERTION_CONFLICT = "readiness_assertion_conflict"
    MANUAL_REVIEW_GATE_MISSING_OR_INVALID = "manual_review_gate_missing_or_invalid"


class MissionOrchestrationQuery(AstroModel):
    schema_version: Literal["1.0"] = "1.0"
    baseline_id: str = Field(min_length=1)
    operator_objective_id: str = Field(min_length=1)
    disposition_claim_id: str = Field(min_length=1)
    manual_review_gate_predicate_id: str = Field(min_length=1)
    disposition_scope: Literal["manual_review_readiness"] = "manual_review_readiness"


class PredicateCheck(AstroModel):
    predicate_id: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    passed: bool
    assertion_ids: tuple[str, ...]
    conflict_ids: tuple[str, ...]


class MissionOrchestrationDecision(AstroModel):
    schema_version: Literal["1.0"] = "1.0"
    graph_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    query: MissionOrchestrationQuery
    disposition: OrchestrationDisposition
    baseline_node_id: str | None = None
    operator_run_node_id: str | None = None
    claim_node_id: str | None = None
    predicate_checks: tuple[PredicateCheck, ...] = ()
    relevant_conflict_ids: tuple[str, ...] = ()
    reason_codes: tuple[OrchestrationReason, ...] = Field(min_length=1)
    authority_boundary: Literal[
        "manual_review_routing_only_no_command_approval_execution_or_operational_authority"
    ] = "manual_review_routing_only_no_command_approval_execution_or_operational_authority"
    decision_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def decision_must_be_canonical_and_digest_bound(
        self,
    ) -> MissionOrchestrationDecision:
        predicate_ids = [item.predicate_id for item in self.predicate_checks]
        if predicate_ids != sorted(predicate_ids) or len(set(predicate_ids)) != len(
            predicate_ids
        ):
            raise ValueError("predicate checks must be unique and predicate-ID-sorted")
        if list(self.relevant_conflict_ids) != sorted(
            self.relevant_conflict_ids
        ) or len(set(self.relevant_conflict_ids)) != len(self.relevant_conflict_ids):
            raise ValueError("relevant conflict IDs must be unique and sorted")
        reason_values = [item.value for item in self.reason_codes]
        if reason_values != sorted(reason_values) or len(set(reason_values)) != len(
            reason_values
        ):
            raise ValueError("reason codes must be unique and sorted")
        payload = self.model_dump(mode="json", exclude={"decision_sha256"})
        if self.decision_sha256 != _canonical_digest(payload):
            raise ValueError("mission orchestration decision digest mismatch")
        return self


def evaluate_mission_orchestration(
    graph: MissionKnowledgeGraph,
    query: MissionOrchestrationQuery,
) -> MissionOrchestrationDecision:
    """Reduce one baseline-bound checked claim into a non-executing route."""

    graph_payload = graph.model_dump(mode="json", exclude={"graph_sha256"})
    if graph.graph_sha256 != _canonical_digest(graph_payload):
        raise ValueError("mission knowledge graph digest mismatch")
    nodes = {node.node_id: node for node in graph.nodes}
    baselines = [
        node
        for node in graph.nodes
        if node.kind == KnowledgeNodeKind.BASELINE
        and (node.source_record_id == query.baseline_id or node.node_id == query.baseline_id)
    ]
    if len(baselines) != 1:
        return _decision(
            graph,
            query,
            OrchestrationDisposition.ABSTAIN,
            (OrchestrationReason.BASELINE_BINDING_MISMATCH,),
        )
    baseline = baselines[0]
    run_candidates: list[KnowledgeNode] = []
    for edge in graph.edges:
        if edge.relation != "operates_against" or edge.target_node_id != baseline.node_id:
            continue
        run_node = nodes[edge.source_node_id]
        declares = [
            nodes[item.target_node_id]
            for item in graph.edges
            if item.source_node_id == run_node.node_id
            and item.relation == "declares"
            and nodes[item.target_node_id].kind == KnowledgeNodeKind.OBJECTIVE
        ]
        if any(item.source_record_id == query.operator_objective_id for item in declares):
            run_candidates.append(run_node)
    if len(run_candidates) != 1:
        return _decision(
            graph,
            query,
            OrchestrationDisposition.ABSTAIN,
            (OrchestrationReason.TARGET_OPERATOR_MISSING_OR_AMBIGUOUS,),
            baseline=baseline,
        )
    run_node = run_candidates[0]
    try:
        context = MissionBaselineContext.model_validate(run_node.properties["mission_context"])
    except (KeyError, ValueError):
        return _decision(
            graph,
            query,
            OrchestrationDisposition.ABSTAIN,
            (OrchestrationReason.BASELINE_BINDING_MISMATCH,),
            baseline=baseline,
            run=run_node,
        )
    baseline_sha256 = baseline.properties.get("baseline_sha256")
    baseline_version = baseline.properties.get("version")
    if (
        context.mission_id != graph.mission_id
        or context.baseline_id != baseline.source_record_id
        or context.baseline_sha256 != baseline_sha256
        or context.baseline_version != baseline_version
    ):
        return _decision(
            graph,
            query,
            OrchestrationDisposition.ABSTAIN,
            (OrchestrationReason.BASELINE_BINDING_MISMATCH,),
            baseline=baseline,
            run=run_node,
        )
    if (
        run_node.properties.get("status") != "completed"
        or run_node.properties.get("schema_version") != "1.4"
    ):
        return _decision(
            graph,
            query,
            OrchestrationDisposition.ABSTAIN,
            (OrchestrationReason.OPERATOR_RUN_INCOMPLETE,),
            baseline=baseline,
            run=run_node,
        )
    authority = run_node.properties.get("authority")
    allowed_actions = authority.get("allowed_actions") if isinstance(authority, dict) else None
    if not isinstance(allowed_actions, list) or any(
        item
        in {
            OperatorActionKind.PROPOSE_COMMAND.value,
            OperatorActionKind.EXECUTE_COMMAND.value,
        }
        for item in allowed_actions
    ):
        return _decision(
            graph,
            query,
            OrchestrationDisposition.ABSTAIN,
            (OrchestrationReason.OPERATIONAL_AUTHORITY_PRESENT,),
            baseline=baseline,
            run=run_node,
        )
    decisions = [
        nodes[edge.target_node_id]
        for edge in graph.edges
        if edge.source_node_id == run_node.node_id
        and edge.relation == "produced"
        and nodes[edge.target_node_id].kind == KnowledgeNodeKind.DECISION
    ]
    claim_nodes = [
        node
        for node in graph.nodes
        if node.kind == KnowledgeNodeKind.CLAIM
        and node.source_id == run_node.source_id
        and node.source_record_id == query.disposition_claim_id
        and any(
            edge.source_node_id == node.node_id
            and edge.target_node_id in {item.node_id for item in decisions}
            and edge.relation in {"supports", "qualifies", "disputes"}
            for edge in graph.edges
        )
    ]
    if len(claim_nodes) != 1:
        return _decision(
            graph,
            query,
            OrchestrationDisposition.ABSTAIN,
            (OrchestrationReason.TARGET_CLAIM_MISSING_OR_AMBIGUOUS,),
            baseline=baseline,
            run=run_node,
        )
    claim_node = claim_nodes[0]
    try:
        claim = ConclusionClaim.model_validate(claim_node.properties)
        cited_nodes = [
            nodes[edge.target_node_id]
            for edge in graph.edges
            if edge.source_node_id == claim_node.node_id and edge.relation == "cites"
        ]
        assertions = {
            node.source_record_id: EvidenceAssertion.model_validate(node.properties)
            for node in cited_nodes
            if node.kind == KnowledgeNodeKind.ASSERTION
        }
        if set(assertions) != set(claim.assertion_ids):
            raise ValueError("claim citation inventory is incomplete")
    except (KeyError, ValueError):
        return _decision(
            graph,
            query,
            OrchestrationDisposition.ABSTAIN,
            (OrchestrationReason.PREDICATE_CHECK_ERROR,),
            baseline=baseline,
            run=run_node,
            claim=claim_node,
        )
    conflicts_by_assertion: dict[str, set[str]] = {}
    for edge in graph.edges:
        if edge.relation != "participates_in":
            continue
        assertion = nodes[edge.source_node_id]
        if assertion.source_record_id in assertions:
            conflicts_by_assertion.setdefault(assertion.source_record_id, set()).add(
                nodes[edge.target_node_id].source_record_id
            )
    checks: list[PredicateCheck] = []
    try:
        for predicate in claim.predicates:
            assertion_ids = _predicate_assertion_ids(predicate)
            checks.append(
                PredicateCheck(
                    predicate_id=predicate.predicate_id,
                    kind=predicate.kind,
                    passed=evaluate_claim_predicate(predicate, assertions),
                    assertion_ids=tuple(sorted(assertion_ids)),
                    conflict_ids=tuple(
                        sorted(
                            {
                                conflict_id
                                for assertion_id in assertion_ids
                                for conflict_id in conflicts_by_assertion.get(assertion_id, set())
                            }
                        )
                    ),
                )
            )
    except (KeyError, TypeError, ValueError):
        return _decision(
            graph,
            query,
            OrchestrationDisposition.ABSTAIN,
            (OrchestrationReason.PREDICATE_CHECK_ERROR,),
            baseline=baseline,
            run=run_node,
            claim=claim_node,
            checks=checks,
        )
    configuration_pairs = [
        predicate
        for predicate in claim.predicates
        if isinstance(predicate, ApplicabilityPredicate)
        and predicate.actual_assertion_predicate.endswith("configuration_id")
        and predicate.required_assertion_predicate.endswith("configuration_id")
    ]
    if not configuration_pairs or any(
        assertions[predicate.actual_assertion_id].value != context.operational_configuration_id
        or assertions[predicate.required_assertion_id].value != context.operational_configuration_id
        for predicate in configuration_pairs
    ):
        return _decision(
            graph,
            query,
            OrchestrationDisposition.ABSTAIN,
            (OrchestrationReason.BASELINE_BINDING_MISMATCH,),
            baseline=baseline,
            run=run_node,
            claim=claim_node,
            checks=checks,
        )
    applicability = [item for item in checks if item.kind == "applicability"]
    if any(item.conflict_ids for item in applicability):
        return _decision(
            graph,
            query,
            OrchestrationDisposition.ABSTAIN,
            (OrchestrationReason.APPLICABILITY_ASSERTION_CONFLICT,),
            baseline=baseline,
            run=run_node,
            claim=claim_node,
            checks=checks,
        )
    if any(not item.passed for item in applicability):
        return _decision(
            graph,
            query,
            OrchestrationDisposition.ABSTAIN,
            (OrchestrationReason.APPLICABILITY_PREDICATE_FAILED,),
            baseline=baseline,
            run=run_node,
            claim=claim_node,
            checks=checks,
        )
    gates = [
        predicate
        for predicate in claim.predicates
        if predicate.predicate_id == query.manual_review_gate_predicate_id
    ]
    if (
        len(gates) != 1
        or not isinstance(gates[0], ExactValuePredicate)
        or gates[0].expected_value is not True
    ):
        return _decision(
            graph,
            query,
            OrchestrationDisposition.ABSTAIN,
            (OrchestrationReason.MANUAL_REVIEW_GATE_MISSING_OR_INVALID,),
            baseline=baseline,
            run=run_node,
            claim=claim_node,
            checks=checks,
        )
    readiness_conflicts = {
        conflict_id
        for item in checks
        if item.kind != "applicability"
        for conflict_id in item.conflict_ids
    }
    reasons: list[OrchestrationReason] = []
    if readiness_conflicts:
        reasons.append(OrchestrationReason.READINESS_ASSERTION_CONFLICT)
    if any(not item.passed for item in checks if item.kind != "applicability"):
        reasons.append(OrchestrationReason.READINESS_PREDICATE_FAILED)
    if claim.disposition == ClaimDisposition.QUALIFIED:
        reasons.append(OrchestrationReason.CLAIM_QUALIFIED)
    elif claim.disposition == ClaimDisposition.DISPUTED:
        reasons.append(OrchestrationReason.CLAIM_DISPUTED)
    if reasons:
        return _decision(
            graph,
            query,
            OrchestrationDisposition.HOLD,
            tuple(reasons),
            baseline=baseline,
            run=run_node,
            claim=claim_node,
            checks=checks,
        )
    return _decision(
        graph,
        query,
        OrchestrationDisposition.CONTINUE,
        (OrchestrationReason.ALL_MANUAL_REVIEW_GATES_SATISFIED,),
        baseline=baseline,
        run=run_node,
        claim=claim_node,
        checks=checks,
    )


def _predicate_assertion_ids(
    predicate: (
        NumericThresholdPredicate
        | FreshnessPredicate
        | ApplicabilityPredicate
        | ExactValuePredicate
    ),
) -> set[str]:
    if isinstance(predicate, NumericThresholdPredicate):
        result = {predicate.assertion_id}
        if predicate.threshold_assertion_id is not None:
            result.add(predicate.threshold_assertion_id)
        return result
    if isinstance(predicate, FreshnessPredicate):
        result = {predicate.assertion_id, predicate.reference_assertion_id}
        if predicate.max_age_assertion_id is not None:
            result.add(predicate.max_age_assertion_id)
        return result
    if isinstance(predicate, ApplicabilityPredicate):
        return {predicate.actual_assertion_id, predicate.required_assertion_id}
    return {predicate.assertion_id}


def _decision(
    graph: MissionKnowledgeGraph,
    query: MissionOrchestrationQuery,
    disposition: OrchestrationDisposition,
    reasons: tuple[OrchestrationReason, ...],
    *,
    baseline: KnowledgeNode | None = None,
    run: KnowledgeNode | None = None,
    claim: KnowledgeNode | None = None,
    checks: list[PredicateCheck] | None = None,
) -> MissionOrchestrationDecision:
    ordered_checks = tuple(sorted(checks or (), key=lambda item: item.predicate_id))
    payload = {
        "schema_version": "1.0",
        "graph_sha256": graph.graph_sha256,
        "query": query.model_dump(mode="json"),
        "disposition": disposition,
        "baseline_node_id": baseline.node_id if baseline is not None else None,
        "operator_run_node_id": run.node_id if run is not None else None,
        "claim_node_id": claim.node_id if claim is not None else None,
        "predicate_checks": [item.model_dump(mode="json") for item in ordered_checks],
        "relevant_conflict_ids": sorted(
            {conflict_id for check in ordered_checks for conflict_id in check.conflict_ids}
        ),
        "reason_codes": sorted(set(reasons), key=lambda item: item.value),
        "authority_boundary": (
            "manual_review_routing_only_no_command_approval_execution_or_operational_authority"
        ),
    }
    return MissionOrchestrationDecision.model_validate(
        {**payload, "decision_sha256": _canonical_digest(payload)}
    )


def _canonical_digest(value: object) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


__all__ = [
    "MissionOrchestrationDecision",
    "MissionOrchestrationQuery",
    "OrchestrationDisposition",
    "OrchestrationReason",
    "PredicateCheck",
    "evaluate_mission_orchestration",
]
