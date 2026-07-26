"""Deterministic, evidence-grounded mission knowledge read model."""

from __future__ import annotations

import json
from enum import StrEnum
from hashlib import sha256
from typing import Any, Literal

from pydantic import Field, JsonValue, model_validator

from astro_core.models import AstroModel
from astro_operator.director import MissionDesignRun
from astro_operator.models import (
    ClaimDisposition,
    EvidenceReference,
    OperatorActionKind,
    OperatorRun,
)


def _canonical_digest(value: object) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class KnowledgeSourceKind(StrEnum):
    MISSION_DESIGN_DIRECTOR = "mission_design_director"
    OPERATOR_RUN = "operator_run"


class KnowledgeSourceSpec(AstroModel):
    source_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
    kind: KnowledgeSourceKind
    path: str = Field(min_length=1)


class MissionKnowledgeGraphSpec(AstroModel):
    schema_version: Literal["1.0"] = "1.0"
    graph_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    mission_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    sources: tuple[KnowledgeSourceSpec, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def source_ids_must_be_unique(self) -> MissionKnowledgeGraphSpec:
        source_ids = [item.source_id for item in self.sources]
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("knowledge source IDs must be unique")
        return self


class KnowledgeNodeKind(StrEnum):
    MISSION = "mission"
    RUN = "run"
    INTENT = "intent"
    OBJECTIVE = "objective"
    REQUIREMENT = "requirement"
    CANDIDATE = "candidate"
    ASSESSMENT = "assessment"
    EVIDENCE = "evidence"
    ASSERTION = "assertion"
    CONFLICT = "conflict"
    CLAIM = "claim"
    DECISION = "decision"
    BASELINE = "baseline"
    TOOL = "tool"


class KnowledgeNode(AstroModel):
    node_id: str = Field(min_length=1)
    kind: KnowledgeNodeKind
    label: str = Field(min_length=1)
    source_id: str | None = None
    source_record_id: str = Field(min_length=1)
    properties: dict[str, JsonValue] = Field(default_factory=dict)


class KnowledgeEdge(AstroModel):
    edge_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_node_id: str = Field(min_length=1)
    relation: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    target_node_id: str = Field(min_length=1)
    source_id: str | None = None
    properties: dict[str, JsonValue] = Field(default_factory=dict)


class KnowledgeSource(AstroModel):
    source_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
    kind: KnowledgeSourceKind
    captured_path: str = Field(min_length=1)
    source_tree_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    workflow: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)


class MissionKnowledgeGraph(AstroModel):
    schema_version: Literal["1.0", "1.1"] = "1.0"
    extractor_version: Literal[
        "astro.mission_knowledge_graph/1.0",
        "astro.mission_knowledge_graph/1.1",
    ] = "astro.mission_knowledge_graph/1.0"
    graph_id: str = Field(min_length=1)
    mission_id: str = Field(min_length=1)
    sources: tuple[KnowledgeSource, ...] = Field(min_length=2)
    nodes: tuple[KnowledgeNode, ...] = Field(min_length=1)
    edges: tuple[KnowledgeEdge, ...]
    graph_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def graph_must_be_canonical_and_closed(self) -> MissionKnowledgeGraph:
        if self.extractor_version != f"astro.mission_knowledge_graph/{self.schema_version}":
            raise ValueError("knowledge graph schema and extractor versions must match")
        source_ids = [item.source_id for item in self.sources]
        node_ids = [item.node_id for item in self.nodes]
        edge_ids = [item.edge_id for item in self.edges]
        if source_ids != sorted(source_ids) or len(set(source_ids)) != len(source_ids):
            raise ValueError("knowledge sources must be unique and source-ID sorted")
        if node_ids != sorted(node_ids) or len(set(node_ids)) != len(node_ids):
            raise ValueError("knowledge nodes must be unique and node-ID sorted")
        if edge_ids != sorted(edge_ids) or len(set(edge_ids)) != len(edge_ids):
            raise ValueError("knowledge edges must be unique and edge-ID sorted")
        known_nodes = set(node_ids)
        if any(
            edge.source_node_id not in known_nodes or edge.target_node_id not in known_nodes
            for edge in self.edges
        ):
            raise ValueError("knowledge graph contains a dangling edge")
        known_sources = set(source_ids)
        if any(
            node.source_id is not None and node.source_id not in known_sources
            for node in self.nodes
        ) or any(
            edge.source_id is not None and edge.source_id not in known_sources
            for edge in self.edges
        ):
            raise ValueError("knowledge graph item references an unknown source")
        return self


class BaselineToolReference(AstroModel):
    tool_node_id: str = Field(min_length=1)
    tool_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    qualification_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class BaselineJustificationTrace(AstroModel):
    graph_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline_node_id: str = Field(min_length=1)
    decision_node_id: str = Field(min_length=1)
    selected_candidate_node_id: str = Field(min_length=1)
    requirement_node_ids: tuple[str, ...]
    assessment_node_ids: tuple[str, ...]
    evidence_node_ids: tuple[str, ...]
    tool_node_ids: tuple[str, ...]
    tool_references: tuple[BaselineToolReference, ...]
    claim_boundary: str = Field(min_length=1)


VerifiedSourcePayload = tuple[
    MissionDesignRun | OperatorRun,
    OperatorRun | None,
    str,
]


class _GraphCollector:
    def __init__(self, mission_id: str) -> None:
        self.nodes: dict[str, KnowledgeNode] = {}
        self.edges: dict[str, KnowledgeEdge] = {}
        self.mission_node_id = f"mission:{mission_id}"
        self.add_node(
            KnowledgeNode(
                node_id=self.mission_node_id,
                kind=KnowledgeNodeKind.MISSION,
                label=mission_id,
                source_record_id=mission_id,
                properties={"mission_id": mission_id},
            )
        )

    def add_node(self, node: KnowledgeNode) -> str:
        existing = self.nodes.get(node.node_id)
        if existing is not None and existing != node:
            raise ValueError(f"knowledge node collision: {node.node_id}")
        self.nodes[node.node_id] = node
        return node.node_id

    def node(
        self,
        source_id: str,
        kind: KnowledgeNodeKind,
        record_id: str,
        label: str,
        properties: dict[str, JsonValue],
    ) -> str:
        return self.add_node(
            KnowledgeNode(
                node_id=f"{source_id}:{kind.value}:{record_id}",
                kind=kind,
                label=label,
                source_id=source_id,
                source_record_id=record_id,
                properties=properties,
            )
        )

    def edge(
        self,
        source_node_id: str,
        relation: str,
        target_node_id: str,
        source_id: str | None,
        properties: dict[str, JsonValue] | None = None,
    ) -> None:
        payload: dict[str, JsonValue] = {
            "source_node_id": source_node_id,
            "relation": relation,
            "target_node_id": target_node_id,
            "source_id": source_id,
            "properties": properties or {},
        }
        edge = KnowledgeEdge(
            edge_id=_canonical_digest(payload),
            source_node_id=source_node_id,
            relation=relation,
            target_node_id=target_node_id,
            source_id=source_id,
            properties=properties or {},
        )
        existing = self.edges.get(edge.edge_id)
        if existing is not None and existing != edge:
            raise ValueError(f"knowledge edge collision: {edge.edge_id}")
        self.edges[edge.edge_id] = edge


def build_mission_knowledge_graph(
    spec: MissionKnowledgeGraphSpec,
    verified_sources: dict[str, VerifiedSourcePayload],
    *,
    schema_version: Literal["1.0", "1.1"] = "1.1",
) -> MissionKnowledgeGraph:
    """Build the canonical graph from already verified, captured source bundles."""

    if set(verified_sources) != {item.source_id for item in spec.sources}:
        raise ValueError("verified knowledge sources do not match the graph specification")
    if schema_version == "1.0" and any(
        isinstance(primary, OperatorRun) and primary.mission_context is not None
        for primary, _nested, _digest in verified_sources.values()
    ):
        raise ValueError("knowledge graph schema 1.0 cannot contain baseline-bound runs")
    collector = _GraphCollector(spec.mission_id)
    graph_sources: list[KnowledgeSource] = []
    for source_spec in sorted(spec.sources, key=lambda item: item.source_id):
        primary, nested_operator, tree_sha256 = verified_sources[source_spec.source_id]
        expected_kind = (
            KnowledgeSourceKind.MISSION_DESIGN_DIRECTOR
            if isinstance(primary, MissionDesignRun)
            else KnowledgeSourceKind.OPERATOR_RUN
        )
        if source_spec.kind != expected_kind:
            raise ValueError(
                f"knowledge source {source_spec.source_id!r} has the wrong verified type"
            )
        workflow = primary.workflow if isinstance(primary, MissionDesignRun) else "mission_operator"
        graph_sources.append(
            KnowledgeSource(
                source_id=source_spec.source_id,
                kind=source_spec.kind,
                captured_path=f"sources/{source_spec.source_id}",
                source_tree_sha256=tree_sha256,
                workflow=workflow,
                schema_version=primary.schema_version,
            )
        )
        if isinstance(primary, MissionDesignRun):
            if nested_operator is None:
                raise ValueError("Director knowledge source lacks its verified operator journal")
            _add_director_source(collector, source_spec.source_id, primary, nested_operator)
        else:
            if nested_operator is not None:
                raise ValueError("operator knowledge source contains an unexpected nested journal")
            _add_operator_source(
                collector,
                source_spec.source_id,
                primary,
                run_record_id="operator",
            )
    if schema_version == "1.1":
        _add_baseline_context_edges(collector, spec, verified_sources)
    payload: dict[str, Any] = {
        "schema_version": schema_version,
        "extractor_version": f"astro.mission_knowledge_graph/{schema_version}",
        "graph_id": spec.graph_id,
        "mission_id": spec.mission_id,
        "sources": [item.model_dump(mode="json") for item in graph_sources],
        "nodes": [
            item.model_dump(mode="json")
            for item in sorted(collector.nodes.values(), key=lambda item: item.node_id)
        ],
        "edges": [
            item.model_dump(mode="json")
            for item in sorted(collector.edges.values(), key=lambda item: item.edge_id)
        ],
    }
    return MissionKnowledgeGraph.model_validate(
        {**payload, "graph_sha256": _canonical_digest(payload)}
    )


def _add_director_source(
    collector: _GraphCollector,
    source_id: str,
    run: MissionDesignRun,
    operator_run: OperatorRun,
) -> None:
    run_node = collector.node(
        source_id,
        KnowledgeNodeKind.RUN,
        "director",
        run.workflow,
        {
            "workflow": run.workflow,
            "schema_version": run.schema_version,
            "run_sha256": run.run_sha256,
            "claim_boundary": run.claim_boundary,
        },
    )
    collector.edge(collector.mission_node_id, "contains", run_node, source_id)
    intent_node = collector.node(
        source_id,
        KnowledgeNodeKind.INTENT,
        run.intent.intent_id,
        run.intent.summary,
        run.intent.model_dump(mode="json"),
    )
    collector.edge(run_node, "declares", intent_node, source_id)
    for capability in run.capability_catalog.capabilities:
        tool_node = collector.node(
            source_id,
            KnowledgeNodeKind.TOOL,
            capability.capability_id,
            f"{capability.capability_id}@{capability.version}",
            capability.model_dump(mode="json"),
        )
        collector.edge(intent_node, "allows", tool_node, source_id)
    for requirement in run.requirement_graph.requirements:
        requirement_node = collector.node(
            source_id,
            KnowledgeNodeKind.REQUIREMENT,
            requirement.requirement_id,
            requirement.statement,
            requirement.model_dump(mode="json"),
        )
        collector.edge(intent_node, "requires", requirement_node, source_id)
        for parent_id in requirement.parent_ids:
            collector.edge(
                collector.node(
                    source_id,
                    KnowledgeNodeKind.REQUIREMENT,
                    parent_id,
                    next(
                        item.statement
                        for item in run.requirement_graph.requirements
                        if item.requirement_id == parent_id
                    ),
                    next(
                        item.model_dump(mode="json")
                        for item in run.requirement_graph.requirements
                        if item.requirement_id == parent_id
                    ),
                ),
                "parent_of",
                requirement_node,
                source_id,
            )
    operator_node = _add_operator_source(
        collector,
        source_id,
        operator_run,
        run_record_id="director-operator",
    )
    collector.edge(run_node, "derived_from", operator_node, source_id)
    decision_node = collector.node(
        source_id,
        KnowledgeNodeKind.DECISION,
        run.decision.decision_sha256,
        f"design decision: {run.decision.disposition.value}",
        run.decision.model_dump(mode="json"),
    )
    collector.edge(run_node, "produced", decision_node, source_id)
    candidate_ids = {
        step.observation.candidate.candidate_id
        for step in operator_run.steps
        if step.observation is not None
    }
    for assessment in run.assessments:
        assessment_node = collector.node(
            source_id,
            KnowledgeNodeKind.ASSESSMENT,
            assessment.assessment_id,
            assessment.requirement_id,
            assessment.model_dump(mode="json"),
        )
        requirement_node = f"{source_id}:requirement:{assessment.requirement_id}"
        candidate_node = f"{source_id}:candidate:{assessment.candidate_id}"
        collector.edge(requirement_node, "assessed_by", assessment_node, source_id)
        collector.edge(candidate_node, "assessed_by", assessment_node, source_id)
        collector.edge(assessment_node, "supports", decision_node, source_id)
        requirement = next(
            item
            for item in run.requirement_graph.requirements
            if item.requirement_id == assessment.requirement_id
        )
        collector.edge(
            f"{source_id}:tool:{requirement.verification_capability_id}",
            "produced",
            assessment_node,
            source_id,
        )
        for evidence_id in assessment.evidence_ids:
            collector.edge(
                f"{source_id}:evidence:{evidence_id}",
                "supports",
                assessment_node,
                source_id,
            )
    if (
        run.decision.selected_candidate_id is not None
        and run.decision.selected_candidate_id in candidate_ids
    ):
        collector.edge(
            decision_node,
            "selected",
            f"{source_id}:candidate:{run.decision.selected_candidate_id}",
            source_id,
        )
    for candidate_id in run.decision.rejected_candidate_ids:
        if candidate_id in candidate_ids:
            collector.edge(
                decision_node,
                "rejected",
                f"{source_id}:candidate:{candidate_id}",
                source_id,
            )
    for evidence_id in run.decision.evidence_ids:
        collector.edge(
            f"{source_id}:evidence:{evidence_id}",
            "informed",
            decision_node,
            source_id,
        )
    if run.baseline is not None:
        baseline_node = collector.node(
            source_id,
            KnowledgeNodeKind.BASELINE,
            run.baseline.baseline_id,
            run.baseline.baseline_id,
            run.baseline.model_dump(mode="json"),
        )
        collector.edge(decision_node, "established", baseline_node, source_id)


def _add_operator_source(
    collector: _GraphCollector,
    source_id: str,
    run: OperatorRun,
    *,
    run_record_id: str,
) -> str:
    run_node = collector.node(
        source_id,
        KnowledgeNodeKind.RUN,
        run_record_id,
        f"mission operator {run.status.value}",
        {
            "schema_version": run.schema_version,
            "status": run.status.value,
            "selected_candidate_id": run.selected_candidate_id,
            "conclusion": run.conclusion,
            "authority": run.authority.model_dump(mode="json"),
            **(
                {"mission_context": run.mission_context.model_dump(mode="json")}
                if run.mission_context is not None
                else {}
            ),
        },
    )
    collector.edge(collector.mission_node_id, "contains", run_node, source_id)
    objective_node = collector.node(
        source_id,
        KnowledgeNodeKind.OBJECTIVE,
        run.objective.objective_id,
        run.objective.summary,
        run.objective.model_dump(mode="json", exclude={"base_evidence", "base_assertions"}),
    )
    collector.edge(run_node, "declares", objective_node, source_id)
    evidence_nodes = _add_evidence_nodes(collector, source_id, run.known_evidence)
    for evidence_node in evidence_nodes.values():
        collector.edge(run_node, "contains", evidence_node, source_id)
    assertion_nodes: dict[str, str] = {}
    if run.world_state is not None:
        for assertion in run.world_state.assertions:
            assertion_node = collector.node(
                source_id,
                KnowledgeNodeKind.ASSERTION,
                assertion.assertion_id,
                f"{assertion.subject} {assertion.predicate}",
                assertion.model_dump(mode="json"),
            )
            assertion_nodes[assertion.assertion_id] = assertion_node
            for evidence_id in assertion.source_evidence_ids:
                collector.edge(
                    evidence_nodes[evidence_id],
                    "supports",
                    assertion_node,
                    source_id,
                )
        for conflict in run.world_state.conflicts:
            conflict_node = collector.node(
                source_id,
                KnowledgeNodeKind.CONFLICT,
                conflict.conflict_id,
                f"conflict: {conflict.subject} {conflict.predicate}",
                conflict.model_dump(mode="json"),
            )
            for assertion_id in conflict.assertion_ids:
                collector.edge(
                    assertion_nodes[assertion_id],
                    "participates_in",
                    conflict_node,
                    source_id,
                )
    for step in run.steps:
        if step.observation is not None:
            candidate = step.observation.candidate
            candidate_node = collector.node(
                source_id,
                KnowledgeNodeKind.CANDIDATE,
                candidate.candidate_id,
                candidate.description or candidate.candidate_id,
                candidate.model_dump(mode="json"),
            )
            collector.edge(objective_node, "evaluated", candidate_node, source_id)
            for evidence in step.observation.evidence:
                collector.edge(
                    evidence_nodes[evidence.evidence_id],
                    "describes",
                    candidate_node,
                    source_id,
                )
        if step.acquisition_result is not None:
            tool = step.acquisition_result.tool
            tool_node = collector.node(
                source_id,
                KnowledgeNodeKind.TOOL,
                tool.tool_id,
                f"{tool.tool_id}@{tool.version}",
                tool.model_dump(mode="json"),
            )
            for evidence in step.acquisition_result.evidence:
                collector.edge(
                    tool_node,
                    "produced",
                    evidence_nodes[evidence.evidence_id],
                    source_id,
                )
            for assertion in step.acquisition_result.assertions:
                collector.edge(
                    tool_node,
                    "produced",
                    assertion_nodes[assertion.assertion_id],
                    source_id,
                )
        if step.action.kind == OperatorActionKind.FINISH:
            decision_node = collector.node(
                source_id,
                KnowledgeNodeKind.DECISION,
                step.action.action_id,
                step.action.conclusion or step.action.action_id,
                step.action.model_dump(mode="json"),
            )
            collector.edge(run_node, "produced", decision_node, source_id)
            for evidence_id in step.action.evidence_ids:
                collector.edge(
                    evidence_nodes[evidence_id],
                    "informed",
                    decision_node,
                    source_id,
                )
            for claim in step.action.conclusion_claims:
                claim_node = collector.node(
                    source_id,
                    KnowledgeNodeKind.CLAIM,
                    claim.claim_id,
                    claim.statement,
                    claim.model_dump(mode="json"),
                )
                claim_relation = _claim_decision_relation(claim.disposition)
                collector.edge(claim_node, claim_relation, decision_node, source_id)
                for assertion_id in claim.assertion_ids:
                    collector.edge(
                        claim_node,
                        "cites",
                        assertion_nodes[assertion_id],
                        source_id,
                    )
            if step.action.selected_candidate_id is not None:
                collector.edge(
                    decision_node,
                    "selected",
                    f"{source_id}:candidate:{step.action.selected_candidate_id}",
                    source_id,
                )
    return run_node


def _add_baseline_context_edges(
    collector: _GraphCollector,
    spec: MissionKnowledgeGraphSpec,
    verified_sources: dict[str, VerifiedSourcePayload],
) -> None:
    directors = [
        (source_id, primary)
        for source_id, (primary, _nested, _digest) in verified_sources.items()
        if isinstance(primary, MissionDesignRun)
    ]
    for source_id, (primary, nested, _digest) in verified_sources.items():
        if not isinstance(primary, OperatorRun) or nested is not None:
            continue
        context = primary.mission_context
        if context is None:
            continue
        if context.mission_id != spec.mission_id:
            raise ValueError(
                f"operator source {source_id!r} mission context does not match the graph"
            )
        matches = [
            (director_source_id, director)
            for director_source_id, director in directors
            if director.run_sha256 == context.mission_design_run_sha256
        ]
        if len(matches) != 1:
            raise ValueError(
                f"operator source {source_id!r} must match exactly one Director run"
            )
        director_source_id, director = matches[0]
        baseline = director.baseline
        if (
            baseline is None
            or baseline.baseline_id != context.baseline_id
            or baseline.version != context.baseline_version
            or baseline.baseline_sha256 != context.baseline_sha256
            or director.decision.disposition.value != "selected"
            or director.verification_plan.baseline_id != baseline.baseline_id
            or director.verification_plan.remaining_hard_requirement_ids
            or any(check.status != "passed" for check in director.verification_plan.checks)
        ):
            raise ValueError(
                f"operator source {source_id!r} baseline context does not match "
                "an eligible Director baseline"
            )
        collector.edge(
            f"{source_id}:run:operator",
            "operates_against",
            f"{director_source_id}:baseline:{baseline.baseline_id}",
            source_id,
            properties=context.model_dump(mode="json"),
        )


def _add_evidence_nodes(
    collector: _GraphCollector,
    source_id: str,
    evidence_references: tuple[EvidenceReference, ...],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for evidence in evidence_references:
        result[evidence.evidence_id] = collector.node(
            source_id,
            KnowledgeNodeKind.EVIDENCE,
            evidence.evidence_id,
            evidence.kind,
            evidence.model_dump(mode="json"),
        )
    return result


def _claim_decision_relation(disposition: ClaimDisposition) -> str:
    return {
        ClaimDisposition.SUPPORTED: "supports",
        ClaimDisposition.QUALIFIED: "qualifies",
        ClaimDisposition.DISPUTED: "disputes",
    }[disposition]


def trace_baseline_justification(
    graph: MissionKnowledgeGraph,
    baseline_id: str,
) -> BaselineJustificationTrace:
    """Return the checked evidence/tool chain behind one unambiguous baseline."""

    baselines = [
        node
        for node in graph.nodes
        if node.kind == KnowledgeNodeKind.BASELINE
        and (node.source_record_id == baseline_id or node.node_id == baseline_id)
    ]
    if len(baselines) != 1:
        raise ValueError(f"baseline query must identify exactly one node; matched {len(baselines)}")
    baseline = baselines[0]
    decision_edges = [
        edge
        for edge in graph.edges
        if edge.relation == "established" and edge.target_node_id == baseline.node_id
    ]
    if len(decision_edges) != 1:
        raise ValueError("baseline must have exactly one establishing decision")
    decision_id = decision_edges[0].source_node_id
    selected = [
        edge.target_node_id
        for edge in graph.edges
        if edge.source_node_id == decision_id and edge.relation == "selected"
    ]
    if len(selected) != 1:
        raise ValueError("baseline decision must select exactly one candidate")
    assessments = sorted(
        edge.source_node_id
        for edge in graph.edges
        if edge.target_node_id == decision_id
        and edge.relation == "supports"
        and next(node for node in graph.nodes if node.node_id == edge.source_node_id).kind
        == KnowledgeNodeKind.ASSESSMENT
    )
    assessment_set = set(assessments)
    requirements = sorted(
        edge.source_node_id
        for edge in graph.edges
        if edge.target_node_id in assessment_set
        and edge.relation == "assessed_by"
        and next(node for node in graph.nodes if node.node_id == edge.source_node_id).kind
        == KnowledgeNodeKind.REQUIREMENT
    )
    evidence = sorted(
        {
            edge.source_node_id
            for edge in graph.edges
            if (edge.target_node_id in assessment_set and edge.relation == "supports")
            or (edge.target_node_id == decision_id and edge.relation == "informed")
            if next(node for node in graph.nodes if node.node_id == edge.source_node_id).kind
            == KnowledgeNodeKind.EVIDENCE
        }
    )
    tools = sorted(
        {
            edge.source_node_id
            for edge in graph.edges
            if edge.target_node_id in assessment_set
            and edge.relation == "produced"
            and next(node for node in graph.nodes if node.node_id == edge.source_node_id).kind
            == KnowledgeNodeKind.TOOL
        }
    )
    nodes_by_id = {node.node_id: node for node in graph.nodes}
    tool_references: list[BaselineToolReference] = []
    for tool_node_id in tools:
        properties = nodes_by_id[tool_node_id].properties
        tool_id = properties.get("capability_id")
        version = properties.get("version")
        qualification_sha256 = properties.get("qualification_sha256")
        if not all(
            isinstance(value, str) and value for value in (tool_id, version, qualification_sha256)
        ):
            raise ValueError("baseline capability lacks exact versioned qualification identity")
        assert isinstance(tool_id, str)
        assert isinstance(version, str)
        assert isinstance(qualification_sha256, str)
        tool_references.append(
            BaselineToolReference(
                tool_node_id=tool_node_id,
                tool_id=tool_id,
                version=version,
                qualification_sha256=qualification_sha256,
            )
        )
    claim_boundary = baseline.properties.get("claim_boundary")
    if not isinstance(claim_boundary, str) or not claim_boundary:
        raise ValueError("baseline lacks its claim boundary")
    return BaselineJustificationTrace(
        graph_sha256=graph.graph_sha256,
        baseline_node_id=baseline.node_id,
        decision_node_id=decision_id,
        selected_candidate_node_id=selected[0],
        requirement_node_ids=tuple(requirements),
        assessment_node_ids=tuple(assessments),
        evidence_node_ids=tuple(evidence),
        tool_node_ids=tuple(tools),
        tool_references=tuple(tool_references),
        claim_boundary=claim_boundary,
    )


__all__ = [
    "BaselineJustificationTrace",
    "BaselineToolReference",
    "KnowledgeEdge",
    "KnowledgeNode",
    "KnowledgeNodeKind",
    "KnowledgeSource",
    "KnowledgeSourceKind",
    "KnowledgeSourceSpec",
    "MissionKnowledgeGraph",
    "MissionKnowledgeGraphSpec",
    "build_mission_knowledge_graph",
    "trace_baseline_justification",
]
