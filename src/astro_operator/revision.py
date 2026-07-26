"""Pure routing from verified campaign episodes to bounded Director handoffs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from enum import StrEnum
from hashlib import sha256
from typing import Literal

from pydantic import Field, StrictInt, model_validator

from astro_core.models import AstroModel
from astro_operator.knowledge import (
    KnowledgeNode,
    KnowledgeNodeKind,
    MissionKnowledgeGraph,
)


def _canonical_digest(value: object) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class MissionDesignRevisionAction(StrEnum):
    SUPPORT_RETENTION_WITHIN_DECLARED_SCOPE = "support_retention_within_declared_scope"
    OPEN_NEW_DIRECTOR_DECISION = "open_new_director_decision"
    ABSTAIN = "abstain"


class MissionDesignRevisionReason(StrEnum):
    VERIFIED_CAMPAIGN_SUPPORTS_RETENTION = "verified_campaign_supports_retention"
    VERIFIED_CAMPAIGN_REQUESTS_REVISION = "verified_campaign_requests_revision"
    VERIFIED_CAMPAIGN_INCONCLUSIVE = "verified_campaign_inconclusive"
    EPISODE_MISSING_OR_AMBIGUOUS = "episode_missing_or_ambiguous"
    BASELINE_MISSING_OR_AMBIGUOUS = "baseline_missing_or_ambiguous"
    EPISODE_BINDING_INVALID = "episode_binding_invalid"


class MissionDesignRevisionQuery(AstroModel):
    episode_id: str = Field(min_length=1)
    baseline_id: str = Field(min_length=1)


class DirectorRevisionRequest(AstroModel):
    schema_version: Literal["1.0"] = "1.0"
    request_id: str = Field(min_length=1)
    mission_id: str = Field(min_length=1)
    graph_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    verification_episode_node_id: str = Field(min_length=1)
    campaign_outcome_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    campaign_definition_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    campaign_claim_boundary: str = Field(min_length=1)
    prior_director_run_node_id: str = Field(min_length=1)
    prior_director_run_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prior_design_decision_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prior_baseline_id: str = Field(min_length=1)
    prior_baseline_version: StrictInt = Field(ge=1)
    prior_baseline_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prior_candidate_id: str = Field(min_length=1)
    failed_requirement_ids: tuple[str, ...] = Field(min_length=1)
    failed_gate_assessment_sha256s: tuple[str, ...] = Field(min_length=1)
    prior_intent_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prior_capability_catalog_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    capability_allowlist_envelope: tuple[str, ...] = Field(min_length=1)
    prior_analysis_cost_ceiling_units: StrictInt = Field(ge=1)
    prior_consumed_analysis_cost_units: StrictInt = Field(ge=1)
    completed_verification_cost_units: StrictInt = Field(ge=1)
    prior_recorded_cost_units: StrictInt = Field(ge=1)
    prior_authority_grant_id: str = Field(min_length=1)
    prior_authority_grant_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    authority_boundary: Literal[
        "requests_new_bounded_director_decision_does_not_select_or_mutate_baseline_"
        "and_requires_fresh_authority_and_budget"
    ] = (
        "requests_new_bounded_director_decision_does_not_select_or_mutate_baseline_"
        "and_requires_fresh_authority_and_budget"
    )
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def request_must_be_canonical(self) -> DirectorRevisionRequest:
        if self.failed_requirement_ids != tuple(sorted(set(self.failed_requirement_ids))):
            raise ValueError("failed requirement IDs must be unique and sorted")
        if self.failed_gate_assessment_sha256s != tuple(
            sorted(set(self.failed_gate_assessment_sha256s))
        ):
            raise ValueError("failed gate assessment digests must be unique and sorted")
        if self.capability_allowlist_envelope != tuple(
            sorted(set(self.capability_allowlist_envelope))
        ):
            raise ValueError("capability envelope must be unique and sorted")
        if self.prior_recorded_cost_units != (
            self.prior_consumed_analysis_cost_units + self.completed_verification_cost_units
        ):
            raise ValueError("prior recorded cost must include design and verification work")
        payload = self.model_dump(mode="json", exclude={"request_sha256"})
        if self.request_sha256 != _canonical_digest(payload):
            raise ValueError("Director revision request digest mismatch")
        return self


class MissionDesignRevisionRoute(AstroModel):
    schema_version: Literal["1.0"] = "1.0"
    graph_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    query: MissionDesignRevisionQuery
    verification_episode_node_id: str | None = None
    baseline_node_id: str | None = None
    disposition_node_id: str | None = None
    campaign_definition_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    campaign_claim_boundary: str | None = Field(default=None, min_length=1)
    action: MissionDesignRevisionAction
    reason: MissionDesignRevisionReason
    revision_request: DirectorRevisionRequest | None = None
    authority_boundary: Literal[
        "routing_only_no_baseline_selection_mutation_or_director_execution"
    ] = "routing_only_no_baseline_selection_mutation_or_director_execution"
    route_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def route_must_be_canonical(self) -> MissionDesignRevisionRoute:
        requires_request = self.action is MissionDesignRevisionAction.OPEN_NEW_DIRECTOR_DECISION
        if requires_request != (self.revision_request is not None):
            raise ValueError("only a new Director route may carry a revision request")
        if self.revision_request is not None and (
            self.revision_request.graph_sha256 != self.graph_sha256
            or self.revision_request.verification_episode_node_id
            != self.verification_episode_node_id
            or self.revision_request.campaign_definition_digest != self.campaign_definition_digest
            or self.revision_request.campaign_claim_boundary != self.campaign_claim_boundary
        ):
            raise ValueError("Director revision request is not bound to its containing route")
        if self.action is not MissionDesignRevisionAction.ABSTAIN and (
            self.campaign_definition_digest is None or self.campaign_claim_boundary is None
        ):
            raise ValueError("non-abstaining routes must preserve the campaign evidence scope")
        payload = self.model_dump(mode="json", exclude={"route_sha256"})
        if self.route_sha256 != _canonical_digest(payload):
            raise ValueError("mission design revision route digest mismatch")
        return self


def evaluate_mission_design_verification(
    graph: MissionKnowledgeGraph,
    query: MissionDesignRevisionQuery,
) -> MissionDesignRevisionRoute:
    """Route one exact verified episode without modifying its historical baseline."""

    graph_payload = graph.model_dump(mode="json", exclude={"graph_sha256"})
    if graph.graph_sha256 != _canonical_digest(graph_payload):
        raise ValueError("mission knowledge graph digest mismatch")
    episodes = [
        node
        for node in graph.nodes
        if node.kind is KnowledgeNodeKind.VERIFICATION_EPISODE
        and (node.source_record_id == query.episode_id or node.node_id == query.episode_id)
    ]
    if len(episodes) != 1:
        return _route(
            graph,
            query,
            action=MissionDesignRevisionAction.ABSTAIN,
            reason=MissionDesignRevisionReason.EPISODE_MISSING_OR_AMBIGUOUS,
        )
    baselines = [
        node
        for node in graph.nodes
        if node.kind is KnowledgeNodeKind.BASELINE
        and (node.source_record_id == query.baseline_id or node.node_id == query.baseline_id)
    ]
    if len(baselines) != 1:
        return _route(
            graph,
            query,
            episode=episodes[0],
            action=MissionDesignRevisionAction.ABSTAIN,
            reason=MissionDesignRevisionReason.BASELINE_MISSING_OR_AMBIGUOUS,
        )
    episode = episodes[0]
    baseline = baselines[0]
    decisions = _targets(graph, episode.node_id, "produced", KnowledgeNodeKind.DECISION)
    bindings = [
        edge
        for edge in graph.edges
        if edge.source_node_id == episode.node_id
        and edge.relation == "verifies"
        and edge.target_node_id == baseline.node_id
    ]
    if len(decisions) != 1 or len(bindings) != 1:
        return _binding_abstention(graph, query, episode, baseline)
    decision = decisions[0]
    disposition_edges = [
        edge
        for edge in graph.edges
        if edge.source_node_id == decision.node_id
        and edge.target_node_id == baseline.node_id
        and edge.relation
        in {
            "supports_retention_of",
            "requests_revision_of",
            "leaves_unresolved",
        }
    ]
    disposition = episode.properties.get("disposition")
    if not isinstance(disposition, str):
        return _binding_abstention(graph, query, episode, baseline, decision)
    if _campaign_scope(episode) is None:
        return _binding_abstention(graph, query, episode, baseline, decision)
    expected_relation = {
        "retain": "supports_retention_of",
        "revise": "requests_revision_of",
        "abstain": "leaves_unresolved",
    }.get(disposition)
    if (
        len(disposition_edges) != 1
        or disposition_edges[0].relation != expected_relation
        or decision.properties.get("disposition") != disposition
    ):
        return _binding_abstention(graph, query, episode, baseline, decision)
    if disposition == "retain":
        return _route(
            graph,
            query,
            episode=episode,
            baseline=baseline,
            decision=decision,
            action=MissionDesignRevisionAction.SUPPORT_RETENTION_WITHIN_DECLARED_SCOPE,
            reason=MissionDesignRevisionReason.VERIFIED_CAMPAIGN_SUPPORTS_RETENTION,
        )
    if disposition == "abstain":
        return _route(
            graph,
            query,
            episode=episode,
            baseline=baseline,
            decision=decision,
            action=MissionDesignRevisionAction.ABSTAIN,
            reason=MissionDesignRevisionReason.VERIFIED_CAMPAIGN_INCONCLUSIVE,
        )
    try:
        request = _build_revision_request(graph, episode, baseline)
    except (KeyError, TypeError, ValueError):
        return _binding_abstention(graph, query, episode, baseline, decision)
    return _route(
        graph,
        query,
        episode=episode,
        baseline=baseline,
        decision=decision,
        action=MissionDesignRevisionAction.OPEN_NEW_DIRECTOR_DECISION,
        reason=MissionDesignRevisionReason.VERIFIED_CAMPAIGN_REQUESTS_REVISION,
        revision_request=request,
    )


def _build_revision_request(
    graph: MissionKnowledgeGraph,
    episode: KnowledgeNode,
    baseline: KnowledgeNode,
) -> DirectorRevisionRequest:
    directors = _targets(graph, episode.node_id, "derived_from", KnowledgeNodeKind.RUN)
    if len(directors) != 1:
        raise ValueError("verification episode must bind exactly one Director run")
    director = directors[0]
    intents = _targets(graph, director.node_id, "declares", KnowledgeNodeKind.INTENT)
    operator_runs = _targets(graph, director.node_id, "derived_from", KnowledgeNodeKind.RUN)
    verification_tools = _sources(graph, episode.node_id, "produced", KnowledgeNodeKind.TOOL)
    establishing_decisions = _sources(
        graph, baseline.node_id, "established", KnowledgeNodeKind.DECISION
    )
    if (
        len(intents) != 1
        or len(operator_runs) != 1
        or len(establishing_decisions) != 1
        or len(verification_tools) != 1
    ):
        raise ValueError("Director revision context is incomplete")
    intent = intents[0]
    authority = operator_runs[0].properties["authority"]
    if not isinstance(authority, dict):
        raise TypeError("Director authority must be an object")
    gates = episode.properties["gate_assessments"]
    if not isinstance(gates, list):
        raise TypeError("campaign gates must be a list")
    failed = [gate for gate in gates if isinstance(gate, dict) and gate.get("passed") is False]
    if not failed:
        raise ValueError("revision request requires a failed campaign gate")
    allowed = intent.properties["allowed_capability_ids"]
    if not isinstance(allowed, list) or not all(isinstance(item, str) for item in allowed):
        raise TypeError("intent capability allow-list must contain strings")
    allowed_strings = tuple(item for item in allowed if isinstance(item, str))
    prior_consumed = _required_int(director.properties, "consumed_analysis_cost_units")
    verification_cost = _required_int(verification_tools[0].properties, "cost_units")
    payload = {
        "schema_version": "1.0",
        "request_id": f"director-revision:{episode.source_record_id}",
        "mission_id": graph.mission_id,
        "graph_sha256": graph.graph_sha256,
        "verification_episode_node_id": episode.node_id,
        "campaign_outcome_sha256": _required_digest(episode.properties, "outcome_sha256"),
        "campaign_definition_digest": _required_digest(
            episode.properties, "campaign_definition_digest"
        ),
        "campaign_claim_boundary": _required_string(episode.properties, "claim_boundary"),
        "prior_director_run_node_id": director.node_id,
        "prior_director_run_sha256": _required_digest(director.properties, "run_sha256"),
        "prior_design_decision_sha256": _required_digest(
            establishing_decisions[0].properties, "decision_sha256"
        ),
        "prior_baseline_id": _required_string(baseline.properties, "baseline_id"),
        "prior_baseline_version": _required_int(baseline.properties, "version"),
        "prior_baseline_sha256": _required_digest(baseline.properties, "baseline_sha256"),
        "prior_candidate_id": _required_string(baseline.properties, "candidate_id"),
        "failed_requirement_ids": sorted(
            _required_string(gate, "director_requirement_id") for gate in failed
        ),
        "failed_gate_assessment_sha256s": sorted(
            _required_digest(gate, "assessment_sha256") for gate in failed
        ),
        "prior_intent_sha256": _required_digest(director.properties, "intent_sha256"),
        "prior_capability_catalog_sha256": _required_digest(
            director.properties, "capability_catalog_sha256"
        ),
        "capability_allowlist_envelope": sorted(allowed_strings),
        "prior_analysis_cost_ceiling_units": _required_int(
            director.properties, "max_analysis_cost_units"
        ),
        "prior_consumed_analysis_cost_units": prior_consumed,
        "completed_verification_cost_units": verification_cost,
        "prior_recorded_cost_units": prior_consumed + verification_cost,
        "prior_authority_grant_id": _required_string(authority, "grant_id"),
        "prior_authority_grant_sha256": _canonical_digest(authority),
        "authority_boundary": (
            "requests_new_bounded_director_decision_does_not_select_or_mutate_baseline_"
            "and_requires_fresh_authority_and_budget"
        ),
    }
    return DirectorRevisionRequest.model_validate(
        {**payload, "request_sha256": _canonical_digest(payload)}
    )


def _targets(
    graph: MissionKnowledgeGraph,
    source_node_id: str,
    relation: str,
    kind: KnowledgeNodeKind,
) -> list[KnowledgeNode]:
    node_by_id = {node.node_id: node for node in graph.nodes}
    return [
        node_by_id[edge.target_node_id]
        for edge in graph.edges
        if edge.source_node_id == source_node_id
        and edge.relation == relation
        and node_by_id[edge.target_node_id].kind is kind
    ]


def _sources(
    graph: MissionKnowledgeGraph,
    target_node_id: str,
    relation: str,
    kind: KnowledgeNodeKind,
) -> list[KnowledgeNode]:
    node_by_id = {node.node_id: node for node in graph.nodes}
    return [
        node_by_id[edge.source_node_id]
        for edge in graph.edges
        if edge.target_node_id == target_node_id
        and edge.relation == relation
        and node_by_id[edge.source_node_id].kind is kind
    ]


def _binding_abstention(
    graph: MissionKnowledgeGraph,
    query: MissionDesignRevisionQuery,
    episode: KnowledgeNode,
    baseline: KnowledgeNode,
    decision: KnowledgeNode | None = None,
) -> MissionDesignRevisionRoute:
    return _route(
        graph,
        query,
        episode=episode,
        baseline=baseline,
        decision=decision,
        action=MissionDesignRevisionAction.ABSTAIN,
        reason=MissionDesignRevisionReason.EPISODE_BINDING_INVALID,
    )


def _route(
    graph: MissionKnowledgeGraph,
    query: MissionDesignRevisionQuery,
    *,
    action: MissionDesignRevisionAction,
    reason: MissionDesignRevisionReason,
    episode: KnowledgeNode | None = None,
    baseline: KnowledgeNode | None = None,
    decision: KnowledgeNode | None = None,
    revision_request: DirectorRevisionRequest | None = None,
) -> MissionDesignRevisionRoute:
    scope = None if episode is None else _campaign_scope(episode)
    payload = {
        "schema_version": "1.0",
        "graph_sha256": graph.graph_sha256,
        "query": query.model_dump(mode="json"),
        "verification_episode_node_id": None if episode is None else episode.node_id,
        "baseline_node_id": None if baseline is None else baseline.node_id,
        "disposition_node_id": None if decision is None else decision.node_id,
        "campaign_definition_digest": None if scope is None else scope[0],
        "campaign_claim_boundary": None if scope is None else scope[1],
        "action": action.value,
        "reason": reason.value,
        "revision_request": (
            None if revision_request is None else revision_request.model_dump(mode="json")
        ),
        "authority_boundary": ("routing_only_no_baseline_selection_mutation_or_director_execution"),
    }
    return MissionDesignRevisionRoute.model_validate(
        {**payload, "route_sha256": _canonical_digest(payload)}
    )


def _campaign_scope(episode: KnowledgeNode) -> tuple[str, str] | None:
    definition_digest = episode.properties.get("campaign_definition_digest")
    claim_boundary = episode.properties.get("claim_boundary")
    if (
        not isinstance(definition_digest, str)
        or len(definition_digest) != 64
        or any(character not in "0123456789abcdef" for character in definition_digest)
        or not isinstance(claim_boundary, str)
        or not claim_boundary
    ):
        return None
    return definition_digest, claim_boundary


def _required_string(properties: Mapping[str, object], key: str) -> str:
    value = properties[key]
    if not isinstance(value, str) or not value:
        raise TypeError(f"{key} must be a non-empty string")
    return value


def _required_digest(properties: Mapping[str, object], key: str) -> str:
    value = _required_string(properties, key)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise TypeError(f"{key} must be a SHA-256 digest")
    return value


def _required_int(properties: Mapping[str, object], key: str) -> int:
    value = properties[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an integer")
    return value


__all__ = [
    "DirectorRevisionRequest",
    "MissionDesignRevisionAction",
    "MissionDesignRevisionQuery",
    "MissionDesignRevisionReason",
    "MissionDesignRevisionRoute",
    "evaluate_mission_design_verification",
]
