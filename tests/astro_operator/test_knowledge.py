from __future__ import annotations

import json
import os
from hashlib import sha256
from pathlib import Path

import pytest

from astro_core.errors import InvalidScenarioError
from astro_operator.knowledge import (
    KnowledgeEdge,
    KnowledgeNodeKind,
    MissionKnowledgeGraph,
    _claim_decision_relation,
    trace_baseline_justification,
)
from astro_operator.knowledge_io import (
    publish_mission_knowledge_graph,
    verify_mission_knowledge_graph,
    write_mission_knowledge_manifest,
)
from astro_operator.models import ClaimDisposition
from astro_operator.orchestration import (
    MissionOrchestrationDecision,
    MissionOrchestrationQuery,
    OrchestrationDisposition,
    OrchestrationReason,
    evaluate_mission_orchestration,
)
from astro_operator.revision import (
    DirectorRevisionRequest,
    MissionDesignRevisionAction,
    MissionDesignRevisionQuery,
    MissionDesignRevisionReason,
    MissionDesignRevisionRoute,
    evaluate_mission_design_verification,
)


def _canonical_digest(value: object) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _run_source_bundles(tmp_path: Path) -> tuple[Path, Path]:
    from astro_cli.main import app
    from tests.astro_cli.helpers import make_cli_runner

    repository = Path(__file__).resolve().parents[2]
    director = tmp_path / "director"
    result = make_cli_runner().invoke(
        app,
        [
            "run-mission-design-director",
            str(repository / "examples/design/leo_mission_design_director.yaml"),
            "--reasoner-replay",
            str(repository / "examples/operator/leo_lifecycle_trade_study_replay.yaml"),
            "--output-dir",
            str(director),
        ],
    )
    assert result.exit_code == 0, result.output
    operator = tmp_path / "post-launch"
    result = make_cli_runner().invoke(
        app,
        [
            "run-mission-operator",
            str(repository / "examples/operator/post_launch_recovery_review.yaml"),
            "--reasoner-replay",
            str(repository / "examples/operator/post_launch_recovery_review_replay.yaml"),
            "--output-dir",
            str(operator),
            "--mission-design-context",
            str(director),
        ],
    )
    assert result.exit_code == 0, result.output
    return director, operator


def _write_spec(
    path: Path,
    director: Path,
    operator: Path,
    *,
    campaign: Path | None = None,
    reverse: bool = False,
) -> None:
    sources = [
        (
            "design",
            "mission_design_director",
            director.as_posix(),
        ),
        (
            "post-launch",
            "operator_run",
            operator.as_posix(),
        ),
    ]
    if campaign is not None:
        sources.append(
            (
                "verification",
                "conditional_campaign",
                campaign.as_posix(),
            )
        )
    if reverse:
        sources.reverse()
    source_yaml = "\n".join(
        f"  - source_id: {source_id}\n    kind: {kind}\n    path: {source_path}"
        for source_id, kind, source_path in sources
    )
    path.write_text(
        "schema_version: '1.0'\n"
        "graph_id: leo-mission-episode-graph\n"
        "mission_id: leo-reference-mission\n"
        "sources:\n"
        f"{source_yaml}\n",
        encoding="utf-8",
    )


def _run_conditional_campaign_source(tmp_path: Path, director: Path) -> Path:
    from astro_cli.main import app
    from tests.astro_cli.helpers import make_cli_runner

    repository = Path(__file__).resolve().parents[2]
    campaign = tmp_path / "verification"
    result = make_cli_runner().invoke(
        app,
        [
            "run-mission-design-conditional-campaign",
            str(director),
            str(repository / "examples/design/leo_mission_design_conditional_campaign.yaml"),
            "--output-dir",
            str(campaign),
        ],
    )
    assert result.exit_code == 0, result.output
    return campaign


def _build_graph(spec: Path, output: Path) -> None:
    from astro_cli.main import app
    from tests.astro_cli.helpers import make_cli_runner

    result = make_cli_runner().invoke(
        app,
        [
            "build-mission-knowledge-graph",
            str(spec),
            "--output-dir",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output


def test_claim_edges_preserve_disposition_semantics() -> None:
    assert _claim_decision_relation(ClaimDisposition.SUPPORTED) == "supports"
    assert _claim_decision_relation(ClaimDisposition.QUALIFIED) == "qualifies"
    assert _claim_decision_relation(ClaimDisposition.DISPUTED) == "disputes"


def test_cross_run_graph_rebuilds_and_traces_baseline(tmp_path: Path) -> None:
    director, operator = _run_source_bundles(tmp_path)
    spec = tmp_path / "graph.yaml"
    _write_spec(spec, director, operator)
    output = tmp_path / "knowledge"
    _build_graph(spec, output)

    graph = verify_mission_knowledge_graph(output)
    assert [item.source_id for item in graph.sources] == ["design", "post-launch"]
    assert {item.kind for item in graph.nodes} >= {
        KnowledgeNodeKind.MISSION,
        KnowledgeNodeKind.REQUIREMENT,
        KnowledgeNodeKind.EVIDENCE,
        KnowledgeNodeKind.ASSERTION,
        KnowledgeNodeKind.CLAIM,
        KnowledgeNodeKind.DECISION,
        KnowledgeNodeKind.BASELINE,
        KnowledgeNodeKind.TOOL,
    }
    assert any(
        node.kind == KnowledgeNodeKind.CLAIM and node.source_record_id == "post-launch-review-ready"
        for node in graph.nodes
    )
    trace = trace_baseline_justification(graph, "leo-mission-design:baseline")
    assert trace.selected_candidate_node_id == "design:candidate:higher-reserve"
    assert len(trace.requirement_node_ids) == 2
    assert len(trace.assessment_node_ids) == 2
    assert len(trace.evidence_node_ids) == 2
    assert trace.tool_node_ids == ("design:tool:astro.mission_lifecycle_screen",)
    assert len(trace.tool_references) == 1
    assert trace.tool_references[0].tool_id == "astro.mission_lifecycle_screen"
    assert trace.tool_references[0].version == "1.0"
    assert trace.tool_references[0].qualification_sha256 == (
        "4d291360a6ad3b340a424d9229a5f6b38633c1282a9674b1e84832a7ef724938"
    )
    assert trace.claim_boundary == (
        "deterministic multi-domain mission design screening, not flight qualification"
    )

    from astro_cli.main import app
    from tests.astro_cli.helpers import make_cli_runner

    result = make_cli_runner().invoke(
        app,
        [
            "trace-mission-baseline",
            str(output),
            "--baseline-id",
            "leo-mission-design:baseline",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["graph_sha256"] == graph.graph_sha256
    assert json.loads(result.output)["tool_references"][0]["version"] == "1.0"

    claim_node = next(
        node.node_id
        for node in graph.nodes
        if node.kind == KnowledgeNodeKind.CLAIM
        and node.source_record_id == "post-launch-review-ready"
    )
    assert any(
        edge.source_node_id == claim_node and edge.relation == "supports" for edge in graph.edges
    )
    assert all(
        not (edge.target_node_id == claim_node and edge.relation == "supports")
        for edge in graph.edges
    )
    handoffs = [edge for edge in graph.edges if edge.relation == "operates_against"]
    assert len(handoffs) == 1
    assert handoffs[0].source_node_id == "post-launch:run:operator"
    assert handoffs[0].target_node_id == ("design:baseline:leo-mission-design:baseline")
    assert handoffs[0].properties["operational_configuration_id"] == (
        "post-launch-recovery-baseline-v1"
    )

    query = MissionOrchestrationQuery(
        baseline_id="leo-mission-design:baseline",
        operator_objective_id="post-launch-recovery-review",
        disposition_claim_id="post-launch-review-ready",
        manual_review_gate_predicate_id="manual-review-required",
    )
    decision = evaluate_mission_orchestration(graph, query)
    assert decision.disposition == OrchestrationDisposition.CONTINUE
    assert decision.reason_codes == (OrchestrationReason.ALL_MANUAL_REVIEW_GATES_SATISFIED,)
    assert all(item.passed for item in decision.predicate_checks)
    assert "no_command_approval_execution" in decision.authority_boundary

    result = make_cli_runner().invoke(
        app,
        [
            "evaluate-mission-orchestration",
            str(output),
            "--baseline-id",
            query.baseline_id,
            "--operator-objective-id",
            query.operator_objective_id,
            "--claim-id",
            query.disposition_claim_id,
            "--manual-review-gate-predicate-id",
            query.manual_review_gate_predicate_id,
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["decision_sha256"] == decision.decision_sha256


def test_verified_campaign_episode_routes_retention_and_relocates(
    tmp_path: Path,
) -> None:
    director, operator = _run_source_bundles(tmp_path)
    campaign = _run_conditional_campaign_source(tmp_path, director)
    spec = tmp_path / "graph.yaml"
    _write_spec(spec, director, operator, campaign=campaign)
    output = tmp_path / "knowledge"
    _build_graph(spec, output)

    graph = verify_mission_knowledge_graph(output)
    assert graph.schema_version == "1.2"
    episode = next(
        node for node in graph.nodes if node.kind is KnowledgeNodeKind.VERIFICATION_EPISODE
    )
    assert episode.source_record_id == "leo-mission-design-conditional-verification"
    assert episode.properties["disposition"] == "retain"
    assert any(
        edge.source_node_id == episode.node_id
        and edge.relation == "verifies"
        and edge.target_node_id == "design:baseline:leo-mission-design:baseline"
        for edge in graph.edges
    )
    assert any(
        edge.relation == "supports_retention_of"
        and edge.target_node_id == "design:baseline:leo-mission-design:baseline"
        for edge in graph.edges
    )
    route = evaluate_mission_design_verification(
        graph,
        MissionDesignRevisionQuery(
            episode_id=episode.source_record_id,
            baseline_id="leo-mission-design:baseline",
        ),
    )
    assert route.action is MissionDesignRevisionAction.SUPPORT_RETENTION_WITHIN_DECLARED_SCOPE
    assert route.reason is MissionDesignRevisionReason.VERIFIED_CAMPAIGN_SUPPORTS_RETENTION
    assert route.revision_request is None
    assert route.campaign_definition_digest == episode.properties["campaign_definition_digest"]
    assert route.campaign_claim_boundary == (
        "configured_design_space_verification_not_operational_probability_"
        "qualification_or_authority"
    )

    from astro_cli.main import app
    from tests.astro_cli.helpers import make_cli_runner

    result = make_cli_runner().invoke(
        app,
        [
            "evaluate-mission-design-verification",
            str(output),
            "--episode-id",
            episode.source_record_id,
            "--baseline-id",
            "leo-mission-design:baseline",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["route_sha256"] == route.route_sha256

    relocated = tmp_path / "relocated-knowledge"
    output.rename(relocated)
    assert verify_mission_knowledge_graph(relocated).graph_sha256 == graph.graph_sha256


def test_revise_episode_opens_bounded_director_request_without_mutating_baseline(
    tmp_path: Path,
) -> None:
    director, operator = _run_source_bundles(tmp_path)
    campaign = _run_conditional_campaign_source(tmp_path, director)
    spec = tmp_path / "graph.yaml"
    _write_spec(spec, director, operator, campaign=campaign)
    output = tmp_path / "knowledge"
    _build_graph(spec, output)
    graph = verify_mission_knowledge_graph(output)
    revised = _synthetic_revise_graph(graph)
    query = MissionDesignRevisionQuery(
        episode_id="leo-mission-design-conditional-verification",
        baseline_id="leo-mission-design:baseline",
    )

    route = evaluate_mission_design_verification(revised, query)

    assert route.action is MissionDesignRevisionAction.OPEN_NEW_DIRECTOR_DECISION
    assert route.reason is MissionDesignRevisionReason.VERIFIED_CAMPAIGN_REQUESTS_REVISION
    request = route.revision_request
    assert request is not None
    assert request.prior_baseline_id == "leo-mission-design:baseline"
    assert request.prior_baseline_version == 1
    assert request.failed_requirement_ids == ("deorbit-propellant-reserve",)
    assert request.prior_analysis_cost_ceiling_units == 7
    assert request.prior_consumed_analysis_cost_units == 3
    assert request.completed_verification_cost_units == 3
    assert request.prior_recorded_cost_units == 6
    assert request.capability_allowlist_envelope == (
        "astro.mission_lifecycle_screen",
        "astro.mission_lifecycle_uncertainty",
    )
    assert request.authority_boundary == (
        "requests_new_bounded_director_decision_does_not_select_or_mutate_baseline_"
        "and_requires_fresh_authority_and_budget"
    )
    assert request.campaign_definition_digest == route.campaign_definition_digest
    assert request.campaign_claim_boundary == route.campaign_claim_boundary
    baseline = next(node for node in revised.nodes if node.node_id == route.baseline_node_id)
    assert baseline.properties["version"] == 1
    assert baseline.properties["baseline_sha256"] == request.prior_baseline_sha256

    missing = evaluate_mission_design_verification(
        graph,
        query.model_copy(update={"episode_id": "missing"}),
    )
    assert missing.action is MissionDesignRevisionAction.ABSTAIN
    assert missing.reason is MissionDesignRevisionReason.EPISODE_MISSING_OR_AMBIGUOUS

    payload = route.model_dump(mode="json")
    payload["action"] = "support_retention_within_declared_scope"
    with pytest.raises(ValueError, match="only a new Director route"):
        MissionDesignRevisionRoute.model_validate(payload)

    request_payload = request.model_dump(mode="json", exclude={"request_sha256"})
    request_payload["graph_sha256"] = "0" * 64
    spliced_request = DirectorRevisionRequest.model_validate(
        {
            **request_payload,
            "request_sha256": _canonical_digest(request_payload),
        }
    )
    route_payload = route.model_dump(mode="json", exclude={"route_sha256"})
    route_payload["revision_request"] = spliced_request.model_dump(mode="json")
    with pytest.raises(ValueError, match="not bound to its containing route"):
        MissionDesignRevisionRoute.model_validate(
            {
                **route_payload,
                "route_sha256": _canonical_digest(route_payload),
            }
        )


def test_orchestration_reducer_fails_closed_and_holds_readiness(
    tmp_path: Path,
) -> None:
    director, operator = _run_source_bundles(tmp_path)
    spec = tmp_path / "graph.yaml"
    _write_spec(spec, director, operator)
    output = tmp_path / "knowledge"
    _build_graph(spec, output)
    graph = verify_mission_knowledge_graph(output)
    query = MissionOrchestrationQuery(
        baseline_id="leo-mission-design:baseline",
        operator_objective_id="post-launch-recovery-review",
        disposition_claim_id="post-launch-review-ready",
        manual_review_gate_predicate_id="manual-review-required",
    )

    unbound = graph.model_copy(
        update={"edges": tuple(edge for edge in graph.edges if edge.relation != "operates_against")}
    )
    with pytest.raises(ValueError, match="graph digest mismatch"):
        evaluate_mission_orchestration(unbound, query)
    unbound = _redigest_graph(unbound)
    unbound_decision = evaluate_mission_orchestration(unbound, query)
    assert unbound_decision.disposition == OrchestrationDisposition.ABSTAIN
    assert unbound_decision.reason_codes == (
        OrchestrationReason.TARGET_OPERATOR_MISSING_OR_AMBIGUOUS,
    )

    command_capable = _replace_graph_node_property(
        graph,
        "post-launch:run:operator",
        ("authority", "allowed_actions"),
        ["request_evidence", "finish", "propose_command"],
    )
    authority_decision = evaluate_mission_orchestration(command_capable, query)
    assert authority_decision.disposition == OrchestrationDisposition.ABSTAIN
    assert authority_decision.reason_codes == (OrchestrationReason.OPERATIONAL_AUTHORITY_PRESENT,)

    inapplicable = _replace_graph_node_property(
        graph,
        "post-launch:assertion:telemetry.mode",
        ("value",),
        "safe_mode",
    )
    applicability_decision = evaluate_mission_orchestration(inapplicable, query)
    assert applicability_decision.disposition == OrchestrationDisposition.ABSTAIN
    assert applicability_decision.reason_codes == (
        OrchestrationReason.APPLICABILITY_PREDICATE_FAILED,
    )

    not_ready = _replace_graph_node_property(
        graph,
        "post-launch:assertion:estimate.converged",
        ("value",),
        False,
    )
    readiness_decision = evaluate_mission_orchestration(not_ready, query)
    assert readiness_decision.disposition == OrchestrationDisposition.HOLD
    assert readiness_decision.reason_codes == (OrchestrationReason.READINESS_PREDICATE_FAILED,)

    missing_gate = evaluate_mission_orchestration(
        graph,
        query.model_copy(update={"manual_review_gate_predicate_id": "missing-manual-review-gate"}),
    )
    assert missing_gate.disposition == OrchestrationDisposition.ABSTAIN
    assert missing_gate.reason_codes == (OrchestrationReason.MANUAL_REVIEW_GATE_MISSING_OR_INVALID,)

    decision_payload = evaluate_mission_orchestration(graph, query).model_dump(mode="json")
    decision_payload["disposition"] = "hold"
    with pytest.raises(ValueError, match="decision digest mismatch"):
        MissionOrchestrationDecision.model_validate(decision_payload)


def _replace_graph_node_property(
    graph: MissionKnowledgeGraph,
    node_id: str,
    path: tuple[str, ...],
    value: object,
) -> MissionKnowledgeGraph:
    nodes = []
    for node in graph.nodes:
        if node.node_id != node_id:
            nodes.append(node)
            continue
        properties = json.loads(json.dumps(node.properties))
        target = properties
        for part in path[:-1]:
            target = target[part]
        target[path[-1]] = value
        nodes.append(node.model_copy(update={"properties": properties}))
    return _redigest_graph(graph.model_copy(update={"nodes": tuple(nodes)}))


def _redigest_graph(graph: MissionKnowledgeGraph) -> MissionKnowledgeGraph:
    payload = graph.model_dump(mode="json", exclude={"graph_sha256"})
    return graph.model_copy(update={"graph_sha256": _canonical_digest(payload)})


def _synthetic_revise_graph(graph: MissionKnowledgeGraph) -> MissionKnowledgeGraph:
    episode = next(
        node for node in graph.nodes if node.kind is KnowledgeNodeKind.VERIFICATION_EPISODE
    )
    disposition = next(
        node
        for node in graph.nodes
        if node.kind is KnowledgeNodeKind.DECISION and node.source_id == episode.source_id
    )
    nodes = []
    for node in graph.nodes:
        properties = json.loads(json.dumps(node.properties))
        if node.node_id == episode.node_id:
            properties["disposition"] = "revise"
            properties["reason"] = "declared_design_space_gate_failed"
            properties["gate_assessments"][0]["passed"] = False
            properties["gate_assessments"][0]["passed_samples"] = 7
            properties["gate_assessments"][0]["observed_pass_fraction"] = 0.875
        elif node.node_id == disposition.node_id:
            properties["disposition"] = "revise"
            properties["reason"] = "declared_design_space_gate_failed"
        nodes.append(node.model_copy(update={"properties": properties}))
    edges = []
    for edge in graph.edges:
        if edge.source_node_id == disposition.node_id and edge.relation == "supports_retention_of":
            payload = {
                "source_node_id": edge.source_node_id,
                "relation": "requests_revision_of",
                "target_node_id": edge.target_node_id,
                "source_id": edge.source_id,
                "properties": edge.properties,
            }
            edges.append(
                KnowledgeEdge.model_validate({**payload, "edge_id": _canonical_digest(payload)})
            )
        else:
            edges.append(edge)
    graph_payload: dict[str, object] = graph.model_dump(
        mode="json", exclude={"graph_sha256", "nodes", "edges"}
    )
    graph_payload["nodes"] = [
        node.model_dump(mode="json") for node in sorted(nodes, key=lambda item: item.node_id)
    ]
    graph_payload["edges"] = [
        edge.model_dump(mode="json") for edge in sorted(edges, key=lambda item: item.edge_id)
    ]
    return MissionKnowledgeGraph.model_validate(
        {**graph_payload, "graph_sha256": _canonical_digest(graph_payload)}
    )


def test_graph_is_input_order_invariant_and_relocatable(tmp_path: Path) -> None:
    director, operator = _run_source_bundles(tmp_path)
    first_spec = tmp_path / "first.yaml"
    second_spec = tmp_path / "second.yaml"
    _write_spec(first_spec, director, operator)
    _write_spec(second_spec, director, operator, reverse=True)
    first = tmp_path / "first"
    second = tmp_path / "second"
    _build_graph(first_spec, first)
    _build_graph(second_spec, second)
    assert (
        verify_mission_knowledge_graph(first).graph_sha256
        == verify_mission_knowledge_graph(second).graph_sha256
    )
    relocated = tmp_path / "relocated"
    first.rename(relocated)
    assert verify_mission_knowledge_graph(relocated).graph_id == ("leo-mission-episode-graph")


def test_verifier_rejects_inventory_and_self_consistent_graph_forgery(
    tmp_path: Path,
) -> None:
    director, operator = _run_source_bundles(tmp_path)
    spec = tmp_path / "graph.yaml"
    _write_spec(spec, director, operator)
    output = tmp_path / "knowledge"
    _build_graph(spec, output)

    extra = output / "undeclared.txt"
    extra.write_text("not inventoried\n", encoding="utf-8")
    with pytest.raises(InvalidScenarioError, match="inventory"):
        verify_mission_knowledge_graph(output)
    extra.unlink()

    fifo = output / "undeclared-fifo"
    try:
        os.mkfifo(fifo)
    except (AttributeError, OSError):
        pass
    else:
        with pytest.raises(InvalidScenarioError, match="unsupported entry"):
            verify_mission_knowledge_graph(output)
        fifo.unlink()

    graph_path = output / "mission-knowledge-graph.json"
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    baseline = next(node for node in graph["nodes"] if node["kind"] == "baseline")
    baseline["properties"]["claim_boundary"] = "forged broad authority"
    graph["graph_sha256"] = _canonical_digest(
        {key: value for key, value in graph.items() if key != "graph_sha256"}
    )
    graph_path.write_text(json.dumps(graph, indent=2) + "\n", encoding="utf-8")
    write_mission_knowledge_manifest(output)
    with pytest.raises(InvalidScenarioError, match="verified source bundles"):
        verify_mission_knowledge_graph(output)


def test_build_rejects_tampered_or_symbolic_link_source(tmp_path: Path) -> None:
    director, operator = _run_source_bundles(tmp_path)
    result_path = (
        director / "operator" / "candidates" / "higher-reserve" / "mission-lifecycle-result.json"
    )
    result_path.write_text("{}\n", encoding="utf-8")
    spec = tmp_path / "tampered.yaml"
    _write_spec(spec, director, operator)

    with pytest.raises(InvalidScenarioError, match="failed verification"):
        publish_mission_knowledge_graph(spec, tmp_path / "tampered-output")

    clean_director, _ = _run_source_bundles(tmp_path / "clean")
    link = clean_director / "unexpected-link"
    try:
        link.symlink_to(clean_director / "mission-design-run.json")
    except OSError:
        pytest.skip("symbolic links are unavailable")
    link_spec = tmp_path / "link.yaml"
    _write_spec(link_spec, clean_director, operator)
    with pytest.raises(InvalidScenarioError, match="symbolic link"):
        publish_mission_knowledge_graph(link_spec, tmp_path / "link-output")
