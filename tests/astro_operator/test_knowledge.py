from __future__ import annotations

import json
import os
from hashlib import sha256
from pathlib import Path

import pytest

from astro_core.errors import InvalidScenarioError
from astro_operator.knowledge import (
    KnowledgeNodeKind,
    _claim_decision_relation,
    trace_baseline_justification,
)
from astro_operator.knowledge_io import (
    publish_mission_knowledge_graph,
    verify_mission_knowledge_graph,
    write_mission_knowledge_manifest,
)
from astro_operator.models import ClaimDisposition


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
        ],
    )
    assert result.exit_code == 0, result.output
    return director, operator


def _write_spec(path: Path, director: Path, operator: Path, *, reverse: bool = False) -> None:
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
