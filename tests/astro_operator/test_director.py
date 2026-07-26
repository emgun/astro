from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from astro_core.errors import InvalidScenarioError
from astro_operator.director import (
    CapabilityCatalog,
    ConditionalAnalysisDecision,
    MissionDesignDirectorSpec,
    RequirementGraph,
    build_analysis_plan,
    build_mission_design_run,
    lifecycle_screen_capability,
    lifecycle_uncertainty_capability,
    mission_design_run_payload,
)
from astro_operator.director_io import (
    _verify_lifecycle_observations,
    load_mission_design_director_spec,
    verify_mission_design_director,
    write_mission_design_manifest,
    write_mission_design_run,
)
from astro_operator.io import verify_operator_run
from astro_operator.models import OperatorRun

SPEC = Path("examples/design/leo_mission_design_director.yaml")
REPLAY = Path("examples/operator/leo_lifecycle_trade_study_replay.yaml")


def _run_checked_bundle(output: Path) -> None:
    from astro_cli.main import app
    from tests.astro_cli.helpers import make_cli_runner

    result = make_cli_runner().invoke(
        app,
        [
            "run-mission-design-director",
            str(SPEC),
            "--reasoner-replay",
            str(REPLAY),
            "--output-dir",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output


def test_checked_design_director_selects_and_verifies_baseline(tmp_path: Path) -> None:
    output = tmp_path / "mission-design"
    _run_checked_bundle(output)

    run = verify_mission_design_director(output)
    assert run.decision.disposition == "selected"
    assert run.decision.selected_candidate_id == "higher-reserve"
    assert run.baseline is not None
    assert run.baseline.assignments == {"spacecraft_wet_mass_kg": 520.0}
    assert all(item.passed for item in run.assessments)
    assert run.schema_version == "1.1"
    assert run.analysis_plan.total_cost_units == 7
    assert run.consumed_analysis_cost_units == 3
    conditional_node = run.analysis_plan.nodes[1]
    assert conditional_node.activation == "decision_relevant"
    assert conditional_node.depends_on == ("analyze:astro.mission_lifecycle_screen",)
    assert conditional_node.trigger_rule_ids == ("entry-interface-margin-near-boundary",)
    recommendation = run.verification_plan.conditional_analyses[0]
    assert recommendation.disposition == "recommended"
    assert recommendation.candidate_id == "higher-reserve"
    assert recommendation.baseline_id == "leo-mission-design:baseline"
    assert recommendation.requirement_id == "entry-interface-continuity"
    assert recommendation.observed_margin == pytest.approx(0.908437281352235)
    assert recommendation.decision_relevance_score == pytest.approx(0.091562718647765)
    assert run.verification_plan.remaining_hard_requirement_ids == ()
    assert {
        item["role"]
        for item in json.loads((output / "director-manifest.json").read_text(encoding="utf-8"))[
            "artifacts"
        ]
    } >= {
        "declared_design_contract",
        "derived_design_decision",
        "adaptive_operator_journal",
        "candidate_analysis_evidence",
    }


def test_design_director_runs_from_outside_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from astro_cli.main import app
    from tests.astro_cli.helpers import make_cli_runner

    repository = Path(__file__).resolve().parents[2]
    output = tmp_path / "portable-design"
    monkeypatch.chdir(tmp_path)
    result = make_cli_runner().invoke(
        app,
        [
            "run-mission-design-director",
            str(repository / SPEC),
            "--reasoner-replay",
            str(repository / REPLAY),
            "--output-dir",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    assert verify_mission_design_director(output).baseline is not None


def test_verifier_rejects_extra_and_rederived_tampered_decision(tmp_path: Path) -> None:
    output = tmp_path / "mission-design"
    _run_checked_bundle(output)
    extra = output / "undeclared.txt"
    extra.write_text("not inventoried\n", encoding="utf-8")
    with pytest.raises(InvalidScenarioError, match="inventory"):
        verify_mission_design_director(output)
    extra.unlink()

    run_path = output / "mission-design-run.json"
    payload = json.loads(run_path.read_text(encoding="utf-8"))
    payload["decision"]["selected_candidate_id"] = "baseline"
    payload["run_sha256"] = sha256(
        json.dumps(
            {key: value for key, value in payload.items() if key != "run_sha256"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    run_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_mission_design_manifest(output)
    with pytest.raises(InvalidScenarioError, match="derived decision"):
        verify_mission_design_director(output)


def test_verifier_reconstructs_observations_and_manifest_roles(tmp_path: Path) -> None:
    output = tmp_path / "mission-design"
    _run_checked_bundle(output)
    operator_path = output / "operator" / "operator-run.json"
    operator_payload = json.loads(operator_path.read_text(encoding="utf-8"))
    selected = next(
        step
        for step in operator_payload["steps"]
        if step.get("observation", {}).get("candidate", {}).get("candidate_id") == "higher-reserve"
    )
    metric = next(
        item
        for item in selected["observation"]["metrics"]
        if item["metric_id"] == "margin:deorbit:propellant_reserve"
    )
    metric["value"] += 123.0
    with pytest.raises(InvalidScenarioError, match="result artifact"):
        _verify_lifecycle_observations(
            output / "operator", OperatorRun.model_validate(operator_payload)
        )

    scenario_output = tmp_path / "scenario-design"
    _run_checked_bundle(scenario_output)
    scenario_run = verify_operator_run(scenario_output / "operator")
    scenario_path = scenario_output / "operator" / "candidates" / "higher-reserve" / "scenario.json"
    scenario_payload = json.loads(scenario_path.read_text(encoding="utf-8"))
    scenario_payload["input_overrides"]["twin_battery_capacity_wh"] = 999.0
    scenario_path.write_text(json.dumps(scenario_payload, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(InvalidScenarioError, match="exact base scenario"):
        _verify_lifecycle_observations(scenario_output / "operator", scenario_run)

    failure_output = tmp_path / "failure-design"
    _run_checked_bundle(failure_output)
    failure_run = verify_operator_run(failure_output / "operator")
    error_path = (
        failure_output / "operator" / "candidates" / "lighter-spacecraft" / "evaluation-error.json"
    )
    error_payload = json.loads(error_path.read_text(encoding="utf-8"))
    error_payload["operator_scenario_sha256"] = "0" * 64
    error_path.write_text(json.dumps(error_payload, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(InvalidScenarioError, match="error artifact"):
        _verify_lifecycle_observations(failure_output / "operator", failure_run)

    _run_checked_bundle(tmp_path / "role-design")
    role_output = tmp_path / "role-design"
    manifest_path = role_output / "director-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][0]["role"] = "forged_role"
    manifest["manifest_sha256"] = sha256(
        json.dumps(
            {key: value for key, value in manifest.items() if key != "manifest_sha256"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(InvalidScenarioError, match="role mismatch"):
        verify_mission_design_director(role_output)


def test_hard_requirement_failure_abstains_without_baseline(tmp_path: Path) -> None:
    output = tmp_path / "mission-design"
    _run_checked_bundle(output)
    spec = load_mission_design_director_spec(output / "inputs" / "design-spec.yaml")
    requirements = list(spec.requirement_graph.requirements)
    requirements[0] = requirements[0].model_copy(update={"threshold": 100.0})
    strict_spec = spec.model_copy(
        update={
            "requirement_graph": RequirementGraph(
                graph_id=spec.requirement_graph.graph_id,
                requirements=tuple(requirements),
            )
        }
    )
    operator_path = output / "operator" / "operator-run.json"
    run = build_mission_design_run(
        spec=strict_spec,
        operator_run=verify_operator_run(output / "operator"),
        spec_sha256="0" * 64,
        operator_run_sha256=sha256(operator_path.read_bytes()).hexdigest(),
    )
    assert run.decision.disposition == "abstained"
    assert run.baseline is None
    assert run.decision.unresolved_requirement_ids == ("deorbit-propellant-reserve",)
    conditional = run.verification_plan.conditional_analyses[0]
    assert conditional.candidate_id == "higher-reserve"
    assert conditional.baseline_id is None
    assert conditional.disposition == "deferred"
    assert conditional.reason == "baseline_not_eligible"


def test_planner_rejects_unregistered_capability_contract() -> None:
    spec = load_mission_design_director_spec(SPEC)
    capability = lifecycle_screen_capability()
    unregistered = lifecycle_uncertainty_capability().model_copy(
        update={
            "capability_id": "evil.unregistered",
            "version": "999",
            "qualification_sha256": "0" * 64,
        }
    )
    forged = MissionDesignDirectorSpec.model_validate(
        {
            **spec.model_dump(mode="python"),
            "intent": spec.intent.model_copy(
                update={
                    "allowed_capability_ids": (
                        capability.capability_id,
                        unregistered.capability_id,
                    )
                }
            ),
            "capability_catalog": CapabilityCatalog(
                catalog_id=spec.capability_catalog.catalog_id,
                capabilities=(capability, unregistered),
            ),
            "conditional_analysis_rules": (
                spec.conditional_analysis_rules[0].model_copy(
                    update={"capability_id": unregistered.capability_id}
                ),
            ),
        }
    )
    with pytest.raises(ValueError, match="registered capabilities"):
        build_analysis_plan(forged)


def test_conditional_analysis_defers_outside_declared_decision_band(
    tmp_path: Path,
) -> None:
    output = tmp_path / "mission-design"
    _run_checked_bundle(output)
    spec = load_mission_design_director_spec(output / "inputs" / "design-spec.yaml")
    narrow = spec.model_copy(
        update={
            "conditional_analysis_rules": (
                spec.conditional_analysis_rules[0].model_copy(
                    update={"maximum_absolute_margin": 0.5}
                ),
            )
        }
    )
    operator_path = output / "operator" / "operator-run.json"
    run = build_mission_design_run(
        spec=narrow,
        operator_run=verify_operator_run(output / "operator"),
        spec_sha256="0" * 64,
        operator_run_sha256=sha256(operator_path.read_bytes()).hexdigest(),
    )
    decision = run.verification_plan.conditional_analyses[0]
    assert decision.disposition == "deferred"
    assert decision.decision_relevance_score == 0.0
    assert decision.reason == "outside_declared_decision_change_band"


def test_conditional_analysis_reservation_respects_intent_budget() -> None:
    spec = load_mission_design_director_spec(SPEC)
    underfunded = spec.model_copy(
        update={"intent": spec.intent.model_copy(update={"max_analysis_cost_units": 6})}
    )
    with pytest.raises(ValueError, match="cost budget"):
        build_analysis_plan(underfunded)


@pytest.mark.parametrize("margin", [True, "1.0"])
def test_conditional_analysis_band_rejects_coercive_yaml_values(
    margin: object,
) -> None:
    spec = load_mission_design_director_spec(SPEC)
    payload = spec.model_dump(mode="python")
    payload["conditional_analysis_rules"][0]["maximum_absolute_margin"] = margin
    with pytest.raises(ValueError, match="margin must be numeric"):
        MissionDesignDirectorSpec.model_validate(payload)


def test_schema_1_0_wire_payload_and_digests_remain_compatible(
    tmp_path: Path,
) -> None:
    output = tmp_path / "mission-design"
    _run_checked_bundle(output)
    current = load_mission_design_director_spec(output / "inputs" / "design-spec.yaml")
    screen = lifecycle_screen_capability()
    legacy = MissionDesignDirectorSpec.model_validate(
        {
            **current.model_dump(mode="python"),
            "schema_version": "1.0",
            "intent": current.intent.model_copy(
                update={
                    "allowed_capability_ids": (screen.capability_id,),
                    "max_analysis_cost_units": 4,
                }
            ),
            "capability_catalog": CapabilityCatalog(
                catalog_id=current.capability_catalog.catalog_id,
                capabilities=(screen,),
            ),
            "conditional_analysis_rules": (),
        }
    )
    operator_path = output / "operator" / "operator-run.json"
    run = build_mission_design_run(
        spec=legacy,
        operator_run=verify_operator_run(output / "operator"),
        spec_sha256="0" * 64,
        operator_run_sha256=sha256(operator_path.read_bytes()).hexdigest(),
    )

    payload = mission_design_run_payload(run)
    assert "activation" not in payload["analysis_plan"]["nodes"][0]
    assert "trigger_rule_ids" not in payload["analysis_plan"]["nodes"][0]
    assert "conditional_analyses" not in payload["verification_plan"]
    unsigned = {key: value for key, value in payload.items() if key != "run_sha256"}
    assert (
        run.run_sha256
        == sha256(
            json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    )

    path = tmp_path / "legacy-run.json"
    write_mission_design_run(path, run)
    assert json.loads(path.read_text(encoding="utf-8")) == payload


def test_conditional_analysis_decision_rejects_self_consistent_semantic_tamper(
    tmp_path: Path,
) -> None:
    output = tmp_path / "mission-design"
    _run_checked_bundle(output)
    decision = verify_mission_design_director(output).verification_plan.conditional_analyses[0]
    payload = decision.model_dump(mode="json")
    payload["decision_relevance_score"] = 0.75
    unsigned = {key: value for key, value in payload.items() if key != "decision_sha256"}
    payload["decision_sha256"] = sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    with pytest.raises(ValueError, match="relevance score"):
        ConditionalAnalysisDecision.model_validate(payload)
