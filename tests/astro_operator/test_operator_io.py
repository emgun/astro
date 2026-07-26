from __future__ import annotations

import json
from pathlib import Path

import pytest

from astro_core.errors import InvalidScenarioError
from astro_operator.io import verify_operator_run, write_operator_run
from astro_operator.models import (
    AuthorityGrant,
    AuthorityLevel,
    CommandRequest,
    CommandResult,
    DesignVariable,
    MetricGoal,
    MissionObjective,
    OperatorAction,
    OperatorActionKind,
    OperatorRun,
    OperatorRunStatus,
    OperatorStep,
)


def test_checked_operator_example_runs_and_verifies(tmp_path: Path) -> None:
    from astro_cli.main import app
    from tests.astro_cli.helpers import make_cli_runner

    output = tmp_path / "operator-run"
    result = make_cli_runner().invoke(
        app,
        [
            "run-mission-operator",
            "examples/operator/leo_lifecycle_trade_study.yaml",
            "--reasoner-replay",
            "examples/operator/leo_lifecycle_trade_study_replay.yaml",
            "--output-dir",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    run = verify_operator_run(output)
    assert run.selected_candidate_id == "higher-reserve"
    assert [step.observation.evaluation_status for step in run.steps[:3] if step.observation] == [
        "evaluated",
        "evaluation_failed",
        "evaluated",
    ]
    assert all(step.reasoner_invocation is not None for step in run.steps)
    providers = {
        step.reasoner_invocation.provider for step in run.steps if step.reasoner_invocation
    }
    assert providers == {"deterministic-replay"}

    trace_path = output / "operator-run.json"
    trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
    trace_payload["steps"][0]["reasoner_invocation"]["provider"] = "forged-provider"
    trace_path.write_text(json.dumps(trace_payload), encoding="utf-8")
    with pytest.raises(InvalidScenarioError, match="record digest"):
        verify_operator_run(output)
    write_operator_run(trace_path, run)

    trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
    del trace_payload["steps"][0]["reasoner_invocation"]
    trace_path.write_text(json.dumps(trace_payload), encoding="utf-8")
    with pytest.raises(InvalidScenarioError, match="requires reasoner provenance"):
        verify_operator_run(output)
    write_operator_run(trace_path, run)

    trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
    trace_payload["schema_version"] = "1.0"
    trace_path.write_text(json.dumps(trace_payload), encoding="utf-8")
    with pytest.raises(InvalidScenarioError, match="does not contain reasoner provenance"):
        verify_operator_run(output)
    write_operator_run(trace_path, run)

    trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
    trace_payload["schema_version"] = "2.0"
    trace_path.write_text(json.dumps(trace_payload), encoding="utf-8")
    with pytest.raises(InvalidScenarioError, match="Could not verify"):
        verify_operator_run(output)
    write_operator_run(trace_path, run)

    candidate_result = output / "candidates" / "higher-reserve" / "result.json"
    candidate_result.write_text("{}", encoding="utf-8")
    with pytest.raises(InvalidScenarioError, match="digest mismatch"):
        verify_operator_run(output)


def test_perception_to_decision_example_binds_baseline_context_and_fails_tampering(
    tmp_path: Path,
) -> None:
    from astro_cli.main import app
    from tests.astro_cli.helpers import make_cli_runner

    output = tmp_path / "post-launch-review"
    result = make_cli_runner().invoke(
        app,
        [
            "run-mission-operator",
            "examples/operator/post_launch_recovery_review.yaml",
            "--reasoner-replay",
            "examples/operator/post_launch_recovery_review_replay.yaml",
            "--output-dir",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    run = verify_operator_run(output)
    assert run.schema_version == "1.4"
    assert run.mission_context is not None
    assert run.mission_context.mission_id == "leo-reference-mission"
    assert run.mission_context.baseline_id == "leo-mission-design:baseline"
    assert run.mission_context.operational_configuration_id == ("post-launch-recovery-baseline-v1")
    assert len(run.steps) == 4
    assert run.world_state is not None
    assert len(run.world_state.assertions) == 16
    claim = run.steps[-1].action.conclusion_claims[0]
    assert len(claim.predicates) == 10
    assert {item.epistemic_kind.value for item in run.known_evidence[1:]} == {
        "simulated",
        "estimated",
        "declared",
    }

    trace = output / "operator-run.json"
    payload = json.loads(trace.read_text(encoding="utf-8"))
    payload["schema_version"] = "1.3"
    trace.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(InvalidScenarioError, match="older schemas forbid"):
        verify_operator_run(output)
    write_operator_run(trace, run)

    procedure = output / "evidence" / "procedure-post-launch-procedure.yaml"
    procedure.write_text("maximum_position_sigma_km: 99\n", encoding="utf-8")
    with pytest.raises(InvalidScenarioError, match="digest mismatch"):
        verify_operator_run(output)


def test_verifier_rejects_forged_out_of_grant_command_history(tmp_path: Path) -> None:
    command = CommandRequest(command_id="forged", command_type="burn")
    execute = OperatorAction(
        action_id="execute-forged",
        kind=OperatorActionKind.EXECUTE_COMMAND,
        rationale="Forge a command history.",
        command=command,
    )
    finish = OperatorAction(
        action_id="finish",
        kind=OperatorActionKind.FINISH,
        rationale="Finish the forged run.",
        conclusion="Forged.",
    )
    run = OperatorRun(
        objective=MissionObjective(
            objective_id="forged",
            summary="Forged journal fixture.",
            design_variables=(
                DesignVariable(
                    variable_id="mass",
                    target="spacecraft_wet_mass_kg",
                    lower_bound=1.0,
                    upper_bound=2.0,
                    unit="kg",
                ),
            ),
            metric_goals=(MetricGoal(metric_id="margin", objective="maximize", unit="kg"),),
        ),
        authority=AuthorityGrant(
            grant_id="research-only",
            level=AuthorityLevel.RESEARCH,
            mission_scope="forged fixture",
            allowed_actions=(OperatorActionKind.FINISH,),
            max_steps=2,
            max_candidate_evaluations=0,
        ),
        status=OperatorRunStatus.COMPLETED,
        steps=(
            OperatorStep(
                sequence=1,
                action=execute,
                command_result=CommandResult(command=command, status="executed"),
            ),
            OperatorStep(sequence=2, action=finish),
        ),
        known_evidence=(),
        conclusion="Forged.",
    )
    output = tmp_path / "forged"
    output.mkdir()
    write_operator_run(output / "operator-run.json", run)

    with pytest.raises(InvalidScenarioError, match="outside grant"):
        verify_operator_run(output)


def test_verifier_rejects_symlinked_evidence_artifact(tmp_path: Path) -> None:
    from astro_cli.main import app
    from tests.astro_cli.helpers import make_cli_runner

    output = tmp_path / "operator-run"
    result = make_cli_runner().invoke(
        app,
        [
            "run-mission-operator",
            "examples/operator/leo_lifecycle_trade_study.yaml",
            "--reasoner-replay",
            "examples/operator/leo_lifecycle_trade_study_replay.yaml",
            "--output-dir",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    artifact = output / "candidates" / "higher-reserve" / "result.json"
    external = tmp_path / "external.json"
    external.write_bytes(artifact.read_bytes())
    artifact.unlink()
    artifact.symlink_to(external)

    with pytest.raises(InvalidScenarioError, match="symbolic link"):
        verify_operator_run(output)


def test_checked_spec_resolves_from_outside_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from astro_cli.main import app
    from tests.astro_cli.helpers import make_cli_runner

    repository = Path(__file__).resolve().parents[2]
    spec = repository / "examples" / "operator" / "leo_lifecycle_trade_study.yaml"
    replay = repository / "examples" / "operator" / "leo_lifecycle_trade_study_replay.yaml"
    output = tmp_path / "portable-run"
    monkeypatch.chdir(tmp_path)

    result = make_cli_runner().invoke(
        app,
        [
            "run-mission-operator",
            str(spec),
            "--reasoner-replay",
            str(replay),
            "--output-dir",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert verify_operator_run(output).selected_candidate_id == "higher-reserve"
