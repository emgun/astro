from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

import astro_assurance.lifecycle_review as lifecycle_review_module
from astro_assistant.models import ArtifactKind, WorkflowArtifact
from astro_assistant.validators import validate_artifact
from astro_assurance.lifecycle_review import (
    review_mission_lifecycle,
    verify_mission_lifecycle_result,
    verify_mission_lifecycle_review,
)
from astro_assurance.lifecycle_review_io import (
    load_mission_lifecycle_review,
    write_mission_lifecycle_review,
)
from astro_assurance.lifecycle_review_models import LifecycleReviewDisposition
from astro_cli.main import app
from astro_core.errors import InvalidScenarioError
from astro_mission.io import (
    load_mission_lifecycle_scenario,
    write_mission_lifecycle_result,
)
from astro_mission.runner import run_mission_lifecycle

SCENARIO = Path("examples/lifecycle/leo_round_trip.yaml")


@pytest.fixture(scope="module")
def lifecycle_result_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    scenario = load_mission_lifecycle_scenario(SCENARIO)
    result = run_mission_lifecycle(scenario)
    path = tmp_path_factory.mktemp("lifecycle-review") / "lifecycle.json"
    write_mission_lifecycle_result(path, result)
    return path


def test_checked_lifecycle_review_is_verified_and_bounded(
    lifecycle_result_path: Path,
) -> None:
    review = review_mission_lifecycle(lifecycle_result_path, SCENARIO)

    assert review.integrity_verified
    assert review.lifecycle_passed
    assert review.disposition is LifecycleReviewDisposition.ADDITIONAL_REVIEW_REQUIRED
    assert len(review.triage_actions) == 1
    assert review.triage_actions[0].source_finding_id == (
        "margin_unit_digital_twin_mass_budget_rollup_margin_kg"
    )
    assert len(review.findings) == 13
    assert {reference.role for reference in review.referenced_inputs} == {
        "launch_scenario",
        "twin_scenario",
        "reentry_scenario",
    }
    assert {finding.finding_id for finding in review.findings} >= {
        "integrity_verified",
        "phase_manifest",
        "continuity_all_passed",
        "margins_all_passed",
        "claim_boundary",
    }


def test_lifecycle_result_verifier_rejects_schema_valid_tampering(
    tmp_path: Path, lifecycle_result_path: Path
) -> None:
    payload = json.loads(lifecycle_result_path.read_text(encoding="utf-8"))
    payload["warnings"] = payload["warnings"][:-1]
    tampered = tmp_path / "tampered-result.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(InvalidScenarioError, match="does not match"):
        verify_mission_lifecycle_result(tampered, SCENARIO)


def test_lifecycle_review_rejects_optional_backend(
    tmp_path: Path, lifecycle_result_path: Path
) -> None:
    payload = yaml.safe_load(SCENARIO.read_text(encoding="utf-8"))
    payload["launch_backend"] = "rocketpy"
    optional = tmp_path / "optional.yaml"
    optional.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(InvalidScenarioError, match="supports local"):
        verify_mission_lifecycle_result(lifecycle_result_path, optional)


def test_lifecycle_execution_failure_is_a_structured_verification_error(
    tmp_path: Path, lifecycle_result_path: Path
) -> None:
    payload = yaml.safe_load(SCENARIO.read_text(encoding="utf-8"))
    payload["deorbit"]["minimum_propellant_reserve_kg"] = 1000.0
    failing = tmp_path / "failing.yaml"
    failing.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "verify-mission-lifecycle-result",
            str(lifecycle_result_path),
            str(failing),
        ],
    )

    assert result.exit_code == 2
    assert "could not reproduce" in result.output


def test_numeric_scenario_id_produces_valid_normalized_review_id(tmp_path: Path) -> None:
    payload = yaml.safe_load(SCENARIO.read_text(encoding="utf-8"))
    payload["scenario_id"] = "123"
    scenario_path = tmp_path / "numeric.yaml"
    scenario_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    scenario = load_mission_lifecycle_scenario(scenario_path)
    result_path = tmp_path / "numeric.json"
    write_mission_lifecycle_result(result_path, run_mission_lifecycle(scenario))

    review = review_mission_lifecycle(result_path, scenario_path)

    assert review.review_id == "value_123-lifecycle-review-v1"


def test_verification_executes_captured_referenced_bytes_during_aba_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = yaml.safe_load(SCENARIO.read_text(encoding="utf-8"))
    for key in ("launch_scenario", "twin_scenario", "reentry_scenario"):
        source = Path(payload[key])
        copied = tmp_path / source.name
        copied.write_bytes(source.read_bytes())
        payload[key] = str(copied)
    scenario_path = tmp_path / "scenario.yaml"
    scenario_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    scenario = load_mission_lifecycle_scenario(scenario_path)
    result_path = tmp_path / "result.json"
    write_mission_lifecycle_result(result_path, run_mission_lifecycle(scenario))
    original_run = lifecycle_review_module._run_captured_lifecycle

    def run_during_replacement(
        captured_scenario: object,
        referenced_bytes: object,
    ) -> object:
        launch_path = Path(payload["launch_scenario"])
        original = launch_path.read_bytes()
        launch_path.write_text("not: the captured launch scenario\n", encoding="utf-8")
        try:
            return original_run(captured_scenario, referenced_bytes)  # type: ignore[arg-type]
        finally:
            launch_path.write_bytes(original)

    monkeypatch.setattr(
        lifecycle_review_module, "_run_captured_lifecycle", run_during_replacement
    )

    assert verify_mission_lifecycle_result(result_path, scenario_path).passed


def test_review_verifier_and_artifact_validator_reject_tampering(
    tmp_path: Path, lifecycle_result_path: Path
) -> None:
    review = review_mission_lifecycle(lifecycle_result_path, SCENARIO)
    output = tmp_path / "review.json"
    payload = review.model_dump(mode="json")
    payload["lifecycle_passed"] = False
    output.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(InvalidScenarioError, match="does not match"):
        verify_mission_lifecycle_review(output)
    artifact = WorkflowArtifact(path=str(output), kind=ArtifactKind.MISSION_LIFECYCLE_REVIEW)
    assert not validate_artifact(artifact)


def test_lifecycle_review_commands_write_and_verify_artifact(
    tmp_path: Path, lifecycle_result_path: Path
) -> None:
    output = tmp_path / "review.json"
    summary = tmp_path / "review.txt"
    result = CliRunner().invoke(
        app,
        [
            "review-mission-lifecycle",
            str(lifecycle_result_path),
            str(SCENARIO),
            "--output",
            str(output),
            "--summary-output",
            str(summary),
        ],
    )

    assert result.exit_code == 0
    assert "Disposition: additional_review_required" in result.stdout
    assert load_mission_lifecycle_review(output).integrity_verified
    verification = CliRunner().invoke(
        app, ["verify-mission-lifecycle-review", str(output)]
    )
    assert verification.exit_code == 0
    assert "Integrity verified: true" in verification.stdout


def test_lifecycle_review_command_rejects_hard_link_alias(
    tmp_path: Path, lifecycle_result_path: Path
) -> None:
    summary_alias = tmp_path / "summary-alias.json"
    summary_alias.hardlink_to(lifecycle_result_path)
    original = lifecycle_result_path.read_bytes()

    result = CliRunner().invoke(
        app,
        [
            "review-mission-lifecycle",
            str(lifecycle_result_path),
            str(SCENARIO),
            "--output",
            str(tmp_path / "review.json"),
            "--summary-output",
            str(summary_alias),
        ],
    )

    assert result.exit_code == 2
    assert "different files" in result.output
    assert lifecycle_result_path.read_bytes() == original


def test_lifecycle_review_json_is_deterministic(
    tmp_path: Path, lifecycle_result_path: Path
) -> None:
    review = review_mission_lifecycle(lifecycle_result_path, SCENARIO)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write_mission_lifecycle_review(first, review)
    write_mission_lifecycle_review(second, review)

    assert first.read_bytes() == second.read_bytes()
