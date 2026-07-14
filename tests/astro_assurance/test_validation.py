from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

import astro_assurance.validation_cli as validation_cli
import astro_assurance.validation_runner as validation_runner
from astro_assurance.errors import MissionAssuranceError
from astro_assurance.models import MissionAssuranceCase
from astro_assurance.validation_io import (
    load_paired_assurance_validation_protocol,
    load_paired_assurance_validation_result,
    verify_paired_assurance_validation_result,
    write_paired_assurance_validation_result,
)
from astro_assurance.validation_models import (
    AssuranceValidationProfile,
    AssuranceValidationStatus,
    PairedAssuranceValidationProtocol,
    summarize_validation_pairs,
)
from astro_assurance.validation_runner import run_paired_assurance_validation
from astro_cli.main import app
from astro_core.errors import InvalidScenarioError
from astro_core.models import ForceModelName

PROTOCOL = Path("examples/assurance/paired_force_model_validation.yaml")


@pytest.fixture(scope="module")
def one_pair_result():
    protocol = load_paired_assurance_validation_protocol(PROTOCOL)
    limited = protocol.model_copy(update={"realizations": protocol.realizations[:1]})
    return run_paired_assurance_validation(limited)


@pytest.fixture(scope="module")
def full_result():
    protocol = load_paired_assurance_validation_protocol(PROTOCOL)
    return run_paired_assurance_validation(protocol)


def test_paired_profiles_share_realization_and_preserve_role_evidence(one_pair_result) -> None:
    pair = one_pair_result.pairs[0]
    assert pair.paired_complete
    assert pair.matched.profile is AssuranceValidationProfile.MATCHED_TWO_BODY
    assert pair.mismatched.profile is AssuranceValidationProfile.TRUTH_J2_ESTIMATOR_TWO_BODY
    assert pair.matched.status is AssuranceValidationStatus.SUCCESS
    assert pair.mismatched.status is AssuranceValidationStatus.SUCCESS
    assert pair.matched.truth_force_model is ForceModelName.TWO_BODY
    assert pair.mismatched.truth_force_model is ForceModelName.J2
    assert pair.matched.estimation_force_model is ForceModelName.TWO_BODY
    assert pair.mismatched.estimation_force_model is ForceModelName.TWO_BODY

    matched_case = pair.matched.assurance_case
    mismatched_case = pair.mismatched.assurance_case
    assert matched_case is not None
    assert mismatched_case is not None
    assert matched_case.metadata["tracking_noise_seed"] == 7301
    assert mismatched_case.metadata["tracking_noise_seed"] == 7301
    assert matched_case.metadata["truth_force_model"] == "two_body"
    assert mismatched_case.metadata["truth_force_model"] == "j2"
    assert mismatched_case.metadata["estimation_force_model"] == "two_body"
    assert all(
        measurement.epoch <= matched_case.correction_maneuver.epoch
        for measurement in matched_case.measurements
    )
    assert matched_case.metadata["rejected_after_decision_measurement_count"] > 0

    matched_measurement = matched_case.measurements[0]
    mismatched_measurement = mismatched_case.measurements[0]
    assert matched_measurement.metadata["simulation_truth_sigma"] == pytest.approx(0.0084)
    assert matched_measurement.metadata["estimator_assumed_sigma"] == pytest.approx(0.0092)
    assert matched_measurement.metadata["simulation_truth_bias"] == pytest.approx(-0.0035)
    assert matched_measurement.metadata["estimator_bias"] == pytest.approx(-0.0010)
    assert matched_measurement.metadata["simulation_noise_seed"] == 7301
    assert matched_measurement.metadata["simulation_noise_realization"] == pytest.approx(
        mismatched_measurement.metadata["simulation_noise_realization"]
    )


def test_execution_errors_preserve_scaled_magnitude_and_epoch(one_pair_result) -> None:
    pair = one_pair_result.pairs[0]
    case = pair.matched.assurance_case
    assert case is not None
    commanded = np.linalg.norm(case.correction_maneuver.delta_v_km_s)
    executed = case.truth_corrected_scenario.maneuvers[0]
    assert np.linalg.norm(executed.delta_v_km_s) == pytest.approx(commanded * 0.992)
    assert (executed.epoch - case.correction_maneuver.epoch).total_seconds() == -4.0
    assert executed.metadata["execution_pointing_1_deg"] == -0.08
    assert executed.metadata["execution_pointing_basis"]["convention"][0] == "command_frame"
    assert pair.matched.metrics[
        "original_executed_component_delta_v_margin_km_s"
    ] == pytest.approx(0.02 - max(abs(component) for component in executed.delta_v_km_s))


def test_summary_keeps_profiles_unpooled_and_reports_paired_deltas(one_pair_result) -> None:
    summary = one_pair_result.summary
    assert summary.requested_pairs == 1
    assert summary.paired_complete == 1
    assert summary.matched_passed == 1
    assert summary.mismatched_passed == 0
    assert summary.pass_regressions == 1
    assert summary.denominator_policy == "counts_only_profiles_unpooled_no_probability_estimate"
    od_delta = summary.paired_metric_deltas["od_position_error_km"]
    assert od_delta.count == 1
    assert od_delta.minimum == od_delta.median == od_delta.maximum
    assert "probabilities" not in one_pair_result.model_dump(mode="json")
    assert one_pair_result.pairs[0].matched.assurance_case_passed is True
    assert one_pair_result.pairs[0].matched.passed is True


def test_result_verification_rejects_embedded_case_tampering(
    tmp_path: Path, full_result
) -> None:
    result_path = tmp_path / "paired.json"
    write_paired_assurance_validation_result(result_path, full_result)
    verified = verify_paired_assurance_validation_result(result_path)
    assert verified.protocol_id == full_result.protocol_id

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["pairs"][0]["matched"]["assurance_case"]["metadata"]["tampered"] = True
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(InvalidScenarioError, match="assurance result digest"):
        load_paired_assurance_validation_result(result_path)


def test_result_verification_rejects_forged_manifest_digest(
    tmp_path: Path, full_result
) -> None:
    payload = full_result.model_dump(mode="json")
    profile = payload["pairs"][0]["matched"]
    profile["assurance_case"]["manifest"]["entries"][0]["source_digest"] = "0" * 64
    case = MissionAssuranceCase.model_validate(profile["assurance_case"])
    profile["assurance_result_digest"] = sha256(case.model_dump_json().encode()).hexdigest()
    result_path = tmp_path / "forged-manifest.json"
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(InvalidScenarioError, match="embedded artifact digest mismatch"):
        verify_paired_assurance_validation_result(result_path)


def test_result_verification_rejects_forged_derived_metrics(
    tmp_path: Path, full_result
) -> None:
    payload = full_result.model_dump(mode="json")
    pair = payload["pairs"][0]
    metric = "od_position_error_km"
    pair["matched"]["metrics"][metric] += 1.0
    pair["delta_mismatched_minus_matched"][metric] -= 1.0
    delta_summary = payload["summary"]["paired_metric_deltas"][metric]
    values = [
        item["delta_mismatched_minus_matched"][metric] for item in payload["pairs"]
    ]
    delta_summary["minimum"] = min(values)
    delta_summary["median"] = float(np.median(values))
    delta_summary["maximum"] = max(values)
    result_path = tmp_path / "forged-metrics.json"
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(InvalidScenarioError, match="profile metrics do not match"):
        verify_paired_assurance_validation_result(result_path)


def test_result_verification_rejects_forged_pair_reversal(
    tmp_path: Path, full_result
) -> None:
    payload = full_result.model_dump(mode="json")
    payload["pairs"][0]["pass_reversal"] = "unchanged"
    payload["summary"]["pass_regressions"] -= 1
    payload["summary"]["unchanged_pass_disposition"] += 1
    result_path = tmp_path / "forged-reversal.json"
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(InvalidScenarioError, match="pair reversal does not match"):
        verify_paired_assurance_validation_result(result_path)


def test_result_verification_rejects_profile_slot_swap(
    tmp_path: Path, full_result
) -> None:
    payload = full_result.model_dump(mode="json")
    pair = payload["pairs"][0]
    pair["matched"], pair["mismatched"] = pair["mismatched"], pair["matched"]
    result_path = tmp_path / "swapped-profiles.json"
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(InvalidScenarioError, match="matched slot must contain"):
        load_paired_assurance_validation_result(result_path)


def test_result_verification_rejects_coordinate_case_substitution(
    tmp_path: Path, full_result
) -> None:
    pairs = list(full_result.pairs)
    replacement = pairs[1]
    pairs[0] = pairs[0].model_copy(
        update={
            "matched": replacement.matched,
            "mismatched": replacement.mismatched,
            "paired_complete": replacement.paired_complete,
            "delta_mismatched_minus_matched": replacement.delta_mismatched_minus_matched,
            "pass_reversal": replacement.pass_reversal,
        }
    )
    substituted = full_result.model_copy(
        update={
            "pairs": tuple(pairs),
            "summary": summarize_validation_pairs(tuple(pairs)),
        }
    )
    result_path = tmp_path / "substituted-coordinate.json"
    write_paired_assurance_validation_result(result_path, substituted)

    with pytest.raises(InvalidScenarioError, match="does not match protocol coordinate"):
        verify_paired_assurance_validation_result(result_path)


def test_result_verification_rejects_nested_input_drift(
    tmp_path: Path, full_result
) -> None:
    payload = full_result.model_dump(mode="json")
    first_case = payload["pairs"][0]["matched"]["assurance_case"]
    source = Path(first_case["manifest"]["inputs"][1]["path"])
    copied_source = tmp_path / source.name
    copied_source.write_bytes(source.read_bytes())
    for pair in payload["pairs"]:
        for profile_name in ("matched", "mismatched"):
            profile = pair[profile_name]
            case_payload = profile["assurance_case"]
            if case_payload is None:
                continue
            reference = case_payload["manifest"]["inputs"][1]
            reference["path"] = str(copied_source)
            reference["file_digest"] = sha256(copied_source.read_bytes()).hexdigest()
            case = MissionAssuranceCase.model_validate(case_payload)
            profile["assurance_result_digest"] = sha256(case.model_dump_json().encode()).hexdigest()
    result_path = tmp_path / "nested-input.json"
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    copied_source.write_text("changed after campaign\n", encoding="utf-8")

    with pytest.raises(InvalidScenarioError, match="input digest mismatch"):
        verify_paired_assurance_validation_result(result_path)


def test_protocol_requires_exact_integer_noise_seed() -> None:
    protocol = load_paired_assurance_validation_protocol(PROTOCOL)
    payload = protocol.model_dump(mode="python")
    payload["realizations"][0]["input_overrides"]["tracking_noise_seed"] = 1.5
    with pytest.raises(ValueError, match="must be an integer"):
        PairedAssuranceValidationProtocol.model_validate(payload)


def test_protocol_requires_unique_noise_seeds() -> None:
    protocol = load_paired_assurance_validation_protocol(PROTOCOL)
    payload = protocol.model_dump(mode="python")
    payload["realizations"][1]["input_overrides"]["tracking_noise_seed"] = payload[
        "realizations"
    ][0]["input_overrides"]["tracking_noise_seed"]
    with pytest.raises(ValueError, match="tracking noise seeds must be unique"):
        PairedAssuranceValidationProtocol.model_validate(payload)


def test_source_integrity_failure_aborts_protocol_and_cli_without_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_integrity(_scenario: object) -> object:
        raise MissionAssuranceError("source changed", phase="manifest")

    monkeypatch.setattr(validation_runner, "run_post_launch_assurance", fail_integrity)
    protocol = load_paired_assurance_validation_protocol(PROTOCOL)
    limited = protocol.model_copy(update={"realizations": protocol.realizations[:1]})
    with pytest.raises(MissionAssuranceError, match="source changed"):
        run_paired_assurance_validation(limited)

    monkeypatch.setattr(
        validation_cli,
        "run_paired_assurance_validation",
        lambda _protocol: fail_integrity(object()),
    )
    output = tmp_path / "must-not-exist.json"
    result = CliRunner().invoke(
        app,
        [
            "run-assurance-validation",
            str(PROTOCOL),
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 2
    assert not output.exists()


def test_validation_command_resolves_protocol_outside_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol_path = PROTOCOL.resolve()
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["validate-assurance-validation", str(protocol_path)])
    assert result.exit_code == 0
    assert json.loads(result.stdout) == {
        "protocol_id": "leo-paired-assurance-validation-v1",
        "valid": True,
    }


def test_run_command_rejects_output_summary_path_collision(tmp_path: Path) -> None:
    output = tmp_path / "same-output.json"
    result = CliRunner().invoke(
        app,
        [
            "run-assurance-validation",
            str(PROTOCOL),
            "--output",
            str(output),
            "--summary-output",
            str(output),
        ],
    )

    assert result.exit_code == 2
    assert "must be different paths" in result.output
    assert not output.exists()
