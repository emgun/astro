from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from astro_assurance.errors import MissionAssuranceError
from astro_assurance.model_form_io import (
    load_model_form_factorial_protocol,
    verify_model_form_factorial_result,
    write_model_form_factorial_result,
)
from astro_assurance.model_form_models import MODEL_FORM_FACTORIAL_CONTRAST_IDS
from astro_assurance.model_form_runner import run_model_form_factorial
from astro_assurance.validation_models import AssuranceValidationProfile
from astro_cli.main import app
from astro_core.errors import InvalidScenarioError
from astro_core.models import ForceModelName

PROTOCOL = Path("examples/assurance/model_form_matrix_validation.yaml")


def _one_case_protocol(tmp_path: Path) -> Path:
    raw = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    raw["realizations"] = raw["realizations"][:1]
    assurance_source = Path("examples/assurance/post_launch_orbit_acquisition.yaml")
    assurance = yaml.safe_load(assurance_source.read_text(encoding="utf-8"))
    for key in ("launch_scenario", "tracking_scenario", "twin_scenario"):
        assurance[key] = str(Path(assurance[key]).resolve())
    assurance_path = tmp_path / "assurance.yaml"
    assurance_path.write_text(yaml.safe_dump(assurance, sort_keys=False), encoding="utf-8")
    calibration_path = tmp_path / "calibration.yaml"
    calibration_path.write_bytes(
        Path("examples/assurance/paired_force_model_calibration.yaml").read_bytes()
    )
    raw["assurance_scenario"] = str(assurance_path)
    raw["calibration_evidence"] = str(calibration_path)
    path = tmp_path / "model_form_matrix.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def test_protocol_requires_exact_ordered_factorial(tmp_path: Path) -> None:
    raw = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    raw["profiles"] = list(reversed(raw["profiles"]))
    path = tmp_path / "reordered.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(InvalidScenarioError, match="exact required order"):
        load_model_form_factorial_protocol(path)


def test_one_realization_runs_all_force_roles_with_common_seed(tmp_path: Path) -> None:
    protocol = load_model_form_factorial_protocol(_one_case_protocol(tmp_path))
    result = run_model_form_factorial(protocol)
    realization = result.realizations[0]

    assert [cell.profile_result.profile for cell in realization.cells] == list(
        protocol.profiles
    )
    assert [
        (
            cell.profile_result.truth_force_model,
            cell.profile_result.estimation_force_model,
        )
        for cell in realization.cells
    ] == [
        (ForceModelName.TWO_BODY, ForceModelName.TWO_BODY),
        (ForceModelName.TWO_BODY, ForceModelName.J2),
        (ForceModelName.J2, ForceModelName.TWO_BODY),
        (ForceModelName.J2, ForceModelName.J2),
    ]
    assert realization.realization.input_overrides.tracking_noise_seed == 7301
    assert tuple(contrast.contrast_id for contrast in realization.contrasts) == (
        MODEL_FORM_FACTORIAL_CONTRAST_IDS
    )


def test_contrasts_are_exact_and_counts_remain_unpooled(tmp_path: Path) -> None:
    result = run_model_form_factorial(
        load_model_form_factorial_protocol(_one_case_protocol(tmp_path))
    )
    realization = result.realizations[0]
    cells = {
        cell.profile_result.profile: cell.profile_result for cell in realization.cells
    }
    low, high, interaction = realization.contrasts
    metric = next(iter(low.metric_deltas))

    assert low.metric_deltas[metric] == pytest.approx(
        cells[AssuranceValidationProfile.TRUTH_TWO_BODY_ESTIMATOR_J2].metrics[metric]
        - cells[AssuranceValidationProfile.MATCHED_TWO_BODY].metrics[metric]
    )
    assert high.metric_deltas[metric] == pytest.approx(
        cells[AssuranceValidationProfile.MATCHED_J2].metrics[metric]
        - cells[AssuranceValidationProfile.TRUTH_J2_ESTIMATOR_TWO_BODY].metrics[metric]
    )
    assert interaction.metric_deltas[metric] == pytest.approx(
        high.metric_deltas[metric] - low.metric_deltas[metric]
    )
    assert result.summary.denominator_policy.startswith("counts_only")
    assert all(counts.requested == 1 for counts in result.summary.profile_counts.values())


def test_result_verifier_reexecutes_exact_protocol(tmp_path: Path) -> None:
    protocol_path = _one_case_protocol(tmp_path)
    result = run_model_form_factorial(load_model_form_factorial_protocol(protocol_path))
    result_path = tmp_path / "result.json"
    write_model_form_factorial_result(result_path, result)

    verified = verify_model_form_factorial_result(result_path)
    assert verified.model_dump(mode="json") == result.model_dump(mode="json")


def test_result_verifier_rejects_source_tampering(tmp_path: Path) -> None:
    protocol_path = _one_case_protocol(tmp_path)
    result = run_model_form_factorial(load_model_form_factorial_protocol(protocol_path))
    result_path = tmp_path / "result.json"
    write_model_form_factorial_result(result_path, result)
    protocol_path.write_text(protocol_path.read_text() + "\n", encoding="utf-8")

    with pytest.raises(InvalidScenarioError, match="protocol source digest mismatch"):
        verify_model_form_factorial_result(result_path)


@pytest.mark.parametrize("source_name", ["assurance.yaml", "calibration.yaml"])
def test_result_verifier_rejects_bound_input_tampering(
    tmp_path: Path, source_name: str
) -> None:
    protocol_path = _one_case_protocol(tmp_path)
    result = run_model_form_factorial(load_model_form_factorial_protocol(protocol_path))
    result_path = tmp_path / "result.json"
    write_model_form_factorial_result(result_path, result)
    source = tmp_path / source_name
    source.write_text(source.read_text() + "\n", encoding="utf-8")

    with pytest.raises(InvalidScenarioError, match="source digest mismatch"):
        verify_model_form_factorial_result(result_path)


def test_result_verifier_rejects_derived_metric_forgery(tmp_path: Path) -> None:
    protocol_path = _one_case_protocol(tmp_path)
    result = run_model_form_factorial(load_model_form_factorial_protocol(protocol_path))
    result_path = tmp_path / "result.json"
    write_model_form_factorial_result(result_path, result)
    payload = yaml.safe_load(result_path.read_text(encoding="utf-8"))
    payload["realizations"][0]["cells"][0]["profile_result"]["metrics"][
        "od_position_error_km"
    ] += 1.0
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(InvalidScenarioError):
        verify_model_form_factorial_result(result_path)


@pytest.mark.parametrize(
    "source_name", ["model_form_matrix.yaml", "assurance.yaml", "calibration.yaml"]
)
def test_runner_rejects_input_drift_during_execution(
    tmp_path: Path, source_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import astro_assurance.model_form_runner as runner

    protocol_path = _one_case_protocol(tmp_path)
    protocol = load_model_form_factorial_protocol(protocol_path)
    original = runner.run_assurance_validation_profile
    calls = 0

    def mutating_profile(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        value = original(*args, **kwargs)
        if calls == 4:
            source = tmp_path / source_name
            source.write_text(source.read_text() + "\n", encoding="utf-8")
        return value

    monkeypatch.setattr(runner, "run_assurance_validation_profile", mutating_profile)
    with pytest.raises(MissionAssuranceError, match="changed during execution"):
        run_model_form_factorial(protocol)


def test_cli_validate_run_and_verify(tmp_path: Path) -> None:
    protocol_path = _one_case_protocol(tmp_path)
    result_path = tmp_path / "result.json"
    runner = CliRunner()

    validated = runner.invoke(app, ["validate-model-form-matrix", str(protocol_path)])
    executed = runner.invoke(
        app,
        ["run-model-form-matrix", str(protocol_path), "--output", str(result_path)],
    )
    verified = runner.invoke(app, ["verify-model-form-matrix", str(result_path)])

    assert validated.exit_code == 0, validated.output
    assert executed.exit_code == 0, executed.output
    assert verified.exit_code == 0, verified.output
    assert '"valid": true' in verified.output


def test_cli_rejects_output_collision(tmp_path: Path) -> None:
    protocol_path = _one_case_protocol(tmp_path)
    output = tmp_path / "same.txt"
    result = CliRunner().invoke(
        app,
        [
            "run-model-form-matrix",
            str(protocol_path),
            "--output",
            str(output),
            "--summary-output",
            str(output),
        ],
    )

    assert result.exit_code == 2
    assert "must be different paths" in result.output
