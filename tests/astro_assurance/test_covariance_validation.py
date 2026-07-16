from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from astro_assurance.covariance_validation_io import (
    load_covariance_validation_protocol,
    run_covariance_validation,
    verify_covariance_validation_result,
    write_covariance_validation_result,
)
from astro_assurance.covariance_validation_models import CovarianceValidationDisposition
from astro_cli.main import app
from astro_core.errors import InvalidScenarioError
from astro_core.io import load_scenario
from astro_dynamics.local import propagate_local


def _write_trajectory_pair(tmp_path: Path) -> tuple[Path, Path]:
    scenario = load_scenario("examples/scenarios/leo_covariance.yaml").model_copy(
        update={"covariance_process_noise_acceleration_km_s2": 0.0}
    )
    trajectory = propagate_local(scenario)
    trajectory = trajectory.model_copy(
        update={
            "metadata": {
                **trajectory.metadata,
                "covariance_units_policy": {
                    "frame": "EME2000",
                    "representation": "cartesian",
                    "time_scale": "UTC",
                    "state_order": ["x", "y", "z", "vx", "vy", "vz"],
                    "state_units": ["km", "km", "km", "km/s", "km/s", "km/s"],
                    "covariance_units_policy": "outer_product_of_state_units",
                },
            }
        }
    )
    candidate = tmp_path / "candidate.json"
    reference = tmp_path / "reference.json"
    candidate_trajectory = trajectory.model_copy(
        update={
            "backend": "orekit",
            "metadata": {
                **trajectory.metadata,
                "covariance_implementation": "candidate-implementation",
            }
        }
    )
    reference_trajectory = trajectory.model_copy(
        update={
            "backend": "tudat",
            "metadata": {
                **trajectory.metadata,
                "covariance_implementation": "reference-implementation",
            }
        }
    )
    candidate.write_text(candidate_trajectory.model_dump_json(indent=2), encoding="utf-8")
    reference.write_text(reference_trajectory.model_dump_json(indent=2), encoding="utf-8")
    return candidate, reference


def _thresholds() -> dict[str, float | int]:
    return {
        "minimum_epochs": 2,
        "symmetry_tolerance": 1.0e-12,
        "minimum_eigenvalue": 1.0e-16,
        "maximum_condition_number": 1.0e12,
        "maximum_relative_covariance_frobenius_error": 1.0e-12,
        "covariance_trace_ratio_minimum": 0.999999,
        "covariance_trace_ratio_maximum": 1.000001,
        "maximum_accumulated_state_transition_frobenius_error": 1.0e-12,
        "generalized_eigenvalue_minimum": 0.999999,
        "generalized_eigenvalue_maximum": 1.000001,
        "maximum_state_position_delta_km": 1.0e-12,
        "maximum_state_velocity_delta_km_s": 1.0e-12,
        "confidence_level": 0.95,
        "minimum_empirical_samples": 30,
        "minimum_coverage": 0.9,
    }


def _protocol_payload(
    candidate: Path,
    reference: Path,
    *,
    empirical: Path | None = None,
    independence_review: Path | None = None,
    independent: bool = False,
) -> dict[str, Any]:
    return {
        "protocol_id": "covariance-validation-test-v1",
        "candidate_trajectory_path": str(candidate),
        "reference_trajectory_path": str(reference),
        "empirical_evidence_path": None if empirical is None else str(empirical),
        "empirical_scenario_path": (
            None if empirical is None else str(empirical.with_name("empirical-scenario.yaml"))
        ),
        "independence_review_path": (
            None if independence_review is None else str(independence_review)
        ),
        "units_policy": {},
        "thresholds": _thresholds(),
        "independence": {
            "candidate_implementation": "candidate-implementation",
            "reference_implementation": "reference-implementation",
            "independent_implementations": independent,
            "rationale": "Explicit test declaration.",
        },
        "required_force_features": ["two_body"],
    }


def _write_protocol(tmp_path: Path, payload: dict[str, Any]) -> Path:
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _write_empirical(
    tmp_path: Path, predictor: Path, error_scale: float = 1.0
) -> Path:
    error = [float(np.sqrt(6.0) * error_scale), 0.0, 0.0, 0.0, 0.0, 0.0]
    identity = np.eye(6).tolist()
    initial_covariance = np.diag([1.0, 1.0, 1.0, 1.0e-6, 1.0e-6, 1.0e-6]).tolist()
    scenario_path = tmp_path / "empirical-scenario.yaml"
    scenario_payload = yaml.safe_load(
        Path("examples/scenarios/leo_covariance.yaml").read_text(encoding="utf-8")
    )
    scenario_payload["covariance_process_noise_acceleration_km_s2"] = 0.0
    scenario_path.write_text(
        yaml.safe_dump(scenario_payload, sort_keys=False), encoding="utf-8"
    )
    payload = {
        "artifact_id": "empirical-covariance-test-v1",
        "units_policy": {},
        "population_definition": "Deterministic test population.",
        "independent_realizations": True,
        "independence_basis": "Each test sample has a distinct realization epoch.",
        "campaign_provenance": {
            "scenario_id": "leo-covariance",
            "scenario_sha256": sha256(scenario_path.read_bytes()).hexdigest(),
            "predictor_trajectory_sha256": sha256(predictor.read_bytes()).hexdigest(),
            "predictor_backend": "orekit",
            "predictor_implementation": "candidate-implementation",
            "truth_backend": "tudat",
            "seed": 817,
            "evaluation_epoch": "2026-01-01T00:00:00Z",
            "initial_covariance": initial_covariance,
            "sample_count": 30,
            "force_model": {"gravity": "two_body"},
        },
        "samples": [
            {
                "sample_id": f"sample-{index:02d}",
                "epoch": "2026-01-01T00:00:00Z",
                "state_error": error,
                "predicted_covariance": identity,
                "independent_truth": True,
                "initial_state_perturbation": [
                    float(index + 1) * 1.0e-6,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                "nominal_truth_state": [0.0] * 6,
                "realized_truth_state": error,
            }
            for index in range(30)
        ],
    }
    path = tmp_path / "empirical.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_independence_review(tmp_path: Path) -> Path:
    payload = {
        "review_id": "covariance-independence-review-v1",
        "reviewer": "test-reviewer",
        "reviewed_at": "2026-01-01T00:00:00Z",
        "candidate_implementation": "candidate-implementation",
        "reference_implementation": "reference-implementation",
        "evidence_reviewed": ["implementation architecture", "dependency graph"],
        "conclusion": "independent_implementations",
        "rationale": "Test implementations are declared structurally independent.",
        "limitations": ["Synthetic test review only."],
    }
    path = tmp_path / "independence-review.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_identical_local_products_still_require_independent_empirical_evidence(
    tmp_path: Path,
) -> None:
    candidate, reference = _write_trajectory_pair(tmp_path)
    protocol = load_covariance_validation_protocol(
        _write_protocol(tmp_path, _protocol_payload(candidate, reference))
    )

    result = run_covariance_validation(protocol)

    assert result.comparison_summary.passed_epochs == 11
    assert result.disposition is CovarianceValidationDisposition.ADDITIONAL_EVIDENCE_REQUIRED
    assert {blocker.blocker_id for blocker in result.blockers} == {
        "independent_implementation_missing",
        "empirical_consistency_missing",
    }
    assert result.certification_claim == "no_certification_claim"


def test_complete_consistent_evidence_satisfies_only_preregistered_criteria(
    tmp_path: Path,
) -> None:
    candidate, reference = _write_trajectory_pair(tmp_path)
    empirical = _write_empirical(tmp_path, candidate)
    review = _write_independence_review(tmp_path)
    protocol = load_covariance_validation_protocol(
        _write_protocol(
            tmp_path,
            _protocol_payload(
                candidate,
                reference,
                empirical=empirical,
                independence_review=review,
                independent=True,
            ),
        )
    )

    result = run_covariance_validation(protocol)

    assert result.disposition is CovarianceValidationDisposition.CRITERIA_SATISFIED
    assert result.empirical_nees_summary is not None
    assert result.empirical_nees_summary.mean_nees == pytest.approx(6.0)
    assert result.empirical_nees_summary.coverage == 1.0
    assert result.empirical_nees_summary.coverage_lower_confidence_bound > 0.9
    assert not result.blockers
    assert "not_flight_certification" in result.claim_boundary


def test_independent_evidence_rejects_mismatched_producer_provenance(
    tmp_path: Path,
) -> None:
    candidate, reference = _write_trajectory_pair(tmp_path)
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    payload["metadata"]["covariance_implementation"] = "relabeled-implementation"
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    empirical = _write_empirical(tmp_path, candidate)
    review = _write_independence_review(tmp_path)
    protocol = load_covariance_validation_protocol(
        _write_protocol(
            tmp_path,
            _protocol_payload(
                candidate,
                reference,
                empirical=empirical,
                independence_review=review,
                independent=True,
            ),
        )
    )

    result = run_covariance_validation(protocol)

    assert result.disposition is CovarianceValidationDisposition.ADDITIONAL_EVIDENCE_REQUIRED
    assert {blocker.blocker_id for blocker in result.blockers} == {
        "implementation_provenance_mismatch"
    }


def test_empirical_campaign_must_match_comparison_applicability(tmp_path: Path) -> None:
    candidate, reference = _write_trajectory_pair(tmp_path)
    empirical = _write_empirical(tmp_path, candidate)
    empirical_payload = json.loads(empirical.read_text(encoding="utf-8"))
    empirical_payload["campaign_provenance"]["force_model"] = {
        "gravity": "j2"
    }
    empirical.write_text(json.dumps(empirical_payload), encoding="utf-8")
    protocol = load_covariance_validation_protocol(
        _write_protocol(
            tmp_path,
            _protocol_payload(candidate, reference, empirical=empirical),
        )
    )

    with pytest.raises(
        InvalidScenarioError,
        match="empirical campaign provenance does not match comparison applicability",
    ):
        run_covariance_validation(protocol)


def test_zero_process_noise_empirical_campaign_rejects_nonzero_q_comparison(
    tmp_path: Path,
) -> None:
    candidate, reference = _write_trajectory_pair(tmp_path)
    reference_payload = json.loads(reference.read_text(encoding="utf-8"))
    reference_payload["covariance_history"][1]["process_noise_covariance"][0][0] = 1.0e-9
    reference.write_text(json.dumps(reference_payload), encoding="utf-8")
    empirical = _write_empirical(tmp_path, candidate)
    protocol = load_covariance_validation_protocol(
        _write_protocol(
            tmp_path,
            _protocol_payload(candidate, reference, empirical=empirical),
        )
    )

    with pytest.raises(
        InvalidScenarioError,
        match="empirical campaign provenance does not match comparison applicability",
    ):
        run_covariance_validation(protocol)


def test_empirical_campaign_rejects_unbound_scenario_digest(tmp_path: Path) -> None:
    candidate, reference = _write_trajectory_pair(tmp_path)
    empirical = _write_empirical(tmp_path, candidate)
    empirical_payload = json.loads(empirical.read_text(encoding="utf-8"))
    empirical_payload["campaign_provenance"]["scenario_sha256"] = "f" * 64
    empirical.write_text(json.dumps(empirical_payload), encoding="utf-8")
    protocol = load_covariance_validation_protocol(
        _write_protocol(
            tmp_path,
            _protocol_payload(candidate, reference, empirical=empirical),
        )
    )

    with pytest.raises(
        InvalidScenarioError,
        match="empirical campaign provenance does not match comparison applicability",
    ):
        run_covariance_validation(protocol)


def test_native_provenance_cannot_be_satisfied_by_relabeling_local_products(
    tmp_path: Path,
) -> None:
    candidate, reference = _write_trajectory_pair(tmp_path)
    for path, implementation in (
        (candidate, "orekit_native_variational"),
        (reference, "tudat_native_variational"),
    ):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["metadata"]["covariance_implementation"] = implementation
        path.write_text(json.dumps(payload), encoding="utf-8")
    empirical = _write_empirical(tmp_path, candidate)
    empirical_payload = json.loads(empirical.read_text(encoding="utf-8"))
    empirical_payload["campaign_provenance"]["predictor_implementation"] = (
        "orekit_native_variational"
    )
    empirical.write_text(json.dumps(empirical_payload), encoding="utf-8")
    review = _write_independence_review(tmp_path)
    review_payload = json.loads(review.read_text(encoding="utf-8"))
    review_payload["candidate_implementation"] = "orekit_native_variational"
    review_payload["reference_implementation"] = "tudat_native_variational"
    review.write_text(json.dumps(review_payload), encoding="utf-8")
    protocol_payload = _protocol_payload(
        candidate,
        reference,
        empirical=empirical,
        independence_review=review,
        independent=True,
    )
    protocol_payload["independence"]["candidate_implementation"] = (
        "orekit_native_variational"
    )
    protocol_payload["independence"]["reference_implementation"] = (
        "tudat_native_variational"
    )
    protocol = load_covariance_validation_protocol(
        _write_protocol(tmp_path, protocol_payload)
    )

    result = run_covariance_validation(protocol)

    assert {blocker.blocker_id for blocker in result.blockers} == {
        "implementation_provenance_mismatch"
    }


def test_checked_native_orekit_tudat_campaign_only_requires_empirical_evidence() -> None:
    campaign_directory = Path(
        "examples/covariance_validation/live_orekit_tudat_native"
    )
    manifest = json.loads(
        (campaign_directory / "campaign_manifest.json").read_text(encoding="utf-8")
    )
    for relative_path, expected_digest in manifest["files"].items():
        assert sha256((campaign_directory / relative_path).read_bytes()).hexdigest() == (
            expected_digest
        )
    for relative_path, expected_digest in manifest["source_scenarios"].items():
        assert sha256(Path(relative_path).read_bytes()).hexdigest() == expected_digest
    protocol = load_covariance_validation_protocol(
        campaign_directory / "protocol.yaml"
    )

    result = run_covariance_validation(protocol)

    assert result.comparison_summary.requested_epochs == 11
    assert result.comparison_summary.passed_epochs == 11
    assert result.disposition is CovarianceValidationDisposition.ADDITIONAL_EVIDENCE_REQUIRED
    assert {blocker.blocker_id for blocker in result.blockers} == {
        "empirical_consistency_missing"
    }
    summary = result.comparison_summary
    expected = manifest["expected"]
    assert summary.maximum_relative_covariance_frobenius_error <= expected[
        "maximum_relative_covariance_frobenius_error"
    ]
    assert summary.maximum_accumulated_state_transition_frobenius_error <= expected[
        "maximum_accumulated_state_transition_frobenius_error"
    ]
    assert summary.covariance_trace_ratio_minimum >= expected[
        "covariance_trace_ratio_minimum"
    ]
    assert summary.covariance_trace_ratio_maximum <= expected[
        "covariance_trace_ratio_maximum"
    ]
    assert summary.generalized_eigenvalue_minimum >= expected[
        "generalized_eigenvalue_minimum"
    ]
    assert summary.generalized_eigenvalue_maximum <= expected[
        "generalized_eigenvalue_maximum"
    ]
    assert summary.maximum_state_position_delta_km <= expected[
        "maximum_state_position_delta_km"
    ]
    assert summary.maximum_state_velocity_delta_km_s <= expected[
        "maximum_state_velocity_delta_km_s"
    ]


def test_checked_native_empirical_campaign_satisfies_preregistered_criteria() -> None:
    campaign_directory = Path(
        "examples/covariance_validation/live_orekit_tudat_native"
    )
    manifest = json.loads(
        (campaign_directory / "campaign_manifest.json").read_text(encoding="utf-8")
    )
    protocol = load_covariance_validation_protocol(
        campaign_directory / "complete_protocol.yaml"
    )

    result = run_covariance_validation(protocol)

    expected = manifest["empirical_expected"]
    assert result.disposition.value == expected["disposition"]
    assert result.blockers == ()
    assert result.empirical_nees_summary is not None
    summary = result.empirical_nees_summary
    assert summary.sample_count == expected["sample_count"]
    assert expected["mean_nees_minimum"] <= summary.mean_nees <= expected[
        "mean_nees_maximum"
    ]
    assert summary.samples_within_bounds == expected["coverage_count"]
    assert summary.coverage == expected["coverage_fraction"]
    assert summary.coverage_lower_confidence_bound >= expected[
        "coverage_lower_confidence_bound_minimum"
    ]
    assert summary.criteria_satisfied
    assert result.certification_claim == "no_certification_claim"


def test_underdispersed_empirical_covariance_fails_criteria(tmp_path: Path) -> None:
    candidate, reference = _write_trajectory_pair(tmp_path)
    empirical = _write_empirical(tmp_path, candidate, error_scale=0.0)
    review = _write_independence_review(tmp_path)
    protocol = load_covariance_validation_protocol(
        _write_protocol(
            tmp_path,
            _protocol_payload(
                candidate,
                reference,
                empirical=empirical,
                independence_review=review,
                independent=True,
            ),
        )
    )

    result = run_covariance_validation(protocol)

    assert result.disposition is CovarianceValidationDisposition.CRITERIA_FAILED
    assert result.empirical_nees_summary is not None
    assert not result.empirical_nees_summary.criteria_satisfied


def test_underpowered_consistent_empirical_campaign_requires_more_evidence(
    tmp_path: Path,
) -> None:
    candidate, reference = _write_trajectory_pair(tmp_path)
    empirical = _write_empirical(tmp_path, candidate)
    payload = json.loads(empirical.read_text(encoding="utf-8"))
    payload["samples"] = payload["samples"][:2]
    payload["campaign_provenance"]["sample_count"] = 2
    empirical.write_text(json.dumps(payload), encoding="utf-8")
    review = _write_independence_review(tmp_path)
    protocol = load_covariance_validation_protocol(
        _write_protocol(
            tmp_path,
            _protocol_payload(
                candidate,
                reference,
                empirical=empirical,
                independence_review=review,
                independent=True,
            ),
        )
    )

    result = run_covariance_validation(protocol)

    assert result.disposition is CovarianceValidationDisposition.ADDITIONAL_EVIDENCE_REQUIRED
    assert {blocker.blocker_id for blocker in result.blockers} == {
        "empirical_sample_count_insufficient"
    }


def test_comparison_reports_asymmetric_covariance_failure(tmp_path: Path) -> None:
    candidate, reference = _write_trajectory_pair(tmp_path)
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    payload["covariance_history"][0]["covariance"][0][1] = 1.0e-4
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    protocol = load_covariance_validation_protocol(
        _write_protocol(tmp_path, _protocol_payload(candidate, reference))
    )

    result = run_covariance_validation(protocol)

    assert result.disposition is CovarianceValidationDisposition.CRITERIA_FAILED
    assert "symmetry" in result.diagnostics[0].failed_criteria


def test_trace_ratio_and_transition_error_are_required(tmp_path: Path) -> None:
    candidate, reference = _write_trajectory_pair(tmp_path)
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    payload["covariance_history"][0]["covariance"] = (
        np.asarray(payload["covariance_history"][0]["covariance"]) * 1.1
    ).tolist()
    transition = payload["covariance_history"][0]["accumulated_state_transition_matrix"]
    transition[0][0] += 0.1
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    protocol = load_covariance_validation_protocol(
        _write_protocol(tmp_path, _protocol_payload(candidate, reference))
    )

    result = run_covariance_validation(protocol)

    assert result.disposition is CovarianceValidationDisposition.CRITERIA_FAILED
    failed = result.diagnostics[0].failed_criteria
    assert "covariance_trace_ratio_maximum" in failed
    assert "accumulated_state_transition_frobenius_error" in failed


def test_missing_transition_matrix_is_invalid_evidence(tmp_path: Path) -> None:
    candidate, reference = _write_trajectory_pair(tmp_path)
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    payload["covariance_history"][0]["accumulated_state_transition_matrix"] = None
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    protocol = load_covariance_validation_protocol(
        _write_protocol(tmp_path, _protocol_payload(candidate, reference))
    )

    with pytest.raises(InvalidScenarioError, match="requires accumulated"):
        run_covariance_validation(protocol)


def test_verifier_rejects_bound_source_and_result_tampering(tmp_path: Path) -> None:
    candidate, reference = _write_trajectory_pair(tmp_path)
    protocol_path = _write_protocol(tmp_path, _protocol_payload(candidate, reference))
    result = run_covariance_validation(load_covariance_validation_protocol(protocol_path))
    result_path = tmp_path / "result.json"
    write_covariance_validation_result(result_path, result)
    assert verify_covariance_validation_result(result_path).protocol_id == result.protocol_id

    candidate.write_text(candidate.read_text() + "\n", encoding="utf-8")
    with pytest.raises(InvalidScenarioError, match="candidate_trajectory digest mismatch"):
        verify_covariance_validation_result(result_path)


def test_runner_rejects_source_change_after_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import astro_assurance.covariance_validation_runner as validation_runner

    candidate, reference = _write_trajectory_pair(tmp_path)
    protocol = load_covariance_validation_protocol(
        _write_protocol(tmp_path, _protocol_payload(candidate, reference))
    )
    original = validation_runner.assess_covariance_validation

    def mutating_assessment(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        candidate.write_text(candidate.read_text() + "\n", encoding="utf-8")
        return result

    monkeypatch.setattr(validation_runner, "assess_covariance_validation", mutating_assessment)
    with pytest.raises(InvalidScenarioError, match="changed during assessment"):
        run_covariance_validation(protocol)


def test_cli_writes_verifiable_additional_evidence_result_with_exit_one(
    tmp_path: Path,
) -> None:
    candidate, reference = _write_trajectory_pair(tmp_path)
    protocol = _write_protocol(tmp_path, _protocol_payload(candidate, reference))
    output = tmp_path / "result.json"
    runner = CliRunner()

    validated = runner.invoke(app, ["validate-covariance-validation", str(protocol)])
    assessed = runner.invoke(
        app,
        ["assess-covariance-validation", str(protocol), "--output", str(output)],
    )
    verified = runner.invoke(app, ["verify-covariance-validation", str(output)])

    assert validated.exit_code == 0, validated.output
    assert assessed.exit_code == 1, assessed.output
    assert output.exists()
    assert verified.exit_code == 0, verified.output


def test_empirical_artifact_rejects_duplicate_observations(tmp_path: Path) -> None:
    candidate, reference = _write_trajectory_pair(tmp_path)
    empirical = _write_empirical(tmp_path, candidate)
    review = _write_independence_review(tmp_path)
    payload = json.loads(empirical.read_text(encoding="utf-8"))
    payload["samples"][1]["initial_state_perturbation"] = payload["samples"][0][
        "initial_state_perturbation"
    ]
    empirical.write_text(json.dumps(payload), encoding="utf-8")
    protocol = _write_protocol(
        tmp_path,
        _protocol_payload(
            candidate,
            reference,
            empirical=empirical,
            independence_review=review,
            independent=True,
        ),
    )

    result = CliRunner().invoke(
        app,
        [
            "assess-covariance-validation",
            str(protocol),
            "--output",
            str(tmp_path / "result.json"),
        ],
    )

    assert result.exit_code == 2
    assert "must not be duplicated" in result.output
