from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from astro_assurance.validation_calibration_io import (
    inspect_assurance_validation_calibration,
    load_assurance_validation_calibration,
    validate_calibration_against_protocol,
)
from astro_assurance.validation_io import load_paired_assurance_validation_protocol
from astro_assurance.validation_models import (
    AssuranceCalibrationPromotionStatus,
    AssuranceValidationCalibrationManifest,
    InsertionCovarianceEvidence,
    PropulsionExecutionResidualEvidence,
    StationResidualEvidence,
)
from astro_cli.main import app
from astro_core.errors import InvalidScenarioError

EVIDENCE = Path("examples/assurance/mission_calibration_evidence_example.yaml")
CALIBRATION = Path("examples/assurance/paired_force_model_calibration.yaml")
PROTOCOL = Path("examples/assurance/paired_force_model_validation.yaml")


def test_checked_evidence_pack_is_typed_but_non_promoting() -> None:
    calibration = load_assurance_validation_calibration(EVIDENCE)
    report = inspect_assurance_validation_calibration(calibration)

    assert calibration.promotion_status is AssuranceCalibrationPromotionStatus.ILLUSTRATIVE
    assert [type(product) for product in calibration.evidence_products] == [
        StationResidualEvidence,
        PropulsionExecutionResidualEvidence,
        InsertionCovarianceEvidence,
    ]
    covariance = calibration.evidence_products[2]
    assert isinstance(covariance, InsertionCovarianceEvidence)
    assert covariance.standard_deviations == (1.0, 1.0, 1.0, 0.001, 0.001, 0.001)
    assert covariance.correlation_matrix[0] == (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert report["evidence_counts"] == {
        "station_residuals": 1,
        "propulsion_execution_residuals": 1,
        "insertion_covariance": 1,
    }
    assert report["promotion_blockers"] == [
        "illustrative_parameter_bounds_remain",
        "no_mission_or_flight_calibrated_bounds",
        "protocol_coverage_not_checked",
    ]
    assert report["protocol_complete"] is None


def test_station_residual_rejects_observable_unit_mismatch() -> None:
    payload = _product_payload(StationResidualEvidence)
    payload["unit"] = "km/s"
    with pytest.raises(ValidationError, match="unit does not match"):
        StationResidualEvidence.model_validate(payload)


def test_propulsion_residual_rejects_timing_and_scale_mismatch() -> None:
    payload = _product_payload(PropulsionExecutionResidualEvidence)
    payload["timing_residual_s"] = -2.0
    with pytest.raises(ValidationError, match="achieved minus commanded"):
        PropulsionExecutionResidualEvidence.model_validate(payload)

    payload = _product_payload(PropulsionExecutionResidualEvidence)
    payload["magnitude_scale"] = 0.99
    with pytest.raises(ValidationError, match="does not match"):
        PropulsionExecutionResidualEvidence.model_validate(payload)

    payload = _product_payload(PropulsionExecutionResidualEvidence)
    payload["pointing_residual_1_deg"] = 1.0
    with pytest.raises(ValidationError, match="pointing residuals"):
        PropulsionExecutionResidualEvidence.model_validate(payload)


def test_covariance_rejects_ambiguous_or_nonphysical_matrix() -> None:
    payload = _product_payload(InsertionCovarianceEvidence)
    payload["state_units"][-1] = "km"
    with pytest.raises(ValidationError, match="state units"):
        InsertionCovarianceEvidence.model_validate(payload)

    payload = _product_payload(InsertionCovarianceEvidence)
    payload["covariance"][0][1] = 0.5
    with pytest.raises(ValidationError, match="symmetric"):
        InsertionCovarianceEvidence.model_validate(payload)

    payload = _product_payload(InsertionCovarianceEvidence)
    payload["covariance"][0][1] = 2.0
    payload["covariance"][1][0] = 2.0
    with pytest.raises(ValidationError, match="positive semidefinite"):
        InsertionCovarianceEvidence.model_validate(payload)

    payload = _product_payload(InsertionCovarianceEvidence)
    tiny = 1e-14
    payload["covariance"] = [
        [tiny, 2.0 * tiny, 0.0, 0.0, 0.0, 0.0],
        [2.0 * tiny, tiny, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, tiny, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, tiny, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, tiny, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, tiny],
    ]
    with pytest.raises(ValidationError, match="positive semidefinite"):
        InsertionCovarianceEvidence.model_validate(payload)


def test_mission_authority_requires_matching_source_evidence_and_derivation() -> None:
    calibration = load_assurance_validation_calibration(EVIDENCE)
    payload = calibration.model_dump(mode="json")
    payload["sources"][0]["source_kind"] = "mission_test_data"
    station = payload["evidence_products"][0]
    station["authority"] = "mission_test_calibrated"
    payload["parameter_bounds"][0].update(
        {
            "minimum": 0.01,
            "maximum": 0.01,
            "authority": "mission_test_calibrated",
            "evidence_ids": [station["evidence_id"]],
            "derivation": {
                "method": "residual_summary_envelope",
                "rationale": "Reviewed mission-test residual envelope.",
            },
        }
    )
    payload["promotion_status"] = "mission_calibrated"
    promoted = AssuranceValidationCalibrationManifest.model_validate(payload)
    assert promoted.promotion_status is AssuranceCalibrationPromotionStatus.MISSION_CALIBRATED

    payload["parameter_bounds"][0]["parameter"] = "dispersion.position_delta_km[0]"
    with pytest.raises(ValidationError, match="does not apply"):
        AssuranceValidationCalibrationManifest.model_validate(payload)


def test_mission_bound_must_equal_evidence_derived_envelope() -> None:
    calibration = load_assurance_validation_calibration(EVIDENCE)
    payload = calibration.model_dump(mode="json")
    payload["sources"][0]["source_kind"] = "mission_test_data"
    payload["evidence_products"][0]["authority"] = "mission_test_calibrated"
    payload["parameter_bounds"][0].update(
        {
            "minimum": 999.0,
            "maximum": 1000.0,
            "authority": "mission_test_calibrated",
            "evidence_ids": ["synthetic-station-range"],
            "derivation": {
                "method": "residual_summary_envelope",
                "rationale": "Attempted forged envelope.",
            },
        }
    )
    payload["promotion_status"] = "mission_calibrated"
    with pytest.raises(ValidationError, match="evidence-derived envelope"):
        AssuranceValidationCalibrationManifest.model_validate(payload)


def test_range_evidence_cannot_support_range_rate_bound() -> None:
    calibration = load_assurance_validation_calibration(EVIDENCE)
    payload = calibration.model_dump(mode="json")
    payload["sources"][0]["source_kind"] = "mission_test_data"
    payload["evidence_products"][0]["authority"] = "mission_test_calibrated"
    payload["parameter_bounds"][0].update(
        {
            "parameter": "input_overrides.tracking_range_rate_sigma_km_s",
            "minimum": 0.01,
            "maximum": 0.01,
            "unit": "km/s",
            "authority": "mission_test_calibrated",
            "evidence_ids": ["synthetic-station-range"],
            "derivation": {
                "method": "residual_summary_envelope",
                "rationale": "Wrong observable evidence.",
            },
        }
    )
    payload["promotion_status"] = "mission_calibrated"
    with pytest.raises(ValidationError, match="does not apply"):
        AssuranceValidationCalibrationManifest.model_validate(payload)


@pytest.mark.parametrize("index", ["-1", "3", "6", "x"])
def test_insertion_evidence_rejects_invalid_coordinate_index(index: str) -> None:
    calibration = load_assurance_validation_calibration(EVIDENCE)
    payload = calibration.model_dump(mode="json")
    payload["sources"][0]["source_kind"] = "mission_test_data"
    covariance = payload["evidence_products"][2]
    covariance["authority"] = "mission_test_calibrated"
    payload["parameter_bounds"][0].update(
        {
            "parameter": f"dispersion.position_delta_km[{index}]",
            "minimum": -1.0,
            "maximum": 1.0,
            "authority": "mission_test_calibrated",
            "evidence_ids": [covariance["evidence_id"]],
            "derivation": {
                "method": "symmetric_covariance_sigma_envelope",
                "sigma_multiplier": 1.0,
                "rationale": "Invalid coordinate probe.",
            },
        }
    )
    payload["promotion_status"] = "mission_calibrated"
    with pytest.raises(ValidationError, match="does not apply"):
        AssuranceValidationCalibrationManifest.model_validate(payload)


def test_protocol_validation_rejects_unbound_propulsion_context() -> None:
    calibration = load_assurance_validation_calibration(CALIBRATION)
    payload = calibration.model_dump(mode="json")
    payload["sources"].append(
        {
            "source_id": "mission-propulsion-residuals",
            "title": "Reviewed mission propulsion residuals",
            "publisher": "Mission test team",
            "document_id": "propulsion-residuals-v1",
            "revision_or_date": "2026-07-14",
            "location": "reviewed://propulsion-residuals-v1",
            "source_kind": "mission_test_data",
            "applicability": "Unrelated propulsion context probe.",
            "limitations": ["Test-only contract fixture."],
        }
    )
    propulsion = _product_payload(PropulsionExecutionResidualEvidence)
    propulsion.update(
        {
            "source_ids": ["mission-propulsion-residuals"],
            "authority": "mission_test_calibrated",
            "assurance_scenario_id": "post-launch-orbit-acquisition",
            "maneuver_id": "unrelated-maneuver",
            "propulsion_class": "unrelated-propulsion",
            "commanded_epoch": "1990-01-01T00:00:00Z",
            "achieved_epoch": "1990-01-01T00:00:02Z",
        }
    )
    payload["evidence_products"] = [propulsion]
    manifest = AssuranceValidationCalibrationManifest.model_validate(payload)
    protocol = load_paired_assurance_validation_protocol(PROTOCOL)
    with pytest.raises(InvalidScenarioError, match="requires assurance scenario"):
        validate_calibration_against_protocol(manifest, protocol)


def test_forged_mission_evidence_authority_is_rejected() -> None:
    calibration = load_assurance_validation_calibration(EVIDENCE)
    payload = calibration.model_dump(mode="json")
    payload["evidence_products"][0]["authority"] = "mission_test_calibrated"
    with pytest.raises(ValidationError, match="lacks a source"):
        AssuranceValidationCalibrationManifest.model_validate(payload)


def test_protocol_validation_binds_calibrated_station_context() -> None:
    calibration = load_assurance_validation_calibration(CALIBRATION)
    payload = calibration.model_dump(mode="json")
    payload["sources"].append(
        {
            "source_id": "mission-station-residuals",
            "title": "Reviewed mission station residuals",
            "publisher": "Mission test team",
            "document_id": "station-residuals-v1",
            "revision_or_date": "2026-07-14",
            "location": "reviewed://station-residuals-v1",
            "source_kind": "mission_test_data",
            "applicability": "Checked assurance tracking configuration.",
            "limitations": ["Test-only contract fixture."],
        }
    )
    station = _product_payload(StationResidualEvidence)
    station.update(
        {
            "source_ids": ["mission-station-residuals"],
            "authority": "mission_test_calibrated",
            "assurance_scenario_id": "post-launch-orbit-acquisition",
            "tracking_scenario_id": "post-launch-assurance-tracking",
            "station_id": "kourou",
            "sample_standard_deviation": 0.0084,
            "rms": 0.0085,
        }
    )
    second = dict(station)
    second.update(
        {
            "evidence_id": "mission-station-range-second",
            "station_id": "canaveral",
            "sample_standard_deviation": 0.0116,
            "rms": 0.0117,
        }
    )
    payload["evidence_products"] = [station, second]
    bound = next(
        item
        for item in payload["parameter_bounds"]
        if item["parameter"] == "input_overrides.tracking_range_sigma_km"
    )
    bound.update(
        {
            "authority": "mission_test_calibrated",
            "source_ids": ["mission-station-residuals"],
            "evidence_ids": [station["evidence_id"], second["evidence_id"]],
            "derivation": {
                "method": "residual_summary_envelope",
                "rationale": "Exact station standard-deviation envelope.",
            },
        }
    )
    manifest = AssuranceValidationCalibrationManifest.model_validate(payload)
    protocol = load_paired_assurance_validation_protocol(PROTOCOL)
    validate_calibration_against_protocol(manifest, protocol)

    payload["evidence_products"][0]["station_id"] = "unconfigured-station"
    invalid = AssuranceValidationCalibrationManifest.model_validate(payload)
    with pytest.raises(InvalidScenarioError, match="station is not configured"):
        validate_calibration_against_protocol(invalid, protocol)


def test_inspection_command_reports_digest_and_blockers() -> None:
    result = CliRunner().invoke(app, ["inspect-assurance-calibration", str(EVIDENCE)])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["valid"] is True
    assert len(payload["source_digest"]) == 64
    assert payload["promotion_status"] == "illustrative"
    assert payload["calibrated_bound_count"] == 0


def _product_payload(product_type: type[object]) -> dict[str, Any]:
    calibration = load_assurance_validation_calibration(EVIDENCE)
    product = next(
        item for item in calibration.evidence_products if isinstance(item, product_type)
    )
    return product.model_dump(mode="json")
