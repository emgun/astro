from __future__ import annotations

from datetime import timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from astro_assurance.io import load_post_launch_assurance_scenario
from astro_assurance.validation_models import (
    AssuranceCalibrationAuthority,
    AssuranceValidationCalibrationManifest,
    InsertionCovarianceEvidence,
    PairedAssuranceValidationProtocol,
    PropulsionExecutionResidualEvidence,
    StationResidualEvidence,
)
from astro_core.errors import InvalidScenarioError
from astro_core.io import load_scenario

_OVERRIDE_PARAMETERS = (
    ("tracking_range_sigma_km", "km"),
    ("tracking_range_rate_sigma_km_s", "km/s"),
    ("tracking_range_bias_km", "km"),
    ("tracking_range_rate_bias_km_s", "km/s"),
    ("estimation_range_sigma_km", "km"),
    ("estimation_range_rate_sigma_km_s", "km/s"),
    ("estimation_range_bias_km", "km"),
    ("estimation_range_rate_bias_km_s", "km/s"),
    ("correction_execution_scale", "ratio"),
    ("correction_execution_epoch_offset_s", "s"),
    ("correction_execution_pointing_1_deg", "deg"),
    ("correction_execution_pointing_2_deg", "deg"),
    ("twin_solar_array_efficiency", "ratio"),
    ("twin_battery_capacity_wh", "W*h"),
)


def load_assurance_validation_calibration(
    path: Path | str,
) -> AssuranceValidationCalibrationManifest:
    calibration_path = Path(path)
    try:
        source_bytes = calibration_path.read_bytes()
        raw: Any = yaml.safe_load(source_bytes.decode("utf-8"))
    except (OSError, UnicodeError) as exc:
        raise InvalidScenarioError(
            f"Could not read assurance calibration evidence {calibration_path}: {exc}"
        ) from exc
    except yaml.YAMLError as exc:
        raise InvalidScenarioError(
            f"Could not parse assurance calibration evidence {calibration_path}: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise InvalidScenarioError(
            f"Assurance calibration evidence {calibration_path} must contain a mapping"
        )
    try:
        manifest = AssuranceValidationCalibrationManifest.model_validate(raw)
    except ValidationError as exc:
        raise InvalidScenarioError(
            f"Assurance calibration evidence {calibration_path} is invalid: {exc}"
        ) from exc
    return manifest.model_copy(
        update={
            "source_path": str(calibration_path.resolve()),
            "source_digest": sha256(source_bytes).hexdigest(),
        }
    )


def inspect_assurance_validation_calibration(
    calibration: AssuranceValidationCalibrationManifest,
    protocol: PairedAssuranceValidationProtocol | None = None,
) -> dict[str, Any]:
    evidence_counts = {
        kind: sum(
            evidence.kind == kind for evidence in calibration.evidence_products
        )
        for kind in (
            "station_residuals",
            "propulsion_execution_residuals",
            "insertion_covariance",
        )
    }
    calibrated_bounds = [
        bound
        for bound in calibration.parameter_bounds
        if bound.authority
        in {
            AssuranceCalibrationAuthority.MISSION_TEST_CALIBRATED,
            AssuranceCalibrationAuthority.FLIGHT_CALIBRATED,
        }
    ]
    blockers: list[str] = []
    for kind, count in evidence_counts.items():
        if count == 0:
            blockers.append(f"missing_{kind}")
    illustrative_parameters = sorted(
        bound.parameter
        for bound in calibration.parameter_bounds
        if bound.authority is AssuranceCalibrationAuthority.ILLUSTRATIVE
    )
    if illustrative_parameters:
        blockers.append("illustrative_parameter_bounds_remain")
    if not calibrated_bounds:
        blockers.append("no_mission_or_flight_calibrated_bounds")
    protocol_complete: bool | None = None
    if protocol is None:
        blockers.append("protocol_coverage_not_checked")
    else:
        validate_calibration_against_protocol(calibration, protocol)
        protocol_complete = True
    return {
        "calibration_id": calibration.calibration_id,
        "protocol_id": calibration.protocol_id,
        "valid": True,
        "promotion_status": calibration.promotion_status.value,
        "evidence_counts": evidence_counts,
        "parameter_bound_count": len(calibration.parameter_bounds),
        "calibrated_bound_count": len(calibrated_bounds),
        "illustrative_parameters": illustrative_parameters,
        "promotion_blockers": blockers,
        "protocol_complete": protocol_complete,
        "claim_boundary": calibration.claim_boundary,
        "source_digest": calibration.source_digest,
    }


def validate_calibration_against_protocol(
    calibration: AssuranceValidationCalibrationManifest,
    protocol: PairedAssuranceValidationProtocol,
) -> None:
    if calibration.protocol_id != protocol.protocol_id:
        raise InvalidScenarioError("calibration evidence protocol id does not match")
    configured = assurance_validation_parameter_values(protocol)
    declared = {bound.parameter: bound for bound in calibration.parameter_bounds}
    missing = sorted(set(configured) - set(declared))
    extra = sorted(set(declared) - set(configured))
    if missing or extra:
        raise InvalidScenarioError(
            f"calibration parameter coverage mismatch: missing={missing}, extra={extra}"
        )
    for parameter, (unit, values) in configured.items():
        bound = declared[parameter]
        if bound.unit != unit:
            raise InvalidScenarioError(
                f"calibration unit mismatch for {parameter}: {bound.unit} != {unit}"
            )
        outside = [
            value
            for value in values
            if value < float(bound.minimum) or value > float(bound.maximum)
        ]
        if outside:
            raise InvalidScenarioError(
                f"configured values exceed calibration envelope for {parameter}: {outside}"
            )
    _validate_evidence_context(calibration, protocol)


def _validate_evidence_context(
    calibration: AssuranceValidationCalibrationManifest,
    protocol: PairedAssuranceValidationProtocol,
) -> None:
    calibrated_evidence = [
        evidence
        for evidence in calibration.evidence_products
        if evidence.authority
        in {
            AssuranceCalibrationAuthority.MISSION_TEST_CALIBRATED,
            AssuranceCalibrationAuthority.FLIGHT_CALIBRATED,
        }
    ]
    if not calibrated_evidence:
        return
    assurance_path = Path(protocol.assurance_scenario)
    assurance = load_post_launch_assurance_scenario(assurance_path)
    tracking_path = _resolve_reference(assurance_path, assurance.tracking_scenario)
    tracking = load_scenario(tracking_path)
    station_ids = {station.name for station in tracking.ground_stations}
    for evidence in calibrated_evidence:
        if evidence.assurance_scenario_id != assurance.scenario_id:
            raise InvalidScenarioError(
                f"calibration evidence {evidence.evidence_id} assurance scenario mismatch"
            )
        if isinstance(evidence, StationResidualEvidence):
            if evidence.tracking_scenario_id != tracking.scenario_id:
                raise InvalidScenarioError(
                    f"calibration evidence {evidence.evidence_id} tracking scenario mismatch"
                )
            if evidence.station_id not in station_ids:
                raise InvalidScenarioError(
                    f"calibration evidence {evidence.evidence_id} station is not configured"
                )
        elif isinstance(evidence, PropulsionExecutionResidualEvidence):
            expected_maneuver_id = assurance.metadata.get("correction_maneuver_id")
            expected_propulsion_class = assurance.metadata.get("propulsion_class")
            if not isinstance(expected_maneuver_id, str) or not isinstance(
                expected_propulsion_class, str
            ):
                raise InvalidScenarioError(
                    "calibrated propulsion evidence requires assurance scenario maneuver and "
                    "propulsion-class metadata"
                )
            expected_epoch = tracking.initial_state.epoch + timedelta(
                seconds=float(protocol.correction_elapsed_s)
            )
            if (
                evidence.maneuver_id != expected_maneuver_id
                or evidence.propulsion_class != expected_propulsion_class
                or evidence.vector_frame is not tracking.initial_state.frame
                or evidence.commanded_epoch != expected_epoch
            ):
                raise InvalidScenarioError(
                    f"calibration evidence {evidence.evidence_id} propulsion context mismatch"
                )
        elif (
            evidence.tracking_scenario_id != tracking.scenario_id
            or evidence.epoch != tracking.initial_state.epoch
            or evidence.time_scale is not tracking.initial_state.time_scale
            or evidence.central_body is not tracking.initial_state.central_body
            or evidence.frame is not tracking.initial_state.frame
        ):
            raise InvalidScenarioError(
                f"calibration evidence {evidence.evidence_id} insertion context mismatch"
            )
        if isinstance(evidence, InsertionCovarianceEvidence):
            launch_path = _resolve_reference(assurance_path, assurance.launch_scenario)
            evidence_launch_path = _resolve_reference(
                assurance_path, evidence.launcher_configuration
            )
            if evidence_launch_path.resolve() != launch_path.resolve():
                raise InvalidScenarioError(
                    f"calibration evidence {evidence.evidence_id} launcher mismatch"
                )


def _resolve_reference(owner_path: Path, configured_path: str) -> Path:
    configured = Path(configured_path)
    if configured.is_absolute():
        return configured
    for parent in owner_path.resolve().parents:
        candidate = parent / configured
        if candidate.exists():
            return candidate
    return owner_path.parent / configured


def assurance_validation_parameter_values(
    protocol: PairedAssuranceValidationProtocol,
) -> dict[str, tuple[str, tuple[float, ...]]]:
    values: dict[str, tuple[str, list[float]]] = {
        "tracking_duration_s": ("s", [float(protocol.tracking_duration_s)]),
        "correction_elapsed_s": ("s", [float(protocol.correction_elapsed_s)]),
        "verification_elapsed_s": ("s", [float(protocol.verification_elapsed_s)]),
        "diagnostic_maximum_component_delta_v_km_s": (
            "km/s",
            [float(protocol.diagnostic_maximum_component_delta_v_km_s)],
        ),
        "diagnostic_maximum_total_delta_v_km_s": (
            "km/s",
            [float(protocol.diagnostic_maximum_total_delta_v_km_s)],
        ),
    }

    def append(parameter: str, unit: str, value: float) -> None:
        if parameter not in values:
            values[parameter] = (unit, [])
        configured_unit, configured_values = values[parameter]
        if configured_unit != unit:
            raise InvalidScenarioError(f"internal calibration unit mismatch for {parameter}")
        configured_values.append(value)

    for realization in protocol.realizations:
        for index, value in enumerate(realization.dispersion.position_delta_km):
            append(f"dispersion.position_delta_km[{index}]", "km", float(value))
        for index, value in enumerate(realization.dispersion.velocity_delta_km_s):
            append(f"dispersion.velocity_delta_km_s[{index}]", "km/s", float(value))
        overrides = realization.input_overrides
        for field, unit in _OVERRIDE_PARAMETERS:
            value = getattr(overrides, field)
            if value is not None:
                append(f"input_overrides.{field}", unit, float(value))
        for node in overrides.twin_thermal_node_overrides:
            if node.emissivity is not None:
                append(
                    f"input_overrides.twin_thermal_node_overrides.{node.node_name}.emissivity",
                    "ratio",
                    float(node.emissivity),
                )
            if node.internal_heat_fraction is not None:
                append(
                    "input_overrides.twin_thermal_node_overrides."
                    f"{node.node_name}.internal_heat_fraction",
                    "ratio",
                    float(node.internal_heat_fraction),
                )
    return {
        parameter: (unit, tuple(configured_values))
        for parameter, (unit, configured_values) in values.items()
    }
