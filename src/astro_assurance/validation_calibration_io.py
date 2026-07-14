from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from astro_assurance.validation_models import (
    AssuranceValidationCalibrationManifest,
    PairedAssuranceValidationProtocol,
)
from astro_core.errors import InvalidScenarioError

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
