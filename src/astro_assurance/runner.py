from __future__ import annotations

import json
from datetime import datetime, timedelta
from hashlib import sha256
from math import exp, radians, tan
from pathlib import Path
from typing import Any

import numpy as np

from astro_assurance.errors import MissionAssuranceError
from astro_assurance.models import (
    AssuranceContinuityCheck,
    AssuranceContinuityReport,
    AssuranceInputReference,
    AssuranceManifest,
    AssuranceManifestEntry,
    AssuranceMargin,
    AssuranceMarginReport,
    AssuranceStatus,
    MissionAssuranceCase,
    PostLaunchAssuranceScenario,
)
from astro_assurance.targeting import design_candidate_correction, trajectory_sample_at_elapsed
from astro_core.io import load_scenario
from astro_core.models import (
    AstroModel,
    CartesianState,
    ForceModelName,
    MeasurementRecord,
    MeasurementType,
    OrbitState,
    Scenario,
    Trajectory,
    Vector3,
)
from astro_dynamics.local import propagate_local
from astro_launch.backends import propagate_launch_with_backend
from astro_launch.io import load_launch_scenario
from astro_od.estimation import estimate_initial_state
from astro_od.measurements import elevation_deg, generate_synthetic_measurements
from astro_twin.io import load_twin_scenario
from astro_twin.models import DigitalTwinResult, DigitalTwinScenario
from astro_twin.runner import run_digital_twin

_STANDARD_GRAVITY_M_S2 = 9.80665
_STATE_TOLERANCE = 1.0e-12
_MASS_TOLERANCE_KG = 1.0e-6
_RANGE_TYPES = {
    MeasurementType.RANGE,
    MeasurementType.TWO_WAY_RANGE,
    MeasurementType.THREE_WAY_RANGE,
}
_RANGE_RATE_TYPES = {
    MeasurementType.RANGE_RATE,
    MeasurementType.TWO_WAY_RANGE_RATE,
    MeasurementType.THREE_WAY_RANGE_RATE,
}


def run_post_launch_assurance(
    scenario: PostLaunchAssuranceScenario,
) -> MissionAssuranceCase:
    assurance_input = _loaded_assurance_reference(scenario)
    _assert_input_unchanged(assurance_input)
    launch_input = _input_reference("launch_scenario", scenario.launch_scenario)
    launch_template = load_launch_scenario(scenario.launch_scenario)
    _assert_input_unchanged(launch_input)
    launch = propagate_launch_with_backend(launch_template, scenario.launch_backend)
    tracking_input = _input_reference("tracking_scenario", scenario.tracking_scenario)
    tracking_template = _resolve_tracking_template(
        load_scenario(scenario.tracking_scenario), scenario
    )
    _assert_input_unchanged(tracking_input)
    _validate_schedule(scenario, tracking_template)

    nominal_scenario = _tracking_scenario_from_launch(
        tracking_template,
        launch.insertion_state,
        launch.samples[-1].mass_kg,
        scenario_id=f"{scenario.scenario_id}-nominal",
    )
    estimation_force_model = _configured_force_model(scenario, truth=False)
    if estimation_force_model is not None:
        nominal_scenario = _scenario_with_force_model(nominal_scenario, estimation_force_model)
    truth_scenario = nominal_scenario.model_copy(
        update={
            "scenario_id": f"{scenario.scenario_id}-truth",
            "initial_state": _dispersed_state(nominal_scenario.initial_state, scenario),
            "metadata": {
                **nominal_scenario.metadata,
                "evidence_scope": "simulation_truth",
                "configured_dispersion": scenario.dispersion.model_dump(mode="json"),
            },
        }
    )
    truth_force_model = _configured_force_model(scenario, truth=True)
    if truth_force_model is not None:
        truth_scenario = _scenario_with_force_model(truth_scenario, truth_force_model)
    nominal_trajectory = propagate_local(nominal_scenario)
    truth_trajectory = propagate_local(truth_scenario)
    generated_measurements = tuple(
        generate_synthetic_measurements(truth_scenario, truth_trajectory)
    )
    visible_measurements = _visible_measurements(
        truth_scenario,
        truth_trajectory,
        generated_measurements,
    )
    decision_epoch = truth_scenario.initial_state.epoch + timedelta(
        seconds=float(scenario.correction.correction_elapsed_s)
    )
    decision_measurements = tuple(
        measurement for measurement in visible_measurements if measurement.epoch <= decision_epoch
    )
    if not decision_measurements:
        raise MissionAssuranceError(
            "no visible tracking measurements are available by the correction decision epoch",
            phase="tracking",
        )
    measurements = _measurements_for_estimation(decision_measurements, scenario)
    estimate = estimate_initial_state(
        nominal_scenario,
        list(measurements),
        backend=scenario.tracking_backend,
    )
    estimated_scenario = nominal_scenario.model_copy(
        update={
            "scenario_id": f"{scenario.scenario_id}-estimated",
            "initial_state": estimate.estimated_state,
            "metadata": {
                **nominal_scenario.metadata,
                "source": "suite_batch_orbit_determination",
            },
        }
    )
    estimated_trajectory = propagate_local(estimated_scenario)
    candidate = design_candidate_correction(
        estimated_scenario,
        estimated_trajectory,
        nominal_trajectory,
        scenario.correction,
    )
    estimated_corrected_scenario = estimated_scenario.model_copy(
        update={
            "scenario_id": f"{scenario.scenario_id}-estimated-corrected",
            "maneuvers": [candidate],
        }
    )
    execution_scale = (
        1.0
        if scenario.input_overrides is None
        or scenario.input_overrides.correction_execution_scale is None
        else float(scenario.input_overrides.correction_execution_scale)
    )
    executed_delta_v, pointing_basis = _executed_delta_v(candidate.delta_v_km_s, scenario)
    execution_epoch_offset_s = _execution_override(
        scenario, "correction_execution_epoch_offset_s", 0.0
    )
    _validate_execution_epoch(
        truth_scenario,
        candidate.epoch,
        execution_epoch_offset_s,
        float(scenario.correction.verification_elapsed_s),
    )
    executed_candidate = candidate.model_copy(
        update={
            "name": f"{candidate.name}-executed",
            "epoch": candidate.epoch + timedelta(seconds=execution_epoch_offset_s),
            "delta_v_km_s": executed_delta_v,
            "metadata": {
                **candidate.metadata,
                "commanded_maneuver": candidate.name,
                "execution_scale": execution_scale,
                "execution_epoch_offset_s": execution_epoch_offset_s,
                "execution_pointing_1_deg": _execution_override(
                    scenario, "correction_execution_pointing_1_deg", 0.0
                ),
                "execution_pointing_2_deg": _execution_override(
                    scenario, "correction_execution_pointing_2_deg", 0.0
                ),
                "execution_pointing_basis": pointing_basis,
                "evidence_scope": "simulation_truth",
            },
        }
    )
    truth_corrected_scenario = truth_scenario.model_copy(
        update={
            "scenario_id": f"{scenario.scenario_id}-truth-corrected",
            "maneuvers": [executed_candidate],
        }
    )
    estimated_corrected = propagate_local(estimated_corrected_scenario)
    truth_corrected = propagate_local(truth_corrected_scenario)
    twin_input = _input_reference("twin_scenario", scenario.twin_scenario)
    twin_template = _resolve_twin_template(load_twin_scenario(scenario.twin_scenario), scenario)
    _assert_input_unchanged(twin_input)
    candidate_propellant_used_kg = _candidate_propellant_used_kg(
        twin_template,
        candidate.delta_v_km_s,
        float(scenario.correction.specific_impulse_s),
    )
    propellant_used_kg = _candidate_propellant_used_kg(
        twin_template,
        executed_candidate.delta_v_km_s,
        float(scenario.correction.specific_impulse_s),
    )
    available_propellant_kg = float(twin_template.spacecraft.propellant_mass_kg)
    if propellant_used_kg > available_propellant_kg:
        raise MissionAssuranceError(
            "candidate correction exceeds available digital-twin propellant",
            phase="digital_twin",
        )
    corrected_spacecraft = twin_template.spacecraft.model_copy(
        update={"propellant_mass_kg": available_propellant_kg - propellant_used_kg}
    )
    corrected_twin_template = twin_template.model_copy(update={"spacecraft": corrected_spacecraft})
    corrected_twin = run_digital_twin(
        corrected_twin_template,
        trajectory_override=truth_corrected,
    )

    continuity = _continuity_report(
        launch_state=launch.insertion_state,
        nominal_scenario=nominal_scenario,
        truth_scenario=truth_scenario,
        estimate_state=estimate.estimated_state,
        correction_epoch=candidate.epoch,
        corrected_truth=truth_corrected,
        corrected_twin=corrected_twin,
        launch_final_mass_kg=float(launch.samples[-1].mass_kg),
        twin_template=twin_template,
        corrected_twin_template=corrected_twin_template,
        propellant_used_kg=propellant_used_kg,
        correction_elapsed_s=float(scenario.correction.correction_elapsed_s),
    )
    margins, metrics = _margin_report(
        scenario,
        truth_scenario=truth_scenario,
        nominal_trajectory=nominal_trajectory,
        truth_trajectory=truth_trajectory,
        estimate_state=estimate.estimated_state,
        candidate_delta_v_km_s=candidate.delta_v_km_s,
        executed_delta_v_km_s=executed_candidate.delta_v_km_s,
        estimated_corrected=estimated_corrected,
        truth_corrected=truth_corrected,
        corrected_twin_template=corrected_twin_template,
        corrected_twin=corrected_twin,
        propellant_used_kg=propellant_used_kg,
    )
    passed = continuity.all_passed and margins.overall_status is not AssuranceStatus.FAIL
    metadata = {
        **scenario.metadata,
        **metrics,
        "measurement_count": len(measurements),
        "generated_measurement_count": len(generated_measurements),
        "visible_measurement_count": len(visible_measurements),
        "rejected_below_mask_measurement_count": (
            len(generated_measurements) - len(visible_measurements)
        ),
        "rejected_after_decision_measurement_count": (
            len(visible_measurements) - len(measurements)
        ),
        "tracking_decision_epoch": decision_epoch.isoformat(),
        "recovery_disposition": "candidate_for_manual_review",
        "tracking_source": "synthetic_simulation_truth",
        "tracking_visibility_filter": "station_elevation_mask",
        "tracking_noise_seed": int(truth_scenario.measurements.noise.seed),
        "truth_force_model": truth_scenario.force_model.gravity.value,
        "estimation_force_model": nominal_scenario.force_model.gravity.value,
        "truth_force_model_config": truth_scenario.force_model.model_dump(mode="json"),
        "estimation_force_model_config": nominal_scenario.force_model.model_dump(mode="json"),
        "correction_execution_scale": execution_scale,
        "correction_execution_epoch_offset_s": execution_epoch_offset_s,
        "correction_elapsed_s": float(scenario.correction.correction_elapsed_s),
        "verification_elapsed_s": float(scenario.correction.verification_elapsed_s),
        "maximum_component_delta_v_km_s": float(
            scenario.correction.maximum_component_delta_v_km_s
        ),
        "maximum_total_delta_v_km_s": float(scenario.correction.maximum_total_delta_v_km_s),
        "executed_delta_v_km_s": float(
            np.linalg.norm(np.asarray(executed_candidate.delta_v_km_s, dtype=np.float64))
        ),
        "candidate_propellant_used_kg": candidate_propellant_used_kg,
        "executed_propellant_used_kg": propellant_used_kg,
        "resolved_input_overrides": (
            scenario.input_overrides.model_dump(mode="json")
            if scenario.input_overrides is not None
            else None
        ),
    }
    warnings = [
        "Post-launch assurance v1 is deterministic simulation and design-screening evidence, "
        "not operational acquisition or navigation authority.",
        "Synthetic measurements do not prove RF contact or authenticated spacecraft identity.",
        "The correction is a candidate for manual review, not a flight command.",
        *corrected_twin.warnings,
    ]
    input_references = (assurance_input, launch_input, tracking_input, twin_input)
    for reference in input_references:
        _assert_input_unchanged(reference)
    manifest = _manifest(
        scenario,
        input_references=input_references,
        launch=launch,
        nominal_scenario=nominal_scenario,
        truth_scenario=truth_scenario,
        nominal_trajectory=nominal_trajectory,
        truth_trajectory=truth_trajectory,
        measurements=measurements,
        estimate=estimate,
        candidate=candidate,
        estimated_corrected_scenario=estimated_corrected_scenario,
        truth_corrected_scenario=truth_corrected_scenario,
        estimated_corrected=estimated_corrected,
        truth_corrected=truth_corrected,
        corrected_twin=corrected_twin,
        continuity=continuity,
        margins=margins,
        passed=passed,
        metadata=metadata,
        warnings=warnings,
    )
    return MissionAssuranceCase(
        scenario_id=scenario.scenario_id,
        launch_trajectory=launch,
        nominal_scenario=nominal_scenario,
        truth_scenario=truth_scenario,
        nominal_trajectory=nominal_trajectory,
        truth_trajectory=truth_trajectory,
        measurements=measurements,
        estimate=estimate,
        correction_maneuver=candidate,
        estimated_corrected_scenario=estimated_corrected_scenario,
        truth_corrected_scenario=truth_corrected_scenario,
        estimated_corrected_trajectory=estimated_corrected,
        truth_corrected_trajectory=truth_corrected,
        corrected_digital_twin=corrected_twin,
        continuity_report=continuity,
        margin_report=margins,
        manifest=manifest,
        passed=passed,
        metadata=metadata,
        warnings=warnings,
    )


def _resolve_tracking_template(
    template: Scenario,
    scenario: PostLaunchAssuranceScenario,
) -> Scenario:
    overrides = scenario.input_overrides
    if overrides is None:
        return template
    noise_payload = template.measurements.noise.model_dump(mode="python")
    if overrides.tracking_range_sigma_km is not None:
        noise_payload["range_sigma_km"] = overrides.tracking_range_sigma_km
    if overrides.tracking_range_rate_sigma_km_s is not None:
        noise_payload["range_rate_sigma_km_s"] = overrides.tracking_range_rate_sigma_km_s
    if overrides.tracking_noise_seed is not None:
        noise_payload["seed"] = overrides.tracking_noise_seed
    measurements = template.measurements.model_copy(
        update={"noise": type(template.measurements.noise).model_validate(noise_payload)}
    )
    propagation = template.propagation
    if overrides.tracking_duration_s is not None:
        propagation = propagation.model_copy(update={"duration_s": overrides.tracking_duration_s})
    return Scenario.model_validate(
        template.model_copy(
            update={"measurements": measurements, "propagation": propagation}
        ).model_dump(mode="python")
    )


def _configured_force_model(
    scenario: PostLaunchAssuranceScenario,
    *,
    truth: bool,
) -> ForceModelName | None:
    overrides = scenario.input_overrides
    if overrides is None:
        return None
    return overrides.truth_force_model if truth else overrides.estimation_force_model


def _scenario_with_force_model(scenario: Scenario, gravity: ForceModelName) -> Scenario:
    if scenario.force_model.enabled_high_fidelity_flags():
        raise MissionAssuranceError(
            "paired local assurance force-model overrides do not support high-fidelity flags",
            phase="configuration",
        )
    payload = scenario.force_model.model_dump(mode="python")
    payload["gravity"] = gravity
    if gravity is ForceModelName.J2:
        payload["gravity_degree"] = 2
        payload["gravity_order"] = 0
    else:
        payload["gravity_degree"] = None
        payload["gravity_order"] = None
    force_model = type(scenario.force_model).model_validate(payload)
    return Scenario.model_validate(
        scenario.model_copy(update={"force_model": force_model}).model_dump(mode="python")
    )


def _measurements_for_estimation(
    measurements: tuple[MeasurementRecord, ...],
    scenario: PostLaunchAssuranceScenario,
) -> tuple[MeasurementRecord, ...]:
    overrides = scenario.input_overrides
    resolved: list[MeasurementRecord] = []
    for measurement in measurements:
        truth_bias = 0.0
        estimator_bias = 0.0
        assumed_sigma = float(measurement.sigma)
        if overrides is not None and measurement.measurement_type in _RANGE_TYPES:
            truth_bias = float(overrides.tracking_range_bias_km or 0.0)
            estimator_bias = float(overrides.estimation_range_bias_km or 0.0)
            if overrides.estimation_range_sigma_km is not None:
                assumed_sigma = float(overrides.estimation_range_sigma_km)
        elif overrides is not None and measurement.measurement_type in _RANGE_RATE_TYPES:
            truth_bias = float(overrides.tracking_range_rate_bias_km_s or 0.0)
            estimator_bias = float(overrides.estimation_range_rate_bias_km_s or 0.0)
            if overrides.estimation_range_rate_sigma_km_s is not None:
                assumed_sigma = float(overrides.estimation_range_rate_sigma_km_s)
        ideal_truth = float(measurement.metadata.get("truth", measurement.value))
        noise_realization = float(measurement.value) - ideal_truth
        resolved.append(
            measurement.model_copy(
                update={
                    "value": float(measurement.value) + truth_bias,
                    "sigma": assumed_sigma,
                    "metadata": {
                        **measurement.metadata,
                        "simulation_truth_sigma": float(measurement.sigma),
                        "simulation_truth_bias": truth_bias,
                        "simulation_noise_realization": noise_realization,
                        "simulation_noise_seed": (
                            None if overrides is None else overrides.tracking_noise_seed
                        ),
                        "estimator_assumed_sigma": assumed_sigma,
                        "estimator_bias": estimator_bias,
                    },
                }
            )
        )
    return tuple(resolved)


def _execution_override(scenario: PostLaunchAssuranceScenario, field: str, default: float) -> float:
    overrides = scenario.input_overrides
    value = None if overrides is None else getattr(overrides, field)
    return default if value is None else float(value)


def _validate_execution_epoch(
    truth_scenario: Scenario,
    commanded_epoch: datetime,
    offset_s: float,
    verification_elapsed_s: float,
) -> None:
    executed_epoch = commanded_epoch + timedelta(seconds=offset_s)
    verification_epoch = truth_scenario.initial_state.epoch + timedelta(
        seconds=verification_elapsed_s
    )
    if executed_epoch < truth_scenario.initial_state.epoch:
        raise MissionAssuranceError(
            "executed correction precedes the truth scenario epoch", phase="configuration"
        )
    if executed_epoch >= verification_epoch:
        raise MissionAssuranceError(
            "executed correction must precede the verification epoch", phase="configuration"
        )


def _executed_delta_v(
    commanded_delta_v_km_s: tuple[float, float, float],
    scenario: PostLaunchAssuranceScenario,
) -> tuple[tuple[float, float, float], dict[str, Any]]:
    commanded = np.asarray(commanded_delta_v_km_s, dtype=np.float64)
    magnitude = float(np.linalg.norm(commanded))
    if magnitude <= 0.0:
        raise MissionAssuranceError("candidate correction has zero delta-v", phase="correction")
    direction = commanded / magnitude
    reference = np.eye(3, dtype=np.float64)[int(np.argmin(np.abs(direction)))]
    axis_1 = np.cross(direction, reference)
    axis_1 /= np.linalg.norm(axis_1)
    axis_2 = np.cross(direction, axis_1)
    pointing_1 = radians(_execution_override(scenario, "correction_execution_pointing_1_deg", 0.0))
    pointing_2 = radians(_execution_override(scenario, "correction_execution_pointing_2_deg", 0.0))
    perturbed = direction + tan(pointing_1) * axis_1 + tan(pointing_2) * axis_2
    perturbed /= np.linalg.norm(perturbed)
    scale = _execution_override(scenario, "correction_execution_scale", 1.0)
    executed = magnitude * scale * perturbed
    return (
        (float(executed[0]), float(executed[1]), float(executed[2])),
        {
            "command_direction": direction.tolist(),
            "axis_1": axis_1.tolist(),
            "axis_2": axis_2.tolist(),
            "convention": [
                "command_frame",
                "axis_1_cross_command_with_least_aligned_inertial_axis",
                "axis_2_cross_command_with_axis_1",
            ],
        },
    )


def _resolve_twin_template(
    template: DigitalTwinScenario,
    scenario: PostLaunchAssuranceScenario,
) -> DigitalTwinScenario:
    overrides = scenario.input_overrides
    if overrides is None:
        return template
    payload = template.model_dump(mode="python")
    if overrides.twin_solar_array_efficiency is not None:
        payload["power"]["solar_array_efficiency"] = overrides.twin_solar_array_efficiency
    if overrides.twin_battery_capacity_wh is not None:
        payload["power"]["battery_capacity_wh"] = overrides.twin_battery_capacity_wh
    positions = {node["name"]: index for index, node in enumerate(payload["thermal_nodes"])}
    if len(positions) != len(payload["thermal_nodes"]):
        raise MissionAssuranceError(
            "digital twin thermal node names must be unique", phase="digital_twin"
        )
    for override in overrides.twin_thermal_node_overrides:
        if override.node_name not in positions:
            raise MissionAssuranceError(
                f"digital twin thermal override references missing node: {override.node_name}",
                phase="digital_twin",
            )
        node = payload["thermal_nodes"][positions[override.node_name]]
        if override.emissivity is not None:
            node["emissivity"] = override.emissivity
        if override.internal_heat_fraction is not None:
            node["internal_heat_fraction"] = override.internal_heat_fraction
    try:
        return DigitalTwinScenario.model_validate(payload)
    except ValueError as exc:
        raise MissionAssuranceError(
            f"resolved digital twin scenario is invalid: {exc}", phase="digital_twin"
        ) from exc


def _validate_schedule(
    scenario: PostLaunchAssuranceScenario,
    tracking_template: Scenario,
) -> None:
    step_s = float(tracking_template.propagation.step_s)
    duration_s = float(tracking_template.propagation.duration_s)
    correction_s = float(scenario.correction.correction_elapsed_s)
    verification_s = float(scenario.correction.verification_elapsed_s)
    if verification_s > duration_s:
        raise MissionAssuranceError(
            "verification_elapsed_s exceeds tracking duration", phase="configuration"
        )
    schedule = (
        ("correction_elapsed_s", correction_s),
        ("verification_elapsed_s", verification_s),
    )
    for name, value in schedule:
        if abs(value / step_s - round(value / step_s)) > 1.0e-9:
            raise MissionAssuranceError(
                f"{name} must align with the tracking propagation step",
                phase="configuration",
            )
    if not tracking_template.ground_stations:
        raise MissionAssuranceError(
            "tracking scenario must configure at least one ground station",
            phase="configuration",
        )


def _tracking_scenario_from_launch(
    template: Scenario,
    insertion_state: OrbitState,
    insertion_mass_kg: float,
    *,
    scenario_id: str,
) -> Scenario:
    return template.model_copy(
        update={
            "scenario_id": scenario_id,
            "initial_state": insertion_state,
            "spacecraft": template.spacecraft.model_copy(update={"mass_kg": insertion_mass_kg}),
            "maneuvers": [],
            "metadata": {
                **template.metadata,
                "workflow": "post_launch_mission_assurance_v1",
                "handoff_source": "launch_insertion_state",
            },
        }
    )


def _visible_measurements(
    scenario: Scenario,
    trajectory: Trajectory,
    measurements: tuple[MeasurementRecord, ...],
) -> tuple[MeasurementRecord, ...]:
    states_by_epoch = {sample.epoch: sample.state for sample in trajectory.samples}
    stations_by_name = {station.name: station for station in scenario.ground_stations}
    visible: list[MeasurementRecord] = []
    for measurement in measurements:
        station = stations_by_name.get(measurement.observer)
        state = states_by_epoch.get(measurement.epoch)
        if station is None or state is None:
            raise MissionAssuranceError(
                "synthetic measurement cannot be bound to its station and truth state",
                phase="tracking",
            )
        station_position = station.position_array(
            measurement.epoch,
            scenario.earth_orientation,
        )
        observed_elevation_deg = elevation_deg(
            state.position_km,
            station_position,
        )
        if observed_elevation_deg >= float(station.elevation_mask_deg):
            visible.append(measurement)
    if len(visible) < 6:
        raise MissionAssuranceError(
            "tracking visibility filter retained fewer than six measurements",
            phase="tracking",
        )
    return tuple(visible)


def _dispersed_state(
    state: OrbitState,
    scenario: PostLaunchAssuranceScenario,
) -> OrbitState:
    position: Vector3 = (
        state.cartesian.position_km[0] + scenario.dispersion.position_delta_km[0],
        state.cartesian.position_km[1] + scenario.dispersion.position_delta_km[1],
        state.cartesian.position_km[2] + scenario.dispersion.position_delta_km[2],
    )
    velocity: Vector3 = (
        state.cartesian.velocity_km_s[0] + scenario.dispersion.velocity_delta_km_s[0],
        state.cartesian.velocity_km_s[1] + scenario.dispersion.velocity_delta_km_s[1],
        state.cartesian.velocity_km_s[2] + scenario.dispersion.velocity_delta_km_s[2],
    )
    return state.model_copy(
        update={"cartesian": CartesianState(position_km=position, velocity_km_s=velocity)}
    )


def _continuity_report(
    *,
    launch_state: OrbitState,
    nominal_scenario: Scenario,
    truth_scenario: Scenario,
    estimate_state: OrbitState,
    correction_epoch: datetime,
    corrected_truth: Trajectory,
    corrected_twin: DigitalTwinResult,
    launch_final_mass_kg: float,
    twin_template: DigitalTwinScenario,
    corrected_twin_template: DigitalTwinScenario,
    propellant_used_kg: float,
    correction_elapsed_s: float,
) -> AssuranceContinuityReport:
    nominal_state = nominal_scenario.initial_state
    twin_wet_mass_kg = float(
        twin_template.spacecraft.dry_mass_kg
        + twin_template.spacecraft.payload_mass_kg
        + twin_template.spacecraft.propellant_mass_kg
    )
    corrected_twin_wet_mass_kg = float(
        corrected_twin_template.spacecraft.dry_mass_kg
        + corrected_twin_template.spacecraft.payload_mass_kg
        + corrected_twin_template.spacecraft.propellant_mass_kg
    )
    checks = (
        _check(
            "launch_nominal_epoch",
            "launch",
            "nominal",
            abs((launch_state.epoch - nominal_state.epoch).total_seconds()),
            0.0,
            "s",
        ),
        _check(
            "launch_nominal_position",
            "launch",
            "nominal",
            _position_error(launch_state.cartesian, nominal_state.cartesian),
            _STATE_TOLERANCE,
            "km",
        ),
        _check(
            "launch_nominal_velocity",
            "launch",
            "nominal",
            _velocity_error(launch_state.cartesian, nominal_state.cartesian),
            _STATE_TOLERANCE,
            "km/s",
        ),
        _check(
            "launch_tracking_mass",
            "launch",
            "tracking",
            abs(launch_final_mass_kg - float(nominal_scenario.spacecraft.mass_kg)),
            _MASS_TOLERANCE_KG,
            "kg",
        ),
        _check(
            "launch_twin_mass",
            "launch",
            "digital_twin",
            abs(launch_final_mass_kg - twin_wet_mass_kg),
            _MASS_TOLERANCE_KG,
            "kg",
        ),
        _check(
            "correction_twin_mass",
            "correction",
            "digital_twin",
            abs(launch_final_mass_kg - propellant_used_kg - corrected_twin_wet_mass_kg),
            _MASS_TOLERANCE_KG,
            "kg",
        ),
        _check(
            "truth_estimate_epoch",
            "truth",
            "estimate",
            abs((truth_scenario.initial_state.epoch - estimate_state.epoch).total_seconds()),
            0.0,
            "s",
        ),
        _check(
            "estimate_correction_epoch",
            "estimate",
            "correction",
            abs((correction_epoch - estimate_state.epoch).total_seconds() - correction_elapsed_s),
            1.0e-9,
            "s",
        ),
        _check(
            "truth_twin_start_epoch",
            "truth_correction",
            "digital_twin",
            abs(
                (
                    corrected_truth.samples[0].epoch - corrected_twin.geometry[0].epoch
                ).total_seconds()
            ),
            0.0,
            "s",
        ),
        _check(
            "truth_twin_end_epoch",
            "truth_correction",
            "digital_twin",
            abs(
                (
                    corrected_truth.samples[-1].epoch - corrected_twin.geometry[-1].epoch
                ).total_seconds()
            ),
            0.0,
            "s",
        ),
    )
    return AssuranceContinuityReport(checks=checks, all_passed=all(item.passed for item in checks))


def _check(
    name: str,
    upstream: str,
    downstream: str,
    error: float,
    tolerance: float,
    unit: str,
) -> AssuranceContinuityCheck:
    return AssuranceContinuityCheck(
        name=name,
        upstream_phase=upstream,
        downstream_phase=downstream,
        error=error,
        tolerance=tolerance,
        unit=unit,
        passed=error <= tolerance,
    )


def _margin_report(
    scenario: PostLaunchAssuranceScenario,
    *,
    truth_scenario: Scenario,
    nominal_trajectory: Trajectory,
    truth_trajectory: Trajectory,
    estimate_state: OrbitState,
    candidate_delta_v_km_s: tuple[float, float, float],
    executed_delta_v_km_s: tuple[float, float, float],
    estimated_corrected: Trajectory,
    truth_corrected: Trajectory,
    corrected_twin_template: DigitalTwinScenario,
    corrected_twin: DigitalTwinResult,
    propellant_used_kg: float,
) -> tuple[AssuranceMarginReport, dict[str, float]]:
    verification_s = float(scenario.correction.verification_elapsed_s)
    nominal_final = trajectory_sample_at_elapsed(nominal_trajectory, verification_s)
    truth_uncorrected_final = trajectory_sample_at_elapsed(truth_trajectory, verification_s)
    estimated_corrected_final = trajectory_sample_at_elapsed(estimated_corrected, verification_s)
    truth_corrected_final = trajectory_sample_at_elapsed(truth_corrected, verification_s)
    od_position_error = _position_error(
        estimate_state.cartesian, truth_scenario.initial_state.cartesian
    )
    od_velocity_error = _velocity_error(
        estimate_state.cartesian, truth_scenario.initial_state.cartesian
    )
    pre_position_error = _position_error(truth_uncorrected_final, nominal_final)
    truth_position_error = _position_error(truth_corrected_final, nominal_final)
    truth_velocity_error = _velocity_error(truth_corrected_final, nominal_final)
    predicted_position_error = _position_error(estimated_corrected_final, nominal_final)
    predicted_velocity_error = _velocity_error(estimated_corrected_final, nominal_final)
    reduction_fraction = (
        1.0 - truth_position_error / pre_position_error if pre_position_error > 0.0 else 1.0
    )
    delta_v = float(np.linalg.norm(np.asarray(candidate_delta_v_km_s, dtype=np.float64)))
    executed_delta_v = float(np.linalg.norm(np.asarray(executed_delta_v_km_s, dtype=np.float64)))
    reserve = float(corrected_twin_template.spacecraft.propellant_mass_kg)
    minimum_soc = min(sample.battery_soc_fraction for sample in corrected_twin.power)
    failed_twin_margin_count = sum(
        margin.status.value == "fail" for margin in corrected_twin.margin_report.margins
    )
    requirements = scenario.requirements
    margins = (
        _maximum_margin(
            "od_position_error",
            od_position_error,
            float(requirements.maximum_od_position_error_km),
            "km",
            "simulation_truth",
        ),
        _maximum_margin(
            "od_velocity_error",
            od_velocity_error,
            float(requirements.maximum_od_velocity_error_km_s),
            "km/s",
            "simulation_truth",
        ),
        _maximum_margin(
            "candidate_delta_v",
            delta_v,
            float(scenario.correction.maximum_total_delta_v_km_s),
            "km/s",
            "decision_available",
        ),
        _maximum_margin(
            "executed_delta_v",
            executed_delta_v,
            float(scenario.correction.maximum_total_delta_v_km_s),
            "km/s",
            "simulation_truth",
        ),
        _maximum_margin(
            "truth_recovery_position_error",
            truth_position_error,
            float(requirements.maximum_truth_recovery_position_error_km),
            "km",
            "simulation_truth",
        ),
        _maximum_margin(
            "truth_recovery_velocity_error",
            truth_velocity_error,
            float(requirements.maximum_truth_recovery_velocity_error_km_s),
            "km/s",
            "simulation_truth",
        ),
        _minimum_margin(
            "truth_position_error_reduction",
            reduction_fraction,
            float(requirements.minimum_position_error_reduction_fraction),
            "1",
            "simulation_truth",
        ),
        _minimum_margin(
            "propellant_reserve",
            reserve,
            float(requirements.minimum_propellant_reserve_kg),
            "kg",
            "design_screening",
        ),
        _minimum_margin(
            "battery_soc",
            minimum_soc,
            float(requirements.minimum_battery_soc_fraction),
            "1",
            "design_screening",
        ),
        AssuranceMargin(
            name="digital_twin_failed_margin_count",
            value=float(failed_twin_margin_count),
            threshold=0.0,
            margin=-float(failed_twin_margin_count),
            normalized_margin=(
                1.0 if failed_twin_margin_count == 0 else -float(failed_twin_margin_count)
            ),
            unit="count",
            status=(
                AssuranceStatus.PASS if failed_twin_margin_count == 0 else AssuranceStatus.FAIL
            ),
            evidence_scope="design_screening",
        ),
    )
    limiting = min(margins, key=lambda item: item.normalized_margin)
    severity = {AssuranceStatus.PASS: 0, AssuranceStatus.WARN: 1, AssuranceStatus.FAIL: 2}
    overall = max(margins, key=lambda item: severity[item.status]).status
    return (
        AssuranceMarginReport(margins=margins, limiting_margin=limiting, overall_status=overall),
        {
            "uncorrected_truth_position_error_km": pre_position_error,
            "predicted_recovery_position_error_km": predicted_position_error,
            "predicted_recovery_velocity_error_km_s": predicted_velocity_error,
            "truth_recovery_position_error_km": truth_position_error,
            "truth_recovery_velocity_error_km_s": truth_velocity_error,
            "truth_position_error_reduction_fraction": reduction_fraction,
            "candidate_delta_v_km_s": delta_v,
            "executed_propellant_used_kg": propellant_used_kg,
        },
    )


def _candidate_propellant_used_kg(
    twin_template: DigitalTwinScenario,
    candidate_delta_v_km_s: tuple[float, float, float],
    specific_impulse_s: float,
) -> float:
    wet_mass_kg = float(
        twin_template.spacecraft.dry_mass_kg
        + twin_template.spacecraft.payload_mass_kg
        + twin_template.spacecraft.propellant_mass_kg
    )
    delta_v_km_s = float(np.linalg.norm(np.asarray(candidate_delta_v_km_s, dtype=np.float64)))
    return wet_mass_kg * (
        1.0 - exp(-(delta_v_km_s * 1000.0) / (specific_impulse_s * _STANDARD_GRAVITY_M_S2))
    )


def _maximum_margin(
    name: str, value: float, threshold: float, unit: str, evidence_scope: str
) -> AssuranceMargin:
    margin = threshold - value
    normalized = margin / threshold
    return AssuranceMargin(
        name=name,
        value=value,
        threshold=threshold,
        margin=margin,
        normalized_margin=normalized,
        unit=unit,
        status=_status(normalized),
        evidence_scope=evidence_scope,
    )


def _minimum_margin(
    name: str, value: float, threshold: float, unit: str, evidence_scope: str
) -> AssuranceMargin:
    margin = value - threshold
    normalized = margin / max(abs(threshold), 1.0)
    return AssuranceMargin(
        name=name,
        value=value,
        threshold=threshold,
        margin=margin,
        normalized_margin=normalized,
        unit=unit,
        status=_status(normalized),
        evidence_scope=evidence_scope,
    )


def _status(normalized_margin: float) -> AssuranceStatus:
    if normalized_margin < 0.0:
        return AssuranceStatus.FAIL
    if normalized_margin < 0.1:
        return AssuranceStatus.WARN
    return AssuranceStatus.PASS


def _position_error(left: CartesianState, right: CartesianState) -> float:
    return float(np.linalg.norm(left.position_array() - right.position_array()))


def _velocity_error(left: CartesianState, right: CartesianState) -> float:
    return float(np.linalg.norm(left.velocity_array() - right.velocity_array()))


def _manifest(
    scenario: PostLaunchAssuranceScenario,
    *,
    input_references: tuple[AssuranceInputReference, ...],
    launch: AstroModel,
    nominal_scenario: AstroModel,
    truth_scenario: AstroModel,
    nominal_trajectory: AstroModel,
    truth_trajectory: AstroModel,
    measurements: tuple[MeasurementRecord, ...],
    estimate: AstroModel,
    candidate: AstroModel,
    estimated_corrected_scenario: AstroModel,
    truth_corrected_scenario: AstroModel,
    estimated_corrected: AstroModel,
    truth_corrected: AstroModel,
    corrected_twin: AstroModel,
    continuity: AstroModel,
    margins: AstroModel,
    passed: bool,
    metadata: dict[str, Any],
    warnings: list[str],
) -> AssuranceManifest:
    measurement_payload = {
        "scenario_id": f"{scenario.scenario_id}-truth",
        "measurements": [item.model_dump(mode="json") for item in measurements],
    }
    decision_payload = {
        "scenario_id": scenario.scenario_id,
        "workflow": "post_launch_mission_assurance_v1",
        "passed": passed,
        "metadata": metadata,
        "warnings": warnings,
    }
    products: tuple[tuple[str, str, str, str, str, Any], ...] = (
        (
            "launch",
            "LaunchTrajectory",
            "launch.json",
            scenario.launch_scenario,
            scenario.launch_backend,
            launch.model_dump(mode="json"),
        ),
        (
            "nominal",
            "Scenario",
            "nominal-scenario.yaml",
            f"{scenario.scenario_id}-nominal",
            "local",
            nominal_scenario.model_dump(mode="json"),
        ),
        (
            "truth",
            "Scenario",
            "truth-scenario.yaml",
            f"{scenario.scenario_id}-truth",
            "local",
            truth_scenario.model_dump(mode="json"),
        ),
        (
            "tracking",
            "MeasurementRecord[]",
            "measurements.json",
            f"{scenario.scenario_id}-truth",
            "synthetic_visible_only",
            measurement_payload,
        ),
        (
            "estimation",
            "EstimateResult",
            "estimate.json",
            f"{scenario.scenario_id}-nominal",
            scenario.tracking_backend,
            estimate.model_dump(mode="json"),
        ),
        (
            "correction",
            "Maneuver",
            "candidate-maneuver.json",
            "post-launch-recovery-candidate",
            "local",
            candidate.model_dump(mode="json"),
        ),
        (
            "nominal",
            "Trajectory",
            "nominal-trajectory.json",
            f"{scenario.scenario_id}-nominal",
            "local",
            nominal_trajectory.model_dump(mode="json"),
        ),
        (
            "truth",
            "Trajectory",
            "truth-trajectory.json",
            f"{scenario.scenario_id}-truth",
            "local",
            truth_trajectory.model_dump(mode="json"),
        ),
        (
            "correction",
            "Scenario",
            "estimated-corrected-scenario.yaml",
            f"{scenario.scenario_id}-estimated-corrected",
            "local",
            estimated_corrected_scenario.model_dump(mode="json"),
        ),
        (
            "correction",
            "Scenario",
            "truth-corrected-scenario.yaml",
            f"{scenario.scenario_id}-truth-corrected",
            "local",
            truth_corrected_scenario.model_dump(mode="json"),
        ),
        (
            "verification",
            "Trajectory",
            "estimated-corrected-trajectory.json",
            f"{scenario.scenario_id}-estimated-corrected",
            "local",
            estimated_corrected.model_dump(mode="json"),
        ),
        (
            "verification",
            "Trajectory",
            "truth-corrected-trajectory.json",
            f"{scenario.scenario_id}-truth-corrected",
            "local",
            truth_corrected.model_dump(mode="json"),
        ),
        (
            "digital_twin",
            "DigitalTwinResult",
            "corrected-digital-twin.json",
            scenario.twin_scenario,
            "local",
            corrected_twin.model_dump(mode="json"),
        ),
        (
            "decision",
            "AssuranceContinuityReport",
            "continuity-report.json",
            scenario.scenario_id,
            "suite",
            continuity.model_dump(mode="json"),
        ),
        (
            "decision",
            "AssuranceMarginReport",
            "margin-report.json",
            scenario.scenario_id,
            "suite",
            margins.model_dump(mode="json"),
        ),
        (
            "decision",
            "MissionAssuranceDecision",
            "decision.json",
            scenario.scenario_id,
            "suite",
            decision_payload,
        ),
    )
    entries = tuple(
        AssuranceManifestEntry(
            sequence=index,
            phase=phase,
            product_type=product_type,
            artifact_name=artifact_name,
            source_id=source_id,
            source_digest=_product_digest(payload),
            backend=backend,
        )
        for index, (phase, product_type, artifact_name, source_id, backend, payload) in enumerate(
            products, start=1
        )
    )
    return AssuranceManifest(
        scenario_id=scenario.scenario_id,
        inputs=input_references,
        entries=entries,
    )


def _loaded_assurance_reference(
    scenario: PostLaunchAssuranceScenario,
) -> AssuranceInputReference:
    if scenario.source_path is None or scenario.source_digest is None:
        raise MissionAssuranceError(
            "mission assurance scenario must retain its loaded source path and digest",
            phase="manifest",
        )
    return AssuranceInputReference(
        role="assurance_scenario",
        path=scenario.source_path,
        file_digest=scenario.source_digest,
    )


def _input_reference(role: str, source_path: str) -> AssuranceInputReference:
    path = Path(source_path).resolve()
    try:
        digest = sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise MissionAssuranceError(
            f"could not digest {role} input {path}: {exc}",
            phase="manifest",
        ) from exc
    return AssuranceInputReference.model_validate(
        {"role": role, "path": str(path), "file_digest": digest}
    )


def _assert_input_unchanged(reference: AssuranceInputReference) -> None:
    path = Path(reference.path)
    try:
        digest = sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise MissionAssuranceError(
            f"could not recheck {reference.role} input {path}: {exc}",
            phase="manifest",
        ) from exc
    if digest != reference.file_digest:
        raise MissionAssuranceError(
            f"{reference.role} input changed during mission assurance execution",
            phase="manifest",
        )


def _product_digest(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(canonical.encode("utf-8")).hexdigest()
