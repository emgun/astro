from __future__ import annotations

from math import exp, sqrt

from astro_core.constants import R_EARTH_KM
from astro_core.models import (
    ForceModelConfig,
    Maneuver,
    OrbitRepresentation,
    OrbitState,
    PropagationConfig,
    Scenario,
    Trajectory,
    Vector3,
)
from astro_dynamics.local import propagate_local
from astro_dynamics.maneuvers import apply_impulsive_maneuver
from astro_launch.backends import propagate_launch_with_backend
from astro_launch.handoff import launch_trajectory_to_orbit_scenario
from astro_launch.io import load_launch_scenario
from astro_launch.models import LaunchScenario, LaunchTrajectory
from astro_mission.errors import MissionLifecycleError
from astro_mission.models import (
    LifecycleContinuityCheck,
    LifecycleContinuityReport,
    LifecyclePhaseManifestEntry,
    LifecyclePhaseName,
    LifecycleStatus,
    MissionLifecycleManifest,
    MissionLifecycleMargin,
    MissionLifecycleMarginReport,
    MissionLifecycleResult,
    MissionLifecycleScenario,
)
from astro_reentry.backends import simulate_reentry_with_backend
from astro_reentry.handoff import trajectory_to_reentry_scenario
from astro_reentry.io import load_reentry_scenario
from astro_reentry.models import ReentryMarginStatus, ReentryResult, ReentryScenario
from astro_twin.io import load_twin_scenario
from astro_twin.models import DigitalTwinResult, DigitalTwinScenario, TwinMarginStatus
from astro_twin.runner import run_digital_twin

_STANDARD_GRAVITY_M_S2 = 9.80665
_STATE_CONTINUITY_TOLERANCE = 1.0e-9
_MASS_CONTINUITY_TOLERANCE_KG = 1.0e-6


def run_mission_lifecycle(scenario: MissionLifecycleScenario) -> MissionLifecycleResult:
    launch_scenario = load_launch_scenario(scenario.launch_scenario)
    launch_scenario = _apply_launch_input_overrides(launch_scenario, scenario)
    launch = propagate_launch_with_backend(launch_scenario, scenario.launch_backend)
    launch_margins = _launch_margins(launch_scenario, launch)
    _require_no_failed_margins(launch_margins, "launch insertion")

    orbit_scenario = launch_trajectory_to_orbit_scenario(
        launch,
        duration_s=float(scenario.orbit.duration_s),
        step_s=float(scenario.orbit.step_s),
        spacecraft_name=scenario.orbit.spacecraft_name,
        area_m2=float(scenario.orbit.reference_area_m2),
        drag_coefficient=float(scenario.orbit.drag_coefficient),
        reflectivity_coefficient=float(scenario.orbit.reflectivity_coefficient),
        gravity=scenario.orbit.gravity,
        scenario_id=f"{scenario.scenario_id}-operations",
    )
    operations = propagate_local(orbit_scenario)

    twin_scenario = load_twin_scenario(scenario.twin_scenario)
    twin_scenario = resolve_lifecycle_twin_scenario(twin_scenario, scenario)
    twin_wet_mass_kg = float(
        twin_scenario.spacecraft.dry_mass_kg
        + twin_scenario.spacecraft.payload_mass_kg
        + twin_scenario.spacecraft.propellant_mass_kg
    )
    orbit_mass_kg = float(orbit_scenario.spacecraft.mass_kg)
    if abs(twin_wet_mass_kg - orbit_mass_kg) > _MASS_CONTINUITY_TOLERANCE_KG:
        raise MissionLifecycleError(
            "digital twin wet mass does not match launch payload mass: "
            f"{twin_wet_mass_kg:.6f} kg != {orbit_mass_kg:.6f} kg",
            lifecycle_phase="digital_twin",
        )
    twin = run_digital_twin(twin_scenario, trajectory_override=operations)

    deorbit_maneuver, deorbit_scenario, deorbit, propellant_used_kg = _run_deorbit(
        scenario,
        operations,
        orbit_scenario,
        available_propellant_kg=float(twin_scenario.spacecraft.propellant_mass_kg),
    )
    reentry_template = load_reentry_scenario(scenario.reentry_scenario)
    reentry_template = _apply_reentry_input_overrides(reentry_template, scenario)
    reentry_scenario = trajectory_to_reentry_scenario(
        deorbit,
        reentry_template,
        sample_index=-1,
        scenario_id=f"{scenario.scenario_id}-reentry",
        use_sample_mass=True,
    )
    reentry = simulate_reentry_with_backend(reentry_scenario, scenario.reentry_backend)

    continuity = _build_continuity_report(
        launch=launch,
        orbit_scenario=orbit_scenario,
        operations=operations,
        twin_wet_mass_kg=twin_wet_mass_kg,
        maneuver=deorbit_maneuver,
        deorbit_scenario=deorbit_scenario,
        deorbit=deorbit,
        reentry_scenario=reentry_scenario,
    )
    margins = _build_margin_report(
        scenario=scenario,
        launch_margins=launch_margins,
        twin=twin,
        propellant_used_kg=propellant_used_kg,
        available_propellant_kg=float(twin_scenario.spacecraft.propellant_mass_kg),
        entry_altitude_km=float(reentry_scenario.initial_state.altitude_km),
        reentry=reentry,
    )
    manifest = _build_manifest(
        scenario=scenario,
        launch=launch,
        operations=operations,
        twin=twin,
        deorbit=deorbit,
        reentry_scenario=reentry_scenario,
        reentry=reentry,
    )
    passed = continuity.all_passed and margins.overall_status != LifecycleStatus.FAIL
    return MissionLifecycleResult(
        scenario_id=scenario.scenario_id,
        launch_trajectory=launch,
        orbit_scenario=orbit_scenario,
        operations_trajectory=operations,
        digital_twin=twin,
        deorbit_maneuver=deorbit_maneuver,
        deorbit_scenario=deorbit_scenario,
        deorbit_trajectory=deorbit,
        reentry_scenario=reentry_scenario,
        reentry_result=reentry,
        continuity_report=continuity,
        margin_report=margins,
        manifest=manifest,
        passed=passed,
        metadata={
            **scenario.metadata,
            "phase_order": ["launch", "operations", "digital_twin", "deorbit", "reentry"],
            "propellant_used_kg": propellant_used_kg,
            **(
                {
                    "resolved_input_overrides": scenario.input_overrides.model_dump(
                        mode="json",
                        exclude_none=True,
                        exclude_defaults=True,
                    )
                }
                if scenario.input_overrides is not None
                else {}
            ),
        },
        warnings=[
            "Mission lifecycle evidence is deterministic design screening, "
            "not flight qualification.",
            *twin.warnings,
            *reentry.warnings,
        ],
    )


def _apply_launch_input_overrides(
    launch_scenario: LaunchScenario,
    lifecycle_scenario: MissionLifecycleScenario,
) -> LaunchScenario:
    overrides = lifecycle_scenario.input_overrides
    if overrides is None:
        return launch_scenario

    resolved = launch_scenario.model_dump(mode="python")
    if overrides.launch_upper_stage_thrust_n is not None:
        resolved["vehicle"]["stages"][-1]["engine"]["thrust_n"] = (
            overrides.launch_upper_stage_thrust_n
        )
    if overrides.spacecraft_wet_mass_kg is not None:
        resolved["vehicle"]["payload_mass_kg"] = overrides.spacecraft_wet_mass_kg
    return LaunchScenario.model_validate(resolved)


def resolve_lifecycle_twin_scenario(
    twin_scenario: DigitalTwinScenario,
    lifecycle_scenario: MissionLifecycleScenario,
) -> DigitalTwinScenario:
    """Apply typed lifecycle overrides and validate the resolved twin template."""
    overrides = lifecycle_scenario.input_overrides
    if overrides is None:
        return twin_scenario

    resolved = twin_scenario.model_dump(mode="python")
    if overrides.spacecraft_wet_mass_kg is not None:
        dry_payload_mass_kg = float(
            twin_scenario.spacecraft.dry_mass_kg + twin_scenario.spacecraft.payload_mass_kg
        )
        resolved["spacecraft"]["propellant_mass_kg"] = (
            float(overrides.spacecraft_wet_mass_kg) - dry_payload_mass_kg
        )
    if overrides.twin_solar_array_efficiency is not None:
        resolved["power"]["solar_array_efficiency"] = overrides.twin_solar_array_efficiency
    if overrides.twin_solar_array_area_m2 is not None:
        resolved["power"]["solar_array_area_m2"] = overrides.twin_solar_array_area_m2
    if overrides.twin_battery_capacity_wh is not None:
        resolved["power"]["battery_capacity_wh"] = overrides.twin_battery_capacity_wh
    thermal_names = [node["name"] for node in resolved["thermal_nodes"]]
    if len(set(thermal_names)) != len(thermal_names):
        raise MissionLifecycleError(
            "digital twin thermal node names must be unique",
            lifecycle_phase="digital_twin",
        )
    thermal_positions = {name: index for index, name in enumerate(thermal_names)}
    for thermal_override in overrides.twin_thermal_node_overrides:
        if thermal_override.node_name not in thermal_positions:
            raise MissionLifecycleError(
                f"digital twin thermal override references missing node: "
                f"{thermal_override.node_name}",
                lifecycle_phase="digital_twin",
            )
        node = resolved["thermal_nodes"][thermal_positions[thermal_override.node_name]]
        if thermal_override.emissivity is not None:
            node["emissivity"] = thermal_override.emissivity
        if thermal_override.internal_heat_fraction is not None:
            node["internal_heat_fraction"] = thermal_override.internal_heat_fraction
    try:
        return DigitalTwinScenario.model_validate(resolved)
    except ValueError as exc:
        raise MissionLifecycleError(
            f"resolved digital twin scenario is invalid: {exc}",
            lifecycle_phase="digital_twin",
        ) from exc


def _apply_reentry_input_overrides(
    reentry_scenario: ReentryScenario,
    lifecycle_scenario: MissionLifecycleScenario,
) -> ReentryScenario:
    overrides = lifecycle_scenario.input_overrides
    if overrides is None:
        return reentry_scenario

    resolved = reentry_scenario.model_dump(mode="python")
    if overrides.reentry_atmosphere_density_scale_factor is not None:
        resolved["atmosphere"]["density_scale_factor"] = (
            overrides.reentry_atmosphere_density_scale_factor
        )
    if overrides.reentry_vehicle_drag_coefficient is not None:
        resolved["vehicle"]["drag_coefficient"] = overrides.reentry_vehicle_drag_coefficient
    return ReentryScenario.model_validate(resolved)


def _launch_margins(
    scenario: LaunchScenario,
    launch: LaunchTrajectory,
) -> tuple[MissionLifecycleMargin, ...]:
    definitions = (
        (
            "insertion_altitude",
            "altitude_miss_km",
            scenario.target_orbit.altitude_tolerance_km,
            "km",
        ),
        (
            "insertion_velocity",
            "velocity_miss_km_s",
            scenario.target_orbit.velocity_tolerance_km_s,
            "km/s",
        ),
        (
            "insertion_radial_velocity",
            "radial_velocity_miss_km_s",
            scenario.target_orbit.radial_velocity_tolerance_km_s,
            "km/s",
        ),
    )
    return tuple(
        _maximum_margin(
            phase="launch",
            name=name,
            value=abs(float(launch.target_miss[key])),
            threshold=float(tolerance),
            unit=unit,
        )
        for name, key, tolerance, unit in definitions
    )


def _run_deorbit(
    scenario: MissionLifecycleScenario,
    operations: Trajectory,
    orbit_scenario: Scenario,
    *,
    available_propellant_kg: float,
) -> tuple[Maneuver, Scenario, Trajectory, float]:
    final_sample = operations.samples[-1]
    preburn_state = OrbitState(
        epoch=final_sample.epoch,
        time_scale=orbit_scenario.initial_state.time_scale,
        frame=orbit_scenario.initial_state.frame,
        central_body=orbit_scenario.initial_state.central_body,
        representation=OrbitRepresentation.CARTESIAN,
        cartesian=final_sample.state,
    )
    velocity = final_sample.state.velocity_km_s
    speed = _norm(velocity)
    delta_v: Vector3 = (
        -float(scenario.deorbit.delta_v_km_s) * velocity[0] / speed,
        -float(scenario.deorbit.delta_v_km_s) * velocity[1] / speed,
        -float(scenario.deorbit.delta_v_km_s) * velocity[2] / speed,
    )
    maneuver = Maneuver(
        name="deorbit-retrograde-burn",
        epoch=final_sample.epoch,
        frame=preburn_state.frame,
        delta_v_km_s=delta_v,
        metadata={
            "workflow": "mission_lifecycle_v1",
            "direction": "retrograde",
            "specific_impulse_s": float(scenario.deorbit.specific_impulse_s),
        },
    )
    postburn_state = apply_impulsive_maneuver(preburn_state, maneuver)
    initial_mass_kg = float(orbit_scenario.spacecraft.mass_kg)
    final_mass_kg = initial_mass_kg * exp(
        -float(scenario.deorbit.delta_v_km_s)
        * 1000.0
        / (float(scenario.deorbit.specific_impulse_s) * _STANDARD_GRAVITY_M_S2)
    )
    propellant_used_kg = initial_mass_kg - final_mass_kg
    reserve_kg = available_propellant_kg - propellant_used_kg
    if reserve_kg < float(scenario.deorbit.minimum_propellant_reserve_kg):
        raise MissionLifecycleError(
            "deorbit burn violates propellant reserve: "
            f"{reserve_kg:.6f} kg available after burn, "
            f"{float(scenario.deorbit.minimum_propellant_reserve_kg):.6f} kg required",
            lifecycle_phase="deorbit",
        )
    deorbit_scenario = Scenario(
        scenario_id=f"{scenario.scenario_id}-deorbit",
        description="Two-body coast from the post-deorbit-burn state to entry interface.",
        spacecraft=orbit_scenario.spacecraft.model_copy(update={"mass_kg": final_mass_kg}),
        initial_state=postburn_state,
        force_model=ForceModelConfig(gravity=orbit_scenario.force_model.gravity),
        propagation=PropagationConfig(
            duration_s=scenario.deorbit.coast_duration_s,
            step_s=scenario.deorbit.step_s,
        ),
        metadata={
            "workflow": "mission_lifecycle_v1",
            "source_operations_scenario_id": operations.scenario_id,
            "propellant_used_kg": propellant_used_kg,
        },
    )
    full_coast = propagate_local(deorbit_scenario)
    interface_index = _entry_interface_index(
        full_coast,
        float(scenario.deorbit.entry_interface_altitude_km),
    )
    interface_sample = full_coast.samples[interface_index]
    interface_altitude_km = _altitude_km(interface_sample.state.position_km)
    if abs(interface_altitude_km - float(scenario.deorbit.entry_interface_altitude_km)) > float(
        scenario.deorbit.interface_tolerance_km
    ):
        raise MissionLifecycleError(
            "deorbit coast crossed entry interface outside tolerance: "
            f"{interface_altitude_km:.6f} km",
            lifecycle_phase="deorbit",
        )
    deorbit = full_coast.model_copy(
        update={
            "samples": full_coast.samples[: interface_index + 1],
            "events": [
                event for event in full_coast.events if event.epoch <= interface_sample.epoch
            ],
            "maneuvers": [maneuver],
            "metadata": {
                **full_coast.metadata,
                "termination_reason": "entry_interface",
                "entry_interface_altitude_km": interface_altitude_km,
                "source_full_coast_sample_count": len(full_coast.samples),
            },
        }
    )
    return maneuver, deorbit_scenario, deorbit, propellant_used_kg


def _entry_interface_index(trajectory: Trajectory, interface_altitude_km: float) -> int:
    for index, sample in enumerate(trajectory.samples):
        radial_velocity_km_s = _dot(sample.state.position_km, sample.state.velocity_km_s) / _norm(
            sample.state.position_km
        )
        if (
            _altitude_km(sample.state.position_km) <= interface_altitude_km
            and radial_velocity_km_s < 0.0
        ):
            return index
    raise MissionLifecycleError(
        f"deorbit coast did not reach descending {interface_altitude_km:.3f} km entry interface",
        lifecycle_phase="deorbit",
    )


def _build_continuity_report(
    *,
    launch: LaunchTrajectory,
    orbit_scenario: Scenario,
    operations: Trajectory,
    twin_wet_mass_kg: float,
    maneuver: Maneuver,
    deorbit_scenario: Scenario,
    deorbit: Trajectory,
    reentry_scenario: ReentryScenario,
) -> LifecycleContinuityReport:
    launch_state = launch.insertion_state.cartesian
    orbit_state = orbit_scenario.initial_state.cartesian
    operations_final = operations.samples[-1]
    checks = (
        _continuity(
            "launch_position",
            "launch",
            "operations",
            _vector_error(launch_state.position_km, orbit_state.position_km),
            _STATE_CONTINUITY_TOLERANCE,
            "km",
        ),
        _continuity(
            "launch_velocity",
            "launch",
            "operations",
            _vector_error(launch_state.velocity_km_s, orbit_state.velocity_km_s),
            _STATE_CONTINUITY_TOLERANCE,
            "km/s",
        ),
        _continuity(
            "launch_mass",
            "launch",
            "operations",
            abs(float(launch.samples[-1].mass_kg) - float(orbit_scenario.spacecraft.mass_kg)),
            _MASS_CONTINUITY_TOLERANCE_KG,
            "kg",
        ),
        _continuity(
            "twin_mass",
            "operations",
            "digital_twin",
            abs(float(orbit_scenario.spacecraft.mass_kg) - twin_wet_mass_kg),
            _MASS_CONTINUITY_TOLERANCE_KG,
            "kg",
        ),
        _continuity(
            "deorbit_position",
            "operations",
            "deorbit",
            _vector_error(
                operations_final.state.position_km,
                deorbit_scenario.initial_state.cartesian.position_km,
            ),
            _STATE_CONTINUITY_TOLERANCE,
            "km",
        ),
        _continuity(
            "deorbit_epoch",
            "operations",
            "deorbit",
            abs((operations_final.epoch - maneuver.epoch).total_seconds()),
            _STATE_CONTINUITY_TOLERANCE,
            "s",
        ),
        _continuity(
            "reentry_epoch",
            "deorbit",
            "reentry",
            abs((deorbit.samples[-1].epoch - reentry_scenario.initial_state.epoch).total_seconds()),
            _STATE_CONTINUITY_TOLERANCE,
            "s",
        ),
        _continuity(
            "reentry_mass",
            "deorbit",
            "reentry",
            abs(
                float(deorbit.samples[-1].mass_kg or 0.0) - float(reentry_scenario.vehicle.mass_kg)
            ),
            _MASS_CONTINUITY_TOLERANCE_KG,
            "kg",
        ),
    )
    return LifecycleContinuityReport(
        checks=checks, all_passed=all(check.passed for check in checks)
    )


def _build_margin_report(
    *,
    scenario: MissionLifecycleScenario,
    launch_margins: tuple[MissionLifecycleMargin, ...],
    twin: DigitalTwinResult,
    propellant_used_kg: float,
    available_propellant_kg: float,
    entry_altitude_km: float,
    reentry: ReentryResult,
) -> MissionLifecycleMarginReport:
    twin_limiting = twin.margin_report.limiting_margin
    margins = [*launch_margins]
    margins.append(
        MissionLifecycleMargin(
            phase="digital_twin",
            name=twin_limiting.name,
            value=twin_limiting.value,
            threshold=twin_limiting.threshold,
            margin=twin_limiting.margin,
            unit=twin_limiting.unit,
            status=_lifecycle_status(twin_limiting.status),
        )
    )
    reserve_kg = available_propellant_kg - propellant_used_kg
    margins.append(
        _minimum_margin(
            phase="deorbit",
            name="propellant_reserve",
            value=reserve_kg,
            threshold=float(scenario.deorbit.minimum_propellant_reserve_kg),
            unit="kg",
        )
    )
    interface_error_km = abs(
        entry_altitude_km - float(scenario.deorbit.entry_interface_altitude_km)
    )
    margins.append(
        _maximum_margin(
            phase="deorbit",
            name="entry_interface_altitude_error",
            value=interface_error_km,
            threshold=float(scenario.deorbit.interface_tolerance_km),
            unit="km",
        )
    )
    margins.extend(
        MissionLifecycleMargin(
            phase="reentry",
            name=margin.name,
            value=margin.value,
            threshold=margin.threshold,
            margin=margin.margin,
            unit=margin.unit,
            status=_lifecycle_status(margin.status),
        )
        for margin in reentry.margin_report.margins
    )
    limiting = min(margins, key=_limiting_margin_key)
    overall = max(margins, key=lambda item: _severity(item.status)).status
    return MissionLifecycleMarginReport(
        margins=tuple(margins),
        limiting_margin=limiting,
        overall_status=overall,
    )


def _build_manifest(
    *,
    scenario: MissionLifecycleScenario,
    launch: LaunchTrajectory,
    operations: Trajectory,
    twin: DigitalTwinResult,
    deorbit: Trajectory,
    reentry_scenario: ReentryScenario,
    reentry: ReentryResult,
) -> MissionLifecycleManifest:
    entries = (
        LifecyclePhaseManifestEntry(
            sequence=1,
            phase="launch",
            product_type="LaunchTrajectory",
            artifact_name="launch.json",
            scenario_id=launch.scenario_id,
            backend=launch.backend,
            start_epoch=launch.samples[0].epoch,
            end_epoch=launch.samples[-1].epoch,
            sample_count=len(launch.samples),
            event_count=len(launch.events),
        ),
        LifecyclePhaseManifestEntry(
            sequence=2,
            phase="operations",
            product_type="Trajectory",
            artifact_name="operations-trajectory.json",
            scenario_id=operations.scenario_id,
            backend=operations.backend,
            start_epoch=operations.samples[0].epoch,
            end_epoch=operations.samples[-1].epoch,
            sample_count=len(operations.samples),
            event_count=len(operations.events),
        ),
        LifecyclePhaseManifestEntry(
            sequence=3,
            phase="digital_twin",
            product_type="DigitalTwinResult",
            artifact_name="digital-twin.json",
            scenario_id=twin.scenario_id,
            backend=str(twin.metadata["orbit_backend"]),
            start_epoch=twin.geometry[0].epoch,
            end_epoch=twin.geometry[-1].epoch,
            sample_count=len(twin.geometry),
            event_count=len(twin.access_windows),
        ),
        LifecyclePhaseManifestEntry(
            sequence=4,
            phase="deorbit",
            product_type="Trajectory",
            artifact_name="deorbit-trajectory.json",
            scenario_id=deorbit.scenario_id,
            backend=deorbit.backend,
            start_epoch=deorbit.samples[0].epoch,
            end_epoch=deorbit.samples[-1].epoch,
            sample_count=len(deorbit.samples),
            event_count=len(deorbit.events),
            metadata={"maneuver_artifact_embedded": True},
        ),
        LifecyclePhaseManifestEntry(
            sequence=5,
            phase="reentry",
            product_type="ReentryResult",
            artifact_name="reentry-result.json",
            scenario_id=reentry.scenario_id,
            backend=reentry.backend,
            start_epoch=reentry_scenario.initial_state.epoch,
            end_epoch=reentry.samples[-1].epoch,
            sample_count=len(reentry.samples),
            event_count=len(reentry.events),
        ),
    )
    return MissionLifecycleManifest(
        scenario_id=scenario.scenario_id,
        entries=entries,
        metadata={"suite_owned_products": True, "claim_boundary": "design_screening"},
    )


def _continuity(
    name: str,
    upstream: LifecyclePhaseName,
    downstream: LifecyclePhaseName,
    error: float,
    tolerance: float,
    unit: str,
) -> LifecycleContinuityCheck:
    return LifecycleContinuityCheck(
        name=name,
        upstream_phase=upstream,
        downstream_phase=downstream,
        error=error,
        tolerance=tolerance,
        unit=unit,
        passed=error <= tolerance,
    )


def _maximum_margin(
    *, phase: LifecyclePhaseName, name: str, value: float, threshold: float, unit: str
) -> MissionLifecycleMargin:
    margin = threshold - value
    return MissionLifecycleMargin(
        phase=phase,
        name=name,
        value=value,
        threshold=threshold,
        margin=margin,
        unit=unit,
        status=_status_for_margin(margin, threshold),
    )


def _minimum_margin(
    *, phase: LifecyclePhaseName, name: str, value: float, threshold: float, unit: str
) -> MissionLifecycleMargin:
    margin = value - threshold
    return MissionLifecycleMargin(
        phase=phase,
        name=name,
        value=value,
        threshold=threshold,
        margin=margin,
        unit=unit,
        status=_status_for_margin(margin, max(abs(threshold), abs(value))),
    )


def _status_for_margin(margin: float, scale: float) -> LifecycleStatus:
    if margin < 0.0:
        return LifecycleStatus.FAIL
    if margin <= 0.1 * max(scale, 1.0e-12):
        return LifecycleStatus.WARN
    return LifecycleStatus.PASS


def _lifecycle_status(status: TwinMarginStatus | ReentryMarginStatus) -> LifecycleStatus:
    if status in {TwinMarginStatus.FAIL, ReentryMarginStatus.FAIL}:
        return LifecycleStatus.FAIL
    if status in {TwinMarginStatus.WARN, ReentryMarginStatus.WARN}:
        return LifecycleStatus.WARN
    return LifecycleStatus.PASS


def _limiting_margin_key(margin: MissionLifecycleMargin) -> tuple[int, float]:
    scale = max(abs(float(margin.threshold)), 1.0e-12)
    return (-_severity(margin.status), float(margin.margin) / scale)


def _severity(status: LifecycleStatus) -> int:
    return {LifecycleStatus.PASS: 0, LifecycleStatus.WARN: 1, LifecycleStatus.FAIL: 2}[status]


def _require_no_failed_margins(margins: tuple[MissionLifecycleMargin, ...], label: str) -> None:
    failed = [margin.name for margin in margins if margin.status == LifecycleStatus.FAIL]
    if failed:
        raise MissionLifecycleError(
            f"{label} failed: {', '.join(failed)}",
            lifecycle_phase="launch",
        )


def _altitude_km(position_km: Vector3) -> float:
    return _norm(position_km) - R_EARTH_KM


def _norm(vector: Vector3) -> float:
    return sqrt(_dot(vector, vector))


def _dot(left: Vector3, right: Vector3) -> float:
    return sum(left[index] * right[index] for index in range(3))


def _vector_error(left: Vector3, right: Vector3) -> float:
    return sqrt(sum((left[index] - right[index]) ** 2 for index in range(3)))
