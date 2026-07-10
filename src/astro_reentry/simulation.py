from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from math import atan2, cos, degrees, pi, radians, sin, sqrt, tan

from astro_core.constants import MU_EARTH_KM3_S2, R_EARTH_KM
from astro_reentry.atmosphere import (
    atmospheric_density_kg_m3,
    convective_heat_rate_w_m2,
    radiative_equilibrium_temperature_k,
)
from astro_reentry.guidance import (
    commanded_bank_angle_deg,
    normalize_heading_deg,
    normalize_longitude_deg,
)
from astro_reentry.margins import build_reentry_margin_report
from astro_reentry.models import (
    ReentryEvent,
    ReentryPeakMetric,
    ReentryPeakSummary,
    ReentryResult,
    ReentrySample,
    ReentryScenario,
    ReentryTarget,
    ReentryTargetMiss,
)

STANDARD_GRAVITY_M_S2 = 9.80665
EARTH_RADIUS_M = R_EARTH_KM * 1000.0
EARTH_MU_M3_S2 = MU_EARTH_KM3_S2 * 1.0e9

State = tuple[float, float, float, float, float, float, float, float]


@dataclass(frozen=True)
class _Diagnostics:
    density_kg_m3: float
    dynamic_pressure_pa: float
    drag_acceleration_m_s2: float
    lift_acceleration_m_s2: float
    deceleration_g: float
    heat_rate_w_m2: float
    equilibrium_temperature_k: float


@dataclass(frozen=True)
class _Record:
    time_s: float
    state: State
    bank_angle_deg: float
    diagnostics: _Diagnostics


def simulate_reentry_local(scenario: ReentryScenario) -> ReentryResult:
    state = _initial_state(scenario)
    time_s = 0.0
    previous_bank_sign = 0
    bank_angle_deg, previous_bank_sign = commanded_bank_angle_deg(
        scenario.guidance,
        velocity_km_s=state[3] / 1000.0,
        latitude_deg=degrees(state[1]),
        longitude_deg=degrees(state[2]),
        heading_deg=normalize_heading_deg(degrees(state[5])),
        target=scenario.target,
        previous_bank_sign=previous_bank_sign,
    )
    initial_record = _record(scenario, time_s, state, bank_angle_deg)
    internal_records = [initial_record]
    output_records = [initial_record]
    events: list[ReentryEvent] = []
    entry_recorded = state[0] - EARTH_RADIUS_M <= (
        scenario.propagation.entry_interface_altitude_km * 1000.0
    )
    last_reversal_time_s = float("-inf")
    if entry_recorded:
        events.append(_event(scenario, "entry_interface", initial_record))

    next_output_s = float(scenario.propagation.step_s)
    termination_reason = "duration"
    while time_s < scenario.propagation.duration_s - 1.0e-12:
        dt_s = min(
            float(scenario.propagation.internal_step_s),
            float(scenario.propagation.duration_s) - time_s,
        )
        bank_angle_deg, bank_sign = commanded_bank_angle_deg(
            scenario.guidance,
            velocity_km_s=state[3] / 1000.0,
            latitude_deg=degrees(state[1]),
            longitude_deg=degrees(state[2]),
            heading_deg=normalize_heading_deg(degrees(state[5])),
            target=scenario.target,
            previous_bank_sign=previous_bank_sign,
        )
        if previous_bank_sign and bank_sign and bank_sign != previous_bank_sign:
            if (
                time_s - last_reversal_time_s
                < scenario.guidance.minimum_bank_reversal_interval_s
            ):
                bank_sign = previous_bank_sign
                bank_angle_deg = abs(bank_angle_deg) * previous_bank_sign
            else:
                events.append(
                    _event(
                        scenario,
                        "guidance_bank_reversal",
                        internal_records[-1],
                        metadata={
                            "previous_sign": previous_bank_sign,
                            "commanded_sign": bank_sign,
                        },
                    )
                )
                last_reversal_time_s = time_s
        previous_bank_sign = bank_sign

        next_state = _rk4_step(scenario, state, bank_angle_deg, dt_s)
        next_time_s = time_s + dt_s
        next_record = _record(scenario, next_time_s, next_state, bank_angle_deg)

        if not entry_recorded and _altitude_km(next_state) <= (
            scenario.propagation.entry_interface_altitude_km
        ):
            interface_record = _interpolate_record_at_altitude(
                scenario,
                internal_records[-1],
                next_record,
                float(scenario.propagation.entry_interface_altitude_km),
            )
            events.append(_event(scenario, "entry_interface", interface_record))
            entry_recorded = True

        termination_fraction, termination_reason = _termination_fraction(
            scenario,
            state,
            next_state,
        )
        if termination_fraction is not None:
            terminal_state = _interpolate_state(state, next_state, termination_fraction)
            terminal_state = _snap_terminal_state(
                scenario,
                terminal_state,
                termination_reason,
            )
            terminal_time_s = time_s + termination_fraction * dt_s
            terminal_record = _record(
                scenario,
                terminal_time_s,
                terminal_state,
                bank_angle_deg,
            )
            internal_records.append(terminal_record)
            _append_record(output_records, terminal_record)
            events.append(
                _event(
                    scenario,
                    "terminal",
                    terminal_record,
                    metadata={"reason": termination_reason},
                )
            )
            break

        state = next_state
        time_s = next_time_s
        internal_records.append(next_record)
        if time_s >= next_output_s - 1.0e-9:
            _append_record(output_records, next_record)
            next_output_s += float(scenario.propagation.step_s)
    else:
        final_record = internal_records[-1]
        _append_record(output_records, final_record)
        events.append(
            _event(
                scenario,
                "terminal",
                final_record,
                metadata={"reason": termination_reason},
            )
        )

    peaks, peak_events = _build_peaks_and_events(scenario, internal_records)
    events.extend(peak_events)
    events.sort(key=lambda event: (event.time_s, _event_sort_order(event.event_type)))
    samples = tuple(_sample(scenario, record) for record in output_records)
    target_miss = _target_miss(scenario.target, samples[-1])
    margins = build_reentry_margin_report(
        limits=scenario.limits,
        peaks=peaks,
        target=scenario.target,
        target_miss=target_miss,
    )
    return ReentryResult(
        scenario_id=scenario.scenario_id,
        backend="local",
        samples=samples,
        events=tuple(events),
        peaks=peaks,
        target_miss=target_miss,
        margin_report=margins,
        metadata={
            "dynamics_model": "spherical_earth_3dof_point_mass",
            "atmosphere_model": scenario.atmosphere.model,
            "aerothermal_model": scenario.aerothermal.model,
            "guidance_mode": scenario.guidance.mode,
            "ballistic_coefficient_kg_m2": scenario.vehicle.ballistic_coefficient_kg_m2,
            "lift_to_drag_ratio": scenario.vehicle.lift_to_drag_ratio,
            "internal_step_s": scenario.propagation.internal_step_s,
            "termination_reason": termination_reason,
            "state_velocity_reference": "atmosphere_relative",
        },
        warnings=_warnings(scenario),
    )


def _initial_state(scenario: ReentryScenario) -> State:
    initial = scenario.initial_state
    return (
        EARTH_RADIUS_M + float(initial.altitude_km) * 1000.0,
        radians(float(initial.latitude_deg)),
        radians(float(initial.longitude_deg)),
        float(initial.velocity_km_s) * 1000.0,
        radians(float(initial.flight_path_angle_deg)),
        radians(float(initial.heading_deg)),
        0.0,
        0.0,
    )


def _rk4_step(
    scenario: ReentryScenario,
    state: State,
    bank_angle_deg: float,
    dt_s: float,
) -> State:
    k1 = _derivatives(scenario, state, bank_angle_deg)
    k2 = _derivatives(scenario, _state_add(state, k1, 0.5 * dt_s), bank_angle_deg)
    k3 = _derivatives(scenario, _state_add(state, k2, 0.5 * dt_s), bank_angle_deg)
    k4 = _derivatives(scenario, _state_add(state, k3, dt_s), bank_angle_deg)
    integrated = tuple(
        state[index]
        + dt_s
        * (k1[index] + 2.0 * k2[index] + 2.0 * k3[index] + k4[index])
        / 6.0
        for index in range(len(state))
    )
    latitude_limit = pi / 2.0 - 1.0e-9
    return (
        max(1.0, integrated[0]),
        max(-latitude_limit, min(latitude_limit, integrated[1])),
        (integrated[2] + pi) % (2.0 * pi) - pi,
        max(0.0, integrated[3]),
        max(-pi / 2.0, min(pi / 2.0, integrated[4])),
        integrated[5] % (2.0 * pi),
        max(0.0, integrated[6]),
        max(0.0, integrated[7]),
    )


def _state_add(state: State, derivative: State, scale: float) -> State:
    return tuple(
        state[index] + scale * derivative[index] for index in range(len(state))
    )  # type: ignore[return-value]


def _derivatives(
    scenario: ReentryScenario,
    state: State,
    bank_angle_deg: float,
) -> State:
    radius_m, latitude, _longitude, velocity_m_s, gamma, heading, _downrange, _heat = state
    diagnostics = _diagnostics(scenario, state)
    gravity_m_s2 = EARTH_MU_M3_S2 / radius_m**2
    velocity_safe = max(velocity_m_s, 1.0)
    cos_gamma = cos(gamma)
    cos_latitude = cos(latitude)
    cos_gamma_safe = _signed_floor(cos_gamma, 1.0e-8)
    cos_latitude_safe = _signed_floor(cos_latitude, 1.0e-8)
    bank_angle = radians(bank_angle_deg)
    lift_acceleration = diagnostics.lift_acceleration_m_s2

    radius_rate = velocity_m_s * sin(gamma)
    latitude_rate = velocity_m_s * cos_gamma * cos(heading) / radius_m
    longitude_rate = (
        velocity_m_s * cos_gamma * sin(heading) / (radius_m * cos_latitude_safe)
    )
    velocity_rate = -diagnostics.drag_acceleration_m_s2 - gravity_m_s2 * sin(gamma)
    gamma_rate = lift_acceleration * cos(bank_angle) / velocity_safe + (
        velocity_m_s / radius_m - gravity_m_s2 / velocity_safe
    ) * cos_gamma
    heading_rate = lift_acceleration * sin(bank_angle) / (
        velocity_safe * cos_gamma_safe
    ) + velocity_m_s * cos_gamma * sin(heading) * tan(latitude) / radius_m
    downrange_rate = max(0.0, velocity_m_s * cos_gamma)
    heat_load_rate = diagnostics.heat_rate_w_m2
    return (
        radius_rate,
        latitude_rate,
        longitude_rate,
        velocity_rate,
        gamma_rate,
        heading_rate,
        downrange_rate,
        heat_load_rate,
    )


def _signed_floor(value: float, floor: float) -> float:
    if abs(value) >= floor:
        return value
    return floor if value >= 0.0 else -floor


def _diagnostics(scenario: ReentryScenario, state: State) -> _Diagnostics:
    velocity_m_s = max(0.0, state[3])
    density = atmospheric_density_kg_m3(scenario.atmosphere, _altitude_km(state))
    dynamic_pressure = 0.5 * density * velocity_m_s**2
    drag_force_n = (
        dynamic_pressure * scenario.vehicle.drag_coefficient * scenario.vehicle.reference_area_m2
    )
    drag_acceleration = drag_force_n / scenario.vehicle.mass_kg
    lift_acceleration = drag_acceleration * scenario.vehicle.lift_to_drag_ratio
    heat_rate = convective_heat_rate_w_m2(
        scenario.aerothermal,
        scenario.vehicle,
        density,
        velocity_m_s,
    )
    return _Diagnostics(
        density_kg_m3=float(density),
        dynamic_pressure_pa=float(dynamic_pressure),
        drag_acceleration_m_s2=float(drag_acceleration),
        lift_acceleration_m_s2=float(lift_acceleration),
        deceleration_g=float(drag_acceleration / STANDARD_GRAVITY_M_S2),
        heat_rate_w_m2=float(heat_rate),
        equilibrium_temperature_k=radiative_equilibrium_temperature_k(
            scenario.aerothermal,
            heat_rate,
        ),
    )


def _record(
    scenario: ReentryScenario,
    time_s: float,
    state: State,
    bank_angle_deg: float,
) -> _Record:
    return _Record(
        time_s=float(time_s),
        state=state,
        bank_angle_deg=float(bank_angle_deg),
        diagnostics=_diagnostics(scenario, state),
    )


def _sample(scenario: ReentryScenario, record: _Record) -> ReentrySample:
    state = record.state
    latitude_deg = degrees(state[1])
    longitude_deg = normalize_longitude_deg(degrees(state[2]))
    return ReentrySample(
        epoch=scenario.initial_state.epoch + timedelta(seconds=record.time_s),
        time_s=record.time_s,
        altitude_km=_altitude_km(state),
        latitude_deg=latitude_deg,
        longitude_deg=longitude_deg,
        downrange_km=state[6] / 1000.0,
        velocity_km_s=state[3] / 1000.0,
        flight_path_angle_deg=degrees(state[4]),
        heading_deg=normalize_heading_deg(degrees(state[5])),
        bank_angle_deg=record.bank_angle_deg,
        atmospheric_density_kg_m3=record.diagnostics.density_kg_m3,
        dynamic_pressure_pa=record.diagnostics.dynamic_pressure_pa,
        drag_acceleration_m_s2=record.diagnostics.drag_acceleration_m_s2,
        lift_acceleration_m_s2=record.diagnostics.lift_acceleration_m_s2,
        deceleration_g=record.diagnostics.deceleration_g,
        convective_heat_rate_w_m2=record.diagnostics.heat_rate_w_m2,
        heat_load_j_m2=state[7],
        radiative_equilibrium_temperature_k=record.diagnostics.equilibrium_temperature_k,
        range_to_target_km=(
            great_circle_distance_km(latitude_deg, longitude_deg, scenario.target)
            if scenario.target is not None
            else None
        ),
    )


def great_circle_distance_km(
    latitude_deg: float,
    longitude_deg: float,
    target: ReentryTarget,
) -> float:
    latitude = radians(latitude_deg)
    target_latitude = radians(float(target.latitude_deg))
    delta_latitude = target_latitude - latitude
    delta_longitude = radians(
        normalize_longitude_deg(float(target.longitude_deg) - longitude_deg)
    )
    haversine = sin(delta_latitude / 2.0) ** 2 + cos(latitude) * cos(
        target_latitude
    ) * sin(delta_longitude / 2.0) ** 2
    central_angle = 2.0 * atan2(sqrt(max(0.0, haversine)), sqrt(max(0.0, 1.0 - haversine)))
    return float(R_EARTH_KM * central_angle)


def _target_miss(
    target: ReentryTarget | None,
    final_sample: ReentrySample,
) -> ReentryTargetMiss | None:
    if target is None:
        return None
    return ReentryTargetMiss(
        distance_km=great_circle_distance_km(
            final_sample.latitude_deg,
            final_sample.longitude_deg,
            target,
        ),
        latitude_error_deg=float(final_sample.latitude_deg - target.latitude_deg),
        longitude_error_deg=normalize_longitude_deg(
            float(final_sample.longitude_deg - target.longitude_deg)
        ),
    )


def _build_peaks_and_events(
    scenario: ReentryScenario,
    records: list[_Record],
) -> tuple[ReentryPeakSummary, list[ReentryEvent]]:
    dynamic_pressure = max(records, key=lambda record: record.diagnostics.dynamic_pressure_pa)
    deceleration = max(records, key=lambda record: record.diagnostics.deceleration_g)
    heat_rate = max(records, key=lambda record: record.diagnostics.heat_rate_w_m2)
    peaks = ReentryPeakSummary(
        dynamic_pressure=_peak_metric(dynamic_pressure, "dynamic_pressure_pa", "Pa"),
        deceleration=_peak_metric(deceleration, "deceleration_g", "g"),
        heat_rate=_peak_metric(heat_rate, "heat_rate_w_m2", "W/m^2"),
        total_heat_load_j_m2=records[-1].state[7],
    )
    events = [
        _event(
            scenario,
            "peak_dynamic_pressure",
            dynamic_pressure,
            metadata={"value_pa": dynamic_pressure.diagnostics.dynamic_pressure_pa},
        ),
        _event(
            scenario,
            "peak_deceleration",
            deceleration,
            metadata={"value_g": deceleration.diagnostics.deceleration_g},
        ),
        _event(
            scenario,
            "peak_heating",
            heat_rate,
            metadata={"value_w_m2": heat_rate.diagnostics.heat_rate_w_m2},
        ),
    ]
    return peaks, events


def _peak_metric(record: _Record, attribute: str, unit: str) -> ReentryPeakMetric:
    value = {
        "dynamic_pressure_pa": record.diagnostics.dynamic_pressure_pa,
        "deceleration_g": record.diagnostics.deceleration_g,
        "heat_rate_w_m2": record.diagnostics.heat_rate_w_m2,
    }[attribute]
    return ReentryPeakMetric(
        value=value,
        unit=unit,
        time_s=record.time_s,
        altitude_km=_altitude_km(record.state),
    )


def _event(
    scenario: ReentryScenario,
    event_type: str,
    record: _Record,
    metadata: dict[str, object] | None = None,
) -> ReentryEvent:
    return ReentryEvent(
        event_type=event_type,  # type: ignore[arg-type]
        epoch=scenario.initial_state.epoch + timedelta(seconds=record.time_s),
        time_s=record.time_s,
        altitude_km=_altitude_km(record.state),
        metadata={} if metadata is None else metadata,
    )


def _event_sort_order(event_type: str) -> int:
    return {
        "entry_interface": 0,
        "guidance_bank_reversal": 1,
        "peak_heating": 2,
        "peak_dynamic_pressure": 3,
        "peak_deceleration": 4,
        "terminal": 5,
    }[event_type]


def _termination_fraction(
    scenario: ReentryScenario,
    state: State,
    next_state: State,
) -> tuple[float | None, str]:
    altitude_threshold_m = scenario.propagation.termination_altitude_km * 1000.0
    previous_altitude_m = state[0] - EARTH_RADIUS_M
    next_altitude_m = next_state[0] - EARTH_RADIUS_M
    if next_altitude_m <= altitude_threshold_m:
        denominator = previous_altitude_m - next_altitude_m
        fraction = (
            1.0
            if denominator <= 0.0
            else (previous_altitude_m - altitude_threshold_m) / denominator
        )
        return max(0.0, min(1.0, fraction)), "altitude"
    minimum_velocity_m_s = scenario.propagation.minimum_velocity_km_s * 1000.0
    if next_state[3] <= minimum_velocity_m_s:
        denominator = state[3] - next_state[3]
        fraction = (
            1.0 if denominator <= 0.0 else (state[3] - minimum_velocity_m_s) / denominator
        )
        return max(0.0, min(1.0, fraction)), "minimum_velocity"
    return None, "duration"


def _interpolate_state(start: State, end: State, fraction: float) -> State:
    return tuple(
        start[index] + fraction * (end[index] - start[index]) for index in range(len(start))
    )  # type: ignore[return-value]


def _snap_terminal_state(
    scenario: ReentryScenario,
    state: State,
    reason: str,
) -> State:
    if reason == "altitude":
        return (
            EARTH_RADIUS_M + scenario.propagation.termination_altitude_km * 1000.0,
            *state[1:],
        )
    if reason == "minimum_velocity":
        return (
            state[0],
            state[1],
            state[2],
            scenario.propagation.minimum_velocity_km_s * 1000.0,
            state[4],
            state[5],
            state[6],
            state[7],
        )
    return state


def _interpolate_record_at_altitude(
    scenario: ReentryScenario,
    start: _Record,
    end: _Record,
    altitude_km: float,
) -> _Record:
    start_altitude = _altitude_km(start.state)
    end_altitude = _altitude_km(end.state)
    denominator = start_altitude - end_altitude
    fraction = 1.0 if denominator <= 0.0 else (start_altitude - altitude_km) / denominator
    fraction = max(0.0, min(1.0, fraction))
    state = _interpolate_state(start.state, end.state, fraction)
    return _record(
        scenario,
        start.time_s + fraction * (end.time_s - start.time_s),
        state,
        end.bank_angle_deg,
    )


def _append_record(records: list[_Record], record: _Record) -> None:
    if record.time_s > records[-1].time_s + 1.0e-12:
        records.append(record)


def _altitude_km(state: State) -> float:
    return float((state[0] - EARTH_RADIUS_M) / 1000.0)


def _warnings(scenario: ReentryScenario) -> list[str]:
    warnings = [
        "Reentry v1 is deterministic engineering-screening evidence, not flight "
        "qualification or operational impact prediction.",
        "Dynamics use a spherical non-rotating Earth and a 3-DOF point-mass vehicle "
        "with atmosphere-relative speed.",
    ]
    if scenario.atmosphere.model == "exponential":
        warnings.append(
            "The exponential atmosphere does not represent weather, winds, composition "
            "changes, or density uncertainty campaigns."
        )
    if scenario.aerothermal.model == "sutton_graves":
        warnings.append(
            "Sutton-Graves-style stagnation heating is a convective screening correlation, "
            "not CFD, ablation, or certified TPS analysis."
        )
    if scenario.guidance.mode == "target_tracking":
        warnings.append(
            "Target tracking is a deterministic bank-sign steering law and does not "
            "constitute operational entry guidance."
        )
    return warnings
