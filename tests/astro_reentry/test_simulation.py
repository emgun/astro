from math import isfinite

import pytest

from astro_reentry.models import (
    BankSchedulePoint,
    ReentryAtmosphereConfig,
    ReentryGuidanceConfig,
    ReentryLimits,
    ReentryPropagationConfig,
)
from astro_reentry.simulation import simulate_reentry_local
from tests.astro_reentry.helpers import make_reentry_scenario, reference_target


def test_ballistic_reentry_terminates_and_reports_loads_heating_and_events() -> None:
    result = simulate_reentry_local(make_reentry_scenario())

    assert result.backend == "local"
    assert result.workflow == "reentry_3dof_v1"
    assert result.metadata["atmosphere_model"] == "exponential"
    assert result.metadata["aerothermal_model"] == "sutton_graves"
    assert any("not flight qualification" in warning for warning in result.warnings)
    assert result.metadata["termination_reason"] == "altitude"
    assert result.samples[-1].altitude_km == 0.0
    assert result.samples[-1].velocity_km_s < result.samples[0].velocity_km_s
    assert result.peaks.dynamic_pressure.value > 0.0
    assert result.peaks.deceleration.value > 0.0
    assert result.peaks.heat_rate.value > 0.0
    assert result.peaks.total_heat_load_j_m2 > 0.0
    event_types = {event.event_type for event in result.events}
    assert {
        "entry_interface",
        "peak_heating",
        "peak_dynamic_pressure",
        "peak_deceleration",
        "terminal",
    } <= event_types
    assert all(
        isfinite(float(value))
        for sample in result.samples
        for value in (
            sample.altitude_km,
            sample.velocity_km_s,
            sample.dynamic_pressure_pa,
            sample.convective_heat_rate_w_m2,
            sample.heat_load_j_m2,
        )
    )


def test_constant_bank_lifting_entry_generates_lift_and_crossrange() -> None:
    scenario = make_reentry_scenario(mode="constant_bank", lift_to_drag_ratio=0.3)

    result = simulate_reentry_local(scenario)

    assert max(sample.lift_acceleration_m_s2 for sample in result.samples) > 0.0
    assert abs(result.samples[-1].latitude_deg) > 0.1
    assert all(sample.bank_angle_deg in {0.0, 45.0} for sample in result.samples)


def test_prescribed_bank_schedule_records_bank_reversal() -> None:
    guidance = ReentryGuidanceConfig(
        mode="bank_schedule",
        minimum_bank_reversal_interval_s=20.0,
        bank_schedule=(
            BankSchedulePoint(velocity_km_s=7.8, bank_angle_deg=45.0),
            BankSchedulePoint(velocity_km_s=5.0, bank_angle_deg=45.0),
            BankSchedulePoint(velocity_km_s=2.5, bank_angle_deg=-35.0),
            BankSchedulePoint(velocity_km_s=0.5, bank_angle_deg=-20.0),
        ),
    )
    scenario = make_reentry_scenario(
        mode="bank_schedule",
        lift_to_drag_ratio=0.3,
        guidance=guidance,
    )

    result = simulate_reentry_local(scenario)

    assert any(event.event_type == "guidance_bank_reversal" for event in result.events)


def test_target_tracking_reaches_reference_target_inside_allowable_miss() -> None:
    scenario = make_reentry_scenario(
        mode="target_tracking",
        lift_to_drag_ratio=0.3,
        target=reference_target(),
    )

    result = simulate_reentry_local(scenario)

    assert result.target_miss is not None
    assert result.target_miss.distance_km < scenario.target.allowable_miss_km
    assert result.samples[-1].range_to_target_km < result.samples[0].range_to_target_km
    target_margin = next(
        margin
        for margin in result.margin_report.margins
        if margin.name == "target_miss_margin_km"
    )
    assert target_margin.status == "pass"


def test_none_atmosphere_has_zero_aerodynamic_and_thermal_loads() -> None:
    scenario = make_reentry_scenario(atmosphere=ReentryAtmosphereConfig(model="none"))

    result = simulate_reentry_local(scenario)

    assert all(sample.dynamic_pressure_pa == 0.0 for sample in result.samples)
    assert all(sample.deceleration_g == 0.0 for sample in result.samples)
    assert all(sample.convective_heat_rate_w_m2 == 0.0 for sample in result.samples)


def test_limit_exceedance_is_explicit_in_margin_report() -> None:
    scenario = make_reentry_scenario(
        limits=ReentryLimits(
            maximum_dynamic_pressure_pa=1.0,
            maximum_deceleration_g=0.1,
            maximum_heat_rate_w_m2=1.0,
            maximum_heat_load_j_m2=1.0,
        )
    )

    result = simulate_reentry_local(scenario)

    assert result.margin_report.limiting_margin.status == "fail"
    assert all(margin.status == "fail" for margin in result.margin_report.margins)


def test_heat_load_is_monotonic_and_internal_step_converges() -> None:
    baseline_scenario = make_reentry_scenario(
        propagation=ReentryPropagationConfig(
            duration_s=1000.0,
            step_s=5.0,
            internal_step_s=1.0,
        )
    )
    refined_scenario = baseline_scenario.model_copy(
        update={
            "propagation": ReentryPropagationConfig(
                duration_s=1000.0,
                step_s=5.0,
                internal_step_s=0.5,
            )
        }
    )

    baseline = simulate_reentry_local(baseline_scenario)
    refined = simulate_reentry_local(refined_scenario)

    heat_loads = [sample.heat_load_j_m2 for sample in refined.samples]
    assert all(
        previous <= current
        for previous, current in zip(heat_loads, heat_loads[1:], strict=False)
    )
    assert baseline.samples[-1].time_s == pytest.approx(
        refined.samples[-1].time_s,
        abs=0.01,
    )
    assert baseline.peaks.dynamic_pressure.value == pytest.approx(
        refined.peaks.dynamic_pressure.value,
        rel=1.0e-4,
    )
    assert baseline.peaks.total_heat_load_j_m2 == pytest.approx(
        refined.peaks.total_heat_load_j_m2,
        rel=1.0e-5,
    )
