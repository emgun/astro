from __future__ import annotations

from datetime import UTC, datetime

import pytest

from astro_reentry.backends import simulate_reentry_with_backend
from astro_reentry.models import (
    AerothermalConfig,
    BankSchedulePoint,
    ReentryAtmosphereConfig,
    ReentryGuidanceConfig,
    ReentryGuidanceMode,
    ReentryInitialState,
    ReentryLimits,
    ReentryPropagationConfig,
    ReentryScenario,
    ReentryTarget,
    ReentryVehicle,
)
from astro_uq.adapters.reentry import reentry_metric_registry, reentry_parameter_registry
from astro_uq.models import (
    DistributionKind,
    DistributionSpec,
    MetricSpec,
    MetricValueKind,
    ParameterRealization,
    UncertainParameter,
    UncertaintyKind,
    UncertaintyModel,
)
from astro_uq.parameters import ParameterBindingError


def _scenario(*, mode: ReentryGuidanceMode = "ballistic", **overrides: object) -> ReentryScenario:
    lifting = mode != "ballistic"
    if mode in {"bank_schedule", "target_tracking"}:
        guidance = ReentryGuidanceConfig(
            mode=mode,
            bank_schedule=(
                BankSchedulePoint(velocity_km_s=7.8, bank_angle_deg=45.0),
                BankSchedulePoint(velocity_km_s=0.5, bank_angle_deg=20.0),
            ),
        )
    else:
        guidance = ReentryGuidanceConfig(
            mode=mode, bank_angle_deg=45.0 if mode == "constant_bank" else 0.0
        )
    payload: dict[str, object] = {
        "scenario_id": f"uq-{mode}",
        "initial_state": ReentryInitialState(
            epoch=datetime(2026, 1, 1, tzinfo=UTC),
            altitude_km=120.0,
            velocity_km_s=7.8,
            flight_path_angle_deg=-6.0,
            heading_deg=90.0,
            latitude_deg=0.0,
            longitude_deg=0.0,
        ),
        "vehicle": ReentryVehicle(
            name="test-vehicle",
            mass_kg=5500.0,
            reference_area_m2=12.0,
            drag_coefficient=1.5,
            lift_to_drag_ratio=0.3 if lifting else 0.0,
            nose_radius_m=0.75,
        ),
        "guidance": guidance,
        "target": ReentryTarget(
            name="target", latitude_deg=1.0, longitude_deg=10.0, allowable_miss_km=25.0
        )
        if mode == "target_tracking"
        else None,
        "limits": ReentryLimits(
            maximum_dynamic_pressure_pa=60000.0,
            maximum_deceleration_g=20.0,
            maximum_heat_rate_w_m2=2500000.0,
            maximum_heat_load_j_m2=150000000.0,
        ),
        "propagation": ReentryPropagationConfig(duration_s=1000.0, step_s=5.0, internal_step_s=1.0),
    }
    payload.update(overrides)
    return ReentryScenario.model_validate(payload)


def _apply(target: str, unit: str, value: float, *, scenario: ReentryScenario) -> ReentryScenario:
    uncertainty = UncertaintyModel(
        parameters=(
            UncertainParameter(
                parameter_id="value",
                target=target,
                unit=unit,
                uncertainty_kind=UncertaintyKind.EPISTEMIC,
                distribution=DistributionSpec(kind=DistributionKind.CONSTANT, value=value),
            ),
        )
    )
    realization = ParameterRealization(
        sample_id="sample-0",
        sample_index=0,
        normalized_values={"value": 0.5},
        physical_values={"value": value},
    )
    resolved = reentry_parameter_registry().apply(
        workflow="reentry",
        scenario=scenario,
        uncertainty=uncertainty,
        realization=realization,
    )[0]
    return ReentryScenario.model_validate(resolved)


@pytest.mark.parametrize(
    ("target", "unit", "value", "section", "field"),
    [
        ("reentry.atmosphere.density_scale_factor", "1", 1.2, "atmosphere", "density_scale_factor"),
        ("reentry.vehicle.drag_coefficient", "1", 1.7, "vehicle", "drag_coefficient"),
        ("reentry.vehicle.lift_to_drag_ratio", "1", 0.4, "vehicle", "lift_to_drag_ratio"),
        ("reentry.vehicle.nose_radius_m", "m", 0.9, "vehicle", "nose_radius_m"),
        ("reentry.guidance.bank_angle_deg", "deg", 35.0, "guidance", "bank_angle_deg"),
        (
            "reentry.initial_state.flight_path_angle_deg",
            "deg",
            -5.5,
            "initial_state",
            "flight_path_angle_deg",
        ),
    ],
)
def test_applicable_bindings_update_real_fields_and_revalidate(
    target: str, unit: str, value: float, section: str, field: str
) -> None:
    scenario = _scenario(mode="constant_bank")

    resolved = _apply(target, unit, value, scenario=scenario)

    assert getattr(getattr(resolved, section), field) == value
    assert resolved is not scenario


@pytest.mark.parametrize(
    ("target", "unit", "scenario"),
    [
        (
            "reentry.atmosphere.density_scale_factor",
            "1",
            _scenario(atmosphere=ReentryAtmosphereConfig(model="none")),
        ),
        (
            "reentry.vehicle.lift_to_drag_ratio",
            "1",
            _scenario(mode="ballistic"),
        ),
        (
            "reentry.vehicle.nose_radius_m",
            "m",
            _scenario(aerothermal=AerothermalConfig(model="none")),
        ),
        (
            "reentry.guidance.bank_angle_deg",
            "deg",
            _scenario(mode="bank_schedule"),
        ),
    ],
)
def test_model_inapplicable_bindings_are_rejected(
    target: str, unit: str, scenario: ReentryScenario
) -> None:
    with pytest.raises(ParameterBindingError, match="not applicable"):
        _apply(target, unit, 1.0, scenario=scenario)


def test_binding_revalidates_the_complete_scenario() -> None:
    scenario = _scenario(mode="constant_bank")

    with pytest.raises(ParameterBindingError, match="resolved reentry scenario is invalid"):
        _apply("reentry.vehicle.lift_to_drag_ratio", "1", 0.0, scenario=scenario)


@pytest.mark.parametrize("with_target", [False, True])
def test_metric_registry_extracts_actual_reentry_result(with_target: bool) -> None:
    scenario = _scenario(mode="target_tracking" if with_target else "ballistic")
    result = simulate_reentry_with_backend(scenario, "local")
    definitions = (
        ("peak_dynamic_pressure_pa", "Pa"),
        ("peak_deceleration_g", "g"),
        ("peak_heat_rate_w_m2", "W/m^2"),
        ("total_heat_load_j_m2", "J/m^2"),
        ("target_miss_km", "km"),
        ("terminal_time_s", "s"),
        ("terminal_altitude_km", "km"),
        ("event_count", "1"),
    )
    specifications = tuple(
        MetricSpec(
            metric_id=name,
            extractor=f"reentry.{name}",
            value_kind=MetricValueKind.NUMERIC,
            unit=unit,
        )
        for name, unit in definitions
    )

    values = reentry_metric_registry().extract(
        workflow="reentry", result=result, specifications=specifications
    )

    assert values["peak_dynamic_pressure_pa"] == result.peaks.dynamic_pressure.value
    assert values["peak_deceleration_g"] == result.peaks.deceleration.value
    assert values["peak_heat_rate_w_m2"] == result.peaks.heat_rate.value
    assert values["total_heat_load_j_m2"] == result.peaks.total_heat_load_j_m2
    assert values["target_miss_km"] == (
        result.target_miss.distance_km if result.target_miss is not None else None
    )
    assert values["terminal_time_s"] == result.samples[-1].time_s
    assert values["terminal_altitude_km"] == result.samples[-1].altitude_km
    assert values["event_count"] == float(len(result.events))
