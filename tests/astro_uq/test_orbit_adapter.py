from __future__ import annotations

from datetime import timedelta
from math import sqrt

import pytest

from astro_core.io import load_scenario
from astro_core.models import ForceModelName, Trajectory, TrajectoryEvent, TrajectorySample
from astro_dynamics.backends import propagate_with_backend
from astro_uq.adapters.orbit import orbit_metric_registry, orbit_parameter_registry
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


def _apply(target: str, unit: str, value: float, *, scenario=None):
    base = scenario or load_scenario("examples/scenarios/leo_two_body.yaml")
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
    return orbit_parameter_registry().apply(
        workflow="orbit",
        scenario=base,
        uncertainty=uncertainty,
        realization=realization,
    )[0]


@pytest.mark.parametrize(
    ("target", "unit", "field", "index", "value"),
    [
        ("orbit.initial_state.cartesian.position_x_km", "km", "position_km", 0, 7001.0),
        ("orbit.initial_state.cartesian.position_y_km", "km", "position_km", 1, 2.0),
        ("orbit.initial_state.cartesian.position_z_km", "km", "position_km", 2, 3.0),
        ("orbit.initial_state.cartesian.velocity_x_km_s", "km/s", "velocity_km_s", 0, 0.1),
        ("orbit.initial_state.cartesian.velocity_y_km_s", "km/s", "velocity_km_s", 1, 7.4),
        ("orbit.initial_state.cartesian.velocity_z_km_s", "km/s", "velocity_km_s", 2, 0.2),
    ],
)
def test_cartesian_bindings_update_one_component_and_revalidate(
    target: str, unit: str, field: str, index: int, value: float
) -> None:
    base = load_scenario("examples/scenarios/leo_two_body.yaml")

    resolved = _apply(target, unit, value, scenario=base)

    assert getattr(resolved.initial_state.cartesian, field)[index] == value
    assert resolved is not base


@pytest.mark.parametrize(
    ("target", "unit", "value", "field"),
    [
        ("orbit.spacecraft.mass_kg", "kg", 900.0, "mass_kg"),
        ("orbit.spacecraft.area_m2", "m^2", 8.0, "area_m2"),
        ("orbit.spacecraft.drag_coefficient", "1", 2.3, "drag_coefficient"),
        ("orbit.spacecraft.reflectivity_coefficient", "1", 1.4, "reflectivity_coefficient"),
    ],
)
def test_spacecraft_bindings_apply_when_relevant(
    target: str, unit: str, value: float, field: str
) -> None:
    scenario = load_scenario("examples/scenarios/leo_two_body.yaml")
    force_model = scenario.force_model.model_copy(
        update={"atmospheric_drag": True, "solar_radiation_pressure": True}
    )
    scenario = scenario.model_copy(update={"force_model": force_model})

    resolved = _apply(target, unit, value, scenario=scenario)

    assert getattr(resolved.spacecraft, field) == value


@pytest.mark.parametrize(
    ("target", "unit"),
    [
        ("orbit.spacecraft.area_m2", "m^2"),
        ("orbit.spacecraft.drag_coefficient", "1"),
        ("orbit.spacecraft.reflectivity_coefficient", "1"),
    ],
)
def test_inactive_force_model_bindings_are_rejected(target: str, unit: str) -> None:
    with pytest.raises(ParameterBindingError, match="not applicable"):
        _apply(target, unit, 1.0)


def test_spacecraft_binding_revalidates_nested_model() -> None:
    with pytest.raises(ParameterBindingError, match="greater than 0"):
        _apply("orbit.spacecraft.mass_kg", "kg", 0.0)


def test_model_variant_binding_changes_gravity_model() -> None:
    scenario = load_scenario("examples/scenarios/leo_two_body.yaml")
    resolved, evidence = orbit_parameter_registry().apply(
        workflow="orbit",
        scenario=scenario,
        uncertainty=UncertaintyModel(
            model_variants=(
                {
                    "variant_id": "j2",
                    "target": "orbit.force_model.gravity",
                    "value": "j2",
                },
            )
        ),
        realization=ParameterRealization(
            sample_id="sample-0",
            sample_index=0,
            model_variants={"orbit.force_model.gravity": "j2"},
        ),
    )

    assert resolved.force_model.gravity is ForceModelName.J2  # type: ignore[attr-defined]
    assert evidence.bindings[0].target == "orbit.force_model.gravity"


def test_metric_registry_extracts_final_trajectory_metrics() -> None:
    trajectory = propagate_with_backend(
        load_scenario("examples/scenarios/leo_two_body.yaml"), "local"
    )
    event = TrajectoryEvent(
        event_type="fixture",
        epoch=trajectory.samples[-1].epoch,
        metadata={},
    )
    trajectory = trajectory.model_copy(update={"events": [event]})
    specifications = tuple(
        MetricSpec(
            metric_id=extractor.removeprefix("orbit."),
            extractor=extractor,
            value_kind=MetricValueKind.NUMERIC,
            unit=unit,
        )
        for extractor, unit in (
            ("orbit.final_position_x_km", "km"),
            ("orbit.final_position_y_km", "km"),
            ("orbit.final_position_z_km", "km"),
            ("orbit.final_velocity_x_km_s", "km/s"),
            ("orbit.final_velocity_y_km_s", "km/s"),
            ("orbit.final_velocity_z_km_s", "km/s"),
            ("orbit.final_radius_km", "km"),
            ("orbit.final_altitude_km", "km"),
            ("orbit.duration_s", "s"),
            ("orbit.event_count", "1"),
        )
    )

    values = orbit_metric_registry().extract(
        workflow="orbit", result=trajectory, specifications=specifications
    )

    final = trajectory.samples[-1].state
    assert values["final_position_x_km"] == final.position_km[0]
    assert values["final_velocity_z_km_s"] == final.velocity_km_s[2]
    expected_radius = sqrt(sum(component**2 for component in final.position_km))
    assert values["final_radius_km"] == pytest.approx(expected_radius)
    assert values["final_altitude_km"] == pytest.approx(expected_radius - 6378.1363)
    assert values["duration_s"] == pytest.approx(
        (trajectory.samples[-1].epoch - trajectory.samples[0].epoch).total_seconds()
    )
    assert values["event_count"] == 1.0


def test_duration_comes_from_trajectory_epochs() -> None:
    scenario = load_scenario("examples/scenarios/leo_two_body.yaml")
    sample = TrajectorySample(
        epoch=scenario.initial_state.epoch,
        state=scenario.initial_state.cartesian,
    )
    trajectory = Trajectory(
        scenario_id=scenario.scenario_id,
        samples=[
            sample,
            sample.model_copy(update={"epoch": sample.epoch + timedelta(seconds=12.5)}),
        ],
        force_model=scenario.force_model,
        backend="fixture",
    )
    specification = MetricSpec(
        metric_id="duration",
        extractor="orbit.duration_s",
        value_kind=MetricValueKind.NUMERIC,
        unit="s",
    )

    values = orbit_metric_registry().extract(
        workflow="orbit", result=trajectory, specifications=(specification,)
    )

    assert values == {"duration": 12.5}
