from __future__ import annotations

from collections.abc import Callable
from math import sqrt

from pydantic import ValidationError

from astro_core.constants import R_EARTH_KM
from astro_core.models import AstroModel, CartesianState, ForceModelName, Scenario, Trajectory
from astro_uq.metrics import MetricExtractor, MetricRegistry
from astro_uq.models import MetricValueKind
from astro_uq.parameters import (
    ParameterBinding,
    ParameterBindingError,
    ParameterRegistry,
    ParameterValue,
)

WORKFLOW = "orbit"

_CARTESIAN_BINDINGS = (
    ("orbit.initial_state.cartesian.position_x_km", "position_km", 0, "km"),
    ("orbit.initial_state.cartesian.position_y_km", "position_km", 1, "km"),
    ("orbit.initial_state.cartesian.position_z_km", "position_km", 2, "km"),
    ("orbit.initial_state.cartesian.velocity_x_km_s", "velocity_km_s", 0, "km/s"),
    ("orbit.initial_state.cartesian.velocity_y_km_s", "velocity_km_s", 1, "km/s"),
    ("orbit.initial_state.cartesian.velocity_z_km_s", "velocity_km_s", 2, "km/s"),
)

_FINAL_CARTESIAN_METRICS = (
    ("orbit.final_position_x_km", "position_km", 0, "km"),
    ("orbit.final_position_y_km", "position_km", 1, "km"),
    ("orbit.final_position_z_km", "position_km", 2, "km"),
    ("orbit.final_velocity_x_km_s", "velocity_km_s", 0, "km/s"),
    ("orbit.final_velocity_y_km_s", "velocity_km_s", 1, "km/s"),
    ("orbit.final_velocity_z_km_s", "velocity_km_s", 2, "km/s"),
)


def _scenario(model: AstroModel) -> Scenario:
    return Scenario.model_validate(model)


def _trajectory(model: AstroModel) -> Trajectory:
    return Trajectory.model_validate(model)


def _numeric(value: ParameterValue) -> float:
    if isinstance(value, str):
        raise ParameterBindingError("orbit parameter requires a numeric value")
    return float(value)


def _replace_cartesian_component(
    field: str, index: int
) -> Callable[[AstroModel, ParameterValue], AstroModel]:
    def update(model: AstroModel, value: ParameterValue) -> AstroModel:
        scenario = _scenario(model)
        cartesian_payload = scenario.initial_state.cartesian.model_dump(mode="python")
        components = list(cartesian_payload[field])
        components[index] = _numeric(value)
        cartesian_payload[field] = tuple(components)
        cartesian = CartesianState.model_validate(cartesian_payload)
        initial_state = scenario.initial_state.model_copy(update={"cartesian": cartesian})
        return Scenario.model_validate(
            scenario.model_copy(update={"initial_state": initial_state}).model_dump(mode="python")
        )

    return update


def _cartesian_component(field: str, index: int) -> Callable[[AstroModel], float]:
    def get(model: AstroModel) -> float:
        cartesian = _scenario(model).initial_state.cartesian
        return float(getattr(cartesian, field)[index])

    return get


def _replace_spacecraft_field(
    field: str, *, applicable: Callable[[Scenario], bool] | None = None
) -> Callable[[AstroModel, ParameterValue], AstroModel]:
    def update(model: AstroModel, value: ParameterValue) -> AstroModel:
        scenario = _scenario(model)
        if applicable is not None and not applicable(scenario):
            raise ParameterBindingError(
                f"orbit.spacecraft.{field} is not applicable to the configured force model"
            )
        spacecraft_payload = scenario.spacecraft.model_dump(mode="python")
        spacecraft_payload[field] = _numeric(value)
        try:
            spacecraft = type(scenario.spacecraft).model_validate(spacecraft_payload)
        except ValidationError as exc:
            raise ParameterBindingError(f"resolved spacecraft is invalid: {exc}") from exc
        return Scenario.model_validate(
            scenario.model_copy(update={"spacecraft": spacecraft}).model_dump(mode="python")
        )

    return update


def _spacecraft_field(field: str) -> Callable[[AstroModel], float]:
    def get(model: AstroModel) -> float:
        return float(getattr(_scenario(model).spacecraft, field))

    return get


def register_orbit_parameters(registry: ParameterRegistry) -> None:
    for target, field, index, unit in _CARTESIAN_BINDINGS:
        registry.register(
            ParameterBinding(
                target=target,
                workflow=WORKFLOW,
                unit=unit,
                value_type=float,
                getter=_cartesian_component(field, index),
                updater=_replace_cartesian_component(field, index),
            )
        )

    def drag_enabled(scenario: Scenario) -> bool:
        return scenario.force_model.atmospheric_drag

    def srp_enabled(scenario: Scenario) -> bool:
        return scenario.force_model.solar_radiation_pressure

    def area_enabled(scenario: Scenario) -> bool:
        return drag_enabled(scenario) or srp_enabled(scenario)

    spacecraft_bindings = (
        ("mass_kg", "kg", 0.0, None, None),
        ("area_m2", "m^2", 0.0, None, area_enabled),
        ("drag_coefficient", "1", 0.0, 10.0, drag_enabled),
        ("reflectivity_coefficient", "1", 0.0, 5.0, srp_enabled),
    )
    for field, unit, lower, upper, applicable in spacecraft_bindings:
        registry.register(
            ParameterBinding(
                target=f"orbit.spacecraft.{field}",
                workflow=WORKFLOW,
                unit=unit,
                value_type=float,
                getter=_spacecraft_field(field),
                updater=_replace_spacecraft_field(field, applicable=applicable),
                lower=lower,
                upper=upper,
            )
        )

    def replace_gravity(model: AstroModel, value: ParameterValue) -> AstroModel:
        if not isinstance(value, str):
            raise ParameterBindingError("orbit gravity variant requires a string value")
        scenario = _scenario(model)
        force_model = scenario.force_model.model_copy(
            update={"gravity": ForceModelName(value)}
        )
        return Scenario.model_validate(
            scenario.model_copy(update={"force_model": force_model}).model_dump(mode="python")
        )

    registry.register(
        ParameterBinding(
            target="orbit.force_model.gravity",
            workflow=WORKFLOW,
            unit="1",
            value_type=str,
            getter=lambda model: _scenario(model).force_model.gravity.value,
            updater=replace_gravity,
        )
    )


def orbit_parameter_registry() -> ParameterRegistry:
    registry = ParameterRegistry()
    register_orbit_parameters(registry)
    return registry


def _final_component(field: str, index: int) -> Callable[[AstroModel], float]:
    def extract(model: AstroModel) -> float:
        return float(getattr(_trajectory(model).samples[-1].state, field)[index])

    return extract


def _final_radius_km(model: AstroModel) -> float:
    position = _trajectory(model).samples[-1].state.position_km
    return sqrt(sum(float(component) ** 2 for component in position))


def _duration_s(model: AstroModel) -> float:
    trajectory = _trajectory(model)
    return (trajectory.samples[-1].epoch - trajectory.samples[0].epoch).total_seconds()


def register_orbit_metrics(registry: MetricRegistry) -> None:
    for extractor_id, field, index, unit in _FINAL_CARTESIAN_METRICS:
        registry.register(
            MetricExtractor(
                extractor_id=extractor_id,
                workflow=WORKFLOW,
                value_kind=MetricValueKind.NUMERIC,
                unit=unit,
                extract=_final_component(field, index),
            )
        )

    scalar_metrics = (
        ("orbit.final_radius_km", "km", _final_radius_km),
        ("orbit.final_altitude_km", "km", lambda model: _final_radius_km(model) - R_EARTH_KM),
        ("orbit.duration_s", "s", _duration_s),
        ("orbit.event_count", "1", lambda model: float(len(_trajectory(model).events))),
    )
    for extractor_id, unit, extract in scalar_metrics:
        registry.register(
            MetricExtractor(
                extractor_id=extractor_id,
                workflow=WORKFLOW,
                value_kind=MetricValueKind.NUMERIC,
                unit=unit,
                extract=extract,
            )
        )


def orbit_metric_registry() -> MetricRegistry:
    registry = MetricRegistry()
    register_orbit_metrics(registry)
    return registry
