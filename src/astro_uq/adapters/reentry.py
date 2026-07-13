from __future__ import annotations

from collections.abc import Callable

from pydantic import ValidationError

from astro_core.models import AstroModel
from astro_reentry.models import ReentryResult, ReentryScenario
from astro_uq.metrics import MetricExtractor, MetricRegistry
from astro_uq.models import MetricValue, MetricValueKind
from astro_uq.parameters import (
    ParameterBinding,
    ParameterBindingError,
    ParameterRegistry,
    ParameterValue,
)

WORKFLOW = "reentry"


def _scenario(model: AstroModel) -> ReentryScenario:
    return ReentryScenario.model_validate(model)


def _result(model: AstroModel) -> ReentryResult:
    return ReentryResult.model_validate(model)


def _numeric(value: ParameterValue) -> float:
    if isinstance(value, str):
        raise ParameterBindingError("reentry parameter requires a numeric value")
    return float(value)


def _nested_field(section: str, field: str) -> Callable[[AstroModel], float]:
    def get(model: AstroModel) -> float:
        return float(getattr(getattr(_scenario(model), section), field))

    return get


def _replace_nested_field(
    section: str,
    field: str,
    *,
    applicable: Callable[[ReentryScenario], bool] | None = None,
) -> Callable[[AstroModel, ParameterValue], AstroModel]:
    target = f"reentry.{section}.{field}"

    def update(model: AstroModel, value: ParameterValue) -> AstroModel:
        scenario = _scenario(model)
        if applicable is not None and not applicable(scenario):
            raise ParameterBindingError(f"{target} is not applicable to the configured model")
        section_model = getattr(scenario, section)
        section_payload = section_model.model_dump(mode="python")
        section_payload[field] = _numeric(value)
        scenario_payload = scenario.model_dump(mode="python")
        scenario_payload[section] = section_payload
        try:
            return ReentryScenario.model_validate(scenario_payload)
        except ValidationError as exc:
            raise ParameterBindingError(f"resolved reentry scenario is invalid: {exc}") from exc

    return update


def register_reentry_parameters(registry: ParameterRegistry) -> None:
    bindings = (
        (
            "atmosphere",
            "density_scale_factor",
            "1",
            0.0,
            None,
            lambda scenario: scenario.atmosphere.model == "exponential",
        ),
        ("vehicle", "drag_coefficient", "1", 0.0, 10.0, None),
        (
            "vehicle",
            "lift_to_drag_ratio",
            "1",
            0.0,
            3.0,
            lambda scenario: scenario.guidance.mode != "ballistic",
        ),
        (
            "vehicle",
            "nose_radius_m",
            "m",
            0.0,
            None,
            lambda scenario: scenario.aerothermal.model == "sutton_graves",
        ),
        (
            "guidance",
            "bank_angle_deg",
            "deg",
            -90.0,
            90.0,
            lambda scenario: scenario.guidance.mode == "constant_bank",
        ),
        ("initial_state", "flight_path_angle_deg", "deg", -90.0, 90.0, None),
    )
    for section, field, unit, lower, upper, applicable in bindings:
        registry.register(
            ParameterBinding(
                target=f"reentry.{section}.{field}",
                workflow=WORKFLOW,
                unit=unit,
                value_type=float,
                getter=_nested_field(section, field),
                updater=_replace_nested_field(section, field, applicable=applicable),
                lower=lower,
                upper=upper,
            )
        )


def reentry_parameter_registry() -> ParameterRegistry:
    registry = ParameterRegistry()
    register_reentry_parameters(registry)
    return registry


def _target_miss_km(model: AstroModel) -> MetricValue:
    target_miss = _result(model).target_miss
    return None if target_miss is None else float(target_miss.distance_km)


def register_reentry_metrics(registry: MetricRegistry) -> None:
    metrics: tuple[tuple[str, str, Callable[[AstroModel], MetricValue]], ...] = (
        (
            "reentry.peak_dynamic_pressure_pa",
            "Pa",
            lambda model: float(_result(model).peaks.dynamic_pressure.value),
        ),
        (
            "reentry.peak_deceleration_g",
            "g",
            lambda model: float(_result(model).peaks.deceleration.value),
        ),
        (
            "reentry.peak_heat_rate_w_m2",
            "W/m^2",
            lambda model: float(_result(model).peaks.heat_rate.value),
        ),
        (
            "reentry.total_heat_load_j_m2",
            "J/m^2",
            lambda model: float(_result(model).peaks.total_heat_load_j_m2),
        ),
        ("reentry.target_miss_km", "km", _target_miss_km),
        ("reentry.terminal_time_s", "s", lambda model: float(_result(model).samples[-1].time_s)),
        (
            "reentry.terminal_altitude_km",
            "km",
            lambda model: float(_result(model).samples[-1].altitude_km),
        ),
        ("reentry.event_count", "1", lambda model: float(len(_result(model).events))),
    )
    for extractor_id, unit, extract in metrics:
        registry.register(
            MetricExtractor(
                extractor_id=extractor_id,
                workflow=WORKFLOW,
                value_kind=MetricValueKind.NUMERIC,
                unit=unit,
                extract=extract,
            )
        )


def reentry_metric_registry() -> MetricRegistry:
    registry = MetricRegistry()
    register_reentry_metrics(registry)
    return registry
