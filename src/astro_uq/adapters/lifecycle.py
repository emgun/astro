from __future__ import annotations

from collections.abc import Callable

from astro_core.models import AstroModel
from astro_mission.models import (
    MissionLifecycleInputOverrides,
    MissionLifecycleResult,
    MissionLifecycleScenario,
)
from astro_uq.metrics import MetricExtractor, MetricRegistry
from astro_uq.models import MetricValue, MetricValueKind
from astro_uq.parameters import (
    ParameterBinding,
    ParameterBindingError,
    ParameterRegistry,
    ParameterValue,
)

WORKFLOW = "mission_lifecycle"


def _scenario(model: AstroModel) -> MissionLifecycleScenario:
    return MissionLifecycleScenario.model_validate(model)


def _result(model: AstroModel) -> MissionLifecycleResult:
    return MissionLifecycleResult.model_validate(model)


def _numeric(value: ParameterValue) -> float:
    if isinstance(value, str):
        raise ParameterBindingError("lifecycle parameter requires a numeric value")
    return float(value)


def _deorbit_field(field: str) -> Callable[[AstroModel], float]:
    return lambda model: float(getattr(_scenario(model).deorbit, field))


def _replace_deorbit_field(
    field: str,
) -> Callable[[AstroModel, ParameterValue], AstroModel]:
    def update(model: AstroModel, value: ParameterValue) -> AstroModel:
        scenario = _scenario(model)
        payload = scenario.deorbit.model_dump(mode="python")
        payload[field] = _numeric(value)
        deorbit = type(scenario.deorbit).model_validate(payload)
        return MissionLifecycleScenario.model_validate(
            scenario.model_copy(update={"deorbit": deorbit}).model_dump(mode="python")
        )

    return update


def _override_field(field: str) -> Callable[[AstroModel], float]:
    def get(model: AstroModel) -> float:
        overrides = _scenario(model).input_overrides
        value = None if overrides is None else getattr(overrides, field)
        if value is None:
            raise ParameterBindingError(f"lifecycle input override {field} is not resolved")
        return float(value)

    return get


def _replace_override_field(field: str) -> Callable[[AstroModel, ParameterValue], AstroModel]:
    def update(model: AstroModel, value: ParameterValue) -> AstroModel:
        scenario = _scenario(model)
        payload = (
            {}
            if scenario.input_overrides is None
            else scenario.input_overrides.model_dump(mode="python")
        )
        payload[field] = _numeric(value)
        overrides = MissionLifecycleInputOverrides.model_validate(payload)
        return MissionLifecycleScenario.model_validate(
            scenario.model_copy(update={"input_overrides": overrides}).model_dump(mode="python")
        )

    return update


def lifecycle_parameter_registry() -> ParameterRegistry:
    registry = ParameterRegistry()
    definitions = (
        ("delta_v_km_s", "km/s", 1.0e-12),
        ("specific_impulse_s", "s", 1.0e-12),
        ("entry_interface_altitude_km", "km", 1.0e-12),
    )
    for field, unit, lower in definitions:
        registry.register(
            ParameterBinding(
                target=f"lifecycle.deorbit.{field}",
                workflow=WORKFLOW,
                unit=unit,
                value_type=float,
                getter=_deorbit_field(field),
                updater=_replace_deorbit_field(field),
                lower=lower,
            )
        )
    override_definitions = (
        ("launch.upper_stage_thrust_n", "launch_upper_stage_thrust_n", "N", 1.0e-12, None),
        ("spacecraft.wet_mass_kg", "spacecraft_wet_mass_kg", "kg", 1.0e-12, None),
        (
            "digital_twin.power.solar_array_efficiency",
            "twin_solar_array_efficiency",
            "1",
            1.0e-12,
            1.0,
        ),
        (
            "reentry.atmosphere.density_scale_factor",
            "reentry_atmosphere_density_scale_factor",
            "1",
            1.0e-12,
            None,
        ),
        (
            "reentry.vehicle.drag_coefficient",
            "reentry_vehicle_drag_coefficient",
            "1",
            1.0e-12,
            10.0,
        ),
    )
    for target, field, unit, lower, upper in override_definitions:
        registry.register(
            ParameterBinding(
                target=f"lifecycle.{target}",
                workflow=WORKFLOW,
                unit=unit,
                value_type=float,
                getter=_override_field(field),
                updater=_replace_override_field(field),
                lower=lower,
                upper=upper,
            )
        )
    return registry


def _margin(phase: str, name: str) -> Callable[[AstroModel], float]:
    def extract(model: AstroModel) -> float:
        result = _result(model)
        matches = [
            margin
            for margin in result.margin_report.margins
            if margin.phase == phase and margin.name == name
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one lifecycle margin for {phase}:{name}")
        return float(matches[0].margin)

    return extract


def lifecycle_metric_registry() -> MetricRegistry:
    registry = MetricRegistry()

    def target_miss(model: AstroModel) -> float | None:
        miss = _result(model).reentry_result.target_miss
        return None if miss is None else float(miss.distance_km)

    def twin_limiting_margin(model: AstroModel) -> float:
        return float(_result(model).digital_twin.margin_report.limiting_margin.margin)

    definitions: tuple[
        tuple[
            str,
            MetricValueKind,
            str | None,
            Callable[[AstroModel], MetricValue],
        ],
        ...,
    ] = (
        ("lifecycle.passed", MetricValueKind.BOOLEAN, None, lambda model: _result(model).passed),
        (
            "lifecycle.all_phases_completed",
            MetricValueKind.BOOLEAN,
            None,
            lambda model: len(_result(model).manifest.entries) == 5,
        ),
        (
            "lifecycle.propellant_reserve_margin_kg",
            MetricValueKind.NUMERIC,
            "kg",
            _margin("deorbit", "propellant_reserve"),
        ),
        (
            "lifecycle.entry_interface_margin_km",
            MetricValueKind.NUMERIC,
            "km",
            _margin("deorbit", "entry_interface_altitude_error"),
        ),
        (
            "lifecycle.twin_limiting_margin",
            MetricValueKind.NUMERIC,
            "native",
            twin_limiting_margin,
        ),
        (
            "lifecycle.twin_limiting_status",
            MetricValueKind.CATEGORY,
            None,
            lambda model: _result(model).digital_twin.margin_report.limiting_margin.status.value,
        ),
        (
            "lifecycle.deorbit_propellant_used_kg",
            MetricValueKind.NUMERIC,
            "kg",
            lambda model: float(_result(model).metadata["propellant_used_kg"]),
        ),
        (
            "lifecycle.reentry_peak_dynamic_pressure_pa",
            MetricValueKind.NUMERIC,
            "Pa",
            lambda model: float(_result(model).reentry_result.peaks.dynamic_pressure.value),
        ),
        (
            "lifecycle.reentry_peak_deceleration_g",
            MetricValueKind.NUMERIC,
            "g",
            lambda model: float(_result(model).reentry_result.peaks.deceleration.value),
        ),
        (
            "lifecycle.reentry_peak_heat_rate_w_m2",
            MetricValueKind.NUMERIC,
            "W/m^2",
            lambda model: float(_result(model).reentry_result.peaks.heat_rate.value),
        ),
        (
            "lifecycle.reentry_total_heat_load_j_m2",
            MetricValueKind.NUMERIC,
            "J/m^2",
            lambda model: float(_result(model).reentry_result.peaks.total_heat_load_j_m2),
        ),
        (
            "lifecycle.reentry_target_miss_km",
            MetricValueKind.NUMERIC,
            "km",
            target_miss,
        ),
    )
    for extractor_id, kind, unit, extract in definitions:
        registry.register(
            MetricExtractor(
                extractor_id=extractor_id,
                workflow=WORKFLOW,
                value_kind=kind,
                unit=unit,
                extract=extract,
            )
        )
    return registry
