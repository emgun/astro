from __future__ import annotations

from collections.abc import Callable

from astro_assurance.models import (
    MissionAssuranceCase,
    MissionAssuranceInputOverrides,
    PostLaunchAssuranceScenario,
)
from astro_core.models import AstroModel
from astro_twin.models import DigitalTwinScenario
from astro_uq.metrics import MetricError, MetricExtractor, MetricRegistry
from astro_uq.models import MetricValue, MetricValueKind
from astro_uq.parameters import (
    ParameterBinding,
    ParameterBindingError,
    ParameterRegistry,
    ParameterValue,
)

WORKFLOW = "mission_assurance"


def _scenario(model: AstroModel) -> PostLaunchAssuranceScenario:
    return PostLaunchAssuranceScenario.model_validate(model)


def _result(model: AstroModel) -> MissionAssuranceCase:
    return MissionAssuranceCase.model_validate(model)


def _numeric(value: ParameterValue) -> float:
    if isinstance(value, str):
        raise ParameterBindingError("mission assurance parameter requires a numeric value")
    return float(value)


def _dispersion_component(kind: str, index: int) -> Callable[[AstroModel], float]:
    field = f"{kind}_delta_km" if kind == "position" else f"{kind}_delta_km_s"
    return lambda model: float(getattr(_scenario(model).dispersion, field)[index])


def _replace_dispersion_component(
    kind: str, index: int
) -> Callable[[AstroModel, ParameterValue], AstroModel]:
    field = f"{kind}_delta_km" if kind == "position" else f"{kind}_delta_km_s"

    def update(model: AstroModel, value: ParameterValue) -> AstroModel:
        scenario = _scenario(model)
        payload = scenario.dispersion.model_dump(mode="python")
        vector = list(payload[field])
        vector[index] = _numeric(value)
        payload[field] = tuple(vector)
        dispersion = type(scenario.dispersion).model_validate(payload)
        return scenario.model_copy(update={"dispersion": dispersion})

    return update


def _override_field(field: str) -> Callable[[AstroModel], float]:
    def get(model: AstroModel) -> float:
        overrides = _scenario(model).input_overrides
        value = None if overrides is None else getattr(overrides, field)
        if value is None:
            raise ParameterBindingError(f"mission assurance input override {field} is not resolved")
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
        overrides = MissionAssuranceInputOverrides.model_validate(payload)
        return scenario.model_copy(update={"input_overrides": overrides})

    return update


def _thermal_override_field(name: str, field: str) -> Callable[[AstroModel], float]:
    def get(model: AstroModel) -> float:
        overrides = _scenario(model).input_overrides
        matches = (
            []
            if overrides is None
            else [
                override
                for override in overrides.twin_thermal_node_overrides
                if override.node_name == name
            ]
        )
        if len(matches) != 1 or getattr(matches[0], field) is None:
            raise ParameterBindingError(
                f"mission assurance thermal override {name}.{field} is not resolved"
            )
        return float(getattr(matches[0], field))

    return get


def _replace_thermal_override_field(
    name: str, field: str
) -> Callable[[AstroModel, ParameterValue], AstroModel]:
    def update(model: AstroModel, value: ParameterValue) -> AstroModel:
        scenario = _scenario(model)
        payload = (
            {}
            if scenario.input_overrides is None
            else scenario.input_overrides.model_dump(mode="python")
        )
        thermal = list(payload.get("twin_thermal_node_overrides", ()))
        positions = {str(item["node_name"]): index for index, item in enumerate(thermal)}
        if name in positions:
            thermal[positions[name]][field] = _numeric(value)
        else:
            thermal.append({"node_name": name, field: _numeric(value)})
        payload["twin_thermal_node_overrides"] = tuple(thermal)
        overrides = MissionAssuranceInputOverrides.model_validate(payload)
        return scenario.model_copy(update={"input_overrides": overrides})

    return update


def assurance_parameter_registry(
    twin_scenario: DigitalTwinScenario | None = None,
) -> ParameterRegistry:
    registry = ParameterRegistry()
    for kind, unit in (("position", "km"), ("velocity", "km/s")):
        for index, axis in enumerate("xyz"):
            registry.register(
                ParameterBinding(
                    target=f"mission_assurance.insertion.{kind}_{axis}",
                    workflow=WORKFLOW,
                    unit=unit,
                    value_type=float,
                    getter=_dispersion_component(kind, index),
                    updater=_replace_dispersion_component(kind, index),
                )
            )
    definitions = (
        ("tracking.range_sigma_km", "tracking_range_sigma_km", "km", 1.0e-12, None),
        (
            "tracking.range_rate_sigma_km_s",
            "tracking_range_rate_sigma_km_s",
            "km/s",
            1.0e-12,
            None,
        ),
        ("correction.execution_scale", "correction_execution_scale", "1", 0.0, 2.0),
        (
            "digital_twin.power.solar_array_efficiency",
            "twin_solar_array_efficiency",
            "1",
            1.0e-12,
            1.0,
        ),
        (
            "digital_twin.power.battery_capacity_wh",
            "twin_battery_capacity_wh",
            "Wh",
            1.0e-12,
            None,
        ),
    )
    for target, field, unit, lower, upper in definitions:
        registry.register(
            ParameterBinding(
                target=f"mission_assurance.{target}",
                workflow=WORKFLOW,
                unit=unit,
                value_type=float,
                getter=_override_field(field),
                updater=_replace_override_field(field),
                lower=lower,
                upper=upper,
            )
        )
    names = [] if twin_scenario is None else [node.name for node in twin_scenario.thermal_nodes]
    if len(set(names)) != len(names):
        raise ParameterBindingError("digital twin thermal node names must be unique")
    for name in names:
        for field in ("emissivity", "internal_heat_fraction"):
            registry.register(
                ParameterBinding(
                    target=f"mission_assurance.digital_twin.thermal_nodes.{name}.{field}",
                    workflow=WORKFLOW,
                    unit="1",
                    value_type=float,
                    getter=_thermal_override_field(name, field),
                    updater=_replace_thermal_override_field(name, field),
                    lower=1.0e-12 if field == "emissivity" else 0.0,
                    upper=1.0,
                )
            )
    return registry


def _metadata(key: str) -> Callable[[AstroModel], float]:
    return lambda model: float(_result(model).metadata[key])


def _assurance_margin_value(name: str) -> Callable[[AstroModel], float]:
    def extract(model: AstroModel) -> float:
        matches = [margin for margin in _result(model).margin_report.margins if margin.name == name]
        if len(matches) != 1:
            raise MetricError(f"expected one mission assurance margin: {name}")
        return float(matches[0].value)

    return extract


def _twin_margin(name: str) -> Callable[[AstroModel], float]:
    def extract(model: AstroModel) -> float:
        matches = [
            margin
            for margin in _result(model).corrected_digital_twin.margin_report.margins
            if margin.name == name
        ]
        if len(matches) != 1:
            raise MetricError(f"expected one mission assurance twin margin: {name}")
        return float(matches[0].margin)

    return extract


def _minimum_thermal_margin(model: AstroModel) -> float:
    margins = [
        margin
        for margin in _result(model).corrected_digital_twin.margin_report.margins
        if margin.name.startswith("thermal_")
        and (margin.name.endswith("_cold_margin_k") or margin.name.endswith("_hot_margin_k"))
    ]
    if not margins:
        raise MetricError("missing mission assurance twin thermal margins")
    return min(float(margin.margin) for margin in margins)


def _propellant_reserve(model: AstroModel) -> float:
    mass_budget = _result(model).corrected_digital_twin.mass_budget
    if mass_budget is None:
        raise MetricError("mission assurance twin result is missing a mass budget")
    return float(mass_budget.propellant_mass_kg)


def assurance_metric_registry() -> MetricRegistry:
    registry = MetricRegistry()
    definitions: tuple[
        tuple[str, MetricValueKind, str | None, Callable[[AstroModel], MetricValue]], ...
    ] = (
        ("mission_assurance.passed", MetricValueKind.BOOLEAN, None, lambda m: _result(m).passed),
        (
            "mission_assurance.od_position_error_km",
            MetricValueKind.NUMERIC,
            "km",
            _assurance_margin_value("od_position_error"),
        ),
        (
            "mission_assurance.od_velocity_error_km_s",
            MetricValueKind.NUMERIC,
            "km/s",
            _assurance_margin_value("od_velocity_error"),
        ),
        (
            "mission_assurance.candidate_delta_v_km_s",
            MetricValueKind.NUMERIC,
            "km/s",
            _metadata("candidate_delta_v_km_s"),
        ),
        (
            "mission_assurance.executed_delta_v_km_s",
            MetricValueKind.NUMERIC,
            "km/s",
            _metadata("executed_delta_v_km_s"),
        ),
        (
            "mission_assurance.truth_recovery_position_error_km",
            MetricValueKind.NUMERIC,
            "km",
            _assurance_margin_value("truth_recovery_position_error"),
        ),
        (
            "mission_assurance.truth_recovery_velocity_error_km_s",
            MetricValueKind.NUMERIC,
            "km/s",
            _assurance_margin_value("truth_recovery_velocity_error"),
        ),
        (
            "mission_assurance.position_error_reduction_fraction",
            MetricValueKind.NUMERIC,
            "1",
            _assurance_margin_value("truth_position_error_reduction"),
        ),
        (
            "mission_assurance.propellant_reserve_kg",
            MetricValueKind.NUMERIC,
            "kg",
            _propellant_reserve,
        ),
        (
            "mission_assurance.minimum_battery_soc_fraction",
            MetricValueKind.NUMERIC,
            "1",
            lambda m: min(
                float(sample.battery_soc_fraction)
                for sample in _result(m).corrected_digital_twin.power
            ),
        ),
        (
            "mission_assurance.minimum_thermal_margin_k",
            MetricValueKind.NUMERIC,
            "K",
            _minimum_thermal_margin,
        ),
        (
            "mission_assurance.twin_battery_soc_margin_fraction",
            MetricValueKind.NUMERIC,
            "1",
            _twin_margin("battery_soc_margin_fraction"),
        ),
        (
            "mission_assurance.failed_twin_margin_count",
            MetricValueKind.NUMERIC,
            "1",
            lambda m: float(
                sum(
                    margin.status.value == "fail"
                    for margin in _result(m).corrected_digital_twin.margin_report.margins
                )
            ),
        ),
        (
            "mission_assurance.retained_measurement_count",
            MetricValueKind.NUMERIC,
            "1",
            lambda m: float(len(_result(m).measurements)),
        ),
    )
    for extractor_id, kind, unit, extract in definitions:
        registry.register(MetricExtractor(extractor_id, WORKFLOW, kind, unit, extract))
    return registry
