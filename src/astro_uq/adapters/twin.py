from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import cast

from pydantic import ValidationError

from astro_core.models import AstroModel
from astro_twin.models import DigitalTwinResult, DigitalTwinScenario
from astro_uq.metrics import MetricError, MetricExtractor, MetricRegistry
from astro_uq.models import MetricValue, MetricValueKind
from astro_uq.parameters import (
    ParameterBinding,
    ParameterBindingError,
    ParameterRegistry,
    ParameterValue,
)

WORKFLOW = "digital_twin"


def _scenario(model: AstroModel) -> DigitalTwinScenario:
    try:
        return DigitalTwinScenario.model_validate(model)
    except ValidationError as exc:
        raise ParameterBindingError(f"resolved digital twin scenario is invalid: {exc}") from exc


def _result(model: AstroModel) -> DigitalTwinResult:
    return DigitalTwinResult.model_validate(model)


def _numeric(value: ParameterValue) -> float:
    if isinstance(value, str):
        raise ParameterBindingError("digital twin parameter requires a numeric value")
    return float(value)


def _replace_scenario_field(
    section: str, field: str
) -> Callable[[AstroModel, ParameterValue], AstroModel]:
    def update(model: AstroModel, value: ParameterValue) -> AstroModel:
        scenario = _scenario(model)
        nested = getattr(scenario, section)
        payload = nested.model_dump(mode="python")
        payload[field] = _numeric(value)
        return _validated_scenario_update(scenario, section, type(nested).model_validate(payload))

    return update


def _scenario_field(section: str, field: str) -> Callable[[AstroModel], float]:
    def get(model: AstroModel) -> float:
        return float(getattr(getattr(_scenario(model), section), field))

    return get


def _unique_named(items: Sequence[AstroModel], *, collection: str) -> dict[str, int]:
    positions: dict[str, int] = {}
    for index, item in enumerate(items):
        name = cast(str, getattr(item, "name"))  # noqa: B009
        if name in positions:
            raise ParameterBindingError(f"duplicate {collection} name: {name}")
        positions[name] = index
    return positions


def _replace_named_field(
    collection: str, name: str, field: str
) -> Callable[[AstroModel, ParameterValue], AstroModel]:
    def update(model: AstroModel, value: ParameterValue) -> AstroModel:
        scenario = _scenario(model)
        items = getattr(scenario, collection)
        positions = _unique_named(items, collection=collection)
        if name not in positions:
            raise ParameterBindingError(f"missing {collection} name: {name}")
        index = positions[name]
        payload = items[index].model_dump(mode="python")
        payload[field] = _numeric(value)
        updated_items = list(items)
        updated_items[index] = type(items[index]).model_validate(payload)
        return _validated_scenario_update(scenario, collection, tuple(updated_items))

    return update


def _named_field(collection: str, name: str, field: str) -> Callable[[AstroModel], float]:
    def get(model: AstroModel) -> float:
        scenario = _scenario(model)
        items = getattr(scenario, collection)
        positions = _unique_named(items, collection=collection)
        if name not in positions:
            raise ParameterBindingError(f"missing {collection} name: {name}")
        return float(getattr(items[positions[name]], field))

    return get


def _validated_scenario_update(
    scenario: DigitalTwinScenario, field: str, value: object
) -> DigitalTwinScenario:
    payload = scenario.model_dump(mode="python")
    payload[field] = value
    try:
        return DigitalTwinScenario.model_validate(payload)
    except ValidationError as exc:
        raise ParameterBindingError(f"resolved digital twin scenario is invalid: {exc}") from exc


def register_twin_parameters(registry: ParameterRegistry, scenario: DigitalTwinScenario) -> None:
    scalar_bindings = (
        ("power", "solar_array_efficiency", "1", 0.0, 1.0),
        ("power", "battery_capacity_wh", "Wh", 0.0, None),
        ("power", "battery_charge_efficiency", "1", 0.0, 1.0),
        ("power", "battery_discharge_efficiency", "1", 0.0, 1.0),
        ("adcs", "max_torque_n_m", "N*m", 0.0, None),
        ("spacecraft", "dry_mass_kg", "kg", 0.0, None),
        ("spacecraft", "payload_mass_kg", "kg", 0.0, None),
        ("spacecraft", "propellant_mass_kg", "kg", 0.0, None),
    )
    for section, field, unit, lower, upper in scalar_bindings:
        registry.register(
            ParameterBinding(
                target=f"digital_twin.{section}.{field}",
                workflow=WORKFLOW,
                unit=unit,
                value_type=float,
                getter=_scenario_field(section, field),
                updater=_replace_scenario_field(section, field),
                lower=lower,
                upper=upper,
            )
        )

    for name in _unique_named(scenario.thermal_nodes, collection="thermal_nodes"):
        registry.register(
            ParameterBinding(
                target=f"digital_twin.thermal_nodes.{name}.emissivity",
                workflow=WORKFLOW,
                unit="1",
                value_type=float,
                getter=_named_field("thermal_nodes", name, "emissivity"),
                updater=_replace_named_field("thermal_nodes", name, "emissivity"),
                lower=0.0,
                upper=1.0,
            )
        )
    for name in _unique_named(scenario.links, collection="links"):
        registry.register(
            ParameterBinding(
                target=f"digital_twin.links.{name}.eirp_dbw",
                workflow=WORKFLOW,
                unit="dBW",
                value_type=float,
                getter=_named_field("links", name, "eirp_dbw"),
                updater=_replace_named_field("links", name, "eirp_dbw"),
            )
        )


def twin_parameter_registry(scenario: DigitalTwinScenario) -> ParameterRegistry:
    registry = ParameterRegistry()
    register_twin_parameters(registry, scenario)
    return registry


def _named_max_temperature(name: str) -> Callable[[AstroModel], float]:
    def extract(model: AstroModel) -> float:
        result = _result(model)
        values: list[float] = []
        for sample in result.thermal:
            if name not in sample.node_temperatures_k:
                raise MetricError(f"missing thermal node name: {name}")
            values.append(float(sample.node_temperatures_k[name]))
        return max(values)

    return extract


def _named_worst_link_margin(name: str) -> Callable[[AstroModel], MetricValue]:
    def extract(model: AstroModel) -> MetricValue:
        windows = [window for window in _result(model).link_windows if window.link_name == name]
        if not windows:
            return None
        return min(float(window.worst_ebn0_margin_db) for window in windows)

    return extract


def _margin(name: str) -> Callable[[AstroModel], float]:
    def extract(model: AstroModel) -> float:
        matches = [margin for margin in _result(model).margin_report.margins if margin.name == name]
        if len(matches) != 1:
            qualifier = "missing" if not matches else "duplicate"
            raise MetricError(f"{qualifier} design margin name: {name}")
        return float(matches[0].margin)

    return extract


def register_twin_metrics(
    registry: MetricRegistry, source: DigitalTwinScenario | DigitalTwinResult
) -> None:
    scalar_metrics: tuple[
        tuple[str, MetricValueKind, str | None, Callable[[AstroModel], MetricValue]], ...
    ] = (
        (
            "digital_twin.min_battery_soc_fraction",
            MetricValueKind.NUMERIC,
            "1",
            lambda model: min(
                float(sample.battery_soc_fraction) for sample in _result(model).power
            ),
        ),
        (
            "digital_twin.min_torque_margin_n_m",
            MetricValueKind.NUMERIC,
            "N*m",
            lambda model: min(float(sample.torque_margin_n_m) for sample in _result(model).adcs),
        ),
        (
            "digital_twin.max_pointing_error_deg",
            MetricValueKind.NUMERIC,
            "deg",
            lambda model: max(float(sample.pointing_error_deg) for sample in _result(model).adcs),
        ),
        (
            "digital_twin.min_pointing_margin_deg",
            MetricValueKind.NUMERIC,
            "deg",
            lambda model: min(float(sample.pointing_margin_deg) for sample in _result(model).adcs),
        ),
        (
            "digital_twin.access_window_count",
            MetricValueKind.NUMERIC,
            "1",
            lambda model: float(len(_result(model).access_windows)),
        ),
        (
            "digital_twin.total_access_duration_s",
            MetricValueKind.NUMERIC,
            "s",
            lambda model: sum(float(window.duration_s) for window in _result(model).access_windows),
        ),
        (
            "digital_twin.mass_margin_fraction",
            MetricValueKind.NUMERIC,
            "1",
            _margin("mass_margin_fraction"),
        ),
        (
            "digital_twin.limiting_margin",
            MetricValueKind.NUMERIC,
            "1",
            lambda model: float(_result(model).margin_report.limiting_margin.margin),
        ),
        (
            "digital_twin.limiting_status",
            MetricValueKind.CATEGORY,
            None,
            lambda model: _result(model).margin_report.limiting_margin.status.value,
        ),
    )
    for extractor_id, value_kind, unit, extract in scalar_metrics:
        registry.register(MetricExtractor(extractor_id, WORKFLOW, value_kind, unit, extract))

    if isinstance(source, DigitalTwinScenario):
        thermal_names = tuple(node.name for node in source.thermal_nodes)
        link_names = tuple(link.name for link in source.links)
    else:
        thermal_names = _unique_result_names(
            (name for sample in source.thermal for name in sample.node_temperatures_k),
            expected_repeats=len(source.thermal),
            collection="thermal node",
        )
        link_names = _unique_result_names(
            (window.link_name for window in source.link_windows),
            expected_repeats=None,
            collection="link",
        )
    for name in thermal_names:
        registry.register(
            MetricExtractor(
                f"digital_twin.thermal_nodes.{name}.max_temperature_k",
                WORKFLOW,
                MetricValueKind.NUMERIC,
                "K",
                _named_max_temperature(name),
            )
        )

    for name in link_names:
        registry.register(
            MetricExtractor(
                f"digital_twin.links.{name}.worst_margin_db",
                WORKFLOW,
                MetricValueKind.NUMERIC,
                "dB",
                _named_worst_link_margin(name),
            )
        )


def _unique_result_names(
    names: Iterable[str], *, expected_repeats: int | None, collection: str
) -> tuple[str, ...]:
    values: tuple[str, ...] = tuple(names)
    ordered = tuple(dict.fromkeys(values))
    if expected_repeats is not None:
        for name in ordered:
            if values.count(name) != expected_repeats:
                raise MetricError(f"missing or duplicate {collection} name: {name}")
    return ordered


def twin_metric_registry(source: DigitalTwinScenario | DigitalTwinResult) -> MetricRegistry:
    registry = MetricRegistry()
    register_twin_metrics(registry, source)
    return registry
