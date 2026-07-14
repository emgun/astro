from __future__ import annotations

from typing import cast

import pytest

from astro_twin.io import load_twin_scenario
from astro_twin.models import DigitalTwinScenario
from astro_twin.runner import run_digital_twin
from astro_uq.adapters.twin import twin_metric_registry, twin_parameter_registry
from astro_uq.metrics import MetricError
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


def _apply(
    target: str, unit: str, value: float, scenario: DigitalTwinScenario
) -> DigitalTwinScenario:
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
    return cast(
        DigitalTwinScenario,
        twin_parameter_registry(scenario).apply(
            workflow="digital_twin",
            scenario=scenario,
            uncertainty=uncertainty,
            realization=realization,
        )[0],
    )


@pytest.mark.parametrize(
    ("target", "unit", "value", "section", "field"),
    [
        ("digital_twin.power.solar_array_area_m2", "m^2", 2.6, "power", "solar_array_area_m2"),
        ("digital_twin.power.solar_array_efficiency", "1", 0.31, "power", "solar_array_efficiency"),
        ("digital_twin.power.battery_capacity_wh", "Wh", 1400.0, "power", "battery_capacity_wh"),
        (
            "digital_twin.power.battery_charge_efficiency",
            "1",
            0.91,
            "power",
            "battery_charge_efficiency",
        ),
        (
            "digital_twin.power.battery_discharge_efficiency",
            "1",
            0.89,
            "power",
            "battery_discharge_efficiency",
        ),
        ("digital_twin.adcs.max_torque_n_m", "N*m", 0.1, "adcs", "max_torque_n_m"),
        ("digital_twin.spacecraft.dry_mass_kg", "kg", 125.0, "spacecraft", "dry_mass_kg"),
        ("digital_twin.spacecraft.payload_mass_kg", "kg", 30.0, "spacecraft", "payload_mass_kg"),
        (
            "digital_twin.spacecraft.propellant_mass_kg",
            "kg",
            50.0,
            "spacecraft",
            "propellant_mass_kg",
        ),
    ],
)
def test_scalar_bindings_revalidate(
    target: str, unit: str, value: float, section: str, field: str
) -> None:
    scenario = load_twin_scenario("examples/twin/leo_observer.yaml")
    resolved = _apply(target, unit, value, scenario)
    assert getattr(getattr(resolved, section), field) == value


def test_named_bindings_select_thermal_node_and_link() -> None:
    scenario = load_twin_scenario("examples/twin/leo_observer.yaml")
    thermal = _apply("digital_twin.thermal_nodes.bus.emissivity", "1", 0.8, scenario)
    internal_heat = _apply(
        "digital_twin.thermal_nodes.bus.internal_heat_fraction", "1", 0.5, scenario
    )
    link = _apply("digital_twin.links.xband-downlink.eirp_dbw", "dBW", 20.0, scenario)
    assert thermal.thermal_nodes[0].emissivity == 0.8
    assert internal_heat.thermal_nodes[0].internal_heat_fraction == 0.5
    assert link.links[0].eirp_dbw == 20.0


def test_duplicate_named_selector_is_rejected() -> None:
    scenario = load_twin_scenario("examples/twin/leo_observer.yaml")
    duplicate = scenario.model_copy(update={"thermal_nodes": scenario.thermal_nodes * 2})
    with pytest.raises(ParameterBindingError, match="duplicate thermal_nodes name"):
        twin_parameter_registry(duplicate)


def test_named_selector_rejects_missing_after_registry_creation() -> None:
    scenario = load_twin_scenario("examples/twin/leo_observer.yaml")
    registry = twin_parameter_registry(scenario)
    missing = scenario.model_copy(update={"thermal_nodes": ()})
    binding = registry.resolve("digital_twin", "digital_twin.thermal_nodes.bus.emissivity")
    with pytest.raises(ParameterBindingError, match="missing thermal_nodes name"):
        binding.updater(missing, 0.8)


def test_binding_performs_full_scenario_revalidation() -> None:
    scenario = load_twin_scenario("examples/twin/leo_observer.yaml")
    invalid_link = scenario.links[0].model_copy(update={"ground_site": "missing"})
    invalid = scenario.model_copy(update={"links": (invalid_link,)})
    with pytest.raises(ParameterBindingError, match="resolved digital twin scenario is invalid"):
        _apply("digital_twin.power.battery_capacity_wh", "Wh", 1300.0, invalid)


def test_metric_registry_extracts_requested_twin_metrics() -> None:
    result = run_digital_twin(load_twin_scenario("examples/twin/leo_full_orbit_power_thermal.yaml"))
    specs = tuple(
        MetricSpec(metric_id=metric_id, extractor=extractor, value_kind=kind, unit=unit)
        for metric_id, extractor, kind, unit in (
            (
                "min_battery_soc_fraction",
                "digital_twin.min_battery_soc_fraction",
                MetricValueKind.NUMERIC,
                "1",
            ),
            (
                "min_battery_energy_wh",
                "digital_twin.min_battery_energy_wh",
                MetricValueKind.NUMERIC,
                "Wh",
            ),
            (
                "terminal_battery_energy_wh",
                "digital_twin.terminal_battery_energy_wh",
                MetricValueKind.NUMERIC,
                "Wh",
            ),
            (
                "battery_energy_change_wh",
                "digital_twin.battery_energy_change_wh",
                MetricValueKind.NUMERIC,
                "Wh",
            ),
            (
                "total_unmet_energy_wh",
                "digital_twin.total_unmet_energy_wh",
                MetricValueKind.NUMERIC,
                "Wh",
            ),
            (
                "max_unmet_load_w",
                "digital_twin.max_unmet_load_w",
                MetricValueKind.NUMERIC,
                "W",
            ),
            (
                "eclipse_duration_s",
                "digital_twin.eclipse_duration_s",
                MetricValueKind.NUMERIC,
                "s",
            ),
            (
                "sunlit_fraction",
                "digital_twin.sunlit_fraction",
                MetricValueKind.NUMERIC,
                "1",
            ),
            (
                "bus_cold_margin_k",
                "digital_twin.thermal_nodes.bus.cold_margin_k",
                MetricValueKind.NUMERIC,
                "K",
            ),
            (
                "bus_hot_margin_k",
                "digital_twin.thermal_nodes.bus.hot_margin_k",
                MetricValueKind.NUMERIC,
                "K",
            ),
            (
                "bus_max_temperature_k",
                "digital_twin.thermal_nodes.bus.max_temperature_k",
                MetricValueKind.NUMERIC,
                "K",
            ),
            (
                "min_torque_margin_n_m",
                "digital_twin.min_torque_margin_n_m",
                MetricValueKind.NUMERIC,
                "N*m",
            ),
            (
                "max_pointing_error_deg",
                "digital_twin.max_pointing_error_deg",
                MetricValueKind.NUMERIC,
                "deg",
            ),
            (
                "total_access_duration_s",
                "digital_twin.total_access_duration_s",
                MetricValueKind.NUMERIC,
                "s",
            ),
            (
                "xband_worst_margin_db",
                "digital_twin.links.xband-downlink.worst_margin_db",
                MetricValueKind.NUMERIC,
                "dB",
            ),
            (
                "mass_margin_fraction",
                "digital_twin.mass_margin_fraction",
                MetricValueKind.NUMERIC,
                "1",
            ),
            ("limiting_margin", "digital_twin.limiting_margin", MetricValueKind.NUMERIC, "1"),
            ("limiting_status", "digital_twin.limiting_status", MetricValueKind.CATEGORY, None),
        )
    )
    values = twin_metric_registry(result).extract(
        workflow="digital_twin", result=result, specifications=specs
    )
    assert values["min_battery_soc_fraction"] == min(
        sample.battery_soc_fraction for sample in result.power
    )
    assert values["min_battery_energy_wh"] == min(
        sample.battery_energy_wh for sample in result.power
    )
    assert values["terminal_battery_energy_wh"] == result.power[-1].battery_energy_wh
    assert values["battery_energy_change_wh"] == sum(
        sample.battery_energy_change_wh for sample in result.power
    )
    assert values["total_unmet_energy_wh"] == sum(sample.unmet_energy_wh for sample in result.power)
    assert values["max_unmet_load_w"] == max(sample.unmet_load_w for sample in result.power)
    assert values["eclipse_duration_s"] > 0.0
    assert 0.0 < values["sunlit_fraction"] < 1.0
    margins = {margin.name: margin.margin for margin in result.margin_report.margins}
    assert values["bus_cold_margin_k"] == margins["thermal_bus_cold_margin_k"]
    assert values["bus_hot_margin_k"] == margins["thermal_bus_hot_margin_k"]
    assert values["bus_max_temperature_k"] == max(
        sample.node_temperatures_k["bus"] for sample in result.thermal
    )
    assert values["min_torque_margin_n_m"] == min(
        sample.torque_margin_n_m for sample in result.adcs
    )
    assert values["max_pointing_error_deg"] == max(
        sample.pointing_error_deg for sample in result.adcs
    )
    assert values["total_access_duration_s"] == sum(
        window.duration_s for window in result.access_windows
    )
    assert values["mass_margin_fraction"] == next(
        m.margin for m in result.margin_report.margins if m.name == "mass_margin_fraction"
    )
    assert values["limiting_status"] == result.margin_report.limiting_margin.status.value


def test_named_link_metric_is_none_when_no_window() -> None:
    result = run_digital_twin(load_twin_scenario("examples/twin/leo_observer.yaml"))
    registry = twin_metric_registry(result)
    no_windows = result.model_copy(update={"link_windows": ()})
    spec = MetricSpec(
        metric_id="link",
        extractor="digital_twin.links.xband-downlink.worst_margin_db",
        value_kind=MetricValueKind.NUMERIC,
        unit="dB",
    )
    assert registry.extract(workflow="digital_twin", result=no_windows, specifications=(spec,)) == {
        "link": None
    }


def test_named_temperature_metric_rejects_missing_node() -> None:
    result = run_digital_twin(load_twin_scenario("examples/twin/leo_observer.yaml"))
    registry = twin_metric_registry(result)
    thermal = tuple(
        sample.model_copy(update={"node_temperatures_k": {}}) for sample in result.thermal
    )
    missing = result.model_copy(update={"thermal": thermal})
    spec = MetricSpec(
        metric_id="temperature",
        extractor="digital_twin.thermal_nodes.bus.max_temperature_k",
        value_kind=MetricValueKind.NUMERIC,
        unit="K",
    )
    with pytest.raises(MetricError, match="missing thermal node name"):
        registry.extract(workflow="digital_twin", result=missing, specifications=(spec,))
