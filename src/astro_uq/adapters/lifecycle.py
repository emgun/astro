from __future__ import annotations

from collections.abc import Callable

from astro_core.models import AstroModel
from astro_mission.models import (
    MissionLifecycleInputOverrides,
    MissionLifecycleResult,
    MissionLifecycleScenario,
)
from astro_twin.models import DigitalTwinScenario
from astro_uq.metrics import MetricError, MetricExtractor, MetricRegistry
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
                f"lifecycle thermal node override {name}.{field} is not resolved"
            )
        return float(getattr(matches[0], field))

    return get


def _replace_thermal_override_field(
    name: str, field: str
) -> Callable[[AstroModel, ParameterValue], AstroModel]:
    def update(model: AstroModel, value: ParameterValue) -> AstroModel:
        scenario = _scenario(model)
        overrides_payload = (
            {}
            if scenario.input_overrides is None
            else scenario.input_overrides.model_dump(mode="python")
        )
        thermal_payloads = list(overrides_payload.get("twin_thermal_node_overrides", ()))
        positions = {
            str(payload["node_name"]): index for index, payload in enumerate(thermal_payloads)
        }
        if name in positions:
            thermal_payloads[positions[name]][field] = _numeric(value)
        else:
            thermal_payloads.append({"node_name": name, field: _numeric(value)})
        overrides_payload["twin_thermal_node_overrides"] = tuple(thermal_payloads)
        overrides = MissionLifecycleInputOverrides.model_validate(overrides_payload)
        return MissionLifecycleScenario.model_validate(
            scenario.model_copy(update={"input_overrides": overrides}).model_dump(mode="python")
        )

    return update


def lifecycle_parameter_registry(
    twin_scenario: DigitalTwinScenario | None = None,
) -> ParameterRegistry:
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
            "digital_twin.power.solar_array_area_m2",
            "twin_solar_array_area_m2",
            "m^2",
            1.0e-12,
            None,
        ),
        (
            "digital_twin.power.battery_capacity_wh",
            "twin_battery_capacity_wh",
            "Wh",
            1.0e-12,
            None,
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
    thermal_names = (
        [] if twin_scenario is None else [node.name for node in twin_scenario.thermal_nodes]
    )
    if len(set(thermal_names)) != len(thermal_names):
        raise ParameterBindingError("digital twin thermal node names must be unique")
    for name in thermal_names:
        for field in ("emissivity", "internal_heat_fraction"):
            registry.register(
                ParameterBinding(
                    target=f"lifecycle.digital_twin.thermal_nodes.{name}.{field}",
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


def _twin_margin(name: str) -> Callable[[AstroModel], float]:
    def extract(model: AstroModel) -> float:
        matches = [
            margin
            for margin in _result(model).digital_twin.margin_report.margins
            if margin.name == name
        ]
        if len(matches) != 1:
            qualifier = "missing" if not matches else "duplicate"
            raise MetricError(f"{qualifier} lifecycle twin margin name: {name}")
        return float(matches[0].margin)

    return extract


def _minimum_thermal_margin(model: AstroModel) -> float:
    margins = [
        margin
        for margin in _result(model).digital_twin.margin_report.margins
        if margin.name.startswith("thermal_")
        and (margin.name.endswith("_cold_margin_k") or margin.name.endswith("_hot_margin_k"))
    ]
    if not margins:
        raise MetricError("missing lifecycle twin thermal margins")
    return min(float(margin.margin) for margin in margins)


def _worst_observed_link_margin(model: AstroModel) -> MetricValue:
    windows = _result(model).digital_twin.link_windows
    if not windows:
        return None
    return min(float(window.worst_ebn0_margin_db) for window in windows)


def lifecycle_metric_registry(
    twin_scenario: DigitalTwinScenario | None = None,
) -> MetricRegistry:
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
            "lifecycle.twin_battery_soc_margin_fraction",
            MetricValueKind.NUMERIC,
            "1",
            _twin_margin("battery_soc_margin_fraction"),
        ),
        (
            "lifecycle.twin_minimum_thermal_margin_k",
            MetricValueKind.NUMERIC,
            "K",
            _minimum_thermal_margin,
        ),
        (
            "lifecycle.twin_pointing_margin_deg",
            MetricValueKind.NUMERIC,
            "deg",
            _twin_margin("pointing_margin_deg"),
        ),
        (
            "lifecycle.twin_torque_margin_n_m",
            MetricValueKind.NUMERIC,
            "N*m",
            _twin_margin("torque_margin_n_m"),
        ),
        (
            "lifecycle.twin_slew_rate_margin_deg_s",
            MetricValueKind.NUMERIC,
            "deg/s",
            _twin_margin("slew_rate_margin_deg_s"),
        ),
        (
            "lifecycle.twin_actuator_utilization_margin_fraction",
            MetricValueKind.NUMERIC,
            "1",
            _twin_margin("actuator_utilization_margin_fraction"),
        ),
        (
            "lifecycle.twin_has_contact",
            MetricValueKind.BOOLEAN,
            None,
            lambda model: bool(_result(model).digital_twin.link_windows),
        ),
        (
            "lifecycle.twin_worst_observed_link_margin_db",
            MetricValueKind.NUMERIC,
            "dB",
            _worst_observed_link_margin,
        ),
        (
            "lifecycle.twin_propellant_fraction_margin",
            MetricValueKind.NUMERIC,
            "1",
            _twin_margin("mass_margin_fraction"),
        ),
        (
            "lifecycle.twin_mass_budget_rollup_margin_kg",
            MetricValueKind.NUMERIC,
            "kg",
            _twin_margin("mass_budget_rollup_margin_kg"),
        ),
        (
            "lifecycle.twin_access_window_count",
            MetricValueKind.NUMERIC,
            "1",
            lambda model: float(len(_result(model).digital_twin.access_windows)),
        ),
        (
            "lifecycle.twin_total_access_duration_s",
            MetricValueKind.NUMERIC,
            "s",
            lambda model: sum(
                float(window.duration_s)
                for window in _result(model).digital_twin.access_windows
            ),
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
    thermal_names = (
        [] if twin_scenario is None else [node.name for node in twin_scenario.thermal_nodes]
    )
    if len(set(thermal_names)) != len(thermal_names):
        raise MetricError("digital twin thermal node names must be unique")
    for name in thermal_names:
        for boundary in ("cold", "hot"):
            registry.register(
                MetricExtractor(
                    extractor_id=(
                        f"lifecycle.twin_thermal_nodes.{name}.{boundary}_margin_k"
                    ),
                    workflow=WORKFLOW,
                    value_kind=MetricValueKind.NUMERIC,
                    unit="K",
                    extract=_twin_margin(f"thermal_{name}_{boundary}_margin_k"),
                )
            )
    return registry
