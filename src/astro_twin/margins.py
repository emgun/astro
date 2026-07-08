from __future__ import annotations

from astro_twin.models import (
    ADCSConfig,
    ADCSSample,
    DesignMargin,
    DesignMarginReport,
    LinkBudgetWindow,
    PowerConfig,
    PowerSample,
    SpacecraftBusConfig,
    ThermalNodeConfig,
    ThermalSample,
    TwinMarginStatus,
)


def build_margin_report(
    *,
    spacecraft: SpacecraftBusConfig,
    power_config: PowerConfig,
    thermal_nodes: tuple[ThermalNodeConfig, ...],
    power: tuple[PowerSample, ...],
    thermal: tuple[ThermalSample, ...],
    adcs: tuple[ADCSSample, ...],
    adcs_config: ADCSConfig,
    link_windows: tuple[LinkBudgetWindow, ...],
) -> DesignMarginReport:
    margins = [
        _mass_margin(spacecraft),
        _battery_margin(power_config, power),
        *_thermal_margins(thermal_nodes, thermal),
        _pointing_margin(adcs_config, adcs),
        _torque_margin(adcs_config, adcs),
        _link_margin(link_windows),
    ]
    limiting_margin = min(margins, key=_limiting_key)
    return DesignMarginReport(margins=tuple(margins), limiting_margin=limiting_margin)


def _mass_margin(spacecraft: SpacecraftBusConfig) -> DesignMargin:
    wet_mass_kg = (
        spacecraft.dry_mass_kg + spacecraft.payload_mass_kg + spacecraft.propellant_mass_kg
    )
    value = spacecraft.propellant_mass_kg / wet_mass_kg
    margin = value - spacecraft.mass_margin_fraction_required
    return DesignMargin(
        name="mass_margin_fraction",
        value=value,
        threshold=spacecraft.mass_margin_fraction_required,
        margin=margin,
        status=_status(margin, warn_threshold=0.05),
    )


def _battery_margin(power_config: PowerConfig, power: tuple[PowerSample, ...]) -> DesignMargin:
    min_soc = min(sample.battery_soc_fraction for sample in power)
    margin = min_soc - power_config.minimum_battery_soc_fraction
    return DesignMargin(
        name="battery_soc_margin_fraction",
        value=min_soc,
        threshold=power_config.minimum_battery_soc_fraction,
        margin=margin,
        status=_status(margin, warn_threshold=0.05),
    )


def _thermal_margins(
    thermal_nodes: tuple[ThermalNodeConfig, ...],
    thermal: tuple[ThermalSample, ...],
) -> list[DesignMargin]:
    margins: list[DesignMargin] = []
    for node in thermal_nodes:
        temperatures = [sample.node_temperatures_k[node.name] for sample in thermal]
        min_temp = min(temperatures)
        max_temp = max(temperatures)
        cold_margin = min_temp - node.minimum_temperature_k
        hot_margin = node.maximum_temperature_k - max_temp
        margins.extend(
            [
                DesignMargin(
                    name=f"thermal_{node.name}_cold_margin_k",
                    value=min_temp,
                    threshold=node.minimum_temperature_k,
                    margin=cold_margin,
                    status=_status(cold_margin, warn_threshold=2.0),
                ),
                DesignMargin(
                    name=f"thermal_{node.name}_hot_margin_k",
                    value=max_temp,
                    threshold=node.maximum_temperature_k,
                    margin=hot_margin,
                    status=_status(hot_margin, warn_threshold=2.0),
                ),
            ]
        )
    return margins


def _pointing_margin(adcs_config: ADCSConfig, adcs: tuple[ADCSSample, ...]) -> DesignMargin:
    value = max(sample.pointing_error_deg for sample in adcs)
    margin = min(sample.pointing_margin_deg for sample in adcs)
    return DesignMargin(
        name="pointing_margin_deg",
        value=value,
        threshold=adcs_config.pointing_requirement_deg,
        margin=margin,
        status=_status(margin, warn_threshold=0.01),
    )


def _torque_margin(adcs_config: ADCSConfig, adcs: tuple[ADCSSample, ...]) -> DesignMargin:
    margin = min(sample.torque_margin_n_m for sample in adcs)
    return DesignMargin(
        name="torque_margin_n_m",
        value=adcs_config.required_slew_torque_n_m,
        threshold=adcs_config.max_torque_n_m,
        margin=margin,
        status=_status(margin, warn_threshold=0.005),
    )


def _link_margin(link_windows: tuple[LinkBudgetWindow, ...]) -> DesignMargin:
    if not link_windows:
        return DesignMargin(
            name="link_margin_db",
            value=float("-inf"),
            threshold=0.0,
            margin=-1.0,
            status=TwinMarginStatus.FAIL,
        )
    margin = min(window.worst_ebn0_margin_db for window in link_windows)
    return DesignMargin(
        name="link_margin_db",
        value=margin,
        threshold=0.0,
        margin=margin,
        status=_status(margin, warn_threshold=3.0),
    )


def _status(margin: float, warn_threshold: float) -> TwinMarginStatus:
    if margin < 0.0:
        return TwinMarginStatus.FAIL
    if margin <= warn_threshold:
        return TwinMarginStatus.WARN
    return TwinMarginStatus.PASS


def _limiting_key(margin: DesignMargin) -> tuple[int, float]:
    severity = {
        TwinMarginStatus.FAIL: 0,
        TwinMarginStatus.WARN: 1,
        TwinMarginStatus.PASS: 2,
    }[margin.status]
    normalizer = abs(margin.threshold) if margin.threshold != 0.0 else 1.0
    return severity, margin.margin / normalizer
