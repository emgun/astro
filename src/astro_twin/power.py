from __future__ import annotations

from astro_twin.models import (
    MissionMode,
    PowerConfig,
    PowerLoadSchedule,
    PowerSample,
    TimelineGeometrySample,
)

_SOLAR_CONSTANT_W_M2 = 1361.0


def compute_power_timeline(
    config: PowerConfig,
    geometry: tuple[TimelineGeometrySample, ...],
    mode_by_elapsed_s: dict[float, MissionMode],
    power_loads: tuple[PowerLoadSchedule, ...] = (),
) -> tuple[PowerSample, ...]:
    soc_wh = config.initial_battery_soc_fraction * config.battery_capacity_wh
    samples: list[PowerSample] = []
    previous_elapsed_s = geometry[0].elapsed_s if geometry else 0.0
    for sample in geometry:
        mode = mode_by_elapsed_s.get(sample.elapsed_s, MissionMode.IDLE)
        generated_w = (
            _SOLAR_CONSTANT_W_M2 * config.solar_array_area_m2 * config.solar_array_efficiency
            if sample.sunlit
            else 0.0
        )
        scheduled_load_w = _scheduled_load_w(power_loads, sample.elapsed_s)
        load_w = _mode_load_w(config, mode) + scheduled_load_w
        dt_h = max(0.0, sample.elapsed_s - previous_elapsed_s) / 3600.0
        net_power_w = generated_w - load_w
        previous_soc_wh = soc_wh
        unmet_energy_wh = 0.0
        curtailed_energy_wh = 0.0
        if net_power_w >= 0.0:
            available_charge_wh = net_power_w * config.battery_charge_efficiency * dt_h
            accepted_charge_wh = min(
                available_charge_wh,
                config.battery_capacity_wh - soc_wh,
            )
            soc_wh += accepted_charge_wh
            curtailed_energy_wh = (
                net_power_w * dt_h - accepted_charge_wh / config.battery_charge_efficiency
            )
        else:
            required_discharge_wh = -net_power_w / config.battery_discharge_efficiency * dt_h
            delivered_discharge_wh = min(required_discharge_wh, soc_wh)
            soc_wh -= delivered_discharge_wh
            unmet_energy_wh = (
                required_discharge_wh - delivered_discharge_wh
            ) * config.battery_discharge_efficiency
        battery_energy_change_wh = soc_wh - previous_soc_wh
        unmet_load_w = unmet_energy_wh / dt_h if dt_h > 0.0 else 0.0
        previous_elapsed_s = sample.elapsed_s
        samples.append(
            PowerSample(
                elapsed_s=sample.elapsed_s,
                mode=mode,
                generated_w=generated_w,
                load_w=load_w,
                scheduled_load_w=scheduled_load_w,
                battery_energy_wh=soc_wh,
                battery_soc_fraction=soc_wh / config.battery_capacity_wh,
                net_power_w=net_power_w,
                battery_energy_change_wh=battery_energy_change_wh,
                unmet_load_w=unmet_load_w,
                unmet_energy_wh=unmet_energy_wh,
                curtailed_energy_wh=max(0.0, curtailed_energy_wh),
            )
        )
    return tuple(samples)


def _mode_load_w(config: PowerConfig, mode: MissionMode) -> float:
    if mode is MissionMode.PAYLOAD:
        return config.payload_load_w
    if mode is MissionMode.DOWNLINK:
        return config.downlink_load_w
    return config.idle_load_w


def _scheduled_load_w(
    power_loads: tuple[PowerLoadSchedule, ...],
    elapsed_s: float,
) -> float:
    return sum(
        load.additional_load_w
        for load in power_loads
        if load.start_s <= elapsed_s <= load.end_s
    )
