from __future__ import annotations

from astro_twin.models import MissionMode, PowerConfig, PowerSample, TimelineGeometrySample

_SOLAR_CONSTANT_W_M2 = 1361.0


def compute_power_timeline(
    config: PowerConfig,
    geometry: tuple[TimelineGeometrySample, ...],
    mode_by_elapsed_s: dict[float, MissionMode],
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
        load_w = _mode_load_w(config, mode)
        dt_h = max(0.0, sample.elapsed_s - previous_elapsed_s) / 3600.0
        soc_wh = min(config.battery_capacity_wh, max(0.0, soc_wh + (generated_w - load_w) * dt_h))
        previous_elapsed_s = sample.elapsed_s
        samples.append(
            PowerSample(
                elapsed_s=sample.elapsed_s,
                mode=mode,
                generated_w=generated_w,
                load_w=load_w,
                battery_soc_fraction=soc_wh / config.battery_capacity_wh,
                net_power_w=generated_w - load_w,
            )
        )
    return tuple(samples)


def _mode_load_w(config: PowerConfig, mode: MissionMode) -> float:
    if mode is MissionMode.PAYLOAD:
        return config.payload_load_w
    if mode is MissionMode.DOWNLINK:
        return config.downlink_load_w
    return config.idle_load_w
