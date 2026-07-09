from __future__ import annotations

from astro_twin.models import PowerSample, ThermalNodeConfig, ThermalSample, TimelineGeometrySample

_SOLAR_CONSTANT_W_M2 = 1361.0
_SIGMA_W_M2_K4 = 5.670374419e-8
_SPACE_TEMPERATURE_K = 3.0


def compute_thermal_timeline(
    nodes: tuple[ThermalNodeConfig, ...],
    geometry: tuple[TimelineGeometrySample, ...],
    power: tuple[PowerSample, ...],
) -> tuple[ThermalSample, ...]:
    temperatures = {node.name: node.initial_temperature_k for node in nodes}
    samples: list[ThermalSample] = []
    previous_elapsed_s = geometry[0].elapsed_s if geometry else 0.0
    for geometry_sample, power_sample in zip(geometry, power, strict=True):
        dt_s = max(0.0, geometry_sample.elapsed_s - previous_elapsed_s)
        previous_elapsed_s = geometry_sample.elapsed_s
        next_temperatures: dict[str, float] = {}
        for node in nodes:
            current_k = temperatures[node.name]
            absorbed_w = (
                _SOLAR_CONSTANT_W_M2 * node.radiator_area_m2 * node.absorptivity
                if geometry_sample.sunlit
                else 0.0
            )
            internal_w = power_sample.load_w * node.internal_heat_fraction
            radiated_w = (
                node.emissivity
                * _SIGMA_W_M2_K4
                * node.radiator_area_m2
                * (current_k**4 - _SPACE_TEMPERATURE_K**4)
            )
            next_temperatures[node.name] = current_k + (
                (absorbed_w + internal_w - radiated_w) * dt_s / node.thermal_mass_j_k
            )
        temperatures = next_temperatures
        samples.append(
            ThermalSample(
                elapsed_s=geometry_sample.elapsed_s,
                node_temperatures_k=dict(temperatures),
            )
        )
    return tuple(samples)
