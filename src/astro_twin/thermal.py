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
        heat_balance_w: dict[str, float] = {}
        for node in nodes:
            current_k = temperatures[node.name]
            absorbed_w = (
                _SOLAR_CONSTANT_W_M2 * node.radiator_area_m2 * node.absorptivity
                if geometry_sample.sunlit
                else 0.0
            )
            albedo_w = (
                node.albedo_flux_w_m2 * node.radiator_area_m2 * node.absorptivity
                if geometry_sample.sunlit
                else 0.0
            )
            planet_ir_w = node.planet_ir_flux_w_m2 * node.radiator_area_m2 * node.absorptivity
            mode_heat_scale = node.mode_internal_heat_scale.get(
                power_sample.mode.value,
                1.0,
            )
            internal_w = power_sample.load_w * node.internal_heat_fraction * mode_heat_scale
            radiated_w = (
                node.emissivity
                * _SIGMA_W_M2_K4
                * node.radiator_area_m2
                * (current_k**4 - _SPACE_TEMPERATURE_K**4)
            )
            net_heat_w = absorbed_w + albedo_w + planet_ir_w + internal_w - radiated_w
            heat_balance_w[node.name] = net_heat_w
            next_temperatures[node.name] = current_k + (net_heat_w * dt_s / node.thermal_mass_j_k)
        temperatures = next_temperatures
        samples.append(
            ThermalSample(
                elapsed_s=geometry_sample.elapsed_s,
                node_temperatures_k=dict(temperatures),
                node_heat_balance_w=heat_balance_w,
            )
        )
    return tuple(samples)
