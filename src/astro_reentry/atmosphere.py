from __future__ import annotations

from math import exp, sqrt

from astro_reentry.models import AerothermalConfig, ReentryAtmosphereConfig, ReentryVehicle

STEFAN_BOLTZMANN_W_M2_K4 = 5.670374419e-8


def atmospheric_density_kg_m3(config: ReentryAtmosphereConfig, altitude_km: float) -> float:
    if config.model == "none":
        return 0.0
    altitude_m = max(0.0, altitude_km * 1000.0)
    exponent = -(altitude_m - config.reference_altitude_m) / config.scale_height_m
    return float(config.density_scale_factor * config.reference_density_kg_m3 * exp(exponent))


def convective_heat_rate_w_m2(
    config: AerothermalConfig,
    vehicle: ReentryVehicle,
    density_kg_m3: float,
    velocity_m_s: float,
) -> float:
    if config.model == "none" or density_kg_m3 <= 0.0:
        return 0.0
    return float(
        config.sutton_graves_coefficient
        * sqrt(density_kg_m3 / vehicle.nose_radius_m)
        * velocity_m_s**3
    )


def radiative_equilibrium_temperature_k(
    config: AerothermalConfig,
    heat_rate_w_m2: float,
) -> float:
    if heat_rate_w_m2 <= 0.0:
        return 0.0
    return float(
        (heat_rate_w_m2 / (config.wall_emissivity * STEFAN_BOLTZMANN_W_M2_K4)) ** 0.25
    )
