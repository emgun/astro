import pytest

from astro_reentry.atmosphere import (
    atmospheric_density_kg_m3,
    convective_heat_rate_w_m2,
    radiative_equilibrium_temperature_k,
)
from astro_reentry.models import AerothermalConfig, ReentryAtmosphereConfig
from tests.astro_reentry.helpers import make_reentry_scenario


def test_exponential_density_decreases_with_altitude_and_scales() -> None:
    config = ReentryAtmosphereConfig(density_scale_factor=1.2)

    assert atmospheric_density_kg_m3(config, 0.0) == pytest.approx(1.47)
    assert atmospheric_density_kg_m3(config, 80.0) < atmospheric_density_kg_m3(config, 40.0)
    assert atmospheric_density_kg_m3(ReentryAtmosphereConfig(model="none"), 0.0) == 0.0


def test_sutton_graves_heating_and_equilibrium_temperature_are_positive() -> None:
    vehicle = make_reentry_scenario().vehicle
    config = AerothermalConfig()

    heat_rate = convective_heat_rate_w_m2(config, vehicle, 1.0e-4, 7800.0)
    temperature = radiative_equilibrium_temperature_k(config, heat_rate)

    assert heat_rate > 0.0
    assert temperature > 0.0
    assert convective_heat_rate_w_m2(config, vehicle, 0.0, 7800.0) == 0.0
