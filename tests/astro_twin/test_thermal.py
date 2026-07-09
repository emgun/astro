from datetime import UTC, datetime

from astro_twin.models import (
    MissionMode,
    PowerSample,
    ThermalNodeConfig,
    TimelineGeometrySample,
)
from astro_twin.thermal import compute_thermal_timeline


def test_compute_thermal_timeline_returns_node_temperatures() -> None:
    geometry = (
        TimelineGeometrySample(
            epoch=datetime(2026, 1, 1, tzinfo=UTC),
            elapsed_s=0.0,
            position_km=(7000.0, 0.0, 0.0),
            altitude_km=621.863,
            sunlit=True,
        ),
    )
    power = (
        PowerSample(
            elapsed_s=0.0,
            mode=MissionMode.IDLE,
            generated_w=500.0,
            load_w=100.0,
            battery_soc_fraction=0.8,
            net_power_w=400.0,
        ),
    )
    node = ThermalNodeConfig(
        name="bus",
        thermal_mass_j_k=45000.0,
        radiator_area_m2=1.0,
        absorptivity=0.55,
        emissivity=0.78,
        initial_temperature_k=293.0,
        minimum_temperature_k=273.0,
        maximum_temperature_k=313.0,
        internal_heat_fraction=0.45,
    )

    samples = compute_thermal_timeline((node,), geometry, power)

    assert samples[0].node_temperatures_k["bus"] == 293.0


def test_compute_thermal_timeline_records_flux_and_mode_heat_balance() -> None:
    geometry = (
        TimelineGeometrySample(
            epoch=datetime(2026, 1, 1, tzinfo=UTC),
            elapsed_s=0.0,
            position_km=(7000.0, 0.0, 0.0),
            altitude_km=621.863,
            sunlit=True,
        ),
        TimelineGeometrySample(
            epoch=datetime(2026, 1, 1, 0, 1, tzinfo=UTC),
            elapsed_s=60.0,
            position_km=(7000.0, 0.0, 0.0),
            altitude_km=621.863,
            sunlit=True,
        ),
    )
    power = (
        PowerSample(
            elapsed_s=0.0,
            mode=MissionMode.PAYLOAD,
            generated_w=500.0,
            load_w=100.0,
            battery_soc_fraction=0.8,
            net_power_w=400.0,
        ),
        PowerSample(
            elapsed_s=60.0,
            mode=MissionMode.PAYLOAD,
            generated_w=500.0,
            load_w=100.0,
            battery_soc_fraction=0.8,
            net_power_w=400.0,
        ),
    )
    node = ThermalNodeConfig(
        name="bus",
        thermal_mass_j_k=45000.0,
        radiator_area_m2=1.0,
        absorptivity=0.55,
        emissivity=0.78,
        initial_temperature_k=293.0,
        minimum_temperature_k=273.0,
        maximum_temperature_k=313.0,
        internal_heat_fraction=0.45,
        albedo_flux_w_m2=120.0,
        planet_ir_flux_w_m2=220.0,
        mode_internal_heat_scale={"payload": 2.0},
    )

    samples = compute_thermal_timeline((node,), geometry, power)

    assert samples[0].node_heat_balance_w["bus"] > 0.0
    assert samples[1].node_temperatures_k["bus"] > samples[0].node_temperatures_k["bus"]
