from datetime import UTC, datetime

from astro_twin.models import (
    MissionMode,
    PowerConfig,
    PowerLoadSchedule,
    TimelineGeometrySample,
)
from astro_twin.power import compute_power_timeline


def test_compute_power_timeline_depletes_battery_in_eclipse() -> None:
    geometry = (
        TimelineGeometrySample(
            epoch=datetime(2026, 1, 1, tzinfo=UTC),
            elapsed_s=0.0,
            position_km=(7000.0, 0.0, 0.0),
            altitude_km=621.863,
            sunlit=True,
        ),
        TimelineGeometrySample(
            epoch=datetime(2026, 1, 1, 0, 10, tzinfo=UTC),
            elapsed_s=600.0,
            position_km=(-7000.0, 0.0, 0.0),
            altitude_km=621.863,
            sunlit=False,
        ),
    )
    config = PowerConfig(
        solar_array_area_m2=2.0,
        solar_array_efficiency=0.25,
        battery_capacity_wh=1000.0,
        initial_battery_soc_fraction=0.8,
        minimum_battery_soc_fraction=0.35,
        idle_load_w=100.0,
        payload_load_w=250.0,
        downlink_load_w=350.0,
    )

    samples = compute_power_timeline(config, geometry, {})

    assert samples[0].mode == MissionMode.IDLE
    assert samples[0].generated_w > samples[0].load_w
    assert samples[1].battery_soc_fraction < samples[0].battery_soc_fraction


def test_compute_power_timeline_applies_scheduled_load_and_battery_efficiency() -> None:
    geometry = (
        TimelineGeometrySample(
            epoch=datetime(2026, 1, 1, tzinfo=UTC),
            elapsed_s=0.0,
            position_km=(-7000.0, 0.0, 0.0),
            altitude_km=621.863,
            sunlit=False,
        ),
        TimelineGeometrySample(
            epoch=datetime(2026, 1, 1, 1, tzinfo=UTC),
            elapsed_s=3600.0,
            position_km=(-7000.0, 0.0, 0.0),
            altitude_km=621.863,
            sunlit=False,
        ),
    )
    config = PowerConfig(
        solar_array_area_m2=1.0,
        solar_array_efficiency=0.25,
        battery_capacity_wh=1000.0,
        initial_battery_soc_fraction=1.0,
        minimum_battery_soc_fraction=0.35,
        idle_load_w=100.0,
        payload_load_w=250.0,
        downlink_load_w=350.0,
        battery_charge_efficiency=0.8,
        battery_discharge_efficiency=0.5,
    )

    samples = compute_power_timeline(
        config,
        geometry,
        {},
        power_loads=(
            PowerLoadSchedule(
                name="payload-heater",
                start_s=0.0,
                end_s=3600.0,
                additional_load_w=100.0,
            ),
        ),
    )

    assert samples[1].scheduled_load_w == 100.0
    assert samples[1].load_w == 200.0
    assert samples[1].battery_energy_wh == 600.0
    assert samples[1].battery_soc_fraction == 0.6
