from pathlib import Path

import pytest

from astro_core.io import load_scenario
from astro_core.models import MeasurementRecord
from astro_dynamics.local import propagate_local
from astro_od.calibration import (
    generate_dsn_calibration_product,
    generate_dsn_calibration_product_from_measurements,
)
from astro_od.io import dump_measurements_tdm, load_measurements
from astro_od.measurements import generate_synthetic_measurements


def test_generate_dsn_calibration_product_summarizes_weather_frequency_media() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_radiometric_weather_frequency.yaml"))
    trajectory = propagate_local(scenario)

    product = generate_dsn_calibration_product(scenario, trajectory)

    assert product.scenario_id == "leo-radiometric-weather-frequency"
    assert product.calibration_model == "weather_frequency_range_delay"
    assert product.media_source == "configured-weather-frequency-example"
    assert product.sample_count == len(product.samples)
    assert product.sample_count > 0
    assert product.measurement_types == (
        "two_way_range",
        "two_way_range_rate",
        "three_way_range",
        "three_way_range_rate",
    )
    assert product.total_media_delay_km_mean > 0.0
    assert product.total_media_delay_km_min <= product.total_media_delay_km_mean
    assert product.total_media_delay_km_max >= product.total_media_delay_km_mean
    assert product.uplink_media_delay_km_mean > 0.0
    assert product.downlink_media_delay_km_mean > 0.0
    assert product.metadata["media_frequency_hz"] == 8.4e9
    assert product.metadata["weather_pressure_hpa"] == 1012.0
    assert product.metadata["zenith_total_electron_content_tecu"] == 8.0

    first_sample = product.samples[0]
    assert first_sample.participant_path
    assert first_sample.total_media_delay_km == pytest.approx(
        first_sample.uplink_media_delay_km + first_sample.downlink_media_delay_km
    )
    assert first_sample.uplink_media_elevation_deg is not None
    assert first_sample.downlink_media_elevation_deg is not None


def test_generate_dsn_calibration_product_rejects_scenarios_without_radiometric_media() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))
    trajectory = propagate_local(scenario)

    with pytest.raises(ValueError, match="radiometric media"):
        generate_dsn_calibration_product(scenario, trajectory)


def test_generate_dsn_calibration_product_from_tdm_measurements(tmp_path: Path) -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_radiometric_weather_frequency.yaml"))
    trajectory = propagate_local(scenario)
    records = generate_synthetic_measurements(scenario, trajectory)
    exported = dump_measurements_tdm(scenario.scenario_id, records)

    path = tmp_path / "radiometric-weather-frequency.tdm"
    loaded = load_measurements_from_text(
        exported,
        path,
        expected_scenario_id=scenario.scenario_id,
    )
    product = generate_dsn_calibration_product_from_measurements(
        scenario.scenario_id,
        loaded,
        station_count=len(scenario.ground_stations),
    )

    assert "ASTRO_TOTAL_MEDIA_DELAY_KM" in exported
    assert product.scenario_id == scenario.scenario_id
    assert product.calibration_model == "weather_frequency_range_delay"
    assert product.sample_count == 66
    assert product.metadata["source_measurement_count"] == len(loaded)
    assert product.metadata["source_format"] == "measurement_records"
    assert product.metadata["media_frequency_hz"] == 8.4e9


def load_measurements_from_text(
    payload: str,
    path: Path,
    *,
    expected_scenario_id: str,
) -> list[MeasurementRecord]:
    path.write_text(payload, encoding="utf-8")
    return load_measurements(path, expected_scenario_id=expected_scenario_id)
