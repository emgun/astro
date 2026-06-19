from pathlib import Path

import pytest

from astro_core.io import load_scenario
from astro_core.models import MeasurementRecord, MeasurementType
from astro_dynamics.local import propagate_local
from astro_od.calibration import (
    generate_dsn_calibration_product,
    generate_dsn_calibration_product_from_measurements,
    generate_station_calibration_product_from_measurements,
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


def test_generate_station_calibration_product_summarizes_station_biases() -> None:
    records = _station_calibration_records()

    product = generate_station_calibration_product_from_measurements("dsn-demo", records)

    assert product.scenario_id == "dsn-demo"
    assert product.calibration_model == "station_measurement_bias_from_truth_metadata"
    assert product.station_count == 2
    assert product.entry_count == 2
    assert product.measurement_count == 3
    assert product.uncalibrated_measurement_count == 0
    assert product.truth_metadata_key == "truth"
    dss14 = product.entries[0]
    assert dss14.station == "DSS-14"
    assert dss14.measurement_type == "range"
    assert dss14.bias_mean == pytest.approx(0.2)
    assert dss14.bias_min == pytest.approx(0.15)
    assert dss14.bias_max == pytest.approx(0.25)
    assert dss14.bias_rms == pytest.approx(((0.25**2 + 0.15**2) / 2.0) ** 0.5)
    assert dss14.bias_abs_mean == pytest.approx(0.2)
    assert dss14.bias_std == pytest.approx(0.05)
    assert dss14.sigma_mean == pytest.approx(0.5)
    assert dss14.sigma_min == pytest.approx(0.5)
    assert dss14.sigma_max == pytest.approx(0.5)
    assert dss14.normalized_bias_mean == pytest.approx(0.4)
    assert dss14.normalized_bias_rms == pytest.approx(0.41231056256176607)
    dss43 = product.entries[1]
    assert dss43.station == "DSS-43"
    assert dss43.measurement_type == "range_rate"
    assert dss43.bias_mean == pytest.approx(-0.01)
    assert dss43.bias_abs_mean == pytest.approx(0.01)
    assert dss43.bias_std == pytest.approx(0.0)
    assert dss43.sigma_min == pytest.approx(0.01)
    assert dss43.sigma_max == pytest.approx(0.01)
    assert dss43.normalized_bias_mean == pytest.approx(-1.0)
    assert dss43.normalized_bias_rms == pytest.approx(1.0)
    assert product.metadata["source_measurement_count"] == 3
    assert product.metadata["calibrated_measurement_count"] == 3
    assert product.metadata["calibration_scope"] == "measurement_residual_summary"
    assert product.metadata["grouping_keys"] == ["observer", "measurement_type", "units"]
    assert product.metadata["residual_definition"] == "measurement_value_minus_truth_metadata"


def test_generate_station_calibration_product_counts_uncalibrated_records() -> None:
    records = _station_calibration_records()
    records.append(
        MeasurementRecord(
            measurement_type=MeasurementType.RANGE,
            epoch="2026-01-01T00:02:00+00:00",
            observer="DSS-14",
            observed_object="demo-sat",
            value=12.0,
            sigma=0.5,
            units="km",
            metadata={},
        )
    )

    product = generate_station_calibration_product_from_measurements("dsn-demo", records)

    assert product.measurement_count == 3
    assert product.uncalibrated_measurement_count == 1
    assert product.metadata["source_measurement_count"] == 4
    assert product.metadata["calibrated_measurement_count"] == 3


def _station_calibration_records() -> list[MeasurementRecord]:
    return [
        MeasurementRecord(
            measurement_type=MeasurementType.RANGE,
            epoch="2026-01-01T00:00:00+00:00",
            observer="DSS-14",
            observed_object="demo-sat",
            value=10.25,
            sigma=0.5,
            units="km",
            metadata={"truth": 10.0},
        ),
        MeasurementRecord(
            measurement_type=MeasurementType.RANGE,
            epoch="2026-01-01T00:01:00+00:00",
            observer="DSS-14",
            observed_object="demo-sat",
            value=11.15,
            sigma=0.5,
            units="km",
            metadata={"truth": 11.0},
        ),
        MeasurementRecord(
            measurement_type=MeasurementType.RANGE_RATE,
            epoch="2026-01-01T00:00:00+00:00",
            observer="DSS-43",
            observed_object="demo-sat",
            value=-0.02,
            sigma=0.01,
            units="km/s",
            metadata={"truth": -0.01},
        ),
    ]


def load_measurements_from_text(
    payload: str,
    path: Path,
    *,
    expected_scenario_id: str,
) -> list[MeasurementRecord]:
    path.write_text(payload, encoding="utf-8")
    return load_measurements(path, expected_scenario_id=expected_scenario_id)
