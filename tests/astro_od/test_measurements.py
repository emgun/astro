from datetime import timedelta
from pathlib import Path

import numpy as np

from astro_core.io import load_scenario
from astro_core.models import MeasurementType
from astro_dynamics.local import propagate_local
from astro_od.measurements import generate_synthetic_measurements, range_km, range_rate_km_s


def test_range_km_returns_euclidean_distance() -> None:
    spacecraft_position = np.array([7000.0, 0.0, 0.0])
    station_position = np.array([6378.0, 0.0, 0.0])

    assert range_km(spacecraft_position, station_position) == 622.0


def test_range_rate_km_s_returns_line_of_sight_velocity() -> None:
    spacecraft_position = np.array([7000.0, 0.0, 0.0])
    spacecraft_velocity = np.array([0.0, 7.5, 0.0])
    station_position = np.array([6378.0, 0.0, 0.0])

    assert range_rate_km_s(spacecraft_position, spacecraft_velocity, station_position) == 0.0


def test_generate_synthetic_measurements_is_deterministic_for_local_leo() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))
    trajectory = propagate_local(scenario)

    first = generate_synthetic_measurements(scenario, trajectory)
    second = generate_synthetic_measurements(scenario, trajectory)

    assert first == second
    assert len(first) == 22
    assert {record.measurement_type for record in first} == {
        MeasurementType.RANGE,
        MeasurementType.RANGE_RATE,
    }
    for record in first:
        assert record.epoch in {sample.epoch for sample in trajectory.samples}
        assert record.observer == "equator-eci"
        assert record.observed_object == "demo-sat"
        assert "truth" in record.metadata
        if record.measurement_type is MeasurementType.RANGE:
            assert record.units == "km"
            assert record.sigma == scenario.measurements.noise.range_sigma_km
        else:
            assert record.units == "km/s"
            assert record.sigma == scenario.measurements.noise.range_rate_sigma_km_s


def test_generate_synthetic_measurements_returns_empty_without_ground_stations() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))
    scenario_without_stations = scenario.model_copy(update={"ground_stations": []})
    trajectory = propagate_local(scenario_without_stations)

    assert generate_synthetic_measurements(scenario_without_stations, trajectory) == []


def test_generate_synthetic_measurements_skips_samples_off_scenario_cadence() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))
    trajectory = propagate_local(scenario)
    offset_samples = [
        sample.model_copy(update={"epoch": sample.epoch + timedelta(seconds=30.0)})
        for sample in trajectory.samples
    ]
    offset_trajectory = trajectory.model_copy(update={"samples": offset_samples})

    assert generate_synthetic_measurements(scenario, offset_trajectory) == []
