from collections import Counter
from datetime import timedelta
from pathlib import Path

import numpy as np

import astro_od.measurements as od_measurements
from astro_core.io import load_scenario
from astro_core.models import MeasurementType
from astro_dynamics.local import propagate_local
from astro_od.measurements import (
    generate_synthetic_measurements,
    range_km,
    range_rate_km_s,
)


def test_range_km_returns_euclidean_distance() -> None:
    spacecraft_position = np.array([7000.0, 0.0, 0.0])
    station_position = np.array([6378.0, 0.0, 0.0])

    assert range_km(spacecraft_position, station_position) == 622.0


def test_range_rate_km_s_returns_line_of_sight_velocity() -> None:
    spacecraft_position = np.array([7000.0, 0.0, 0.0])
    spacecraft_velocity = np.array([0.0, 7.5, 0.0])
    station_position = np.array([6378.0, 0.0, 0.0])

    assert range_rate_km_s(spacecraft_position, spacecraft_velocity, station_position) == 0.0


def test_inertial_angle_measurements_return_line_of_sight_ra_dec() -> None:
    spacecraft_position = np.array([1.0, 1.0, 1.0])
    station_position = np.array([0.0, 0.0, 0.0])
    right_ascension_deg = getattr(od_measurements, "right_ascension_deg", None)
    declination_deg = getattr(od_measurements, "declination_deg", None)

    assert right_ascension_deg is not None
    assert declination_deg is not None
    assert right_ascension_deg(spacecraft_position, station_position) == 45.0
    assert declination_deg(spacecraft_position, station_position) == np.degrees(
        np.arcsin(1.0 / np.sqrt(3.0))
    )


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


def test_generate_synthetic_measurements_supports_inertial_angles() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))
    angle_measurements = scenario.measurements.model_copy(
        update={
            "types": (
                MeasurementType.RIGHT_ASCENSION,
                MeasurementType.DECLINATION,
            )
        }
    )
    angle_scenario = scenario.model_copy(update={"measurements": angle_measurements})
    trajectory = propagate_local(angle_scenario)

    records = generate_synthetic_measurements(angle_scenario, trajectory)

    assert len(records) == 22
    assert {record.measurement_type for record in records} == {
        MeasurementType.RIGHT_ASCENSION,
        MeasurementType.DECLINATION,
    }
    for record in records:
        assert record.units == "deg"
        assert record.sigma == angle_scenario.measurements.noise.angle_sigma_deg
        assert isinstance(record.metadata["truth"], float)


def test_generate_synthetic_measurements_uses_scenario_noise_seed() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))
    trajectory = propagate_local(scenario)
    alternate_noise = scenario.measurements.noise.model_copy(update={"seed": 7})
    alternate_measurements = scenario.measurements.model_copy(update={"noise": alternate_noise})
    alternate_scenario = scenario.model_copy(update={"measurements": alternate_measurements})

    default_records = generate_synthetic_measurements(scenario, trajectory)
    alternate_first = generate_synthetic_measurements(alternate_scenario, trajectory)
    alternate_second = generate_synthetic_measurements(alternate_scenario, trajectory)

    assert alternate_first == alternate_second
    assert [record.value for record in default_records] != [
        record.value for record in alternate_first
    ]


def test_generate_synthetic_measurements_respects_scenario_cadence() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))
    sparse_measurements = scenario.measurements.model_copy(update={"cadence_s": 120.0})
    sparse_scenario = scenario.model_copy(update={"measurements": sparse_measurements})
    trajectory = propagate_local(sparse_scenario)

    records = generate_synthetic_measurements(sparse_scenario, trajectory)
    expected_epochs = {
        sparse_scenario.initial_state.epoch + timedelta(seconds=offset_s)
        for offset_s in (0, 120, 240, 360, 480, 600)
    }

    assert len(records) == 12
    assert {record.epoch for record in records} == expected_epochs
    assert Counter(record.epoch for record in records) == {
        epoch: 2 for epoch in expected_epochs
    }


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
