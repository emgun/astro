from pathlib import Path

import pytest

from astro_backends.orekit.estimation import (
    build_orekit_batch_ls_estimator,
    build_orekit_observed_measurements,
    estimate_orekit_native,
)
from astro_core.errors import UnsupportedBackendError
from astro_core.io import load_scenario
from astro_core.models import MeasurementRecord, MeasurementType
from tests.astro_backends.test_orekit_propagation import _fake_runtime


def test_build_orekit_observed_measurements_maps_range_and_range_rate() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_geodetic_topocentric.yaml"))
    epoch = scenario.initial_state.epoch
    records = [
        MeasurementRecord(
            measurement_type=MeasurementType.RANGE,
            epoch=epoch,
            observer="greenwich-geodetic",
            observed_object=scenario.spacecraft.name,
            value=700.0,
            sigma=0.01,
            units="km",
        ),
        MeasurementRecord(
            measurement_type=MeasurementType.RANGE_RATE,
            epoch=epoch,
            observer="greenwich-geodetic",
            observed_object=scenario.spacecraft.name,
            value=0.5,
            sigma=1.0e-5,
            units="km/s",
        ),
    ]

    observed = build_orekit_observed_measurements(scenario, records, _fake_runtime())

    assert [measurement.measurement_type for measurement in observed] == ["Range", "RangeRate"]
    assert observed[0].station.base_frame.name == "greenwich-geodetic"
    assert observed[0].observed_value_m == 700_000.0
    assert observed[0].sigma == 10.0
    assert observed[0].two_way is True
    assert observed[1].observed_value_m_s == 500.0
    assert observed[1].sigma == 0.01
    assert observed[1].two_way is True
    assert observed[0].satellite.index == 0
    assert observed[1].satellite.index == 0


def test_build_orekit_batch_ls_estimator_adds_observed_measurements() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_geodetic_topocentric.yaml"))
    epoch = scenario.initial_state.epoch
    records = [
        MeasurementRecord(
            measurement_type=MeasurementType.RANGE,
            epoch=epoch,
            observer="greenwich-geodetic",
            observed_object=scenario.spacecraft.name,
            value=700.0,
            sigma=0.01,
            units="km",
        ),
        MeasurementRecord(
            measurement_type=MeasurementType.RANGE_RATE,
            epoch=epoch,
            observer="greenwich-geodetic",
            observed_object=scenario.spacecraft.name,
            value=0.5,
            sigma=1.0e-5,
            units="km/s",
        ),
    ]

    estimator = build_orekit_batch_ls_estimator(scenario, records, _fake_runtime())

    assert len(estimator.propagator_builders) == 1
    assert estimator.propagator_builders[0].position_angle_type == "TRUE"
    assert [measurement.measurement_type for measurement in estimator.measurements] == [
        "Range",
        "RangeRate",
    ]


def test_estimate_orekit_native_maps_estimator_output_to_suite_result() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_geodetic_topocentric.yaml"))
    epoch = scenario.initial_state.epoch
    records = [
        MeasurementRecord(
            measurement_type=MeasurementType.RANGE,
            epoch=epoch,
            observer="greenwich-geodetic",
            observed_object=scenario.spacecraft.name,
            value=700.0,
            sigma=0.01,
            units="km",
        ),
        MeasurementRecord(
            measurement_type=MeasurementType.RANGE_RATE,
            epoch=epoch,
            observer="greenwich-geodetic",
            observed_object=scenario.spacecraft.name,
            value=0.5,
            sigma=1.0e-5,
            units="km/s",
        ),
    ]

    result = estimate_orekit_native(scenario, records, runtime_loader=_fake_runtime)

    assert result.converged is True
    assert result.estimated_state.frame == scenario.initial_state.frame
    assert result.estimated_state.cartesian.position_km == pytest.approx((7001.0, 2.0, 3.0))
    assert result.estimated_state.cartesian.velocity_km_s == pytest.approx((0.01, 7.51, 1.02))
    assert result.residuals == [0.1, -0.2]
    assert result.rms == pytest.approx((0.1**2 + (-0.2) ** 2) ** 0.5)
    assert len(result.covariance) == 6
    assert result.covariance[0][0] == 1.0e-6
    assert result.iterations == 3
    assert result.metadata["backend"] == "orekit_batch_ls_estimator"
    assert result.metadata["estimator"] == "Orekit BatchLSEstimator"
    assert result.metadata["evaluations"] == 5
    assert result.metadata["measurement_count"] == 2
    assert result.metadata["wrapper"] == "orekit_jpype"


def test_build_orekit_observed_measurements_rejects_angle_records() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_geodetic_topocentric.yaml"))
    record = MeasurementRecord(
        measurement_type=MeasurementType.AZIMUTH,
        epoch=scenario.initial_state.epoch,
        observer="greenwich-geodetic",
        observed_object=scenario.spacecraft.name,
        value=90.0,
        sigma=0.001,
        units="deg",
    )

    with pytest.raises(UnsupportedBackendError, match="range and range_rate"):
        build_orekit_observed_measurements(scenario, [record], _fake_runtime())


def test_build_orekit_observed_measurements_requires_geodetic_stations() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))
    record = MeasurementRecord(
        measurement_type=MeasurementType.RANGE,
        epoch=scenario.initial_state.epoch,
        observer="equator-eci",
        observed_object=scenario.spacecraft.name,
        value=700.0,
        sigma=0.01,
        units="km",
    )

    with pytest.raises(UnsupportedBackendError, match="geodetic"):
        build_orekit_observed_measurements(scenario, [record], _fake_runtime())
