from datetime import UTC, datetime, tzinfo
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest
from pydantic import ValidationError

from astro_core.models import (
    Body,
    CartesianState,
    CovarianceSample,
    EarthOrientationConfig,
    EstimateResult,
    ForceModelConfig,
    ForceModelName,
    Frame,
    GroundStation,
    Maneuver,
    MeasurementConfig,
    MeasurementNoise,
    MeasurementRecord,
    MeasurementType,
    OrbitRepresentation,
    OrbitState,
    PropagationConfig,
    Scenario,
    Spacecraft,
    TimeScale,
    Trajectory,
    TrajectoryEvent,
    TrajectorySample,
)


class UndefinedOffsetTimezone(tzinfo):
    def utcoffset(self, dt: datetime | None) -> None:
        return None


def make_state() -> OrbitState:
    return OrbitState(
        epoch=datetime(2026, 1, 1, tzinfo=UTC),
        time_scale=TimeScale.UTC,
        frame=Frame.EME2000,
        central_body=Body.EARTH,
        representation=OrbitRepresentation.CARTESIAN,
        cartesian=CartesianState(
            position_km=(7000.0, 0.0, 0.0),
            velocity_km_s=(0.0, 7.5, 1.0),
        ),
    )


def make_covariance() -> list[list[float]]:
    return [[1.0 if row == column else 0.0 for column in range(6)] for row in range(6)]


def make_trajectory_sample(epoch: datetime) -> TrajectorySample:
    return TrajectorySample(epoch=epoch, state=make_state().cartesian)


def make_trajectory(samples: list[TrajectorySample]) -> Trajectory:
    return Trajectory(
        scenario_id="leo-demo",
        samples=samples,
        force_model=ForceModelConfig(gravity=ForceModelName.TWO_BODY),
        backend="test",
    )


def test_force_model_config_defaults_optional_high_fidelity_flags_to_false() -> None:
    force_model = ForceModelConfig(gravity=ForceModelName.TWO_BODY)

    assert force_model.atmospheric_drag is False
    assert force_model.solar_radiation_pressure is False
    assert force_model.third_body_gravity is False


def test_force_model_config_accepts_optional_high_fidelity_flags() -> None:
    force_model = ForceModelConfig(
        gravity=ForceModelName.OREKIT_HIGH_FIDELITY,
        atmospheric_drag=True,
        solar_radiation_pressure=True,
        third_body_gravity=True,
    )

    assert force_model.gravity is ForceModelName.OREKIT_HIGH_FIDELITY
    assert force_model.atmospheric_drag is True
    assert force_model.solar_radiation_pressure is True
    assert force_model.third_body_gravity is True


def make_measurement_record(**overrides: object) -> MeasurementRecord:
    payload = {
        "measurement_type": MeasurementType.RANGE,
        "epoch": datetime(2026, 1, 1, tzinfo=UTC),
        "observer": "station-a",
        "observed_object": "demo",
        "value": 1000.0,
        "sigma": 0.01,
        "units": "km",
    }
    payload.update(overrides)
    return MeasurementRecord(**payload)


def make_estimate_result(**overrides: object) -> EstimateResult:
    payload = {
        "estimated_state": make_state(),
        "residuals": [0.1, -0.1],
        "covariance": make_covariance(),
        "rms": 0.1,
        "iterations": 3,
        "converged": True,
    }
    payload.update(overrides)
    return EstimateResult(**payload)


def assert_epoch_rejected(epoch: object) -> None:
    with pytest.raises(ValidationError, match="datetime"):
        OrbitState(
            epoch=epoch,
            time_scale=TimeScale.UTC,
            frame=Frame.EME2000,
            central_body=Body.EARTH,
            representation=OrbitRepresentation.CARTESIAN,
            cartesian=CartesianState(
                position_km=(7000.0, 0.0, 0.0),
                velocity_km_s=(0.0, 7.5, 1.0),
            ),
        )

    with pytest.raises(ValidationError, match="datetime"):
        make_measurement_record(epoch=epoch)

    with pytest.raises(ValidationError, match="datetime"):
        TrajectorySample(epoch=epoch, state=make_state().cartesian)


def test_model_extra_fields_are_rejected() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Spacecraft(
            name="demo",
            mass_kg=120.0,
            area_m2=2.5,
            drag_coefficient=2.2,
            reflectivity_coefficient=1.3,
            unexpected_field=True,
        )


def test_scalar_fields_reject_string_and_bool_inputs() -> None:
    with pytest.raises(ValidationError, match="numeric scalar"):
        Spacecraft(
            name="demo",
            mass_kg="120.0",
            area_m2=2.5,
            drag_coefficient=2.2,
            reflectivity_coefficient=1.3,
        )

    with pytest.raises(ValidationError, match="integer"):
        MeasurementNoise(seed="42")

    with pytest.raises(ValidationError, match="integer"):
        make_estimate_result(iterations=True)

    with pytest.raises(ValidationError, match="numeric scalar"):
        Spacecraft(
            name="demo",
            mass_kg=np.bool_(True),
            area_m2=2.5,
            drag_coefficient=2.2,
            reflectivity_coefficient=1.3,
        )


def test_cartesian_state_rejects_string_and_bool_vector_components() -> None:
    with pytest.raises(ValidationError, match="numeric scalar"):
        CartesianState(
            position_km=["7000", 0, 0],
            velocity_km_s=(0.0, 7.5, 0.0),
        )

    with pytest.raises(ValidationError, match="numeric scalar"):
        CartesianState(
            position_km=(True, 0, 0),
            velocity_km_s=(0.0, 7.5, 0.0),
        )


def test_cartesian_state_rejects_numpy_vector_inputs() -> None:
    with pytest.raises(ValidationError, match="NumPy arrays"):
        CartesianState(
            position_km=np.array(["7000", 0, 0], dtype=object),
            velocity_km_s=(0.0, 7.5, 0.0),
        )

    with pytest.raises(ValidationError, match="NumPy arrays"):
        CartesianState(
            position_km=np.array([True, 0, 0], dtype=object),
            velocity_km_s=(0.0, 7.5, 0.0),
        )


def test_ground_station_rejects_string_and_bool_numeric_inputs() -> None:
    with pytest.raises(ValidationError, match="numeric scalar"):
        GroundStation(
            name="station-a",
            position_eci_km=["6378", 0, 0],
            frame=Frame.EME2000,
            elevation_mask_deg=0.0,
        )

    with pytest.raises(ValidationError, match="numeric scalar"):
        GroundStation(
            name="station-a",
            position_eci_km=(6378.1363, 0.0, 0.0),
            frame=Frame.EME2000,
            elevation_mask_deg=True,
        )

    with pytest.raises(ValidationError, match="numeric scalar"):
        GroundStation(
            name="station-a",
            position_eci_km=(6378.1363, 0.0, 0.0),
            frame=Frame.EME2000,
            elevation_mask_deg="5.0",
        )


def test_ground_station_rejects_numpy_position_inputs() -> None:
    with pytest.raises(ValidationError, match="NumPy arrays"):
        GroundStation(
            name="station-a",
            position_eci_km=np.array(["6378", 0, 0], dtype=object),
            frame=Frame.EME2000,
            elevation_mask_deg=0.0,
        )


def test_ground_station_accepts_geodetic_position_with_epoch() -> None:
    station = GroundStation(
        name="equator-geodetic",
        latitude_deg=0.0,
        longitude_deg=0.0,
        altitude_km=0.0,
        frame=Frame.EME2000,
        elevation_mask_deg=0.0,
    )

    with pytest.raises(ValueError, match="epoch"):
        station.position_array()

    position = station.position_array(datetime(2026, 1, 1, tzinfo=UTC))

    assert position.shape == (3,)
    assert np.linalg.norm(position) == pytest.approx(6378.137)
    assert position[2] == pytest.approx(0.0)


def test_ground_station_accepts_earth_orientation_for_geodetic_position() -> None:
    station = GroundStation(
        name="equator-geodetic",
        latitude_deg=0.0,
        longitude_deg=0.0,
        altitude_km=0.0,
        frame=Frame.EME2000,
        elevation_mask_deg=0.0,
    )
    epoch = datetime(2026, 1, 1, tzinfo=UTC)

    default_position = station.position_array(epoch)
    zero_eop_position = station.position_array(epoch, EarthOrientationConfig())
    oriented_position = station.position_array(
        epoch,
        EarthOrientationConfig(
            ut1_minus_utc_s=60.0,
            polar_motion_x_arcsec=0.1,
            polar_motion_y_arcsec=-0.2,
            source="unit-test-eop",
        ),
    )

    assert zero_eop_position == pytest.approx(default_position)
    assert np.linalg.norm(oriented_position - default_position) > 0.01


def test_earth_orientation_rejects_string_and_bool_numeric_inputs() -> None:
    with pytest.raises(ValidationError, match="numeric scalar"):
        EarthOrientationConfig(ut1_minus_utc_s="0.1")

    with pytest.raises(ValidationError, match="numeric scalar"):
        EarthOrientationConfig(polar_motion_x_arcsec=True)


def test_ground_station_rejects_missing_or_mixed_position_definitions() -> None:
    with pytest.raises(ValidationError, match="position_eci_km or geodetic"):
        GroundStation(name="station-a", frame=Frame.EME2000, elevation_mask_deg=0.0)

    with pytest.raises(ValidationError, match="latitude_deg, longitude_deg, and altitude_km"):
        GroundStation(
            name="station-a",
            latitude_deg=0.0,
            longitude_deg=0.0,
            frame=Frame.EME2000,
            elevation_mask_deg=0.0,
        )

    with pytest.raises(ValidationError, match="not both"):
        GroundStation(
            name="station-a",
            position_eci_km=(6378.1363, 0.0, 0.0),
            latitude_deg=0.0,
            longitude_deg=0.0,
            altitude_km=0.0,
            frame=Frame.EME2000,
            elevation_mask_deg=0.0,
        )


def test_orbit_state_requires_finite_cartesian_values() -> None:
    with pytest.raises(ValidationError, match="finite"):
        CartesianState(position_km=(7000.0, float("nan"), 0.0), velocity_km_s=(0.0, 7.5, 0.0))


@pytest.mark.parametrize(
    "epoch",
    [
        0,
        "0",
        "0.0",
        "1704067200",
        Decimal("0"),
        Fraction(0, 1),
        np.int64(0),
    ],
)
def test_epoch_fields_reject_timestamp_like_inputs(epoch: object) -> None:
    assert_epoch_rejected(epoch)


def test_iso_datetime_strings_are_accepted_for_epoch_fields() -> None:
    iso_epoch = "2026-01-01T00:00:00Z"
    orbit_state = OrbitState(
        epoch=iso_epoch,
        time_scale=TimeScale.UTC,
        frame=Frame.EME2000,
        central_body=Body.EARTH,
        representation=OrbitRepresentation.CARTESIAN,
        cartesian=CartesianState(
            position_km=(7000.0, 0.0, 0.0),
            velocity_km_s=(0.0, 7.5, 1.0),
        ),
    )
    measurement = make_measurement_record(epoch=iso_epoch)
    sample = TrajectorySample(epoch=iso_epoch, state=make_state().cartesian)

    assert orbit_state.epoch.tzinfo is not None
    assert measurement.epoch.tzinfo is not None
    assert sample.epoch.tzinfo is not None


def test_orbit_state_rejects_timezone_without_utc_offset() -> None:
    with pytest.raises(ValidationError, match="timezone information"):
        OrbitState(
            epoch=datetime(2026, 1, 1, tzinfo=UndefinedOffsetTimezone()),
            time_scale=TimeScale.UTC,
            frame=Frame.EME2000,
            central_body=Body.EARTH,
            representation=OrbitRepresentation.CARTESIAN,
            cartesian=CartesianState(
                position_km=(7000.0, 0.0, 0.0),
                velocity_km_s=(0.0, 7.5, 1.0),
            ),
        )


@pytest.mark.parametrize(
    "epoch",
    [
        datetime(2026, 1, 1),
        datetime(2026, 1, 1, tzinfo=UndefinedOffsetTimezone()),
    ],
)
def test_trajectory_sample_rejects_epoch_without_utc_offset(epoch: datetime) -> None:
    with pytest.raises(ValidationError, match="timezone information"):
        make_trajectory_sample(epoch)


def test_trajectory_requires_samples() -> None:
    with pytest.raises(ValidationError):
        make_trajectory([])


def test_trajectory_requires_non_empty_identifiers() -> None:
    sample = make_trajectory_sample(datetime(2026, 1, 1, tzinfo=UTC))

    with pytest.raises(ValidationError):
        Trajectory(
            scenario_id="",
            samples=[sample],
            force_model=ForceModelConfig(gravity=ForceModelName.TWO_BODY),
            backend="test",
        )

    with pytest.raises(ValidationError):
        Trajectory(
            scenario_id="leo-demo",
            samples=[sample],
            force_model=ForceModelConfig(gravity=ForceModelName.TWO_BODY),
            backend="",
        )


def test_trajectory_rejects_duplicate_epochs() -> None:
    epoch = datetime(2026, 1, 1, tzinfo=UTC)

    with pytest.raises(ValidationError, match="strictly increasing"):
        make_trajectory([make_trajectory_sample(epoch), make_trajectory_sample(epoch)])


def test_trajectory_rejects_descending_epochs() -> None:
    later_epoch = datetime(2026, 1, 1, 0, 1, tzinfo=UTC)
    earlier_epoch = datetime(2026, 1, 1, tzinfo=UTC)

    with pytest.raises(ValidationError, match="strictly increasing"):
        make_trajectory(
            [
                make_trajectory_sample(later_epoch),
                make_trajectory_sample(earlier_epoch),
            ]
        )


def test_trajectory_defaults_operational_product_fields() -> None:
    trajectory = make_trajectory([make_trajectory_sample(datetime(2026, 1, 1, tzinfo=UTC))])

    assert trajectory.events == []
    assert trajectory.maneuvers == []
    assert trajectory.covariance_history == []


def test_trajectory_accepts_events_maneuvers_and_covariance_history() -> None:
    epoch = datetime(2026, 1, 1, tzinfo=UTC)
    event = TrajectoryEvent(
        event_type="eclipse_entry",
        epoch=epoch,
        description="Entered eclipse.",
        metadata={"source": "test"},
    )
    maneuver = Maneuver(
        name="trim-burn",
        epoch=epoch,
        frame=Frame.EME2000,
        delta_v_km_s=(0.0, 0.001, 0.0),
        duration_s=0.0,
    )
    covariance = CovarianceSample(epoch=epoch, covariance=make_covariance())

    trajectory = Trajectory(
        scenario_id="leo-demo",
        samples=[make_trajectory_sample(epoch)],
        force_model=ForceModelConfig(gravity=ForceModelName.TWO_BODY),
        backend="test",
        events=[event],
        maneuvers=[maneuver],
        covariance_history=[covariance],
    )

    assert trajectory.events == [event]
    assert trajectory.maneuvers == [maneuver]
    assert trajectory.covariance_history == [covariance]


def test_trajectory_sample_accepts_optional_mass_history() -> None:
    epoch = datetime(2026, 1, 1, tzinfo=UTC)
    sample = make_trajectory_sample(epoch).model_copy(update={"mass_kg": 118.5})

    assert sample.mass_kg == 118.5


def test_maneuver_accepts_thrust_vector_mass_flow_fields() -> None:
    epoch = datetime(2026, 1, 1, tzinfo=UTC)

    maneuver = Maneuver(
        name="low-thrust",
        epoch=epoch,
        frame=Frame.EME2000,
        delta_v_km_s=(0.0, 0.0, 0.0),
        duration_s=120.0,
        thrust_vector_n=(0.0, 0.25, 0.0),
        specific_impulse_s=220.0,
    )

    assert maneuver.thrust_vector_n == (0.0, 0.25, 0.0)
    assert maneuver.specific_impulse_s == 220.0
    assert maneuver.thrust_direction_mode == "inertial"


def test_maneuver_accepts_velocity_aligned_thrust_direction_mode() -> None:
    maneuver = Maneuver(
        name="velocity-aligned",
        epoch=datetime(2026, 1, 1, 0, 5, tzinfo=UTC),
        frame=Frame.EME2000,
        delta_v_km_s=(0.0, 0.0, 0.0),
        duration_s=120.0,
        thrust_vector_n=(0.0, 0.25, 0.0),
        specific_impulse_s=220.0,
        thrust_direction_mode="velocity_aligned",
    )

    assert maneuver.thrust_direction_mode == "velocity_aligned"


def test_maneuver_and_covariance_validation() -> None:
    epoch = datetime(2026, 1, 1, tzinfo=UTC)

    with pytest.raises(ValidationError, match="greater than or equal"):
        Maneuver(
            name="bad-burn",
            epoch=epoch,
            frame=Frame.EME2000,
            delta_v_km_s=(0.0, 0.001, 0.0),
            duration_s=-1.0,
        )

    with pytest.raises(ValidationError, match="thrust-vector maneuvers require duration_s > 0"):
        Maneuver(
            name="bad-thrust",
            epoch=epoch,
            frame=Frame.EME2000,
            delta_v_km_s=(0.0, 0.0, 0.0),
            thrust_vector_n=(0.0, 0.25, 0.0),
            specific_impulse_s=220.0,
        )

    with pytest.raises(ValidationError, match="specific_impulse_s"):
        Maneuver(
            name="missing-isp",
            epoch=epoch,
            frame=Frame.EME2000,
            delta_v_km_s=(0.0, 0.0, 0.0),
            duration_s=120.0,
            thrust_vector_n=(0.0, 0.25, 0.0),
        )

    with pytest.raises(ValidationError, match="6x6"):
        CovarianceSample(epoch=epoch, covariance=[[1.0]])


@pytest.mark.parametrize("field_name", ["mass_kg", "area_m2"])
def test_spacecraft_rejects_non_finite_mass_and_area(field_name: str) -> None:
    payload = {
        "name": "demo",
        "mass_kg": 120.0,
        "area_m2": 2.5,
        "drag_coefficient": 2.2,
        "reflectivity_coefficient": 1.3,
        field_name: float("inf"),
    }

    with pytest.raises(ValidationError, match="finite"):
        Spacecraft(**payload)


@pytest.mark.parametrize("field_name", ["drag_coefficient", "reflectivity_coefficient"])
def test_spacecraft_rejects_non_finite_coefficients(field_name: str) -> None:
    payload = {
        "name": "demo",
        "mass_kg": 120.0,
        "area_m2": 2.5,
        "drag_coefficient": 2.2,
        "reflectivity_coefficient": 1.3,
        field_name: float("nan"),
    }

    with pytest.raises(ValidationError, match="finite"):
        Spacecraft(**payload)


@pytest.mark.parametrize(
    ("duration_s", "step_s"),
    [
        (float("inf"), 60.0),
        (600.0, float("inf")),
    ],
)
def test_propagation_config_rejects_non_finite_duration_and_step(
    duration_s: float, step_s: float
) -> None:
    with pytest.raises(ValidationError, match="finite"):
        PropagationConfig(duration_s=duration_s, step_s=step_s)


def test_measurement_noise_and_config_reject_non_finite_scalars() -> None:
    with pytest.raises(ValidationError, match="finite"):
        MeasurementNoise(range_sigma_km=float("inf"))

    with pytest.raises(ValidationError, match="finite"):
        MeasurementNoise(range_rate_sigma_km_s=float("nan"))

    with pytest.raises(ValidationError, match="finite"):
        MeasurementConfig(cadence_s=float("inf"))


def test_measurement_config_requires_non_empty_types() -> None:
    with pytest.raises(ValidationError):
        MeasurementConfig(types=())


@pytest.mark.parametrize(
    "epoch",
    [
        datetime(2026, 1, 1),
        datetime(2026, 1, 1, tzinfo=UndefinedOffsetTimezone()),
    ],
)
def test_measurement_record_rejects_epoch_without_utc_offset(epoch: datetime) -> None:
    with pytest.raises(ValidationError, match="timezone information"):
        make_measurement_record(epoch=epoch)


def test_measurement_record_rejects_non_finite_value_or_sigma() -> None:
    with pytest.raises(ValidationError, match="finite"):
        make_measurement_record(value=float("nan"))

    with pytest.raises(ValidationError, match="finite"):
        make_measurement_record(sigma=float("inf"))


@pytest.mark.parametrize(
    ("measurement_type", "units"),
    [
        (MeasurementType.RANGE, "km/s"),
        (MeasurementType.RANGE_RATE, "km"),
    ],
)
def test_measurement_record_rejects_mismatched_type_and_units(
    measurement_type: MeasurementType, units: str
) -> None:
    with pytest.raises(ValidationError, match="units"):
        make_measurement_record(measurement_type=measurement_type, units=units)


def test_estimate_result_rejects_invalid_covariance_shape_and_negative_iterations() -> None:
    with pytest.raises(ValidationError, match="6x6"):
        make_estimate_result(covariance=[[1.0]])

    with pytest.raises(ValidationError):
        make_estimate_result(iterations=-1)


def test_estimate_result_rejects_non_finite_outputs() -> None:
    bad_covariance = make_covariance()
    bad_covariance[0][0] = float("nan")

    with pytest.raises(ValidationError, match="finite"):
        make_estimate_result(residuals=[0.0, float("nan")])

    with pytest.raises(ValidationError, match="finite"):
        make_estimate_result(rms=float("inf"))

    with pytest.raises(ValidationError, match="finite"):
        make_estimate_result(covariance=bad_covariance)


def test_estimate_result_rejects_numpy_array_outputs() -> None:
    with pytest.raises(ValidationError, match="NumPy arrays"):
        make_estimate_result(residuals=np.array(["0.1", 0.2], dtype=object))

    with pytest.raises(ValidationError, match="NumPy arrays"):
        make_estimate_result(covariance=np.array([[True] * 6 for _ in range(6)], dtype=object))


def test_spacecraft_requires_positive_mass_and_area() -> None:
    with pytest.raises(ValidationError):
        Spacecraft(
            name="bad",
            mass_kg=0.0,
            area_m2=3.0,
            drag_coefficient=2.2,
            reflectivity_coefficient=1.2,
        )

    with pytest.raises(ValidationError):
        Spacecraft(
            name="bad",
            mass_kg=100.0,
            area_m2=-1.0,
            drag_coefficient=2.2,
            reflectivity_coefficient=1.2,
        )


def test_scenario_accepts_minimal_valid_orbital_case() -> None:
    scenario = Scenario(
        scenario_id="leo-demo",
        description="LEO propagation demo",
        spacecraft=Spacecraft(
            name="demo",
            mass_kg=120.0,
            area_m2=2.5,
            drag_coefficient=2.2,
            reflectivity_coefficient=1.3,
        ),
        initial_state=make_state(),
        force_model=ForceModelConfig(gravity=ForceModelName.TWO_BODY),
        propagation=PropagationConfig(duration_s=600.0, step_s=60.0),
        ground_stations=[
            GroundStation(
                name="station-a",
                position_eci_km=(6378.1363, 0.0, 0.0),
                frame=Frame.EME2000,
                elevation_mask_deg=0.0,
            )
        ],
    )

    assert scenario.scenario_id == "leo-demo"
    assert scenario.propagation.sample_count == 11


def test_scenario_rejects_negative_covariance_process_noise() -> None:
    payload = Scenario(
        scenario_id="leo-demo",
        description="LEO propagation demo",
        spacecraft=Spacecraft(
            name="demo",
            mass_kg=120.0,
            area_m2=2.5,
            drag_coefficient=2.2,
            reflectivity_coefficient=1.3,
        ),
        initial_state=make_state(),
        force_model=ForceModelConfig(gravity=ForceModelName.TWO_BODY),
        propagation=PropagationConfig(duration_s=600.0, step_s=60.0),
    ).model_dump(mode="json")

    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        Scenario.model_validate(
            payload | {"covariance_process_noise_acceleration_km_s2": -1.0e-9}
        )
