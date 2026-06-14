from datetime import UTC, datetime, tzinfo

import pytest
from pydantic import ValidationError

from astro_core.models import (
    Body,
    CartesianState,
    EstimateResult,
    ForceModelConfig,
    ForceModelName,
    Frame,
    GroundStation,
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


def test_orbit_state_requires_finite_cartesian_values() -> None:
    with pytest.raises(ValidationError, match="finite"):
        CartesianState(position_km=(7000.0, float("nan"), 0.0), velocity_km_s=(0.0, 7.5, 0.0))


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
