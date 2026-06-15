from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from astro_core.models import Body, Frame, TimeScale
from astro_launch.models import (
    LaunchEngine,
    LaunchEvent,
    LaunchPropagationConfig,
    LaunchScenario,
    LaunchStage,
    LaunchTrajectory,
    LaunchTrajectorySample,
)
from tests.astro_launch.helpers import make_launch_scenario


def make_launch_sample(epoch: datetime, time_s: float) -> LaunchTrajectorySample:
    return LaunchTrajectorySample(
        epoch=epoch,
        time_s=time_s,
        altitude_km=1.0 + time_s,
        downrange_km=0.0,
        velocity_km_s=0.1,
        mass_kg=1000.0,
        stage_name="stage-1",
        dynamic_pressure_pa=10.0,
        acceleration_m_s2=15.0,
    )


def test_launch_scenario_accepts_two_stage_vertical_case() -> None:
    scenario = make_launch_scenario()

    assert scenario.scenario_id == "vertical-two-stage"
    assert scenario.launch_site.body is Body.EARTH
    assert scenario.vehicle.initial_mass_kg == pytest.approx(7450.0)
    assert scenario.propagation.sample_count == 15
    assert scenario.frame is Frame.EME2000
    assert scenario.time_scale is TimeScale.UTC


def test_launch_stage_requires_propellant_to_cover_burn_duration() -> None:
    engine = LaunchEngine(name="hungry", thrust_n=100000.0, specific_impulse_s=250.0)

    with pytest.raises(ValidationError, match="propellant_mass_kg must cover"):
        LaunchStage(
            name="underfueled",
            dry_mass_kg=100.0,
            propellant_mass_kg=1.0,
            engine=engine,
            burn_duration_s=60.0,
            reference_area_m2=1.0,
            drag_coefficient=0.5,
        )


def test_launch_propagation_requires_integer_step_schedule() -> None:
    with pytest.raises(ValidationError, match="integer multiple"):
        LaunchPropagationConfig(duration_s=100.0, step_s=30.0)


def test_launch_scenario_rejects_timestamp_like_epoch_inputs() -> None:
    with pytest.raises(ValidationError, match="datetime"):
        make_launch_scenario(epoch="1704067200")


def test_launch_trajectory_requires_monotonic_samples_and_events() -> None:
    epoch = datetime(2026, 1, 1, tzinfo=UTC)
    scenario = make_launch_scenario()
    insertion_state = scenario.insertion_state_from_vertical_state(
        epoch=epoch + timedelta(seconds=10),
        altitude_km=10.0,
        velocity_km_s=1.0,
    )

    with pytest.raises(ValidationError, match="strictly increasing"):
        LaunchTrajectory(
            scenario_id=scenario.scenario_id,
            samples=[
                make_launch_sample(epoch + timedelta(seconds=10), 10.0),
                make_launch_sample(epoch, 0.0),
            ],
            events=[
                LaunchEvent(
                    event_type="stage_ignition",
                    epoch=epoch,
                    time_s=0.0,
                    stage_name="stage-1",
                ),
                LaunchEvent(
                    event_type="stage_burnout",
                    epoch=epoch + timedelta(seconds=1),
                    time_s=1.0,
                    stage_name="stage-1",
                ),
            ],
            insertion_state=insertion_state,
            target_miss={"altitude_miss_km": 150.0, "velocity_miss_km_s": 6.8},
            backend="local",
        )


def test_insertion_state_from_vertical_state_returns_cartesian_orbit_state() -> None:
    epoch = datetime(2026, 1, 1, tzinfo=UTC)
    scenario: LaunchScenario = make_launch_scenario()

    insertion_state = scenario.insertion_state_from_vertical_state(
        epoch=epoch,
        altitude_km=10.0,
        velocity_km_s=1.0,
    )

    assert insertion_state.epoch == epoch
    assert insertion_state.cartesian.position_km[0] > 6378.0
    assert insertion_state.cartesian.velocity_km_s == (1.0, 0.0, 0.0)
