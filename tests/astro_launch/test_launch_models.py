from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from astro_core.models import Body, Frame, TimeScale
from astro_launch.models import (
    GuidanceConfig,
    LaunchEngine,
    LaunchEvent,
    LaunchPropagationConfig,
    LaunchRocketPyConfig,
    LaunchScenario,
    LaunchStage,
    LaunchTrajectory,
    LaunchTrajectorySample,
    PitchProgramPoint,
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
    assert scenario.rocketpy is None


def test_launch_scenario_accepts_rocketpy_backend_configuration() -> None:
    scenario = make_launch_scenario(
        rocketpy=LaunchRocketPyConfig(
            rail_length_m=5.2,
            inclination_deg=85.0,
            heading_deg=90.0,
            rocket_radius_m=0.31,
            rocket_mass_without_motor_kg=145.0,
            rocket_inertia_without_motor_kg_m2=(45.0, 45.0, 1.2),
            rocket_center_of_mass_without_motor_m=1.8,
            motor_thrust_source_n=((0.0, 0.0), (1.0, 25000.0), (3.0, 0.0)),
            motor_burn_time_s=3.0,
            motor_dry_mass_kg=28.0,
            motor_dry_inertia_kg_m2=(2.4, 2.4, 0.08),
            motor_position_m=-1.4,
            motor_center_of_dry_mass_position_m=-1.1,
            motor_nozzle_position_m=-1.9,
            motor_nozzle_radius_m=0.075,
            motor_grain_number=4,
            motor_grain_density_kg_m3=1815.0,
            motor_grain_outer_radius_m=0.12,
            motor_grain_initial_inner_radius_m=0.045,
            motor_grain_initial_height_m=0.42,
            motor_grain_separation_m=0.012,
            motor_grains_center_of_mass_position_m=-0.8,
            rail_button_upper_position_m=0.7,
            rail_button_lower_position_m=-1.1,
            rail_button_angular_position_deg=45.0,
        )
    )

    assert scenario.rocketpy is not None
    assert scenario.rocketpy.rail_length_m == 5.2
    assert scenario.rocketpy.motor_grain_number == 4


def test_rocketpy_configuration_rejects_invalid_motor_geometry() -> None:
    with pytest.raises(ValidationError, match="inner radius"):
        LaunchRocketPyConfig(
            rail_length_m=5.2,
            inclination_deg=85.0,
            heading_deg=90.0,
            rocket_radius_m=0.31,
            rocket_mass_without_motor_kg=145.0,
            rocket_inertia_without_motor_kg_m2=(45.0, 45.0, 1.2),
            rocket_center_of_mass_without_motor_m=1.8,
            motor_thrust_source_n=((0.0, 0.0), (1.0, 25000.0), (3.0, 0.0)),
            motor_burn_time_s=3.0,
            motor_dry_mass_kg=28.0,
            motor_dry_inertia_kg_m2=(2.4, 2.4, 0.08),
            motor_position_m=-1.4,
            motor_center_of_dry_mass_position_m=-1.1,
            motor_nozzle_position_m=-1.9,
            motor_nozzle_radius_m=0.075,
            motor_grain_number=4,
            motor_grain_density_kg_m3=1815.0,
            motor_grain_outer_radius_m=0.12,
            motor_grain_initial_inner_radius_m=0.13,
            motor_grain_initial_height_m=0.42,
            motor_grain_separation_m=0.012,
            motor_grains_center_of_mass_position_m=-0.8,
            rail_button_upper_position_m=0.7,
            rail_button_lower_position_m=-1.1,
            rail_button_angular_position_deg=45.0,
        )


def test_rocketpy_configuration_rejects_invalid_thrust_curve() -> None:
    with pytest.raises(ValidationError, match="thrust curve time_s values"):
        LaunchRocketPyConfig(
            rail_length_m=5.2,
            inclination_deg=85.0,
            heading_deg=90.0,
            rocket_radius_m=0.31,
            rocket_mass_without_motor_kg=145.0,
            rocket_inertia_without_motor_kg_m2=(45.0, 45.0, 1.2),
            rocket_center_of_mass_without_motor_m=1.8,
            motor_thrust_source_n=((0.0, 0.0), (1.0, 25000.0), (1.0, 0.0)),
            motor_burn_time_s=3.0,
            motor_dry_mass_kg=28.0,
            motor_dry_inertia_kg_m2=(2.4, 2.4, 0.08),
            motor_position_m=-1.4,
            motor_center_of_dry_mass_position_m=-1.1,
            motor_nozzle_position_m=-1.9,
            motor_nozzle_radius_m=0.075,
            motor_grain_number=4,
            motor_grain_density_kg_m3=1815.0,
            motor_grain_outer_radius_m=0.12,
            motor_grain_initial_inner_radius_m=0.045,
            motor_grain_initial_height_m=0.42,
            motor_grain_separation_m=0.012,
            motor_grains_center_of_mass_position_m=-0.8,
            rail_button_upper_position_m=0.7,
            rail_button_lower_position_m=-1.1,
            rail_button_angular_position_deg=45.0,
        )


def test_pitch_program_guidance_requires_ordered_pitch_knots() -> None:
    guidance = GuidanceConfig(
        mode="pitch_program",
        pitch_program=[
            PitchProgramPoint(time_s=0.0, pitch_deg=90.0),
            PitchProgramPoint(time_s=40.0, pitch_deg=45.0),
            PitchProgramPoint(time_s=120.0, pitch_deg=5.0),
        ],
    )

    assert guidance.mode == "pitch_program"
    assert guidance.pitch_program[-1].pitch_deg == 5.0

    with pytest.raises(ValidationError, match="requires at least two pitch_program points"):
        GuidanceConfig(mode="pitch_program")

    with pytest.raises(ValidationError, match="time_s values must be strictly increasing"):
        GuidanceConfig(
            mode="pitch_program",
            pitch_program=[
                PitchProgramPoint(time_s=0.0, pitch_deg=90.0),
                PitchProgramPoint(time_s=0.0, pitch_deg=45.0),
            ],
        )

    with pytest.raises(ValidationError, match="first pitch_program point must start at t=0"):
        GuidanceConfig(
            mode="pitch_program",
            pitch_program=[
                PitchProgramPoint(time_s=5.0, pitch_deg=90.0),
                PitchProgramPoint(time_s=40.0, pitch_deg=45.0),
            ],
        )


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


def test_insertion_state_from_local_state_maps_horizontal_velocity_east() -> None:
    epoch = datetime(2026, 1, 1, tzinfo=UTC)
    scenario: LaunchScenario = make_launch_scenario()

    insertion_state = scenario.insertion_state_from_local_state(
        epoch=epoch,
        altitude_km=10.0,
        radial_velocity_km_s=1.0,
        horizontal_velocity_km_s=2.0,
    )

    assert insertion_state.epoch == epoch
    assert insertion_state.cartesian.position_km[0] > 6378.0
    assert insertion_state.cartesian.velocity_km_s == (1.0, 2.0, 0.0)
