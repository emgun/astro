from astro_core.constants import R_EARTH_KM
from astro_launch.local import propagate_launch_local
from astro_launch.models import AtmosphereConfig
from tests.astro_launch.helpers import make_launch_scenario, make_pitch_program_launch_scenario


def test_propagate_launch_local_returns_events_samples_and_insertion_state() -> None:
    scenario = make_launch_scenario()

    trajectory = propagate_launch_local(scenario)

    assert trajectory.backend == "local"
    assert len(trajectory.samples) == scenario.propagation.sample_count
    assert trajectory.samples[0].altitude_km == 0.0
    assert trajectory.samples[0].mass_kg == scenario.vehicle.initial_mass_kg
    assert trajectory.samples[-1].altitude_km > trajectory.samples[0].altitude_km
    assert trajectory.samples[-1].state is not None
    assert trajectory.insertion_state.cartesian == trajectory.samples[-1].state
    assert trajectory.insertion_state.epoch == trajectory.samples[-1].epoch
    assert trajectory.insertion_state.central_body == scenario.launch_site.body
    assert trajectory.insertion_state.cartesian.position_km[0] > R_EARTH_KM
    assert trajectory.target_miss["altitude_miss_km"] != 0.0
    assert "velocity_miss_km_s" in trajectory.target_miss
    assert all(sample.downrange_km == 0.0 for sample in trajectory.samples)
    assert all(sample.horizontal_velocity_km_s == 0.0 for sample in trajectory.samples)


def test_propagate_launch_local_sequences_stage_events() -> None:
    trajectory = propagate_launch_local(make_launch_scenario())

    event_signatures = [
        (event.event_type, event.stage_name, event.time_s) for event in trajectory.events
    ]

    assert event_signatures == [
        ("stage_ignition", "stage-1", 0.0),
        ("stage_burnout", "stage-1", 70.0),
        ("stage_separation", "stage-1", 70.0),
        ("stage_ignition", "stage-2", 70.0),
        ("stage_burnout", "stage-2", 120.0),
        ("stage_separation", "stage-2", 120.0),
        ("insertion", "payload", 140.0),
    ]


def test_propagate_launch_local_mass_never_increases_and_drops_at_staging() -> None:
    trajectory = propagate_launch_local(make_launch_scenario())
    masses = [sample.mass_kg for sample in trajectory.samples]

    assert all(
        previous >= next_mass for previous, next_mass in zip(masses, masses[1:], strict=False)
    )
    assert min(masses) < masses[0] - 1000.0


def test_propagate_launch_local_dynamic_pressure_is_nonnegative() -> None:
    trajectory = propagate_launch_local(make_launch_scenario())

    assert max(sample.dynamic_pressure_pa for sample in trajectory.samples) > 0.0
    assert all(sample.dynamic_pressure_pa >= 0.0 for sample in trajectory.samples)


def test_propagate_launch_local_none_atmosphere_has_zero_dynamic_pressure() -> None:
    scenario = make_launch_scenario(atmosphere=AtmosphereConfig(model="none"))

    trajectory = propagate_launch_local(scenario)

    assert all(sample.dynamic_pressure_pa == 0.0 for sample in trajectory.samples)


def test_propagate_launch_local_pitch_program_builds_downrange_and_horizontal_velocity() -> None:
    trajectory = propagate_launch_local(make_pitch_program_launch_scenario())
    final_sample = trajectory.samples[-1]

    assert trajectory.metadata["model"] == "pitch_program_2d"
    assert trajectory.metadata["guidance_mode"] == "pitch_program"
    assert final_sample.downrange_km > 0.0
    assert final_sample.horizontal_velocity_km_s > 0.0
    assert final_sample.radial_velocity_km_s > 0.0
    assert final_sample.flight_path_angle_deg < 90.0
    assert final_sample.state is not None
    assert final_sample.state.velocity_km_s[1] > 0.0
    assert trajectory.insertion_state.cartesian == final_sample.state
