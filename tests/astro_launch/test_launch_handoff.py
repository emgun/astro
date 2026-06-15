from astro_core.models import ForceModelName
from astro_launch.local import propagate_launch_local
from tests.astro_launch.helpers import make_launch_scenario


def test_launch_trajectory_to_orbit_scenario_uses_insertion_state_and_final_mass() -> None:
    from astro_launch.handoff import launch_trajectory_to_orbit_scenario

    trajectory = propagate_launch_local(make_launch_scenario())

    scenario = launch_trajectory_to_orbit_scenario(
        trajectory,
        duration_s=600.0,
        step_s=60.0,
    )

    assert scenario.scenario_id == "vertical-two-stage-insertion"
    assert scenario.initial_state == trajectory.insertion_state
    assert scenario.spacecraft.name == "launch-payload"
    assert scenario.spacecraft.mass_kg == trajectory.samples[-1].mass_kg
    assert scenario.propagation.duration_s == 600.0
    assert scenario.propagation.step_s == 60.0
    assert scenario.force_model.gravity is ForceModelName.TWO_BODY
    assert scenario.metadata["source_launch_scenario_id"] == trajectory.scenario_id
    assert scenario.metadata["source_launch_backend"] == trajectory.backend
    assert scenario.metadata["source_launch_sample_count"] == len(trajectory.samples)
    assert scenario.metadata["source_launch_event_count"] == len(trajectory.events)
    assert scenario.metadata["source_launch_target_miss"] == trajectory.target_miss


def test_launch_trajectory_to_orbit_scenario_allows_overrides() -> None:
    from astro_launch.handoff import launch_trajectory_to_orbit_scenario

    trajectory = propagate_launch_local(make_launch_scenario())

    scenario = launch_trajectory_to_orbit_scenario(
        trajectory,
        scenario_id="custom-insertion",
        description="Custom insertion scenario.",
        spacecraft_name="payload-a",
        spacecraft_mass_kg=750.0,
        area_m2=4.0,
        drag_coefficient=2.0,
        reflectivity_coefficient=1.1,
        gravity=ForceModelName.J2,
        duration_s=120.0,
        step_s=30.0,
    )

    assert scenario.scenario_id == "custom-insertion"
    assert scenario.description == "Custom insertion scenario."
    assert scenario.spacecraft.name == "payload-a"
    assert scenario.spacecraft.mass_kg == 750.0
    assert scenario.spacecraft.area_m2 == 4.0
    assert scenario.spacecraft.drag_coefficient == 2.0
    assert scenario.spacecraft.reflectivity_coefficient == 1.1
    assert scenario.force_model.gravity is ForceModelName.J2
