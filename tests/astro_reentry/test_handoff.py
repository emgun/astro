import pytest

from astro_core.io import load_scenario
from astro_dynamics.local import propagate_local
from astro_reentry.handoff import trajectory_to_reentry_scenario
from tests.astro_reentry.helpers import make_reentry_scenario


def test_trajectory_handoff_builds_earth_fixed_entry_state_and_provenance() -> None:
    trajectory = propagate_local(load_scenario("examples/scenarios/leo_two_body.yaml"))
    template = make_reentry_scenario()

    scenario = trajectory_to_reentry_scenario(
        trajectory,
        template,
        sample_index=0,
        scenario_id="orbit-handoff-reentry",
    )

    assert scenario.scenario_id == "orbit-handoff-reentry"
    assert scenario.initial_state.epoch == trajectory.samples[0].epoch
    assert scenario.initial_state.altitude_km > 0.0
    assert scenario.initial_state.velocity_km_s > 0.0
    assert scenario.metadata["workflow"] == "trajectory_reentry_handoff"
    assert scenario.metadata["source_trajectory_scenario_id"] == trajectory.scenario_id


def test_trajectory_handoff_rejects_out_of_range_sample() -> None:
    trajectory = propagate_local(load_scenario("examples/scenarios/leo_two_body.yaml"))

    with pytest.raises(ValueError, match="outside trajectory samples"):
        trajectory_to_reentry_scenario(
            trajectory,
            make_reentry_scenario(),
            sample_index=100000,
        )


def test_trajectory_handoff_records_resolved_negative_sample_index() -> None:
    trajectory = propagate_local(load_scenario("examples/scenarios/leo_two_body.yaml"))

    scenario = trajectory_to_reentry_scenario(
        trajectory,
        make_reentry_scenario(),
        sample_index=-1,
    )

    assert scenario.metadata["source_sample_index"] == len(trajectory.samples) - 1
    assert scenario.metadata["requested_sample_index"] == -1
