from __future__ import annotations

import pytest

from astro_mission.errors import MissionLifecycleError
from astro_mission.io import load_mission_lifecycle_scenario
from astro_mission.models import DeorbitPhaseConfig
from astro_mission.runner import run_mission_lifecycle
from astro_twin.io import load_twin_scenario


def test_reference_lifecycle_runs_all_phases_with_checked_continuity() -> None:
    scenario = load_mission_lifecycle_scenario("examples/lifecycle/leo_round_trip.yaml")

    result = run_mission_lifecycle(scenario)

    assert result.passed
    assert result.continuity_report.all_passed
    assert len(result.continuity_report.checks) == 8
    assert [entry.phase for entry in result.manifest.entries] == [
        "launch",
        "operations",
        "digital_twin",
        "deorbit",
        "reentry",
    ]
    assert result.digital_twin.metadata["orbit_trajectory_source"] == "trajectory_override"
    assert result.deorbit_trajectory.metadata["termination_reason"] == "entry_interface"
    assert abs(result.reentry_scenario.initial_state.altitude_km - 120.0) <= 1.0
    assert result.reentry_result.metadata["termination_reason"] == "altitude"
    assert result.reentry_result.samples[-1].altitude_km == pytest.approx(0.0)


def test_lifecycle_fails_closed_when_launch_misses_declared_insertion() -> None:
    scenario = load_mission_lifecycle_scenario("examples/lifecycle/leo_round_trip.yaml")
    scenario = scenario.model_copy(
        update={"launch_scenario": "examples/launch/pitch_program_two_stage.yaml"}
    )

    with pytest.raises(MissionLifecycleError, match="launch insertion failed"):
        run_mission_lifecycle(scenario)


def test_lifecycle_rejects_twin_mass_discontinuity(monkeypatch: pytest.MonkeyPatch) -> None:
    scenario = load_mission_lifecycle_scenario("examples/lifecycle/leo_round_trip.yaml")
    twin = load_twin_scenario(scenario.twin_scenario)
    mismatched_spacecraft = twin.spacecraft.model_copy(update={"dry_mass_kg": 301.0})
    mismatched_twin = twin.model_copy(update={"spacecraft": mismatched_spacecraft})
    monkeypatch.setattr("astro_mission.runner.load_twin_scenario", lambda _: mismatched_twin)

    with pytest.raises(MissionLifecycleError, match="wet mass does not match"):
        run_mission_lifecycle(scenario)


def test_lifecycle_rejects_insufficient_deorbit_reserve() -> None:
    scenario = load_mission_lifecycle_scenario("examples/lifecycle/leo_round_trip.yaml")
    deorbit = scenario.deorbit.model_copy(update={"minimum_propellant_reserve_kg": 18.0})
    scenario = scenario.model_copy(update={"deorbit": deorbit})

    with pytest.raises(MissionLifecycleError, match="violates propellant reserve"):
        run_mission_lifecycle(scenario)


def test_lifecycle_rejects_deorbit_coast_that_misses_entry_interface() -> None:
    scenario = load_mission_lifecycle_scenario("examples/lifecycle/leo_round_trip.yaml")
    deorbit = DeorbitPhaseConfig(
        **{
            **scenario.deorbit.model_dump(),
            "coast_duration_s": 10.0,
        }
    )
    scenario = scenario.model_copy(update={"deorbit": deorbit})

    with pytest.raises(MissionLifecycleError, match="did not reach descending"):
        run_mission_lifecycle(scenario)
