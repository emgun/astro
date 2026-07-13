from __future__ import annotations

import pytest

import astro_mission.runner as mission_runner
from astro_core.models import Trajectory
from astro_launch.models import LaunchScenario, LaunchTrajectory
from astro_mission.errors import MissionLifecycleError
from astro_mission.io import load_mission_lifecycle_scenario
from astro_mission.models import DeorbitPhaseConfig, MissionLifecycleInputOverrides
from astro_mission.runner import run_mission_lifecycle
from astro_reentry.models import ReentryResult, ReentryScenario
from astro_twin.io import load_twin_scenario
from astro_twin.models import DigitalTwinResult, DigitalTwinScenario


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
    assert "resolved_input_overrides" not in result.metadata


def test_lifecycle_applies_validated_input_overrides_with_mass_continuity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = load_mission_lifecycle_scenario("examples/lifecycle/leo_round_trip.yaml")
    scenario = scenario.model_copy(
        update={
            "input_overrides": MissionLifecycleInputOverrides(
                launch_upper_stage_thrust_n=200_000.0,
                spacecraft_wet_mass_kg=510.0,
                twin_solar_array_efficiency=0.3,
                reentry_atmosphere_density_scale_factor=1.1,
                reentry_vehicle_drag_coefficient=1.6,
            )
        }
    )
    captured: dict[str, object] = {}
    propagate_launch = mission_runner.propagate_launch_with_backend
    run_twin = mission_runner.run_digital_twin
    simulate_reentry = mission_runner.simulate_reentry_with_backend

    def capture_launch(resolved: LaunchScenario, backend: str) -> LaunchTrajectory:
        captured["launch"] = resolved
        return propagate_launch(resolved, backend)

    def capture_twin(
        resolved: DigitalTwinScenario,
        *,
        trajectory_override: Trajectory | None,
    ) -> DigitalTwinResult:
        captured["twin"] = resolved
        return run_twin(resolved, trajectory_override=trajectory_override)

    def capture_reentry(resolved: ReentryScenario, backend: str) -> ReentryResult:
        captured["reentry"] = resolved
        return simulate_reentry(resolved, backend)

    monkeypatch.setattr(mission_runner, "propagate_launch_with_backend", capture_launch)
    monkeypatch.setattr(mission_runner, "run_digital_twin", capture_twin)
    monkeypatch.setattr(mission_runner, "simulate_reentry_with_backend", capture_reentry)

    result = run_mission_lifecycle(scenario)

    launch = captured["launch"]
    twin = captured["twin"]
    reentry = captured["reentry"]
    assert isinstance(launch, LaunchScenario)
    assert isinstance(twin, DigitalTwinScenario)
    assert isinstance(reentry, ReentryScenario)
    assert launch.vehicle.stages[-1].engine.thrust_n == 200_000.0
    assert launch.vehicle.payload_mass_kg == 510.0
    assert twin.power.solar_array_efficiency == 0.3
    assert (
        twin.spacecraft.dry_mass_kg
        + twin.spacecraft.payload_mass_kg
        + twin.spacecraft.propellant_mass_kg
        == 510.0
    )
    assert reentry.atmosphere.density_scale_factor == 1.1
    assert reentry.vehicle.drag_coefficient == 1.6
    assert result.launch_trajectory.samples[-1].mass_kg == pytest.approx(510.0)
    assert result.orbit_scenario.spacecraft.mass_kg == pytest.approx(510.0)
    assert result.digital_twin.mass_budget.configured_wet_mass_kg == pytest.approx(510.0)
    assert next(
        check for check in result.continuity_report.checks if check.name == "twin_mass"
    ).passed
    assert result.metadata["resolved_input_overrides"] == {
        "launch_upper_stage_thrust_n": 200_000.0,
        "spacecraft_wet_mass_kg": 510.0,
        "twin_solar_array_efficiency": 0.3,
        "reentry_atmosphere_density_scale_factor": 1.1,
        "reentry_vehicle_drag_coefficient": 1.6,
    }


def test_lifecycle_fails_closed_when_launch_misses_declared_insertion() -> None:
    scenario = load_mission_lifecycle_scenario("examples/lifecycle/leo_round_trip.yaml")
    scenario = scenario.model_copy(
        update={"launch_scenario": "examples/launch/pitch_program_two_stage.yaml"}
    )

    with pytest.raises(MissionLifecycleError, match="launch insertion failed") as exc_info:
        run_mission_lifecycle(scenario)
    assert exc_info.value.lifecycle_phase == "launch"


def test_lifecycle_rejects_twin_mass_discontinuity(monkeypatch: pytest.MonkeyPatch) -> None:
    scenario = load_mission_lifecycle_scenario("examples/lifecycle/leo_round_trip.yaml")
    twin = load_twin_scenario(scenario.twin_scenario)
    mismatched_spacecraft = twin.spacecraft.model_copy(update={"dry_mass_kg": 301.0})
    mismatched_twin = twin.model_copy(update={"spacecraft": mismatched_spacecraft})
    monkeypatch.setattr("astro_mission.runner.load_twin_scenario", lambda _: mismatched_twin)

    with pytest.raises(MissionLifecycleError, match="wet mass does not match") as exc_info:
        run_mission_lifecycle(scenario)
    assert exc_info.value.lifecycle_phase == "digital_twin"


def test_lifecycle_rejects_insufficient_deorbit_reserve() -> None:
    scenario = load_mission_lifecycle_scenario("examples/lifecycle/leo_round_trip.yaml")
    deorbit = scenario.deorbit.model_copy(update={"minimum_propellant_reserve_kg": 18.0})
    scenario = scenario.model_copy(update={"deorbit": deorbit})

    with pytest.raises(MissionLifecycleError, match="violates propellant reserve") as exc_info:
        run_mission_lifecycle(scenario)
    assert exc_info.value.lifecycle_phase == "deorbit"


def test_lifecycle_rejects_deorbit_coast_that_misses_entry_interface() -> None:
    scenario = load_mission_lifecycle_scenario("examples/lifecycle/leo_round_trip.yaml")
    deorbit = DeorbitPhaseConfig(
        **{
            **scenario.deorbit.model_dump(),
            "coast_duration_s": 10.0,
        }
    )
    scenario = scenario.model_copy(update={"deorbit": deorbit})

    with pytest.raises(MissionLifecycleError, match="did not reach descending") as exc_info:
        run_mission_lifecycle(scenario)
    assert exc_info.value.lifecycle_phase == "deorbit"
