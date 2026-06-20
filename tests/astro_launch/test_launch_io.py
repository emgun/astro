from pathlib import Path

import pytest
import yaml

from astro_core.errors import InvalidScenarioError
from astro_launch.io import load_launch_scenario
from astro_launch.local import propagate_launch_local
from tests.astro_launch.helpers import make_launch_scenario


def test_load_launch_scenario_reads_yaml(tmp_path: Path) -> None:
    path = tmp_path / "launch.yaml"
    path.write_text(
        yaml.safe_dump(make_launch_scenario().model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )

    scenario = load_launch_scenario(path)

    assert scenario.scenario_id == "vertical-two-stage"
    assert scenario.vehicle.stages[0].name == "stage-1"
    assert scenario.propagation.sample_count == 15


def test_load_example_launch_scenario() -> None:
    scenario = load_launch_scenario(Path("examples/launch/vertical_two_stage.yaml"))

    assert scenario.scenario_id == "vertical-two-stage"
    assert scenario.vehicle.stages[1].name == "stage-2"
    assert scenario.target_orbit.altitude_km == 160.0


def test_load_pitch_program_launch_scenario() -> None:
    scenario = load_launch_scenario(Path("examples/launch/pitch_program_two_stage.yaml"))

    assert scenario.scenario_id == "pitch-program-two-stage"
    assert scenario.guidance.mode == "pitch_program"
    assert [point.pitch_deg for point in scenario.guidance.pitch_program] == [
        90.0,
        75.0,
        45.0,
        20.0,
        5.0,
    ]


def test_load_rocketpy_configured_launch_scenario() -> None:
    scenario = load_launch_scenario(Path("examples/launch/rocketpy_configured_two_stage.yaml"))

    assert scenario.scenario_id == "rocketpy-configured-two-stage"
    assert scenario.rocketpy is not None
    assert scenario.rocketpy.rail_length_m == 5.2
    assert scenario.rocketpy.motor_grain_number == 4


def test_load_rocketpy_single_stage_launch_scenario() -> None:
    scenario = load_launch_scenario(Path("examples/launch/rocketpy_configured_single_stage.yaml"))

    assert scenario.scenario_id == "rocketpy-configured-single-stage"
    assert len(scenario.vehicle.stages) == 1
    assert scenario.rocketpy is not None
    assert scenario.rocketpy.motor_thrust_source_n[-1] == (3.0, 0.0)


def test_load_rocketpy_multimotor_unsupported_guard_scenario() -> None:
    scenario = load_launch_scenario(
        Path("examples/launch/rocketpy_configured_multimotor_unsupported.yaml")
    )

    assert scenario.scenario_id == "rocketpy-configured-multimotor-unsupported"
    assert len(scenario.vehicle.stages) == 1
    assert scenario.rocketpy is not None
    assert len(scenario.rocketpy.additional_motors) == 1
    assert scenario.rocketpy.additional_motors[0].name == "strap-on"
    assert scenario.rocketpy.additional_motors[0].position_m == -1.2


def test_load_launch_trajectory_reads_json_product(tmp_path: Path) -> None:
    from astro_launch.io import load_launch_trajectory

    trajectory = propagate_launch_local(make_launch_scenario())
    path = tmp_path / "launch.json"
    path.write_text(trajectory.model_dump_json(), encoding="utf-8")

    loaded = load_launch_trajectory(path)

    assert loaded.scenario_id == trajectory.scenario_id
    assert len(loaded.samples) == len(trajectory.samples)
    assert loaded.insertion_state == trajectory.insertion_state


def test_load_launch_scenario_reports_yaml_parse_errors(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("scenario_id: [broken", encoding="utf-8")

    with pytest.raises(InvalidScenarioError, match="Could not parse launch scenario file"):
        load_launch_scenario(path)


def test_load_launch_scenario_requires_mapping_yaml(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(InvalidScenarioError, match="must contain a mapping"):
        load_launch_scenario(path)


def test_load_launch_scenario_wraps_validation_errors(tmp_path: Path) -> None:
    path = tmp_path / "invalid-launch.yaml"
    path.write_text("scenario_id: missing-required-fields\n", encoding="utf-8")

    with pytest.raises(InvalidScenarioError, match="is invalid"):
        load_launch_scenario(path)
