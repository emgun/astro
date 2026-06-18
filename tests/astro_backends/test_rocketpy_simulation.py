import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from astro_backends.rocketpy.runtime import RocketPyRuntime
from astro_backends.rocketpy.simulation import propagate_launch_rocketpy
from astro_core.errors import UnsupportedBackendError
from astro_launch.io import load_launch_scenario
from astro_launch.local import propagate_launch_local
from astro_launch.models import LaunchRocketPyConfig, LaunchScenario, LaunchTrajectory
from tests.astro_launch.helpers import make_launch_scenario


def _fake_runtime() -> RocketPyRuntime:
    return RocketPyRuntime(
        package="rocketpy",
        package_version="1.12.1",
        module=SimpleNamespace(),
        environment=object,
        solid_motor=object,
        rocket=object,
        flight=object,
    )


class _FakeRocketPyModel:
    def __init__(self, *, solution_end_s: float | None = None) -> None:
        self.solution_end_s = solution_end_s
        self.environment_kwargs: dict[str, Any] | None = None
        self.motor_kwargs: dict[str, Any] | None = None
        self.rocket_kwargs: dict[str, Any] | None = None
        self.motor_position_m: float | None = None
        self.rail_buttons_kwargs: dict[str, Any] | None = None
        self.flight_kwargs: dict[str, Any] | None = None

    def runtime(self) -> RocketPyRuntime:
        model = self

        class Environment:
            def __init__(self, **kwargs: Any) -> None:
                model.environment_kwargs = kwargs

        class SolidMotor:
            def __init__(self, **kwargs: Any) -> None:
                model.motor_kwargs = kwargs

        class _TotalMass:
            def get_value_opt(self, time_s: float) -> float:
                return 173.0 - time_s

        class Rocket:
            def __init__(self, **kwargs: Any) -> None:
                model.rocket_kwargs = kwargs
                self.total_mass = _TotalMass()

            def add_motor(self, motor: SolidMotor, position: float) -> None:
                model.motor_position_m = position

            def set_rail_buttons(self, **kwargs: Any) -> None:
                model.rail_buttons_kwargs = kwargs

        class Flight:
            def __init__(self, *, rocket: Rocket, environment: Environment, **kwargs: Any) -> None:
                self.rocket = rocket
                self.environment = environment
                model.flight_kwargs = kwargs
                if model.solution_end_s is not None:
                    self.solution_array = [[0.0], [model.solution_end_s]]

            def get_solution_at_time(self, time_s: float, atol: float = 1.0e-3) -> list[float]:
                return [
                    time_s,
                    100.0 * time_s,
                    10.0 * time_s,
                    20.0 * time_s,
                    100.0,
                    10.0,
                    20.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]

        return RocketPyRuntime(
            package="rocketpy",
            package_version="1.11.0",
            module=SimpleNamespace(),
            environment=Environment,
            solid_motor=SolidMotor,
            rocket=Rocket,
            flight=Flight,
        )


def _rocketpy_config() -> LaunchRocketPyConfig:
    return LaunchRocketPyConfig(
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


def _single_stage_scenario() -> LaunchScenario:
    scenario = make_launch_scenario(rocketpy=_rocketpy_config())
    return scenario.model_copy(
        update={
            "vehicle": scenario.vehicle.model_copy(
                update={"stages": [scenario.vehicle.stages[0]]}
            )
        }
    )


def test_propagate_launch_rocketpy_reports_runtime_unavailable() -> None:
    def fail_runtime() -> RocketPyRuntime:
        raise UnsupportedBackendError("RocketPy backend unavailable: install astro-suite[launch]")

    with pytest.raises(UnsupportedBackendError, match=r"install astro-suite\[launch\]"):
        propagate_launch_rocketpy(make_launch_scenario(), runtime_loader=fail_runtime)


def test_propagate_launch_rocketpy_requires_live_adapter_configuration() -> None:
    with pytest.raises(UnsupportedBackendError, match="requires scenario.rocketpy"):
        propagate_launch_rocketpy(make_launch_scenario(), runtime_loader=_fake_runtime)


def test_propagate_launch_rocketpy_composes_multistage_stage_schedule() -> None:
    model = _FakeRocketPyModel()
    scenario = make_launch_scenario(rocketpy=_rocketpy_config())

    trajectory = propagate_launch_rocketpy(scenario, runtime_loader=model.runtime)

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
    assert trajectory.backend == "rocketpy"
    assert trajectory.metadata["source_backend"] == "rocketpy_direct"
    assert trajectory.metadata["model"] == "rocketpy_configured_multistage_composition"
    assert trajectory.metadata["rocketpy_stage_count"] == 2
    assert trajectory.metadata["rocketpy_stage_schedule_duration_s"] == 120.0
    assert trajectory.metadata["rocketpy_stage_schedule_complete"] is True
    assert trajectory.metadata["rocketpy_composition"] == "single_flight_suite_stage_schedule"
    assert trajectory.metadata["rocketpy_multistage_adapter_contract"] == {
        "execution_scope": "single_configured_rocketpy_flight_with_suite_stage_annotations",
        "native_multistage_execution": False,
        "suite_stage_count": 2,
        "suite_stage_names": ["stage-1", "stage-2"],
        "rocketpy_config_count": 1,
        "annotated_stage_events": True,
        "stage_schedule_complete": True,
        "stage_schedule_duration_s": 120.0,
        "native_multistage_gap": (
            "runtime has no suite-validated staged multi-motor flight API"
        ),
    }
    assert trajectory.samples[0].stage_name == "stage-1"
    assert trajectory.samples[7].stage_name == "stage-2"
    assert trajectory.samples[-1].stage_name == "payload"


def test_propagate_launch_rocketpy_returns_suite_product_with_fake_runner() -> None:
    scenario = make_launch_scenario(rocketpy=_rocketpy_config())
    seen_runtime: list[RocketPyRuntime] = []
    seen_config: list[LaunchRocketPyConfig] = []

    def fake_runner(
        candidate: LaunchScenario,
        runtime: RocketPyRuntime,
        config: LaunchRocketPyConfig,
    ) -> LaunchTrajectory:
        assert candidate is scenario
        seen_runtime.append(runtime)
        seen_config.append(config)
        return propagate_launch_local(candidate)

    trajectory = propagate_launch_rocketpy(
        scenario,
        runtime_loader=_fake_runtime,
        flight_runner=fake_runner,
    )

    assert len(seen_runtime) == 1
    assert seen_config == [scenario.rocketpy]
    assert trajectory.backend == "rocketpy"
    assert trajectory.metadata["adapter"] == "rocketpy"
    assert trajectory.metadata["rocketpy_version"] == "1.12.1"
    assert trajectory.metadata["source_backend"] == "local"
    assert trajectory.metadata["rocketpy_configured"] is True
    assert trajectory.metadata["rocketpy_rail_length_m"] == 5.2


def test_propagate_launch_rocketpy_runs_single_stage_solid_flight() -> None:
    model = _FakeRocketPyModel()
    scenario = _single_stage_scenario()

    trajectory = propagate_launch_rocketpy(scenario, runtime_loader=model.runtime)

    assert model.environment_kwargs == {
        "date": scenario.epoch,
        "latitude": scenario.launch_site.latitude_deg,
        "longitude": scenario.launch_site.longitude_deg,
        "elevation": scenario.launch_site.altitude_m,
    }
    assert model.motor_kwargs == {
        "thrust_source": [(0.0, 0.0), (1.0, 25000.0), (3.0, 0.0)],
        "dry_mass": 28.0,
        "dry_inertia": (2.4, 2.4, 0.08),
        "nozzle_radius": 0.075,
        "grain_number": 4,
        "grain_density": 1815.0,
        "grain_outer_radius": 0.12,
        "grain_initial_inner_radius": 0.045,
        "grain_initial_height": 0.42,
        "grain_separation": 0.012,
        "grains_center_of_mass_position": -0.8,
        "center_of_dry_mass_position": -1.1,
        "nozzle_position": -1.9,
        "burn_time": 3.0,
    }
    assert model.rocket_kwargs == {
        "radius": 0.31,
        "mass": 145.0,
        "inertia": (45.0, 45.0, 1.2),
        "power_off_drag": 0.5,
        "power_on_drag": 0.5,
        "center_of_mass_without_motor": 1.8,
    }
    assert model.motor_position_m == -1.4
    assert model.rail_buttons_kwargs == {
        "upper_button_position": 0.7,
        "lower_button_position": -1.1,
        "angular_position": 45.0,
    }
    assert model.flight_kwargs == {
        "rail_length": 5.2,
        "inclination": 85.0,
        "heading": 90.0,
        "max_time": scenario.propagation.duration_s,
        "max_time_step": scenario.propagation.step_s,
        "terminate_on_apogee": False,
        "verbose": False,
    }
    assert trajectory.backend == "rocketpy"
    assert trajectory.metadata["source_backend"] == "rocketpy_direct"
    assert trajectory.metadata["rocketpy_version"] == "1.11.0"
    assert trajectory.samples[0].time_s == 0.0
    assert trajectory.samples[-1].time_s == scenario.propagation.duration_s
    assert trajectory.samples[-1].altitude_km == pytest.approx(2.8)
    assert trajectory.samples[-1].horizontal_velocity_km_s == pytest.approx(0.1004987562)


def test_propagate_launch_rocketpy_stops_at_actual_solution_end() -> None:
    model = _FakeRocketPyModel(solution_end_s=25.0)
    scenario = _single_stage_scenario()

    trajectory = propagate_launch_rocketpy(scenario, runtime_loader=model.runtime)

    assert [sample.time_s for sample in trajectory.samples] == [0.0, 10.0, 20.0, 25.0]
    assert trajectory.events[-1].time_s == 25.0
    assert trajectory.metadata["rocketpy_solution_end_s"] == 25.0
    assert trajectory.metadata["rocketpy_stage_schedule_complete"] is False


@pytest.mark.rocketpy_live
def test_live_rocketpy_configured_launch_examples_return_suite_products() -> None:
    if os.environ.get("ASTRO_RUN_ROCKETPY_LIVE") != "1":
        pytest.skip("set ASTRO_RUN_ROCKETPY_LIVE=1 to run live RocketPy launch simulation")
    pytest.importorskip("rocketpy")

    single_stage = load_launch_scenario(
        Path("examples/launch/rocketpy_configured_single_stage.yaml")
    )
    two_stage = load_launch_scenario(Path("examples/launch/rocketpy_configured_two_stage.yaml"))

    single_stage_trajectory = propagate_launch_rocketpy(single_stage)
    two_stage_trajectory = propagate_launch_rocketpy(two_stage)

    assert single_stage_trajectory.backend == "rocketpy"
    assert single_stage_trajectory.metadata["source_backend"] == "rocketpy_direct"
    assert single_stage_trajectory.metadata["model"] == "rocketpy_single_stage_solid"
    assert single_stage_trajectory.metadata["rocketpy_configured"] is True
    assert single_stage_trajectory.metadata["rocketpy_stage_count"] == 1
    assert len(single_stage_trajectory.samples) >= 2
    assert single_stage_trajectory.insertion_state.epoch == (
        single_stage_trajectory.samples[-1].epoch
    )

    assert two_stage_trajectory.backend == "rocketpy"
    assert two_stage_trajectory.metadata["source_backend"] == "rocketpy_direct"
    assert two_stage_trajectory.metadata["model"] == "rocketpy_configured_multistage_composition"
    assert two_stage_trajectory.metadata["rocketpy_stage_count"] == 2
    assert two_stage_trajectory.metadata["rocketpy_composition"] == (
        "single_flight_suite_stage_schedule"
    )
    assert two_stage_trajectory.metadata["rocketpy_multistage_adapter_contract"][
        "native_multistage_execution"
    ] is False
    assert two_stage_trajectory.metadata["rocketpy_multistage_adapter_contract"][
        "suite_stage_names"
    ] == ["stage-1", "stage-2"]
    assert len(two_stage_trajectory.samples) >= 2
    assert all(
        sample.stage_name in {"stage-1", "stage-2", "payload"}
        for sample in two_stage_trajectory.samples
    )
    assert two_stage_trajectory.insertion_state.epoch == (
        two_stage_trajectory.samples[-1].epoch
    )
