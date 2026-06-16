from types import SimpleNamespace

import pytest

from astro_backends.rocketpy.runtime import RocketPyRuntime
from astro_backends.rocketpy.simulation import propagate_launch_rocketpy
from astro_core.errors import UnsupportedBackendError
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


def _rocketpy_config() -> LaunchRocketPyConfig:
    return LaunchRocketPyConfig(
        rail_length_m=5.2,
        inclination_deg=85.0,
        heading_deg=90.0,
        rocket_radius_m=0.31,
        rocket_mass_without_motor_kg=145.0,
        rocket_inertia_without_motor_kg_m2=(45.0, 45.0, 1.2),
        rocket_center_of_mass_without_motor_m=1.8,
        motor_dry_mass_kg=28.0,
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
    )


def test_propagate_launch_rocketpy_reports_runtime_unavailable() -> None:
    def fail_runtime() -> RocketPyRuntime:
        raise UnsupportedBackendError("RocketPy backend unavailable: install astro-suite[launch]")

    with pytest.raises(UnsupportedBackendError, match=r"install astro-suite\[launch\]"):
        propagate_launch_rocketpy(make_launch_scenario(), runtime_loader=fail_runtime)


def test_propagate_launch_rocketpy_requires_live_adapter_configuration() -> None:
    with pytest.raises(UnsupportedBackendError, match="requires scenario.rocketpy"):
        propagate_launch_rocketpy(make_launch_scenario(), runtime_loader=_fake_runtime)


def test_propagate_launch_rocketpy_requires_validated_runner_with_configuration() -> None:
    scenario = make_launch_scenario(rocketpy=_rocketpy_config())

    with pytest.raises(UnsupportedBackendError, match="validated flight runner"):
        propagate_launch_rocketpy(scenario, runtime_loader=_fake_runtime)


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
