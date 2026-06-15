from types import SimpleNamespace

import pytest

from astro_backends.rocketpy.runtime import RocketPyRuntime
from astro_backends.rocketpy.simulation import propagate_launch_rocketpy
from astro_core.errors import UnsupportedBackendError
from astro_launch.local import propagate_launch_local
from astro_launch.models import LaunchScenario, LaunchTrajectory
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


def test_propagate_launch_rocketpy_reports_runtime_unavailable() -> None:
    def fail_runtime() -> RocketPyRuntime:
        raise UnsupportedBackendError("RocketPy backend unavailable: install astro-suite[launch]")

    with pytest.raises(UnsupportedBackendError, match=r"install astro-suite\[launch\]"):
        propagate_launch_rocketpy(make_launch_scenario(), runtime_loader=fail_runtime)


def test_propagate_launch_rocketpy_requires_live_adapter_configuration() -> None:
    with pytest.raises(UnsupportedBackendError, match="requires backend-specific"):
        propagate_launch_rocketpy(make_launch_scenario(), runtime_loader=_fake_runtime)


def test_propagate_launch_rocketpy_returns_suite_product_with_fake_runner() -> None:
    scenario = make_launch_scenario()
    seen_runtime: list[RocketPyRuntime] = []

    def fake_runner(candidate: LaunchScenario, runtime: RocketPyRuntime) -> LaunchTrajectory:
        assert candidate is scenario
        seen_runtime.append(runtime)
        return propagate_launch_local(candidate)

    trajectory = propagate_launch_rocketpy(
        scenario,
        runtime_loader=_fake_runtime,
        flight_runner=fake_runner,
    )

    assert len(seen_runtime) == 1
    assert trajectory.backend == "rocketpy"
    assert trajectory.metadata["adapter"] == "rocketpy"
    assert trajectory.metadata["rocketpy_version"] == "1.12.1"
    assert trajectory.metadata["source_backend"] == "local"
