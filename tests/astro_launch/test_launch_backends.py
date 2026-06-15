import pytest

from astro_core.errors import UnsupportedBackendError
from astro_launch.backends import propagate_launch_with_backend
from astro_launch.local import propagate_launch_local
from astro_launch.models import LaunchScenario, LaunchTrajectory
from tests.astro_launch.helpers import make_launch_scenario


def test_propagate_launch_with_backend_runs_local_reference() -> None:
    scenario = make_launch_scenario()

    trajectory = propagate_launch_with_backend(scenario, "local")

    assert trajectory.backend == "local"
    assert trajectory.model_dump(mode="json") == propagate_launch_local(scenario).model_dump(
        mode="json"
    )


def test_propagate_launch_with_backend_dispatches_rocketpy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = make_launch_scenario()

    def fake_rocketpy(candidate: LaunchScenario) -> LaunchTrajectory:
        assert candidate is scenario
        return propagate_launch_local(candidate).model_copy(update={"backend": "rocketpy"})

    monkeypatch.setattr("astro_launch.backends.propagate_launch_rocketpy", fake_rocketpy)

    trajectory = propagate_launch_with_backend(scenario, "rocketpy")

    assert trajectory.backend == "rocketpy"


def test_propagate_launch_with_backend_rejects_unknown_backend() -> None:
    with pytest.raises(UnsupportedBackendError, match="unsupported launch backend"):
        propagate_launch_with_backend(make_launch_scenario(), "missing")
