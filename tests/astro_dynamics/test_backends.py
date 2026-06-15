from pathlib import Path

import pytest

from astro_core.errors import UnsupportedBackendError
from astro_core.io import load_scenario
from astro_dynamics.backends import propagate_with_backend
from astro_dynamics.local import propagate_local


def test_propagate_with_backend_dispatches_local() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))

    trajectory = propagate_with_backend(scenario, "local")

    assert trajectory.backend == "local"
    assert len(trajectory.samples) == scenario.propagation.sample_count


def test_propagate_with_backend_dispatches_orekit(monkeypatch: pytest.MonkeyPatch) -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))

    def fake_orekit() -> object:
        trajectory = propagate_local(scenario)
        return trajectory.model_copy(update={"backend": "orekit"})

    monkeypatch.setattr("astro_dynamics.backends.propagate_orekit", lambda _scenario: fake_orekit())

    trajectory = propagate_with_backend(scenario, "orekit")

    assert trajectory.backend == "orekit"


def test_propagate_with_backend_rejects_unknown_backend() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))

    with pytest.raises(UnsupportedBackendError, match="unsupported propagation backend"):
        propagate_with_backend(scenario, "bad")
