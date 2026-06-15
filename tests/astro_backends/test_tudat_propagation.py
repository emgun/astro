from types import SimpleNamespace

import pytest

from astro_backends.tudat.propagation import propagate_tudat
from astro_backends.tudat.runtime import TudatRuntime
from astro_core.errors import UnsupportedBackendError
from astro_core.io import load_scenario
from astro_core.models import Scenario, Trajectory
from astro_dynamics.local import propagate_local


def _fake_runtime() -> TudatRuntime:
    return TudatRuntime(
        package="tudatpy",
        package_version="1.0.0",
        module=SimpleNamespace(),
    )


def test_propagate_tudat_reports_runtime_unavailable() -> None:
    scenario = load_scenario("examples/scenarios/leo_two_body.yaml")

    def fail_runtime() -> TudatRuntime:
        raise UnsupportedBackendError("Tudat backend unavailable: TudatPy is not installed")

    with pytest.raises(UnsupportedBackendError, match="TudatPy is not installed"):
        propagate_tudat(scenario, runtime_loader=fail_runtime)


def test_propagate_tudat_requires_validated_runner() -> None:
    scenario = load_scenario("examples/scenarios/leo_two_body.yaml")

    with pytest.raises(UnsupportedBackendError, match="requires a validated Tudat runner"):
        propagate_tudat(scenario, runtime_loader=_fake_runtime)


def test_propagate_tudat_returns_suite_product_with_fake_runner() -> None:
    scenario = load_scenario("examples/scenarios/leo_two_body.yaml")
    seen_runtime: list[TudatRuntime] = []

    def fake_runner(candidate: Scenario, runtime: TudatRuntime) -> Trajectory:
        assert candidate is scenario
        seen_runtime.append(runtime)
        return propagate_local(candidate)

    trajectory = propagate_tudat(
        scenario,
        runtime_loader=_fake_runtime,
        tudat_runner=fake_runner,
    )

    assert len(seen_runtime) == 1
    assert trajectory.backend == "tudat"
    assert trajectory.metadata["adapter"] == "tudat"
    assert trajectory.metadata["source_backend"] == "local"
    assert trajectory.metadata["tudat_version"] == "1.0.0"
