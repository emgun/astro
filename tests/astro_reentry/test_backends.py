import pytest

from astro_core.errors import UnsupportedBackendError
from astro_reentry.backends import simulate_reentry_with_backend
from astro_reentry.simulation import simulate_reentry_local
from tests.astro_reentry.helpers import make_reentry_scenario


def test_reentry_backend_dispatch_returns_suite_owned_local_result() -> None:
    scenario = make_reentry_scenario()

    result = simulate_reentry_with_backend(scenario, "local")

    assert result == simulate_reentry_local(scenario)
    assert result.backend == "local"
    assert result.metadata["dynamics_model"] == "spherical_earth_3dof_point_mass"


def test_reentry_backend_dispatch_rejects_unregistered_engine() -> None:
    with pytest.raises(UnsupportedBackendError, match="unsupported reentry backend: external"):
        simulate_reentry_with_backend(make_reentry_scenario(), "external")
