from __future__ import annotations

from typing import Literal

from astro_backends.orekit import propagate_orekit
from astro_backends.tudat import propagate_tudat
from astro_core.errors import UnsupportedBackendError
from astro_core.models import Scenario, Trajectory
from astro_dynamics.local import propagate_local

PropagationBackend = Literal["local", "orekit", "tudat"]


def propagate_with_backend(scenario: Scenario, backend: str) -> Trajectory:
    if backend == "local":
        return propagate_local(scenario)
    if backend == "orekit":
        return propagate_orekit(scenario)
    if backend == "tudat":
        return propagate_tudat(scenario)
    raise UnsupportedBackendError(f"unsupported propagation backend: {backend}")
