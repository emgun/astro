import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from astro_backends.orekit.conversion import km_s_to_m_s, km_to_m, m_s_to_km_s, m_to_km
from astro_backends.orekit.propagation import propagate_orekit
from astro_backends.orekit.runtime import OrekitRuntime
from astro_core.errors import UnsupportedBackendError
from astro_core.io import load_scenario
from astro_dynamics.local import propagate_local


def test_orekit_unit_conversions_are_reversible() -> None:
    assert km_to_m(7.5) == 7500.0
    assert km_s_to_m_s(7.5) == 7500.0
    assert m_to_km(7500.0) == 7.5
    assert m_s_to_km_s(7500.0) == 7.5


def test_orekit_epoch_requires_utc() -> None:
    epoch = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    assert epoch.isoformat() == "2026-01-01T00:00:00+00:00"


def test_propagate_orekit_reports_runtime_unavailable() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))

    def fail_loader() -> OrekitRuntime:
        raise UnsupportedBackendError("Orekit backend unavailable: install astro-suite[orekit]")

    with pytest.raises(UnsupportedBackendError, match=r"install astro-suite\[orekit\]"):
        propagate_orekit(scenario, runtime_loader=fail_loader)


def test_propagate_orekit_returns_suite_trajectory_with_fake_runtime() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))

    trajectory = propagate_orekit(scenario, runtime_loader=_fake_runtime)

    assert trajectory.scenario_id == "leo-two-body"
    assert trajectory.backend == "orekit"
    assert len(trajectory.samples) == scenario.propagation.sample_count
    assert trajectory.metadata == {
        "wrapper": "orekit_jpype",
        "wrapper_version": "13.1.0",
        "propagator": "KeplerianPropagator",
        "frame": "EME2000",
        "units": "suite km/km_s converted to Orekit m/m_s",
    }
    final_state = trajectory.samples[-1].state
    assert final_state.position_km == pytest.approx((7000.0, 4500.0, 600.0))
    assert final_state.velocity_km_s == pytest.approx((0.0, 7.5, 1.0))


@pytest.mark.orekit_live
def test_live_orekit_two_body_matches_local_reference() -> None:
    if os.environ.get("ASTRO_RUN_OREKIT_LIVE") != "1":
        pytest.skip("set ASTRO_RUN_OREKIT_LIVE=1 to run live Orekit propagation")
    pytest.importorskip("orekit_jpype")

    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))
    orekit_trajectory = propagate_orekit(scenario)
    local_trajectory = propagate_local(scenario)

    assert orekit_trajectory.backend == "orekit"
    assert len(orekit_trajectory.samples) == len(local_trajectory.samples)
    final_orekit = orekit_trajectory.samples[-1].state
    final_local = local_trajectory.samples[-1].state
    assert final_orekit.position_km == pytest.approx(final_local.position_km, abs=1.0)
    assert final_orekit.velocity_km_s == pytest.approx(final_local.velocity_km_s, abs=1.0e-3)


class _FakeFramesFactory:
    @staticmethod
    def getEME2000() -> str:
        return "EME2000"


class _FakeTimeScalesFactory:
    @staticmethod
    def getUTC() -> str:
        return "UTC"


class _FakeAbsoluteDate:
    def __init__(
        self,
        year: int,
        month: int,
        day: int,
        hour: int,
        minute: int,
        seconds: float,
        _time_scale: str,
    ) -> None:
        whole_seconds = int(seconds)
        microsecond = int(round((seconds - whole_seconds) * 1_000_000))
        self.epoch = datetime(
            year,
            month,
            day,
            hour,
            minute,
            whole_seconds,
            microsecond,
            tzinfo=timezone.utc,
        )


class _FakeVector3D:
    def __init__(self, x: float, y: float, z: float) -> None:
        self._x = x
        self._y = y
        self._z = z

    def getX(self) -> float:
        return self._x

    def getY(self) -> float:
        return self._y

    def getZ(self) -> float:
        return self._z


class _FakePVCoordinates:
    def __init__(self, position: _FakeVector3D, velocity: _FakeVector3D) -> None:
        self._position = position
        self._velocity = velocity

    def getPosition(self) -> _FakeVector3D:
        return self._position

    def getVelocity(self) -> _FakeVector3D:
        return self._velocity


class _FakeCartesianOrbit:
    def __init__(
        self,
        pv_coordinates: _FakePVCoordinates,
        frame: str,
        date: _FakeAbsoluteDate,
        mu: float,
    ) -> None:
        self.pv_coordinates = pv_coordinates
        self.frame = frame
        self.date = date
        self.mu = mu


class _FakeSpacecraftState:
    def __init__(self, pv_coordinates: _FakePVCoordinates) -> None:
        self._pv_coordinates = pv_coordinates

    def getPVCoordinates(self, _frame: str) -> _FakePVCoordinates:
        return self._pv_coordinates


class _FakeKeplerianPropagator:
    def __init__(self, orbit: _FakeCartesianOrbit) -> None:
        self._orbit = orbit

    def propagate(self, target_date: _FakeAbsoluteDate) -> _FakeSpacecraftState:
        dt_s = (target_date.epoch - self._orbit.date.epoch).total_seconds()
        initial_position = self._orbit.pv_coordinates.getPosition()
        initial_velocity = self._orbit.pv_coordinates.getVelocity()
        propagated_position = _FakeVector3D(
            initial_position.getX() + initial_velocity.getX() * dt_s,
            initial_position.getY() + initial_velocity.getY() * dt_s,
            initial_position.getZ() + initial_velocity.getZ() * dt_s,
        )
        return _FakeSpacecraftState(
            _FakePVCoordinates(
                propagated_position,
                initial_velocity,
            )
        )


def _fake_runtime() -> OrekitRuntime:
    return OrekitRuntime(
        wrapper="orekit_jpype",
        wrapper_version="13.1.0",
        frames_factory=_FakeFramesFactory,
        time_scales_factory=_FakeTimeScalesFactory,
        absolute_date=_FakeAbsoluteDate,
        vector3d=_FakeVector3D,
        pv_coordinates=_FakePVCoordinates,
        cartesian_orbit=_FakeCartesianOrbit,
        keplerian_propagator=_FakeKeplerianPropagator,
    )
