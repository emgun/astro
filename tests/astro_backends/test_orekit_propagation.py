import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from astro_backends.orekit.conversion import km_s_to_m_s, km_to_m, m_s_to_km_s, m_to_km
from astro_backends.orekit.propagation import propagate_orekit
from astro_backends.orekit.runtime import OrekitRuntime
from astro_core.errors import UnsupportedBackendError
from astro_core.io import load_scenario
from astro_core.models import ForceModelConfig, ForceModelName
from astro_dynamics.local import propagate_local


def test_orekit_unit_conversions_are_reversible() -> None:
    assert km_to_m(7.5) == 7500.0
    assert km_s_to_m_s(7.5) == 7500.0
    assert m_to_km(7500.0) == 7.5
    assert m_s_to_km_s(7500.0) == 7.5


def test_orekit_epoch_requires_utc() -> None:
    epoch = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)

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
        "data_path": "fake-orekit-data.zip",
        "propagator": "KeplerianPropagator",
        "frame": "EME2000",
        "units": "suite km/km_s converted to Orekit m/m_s",
    }
    final_state = trajectory.samples[-1].state
    assert final_state.position_km == pytest.approx((7000.0, 4500.0, 600.0))
    assert final_state.velocity_km_s == pytest.approx((0.0, 7.5, 1.0))


def test_propagate_orekit_j2_uses_numerical_force_model_with_fake_runtime() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml")).model_copy(
        update={"force_model": ForceModelConfig(gravity=ForceModelName.J2)}
    )

    trajectory = propagate_orekit(scenario, runtime_loader=_fake_runtime)

    assert trajectory.scenario_id == "leo-two-body"
    assert trajectory.backend == "orekit"
    assert len(trajectory.samples) == scenario.propagation.sample_count
    assert trajectory.metadata == {
        "wrapper": "orekit_jpype",
        "wrapper_version": "13.1.0",
        "data_path": "fake-orekit-data.zip",
        "propagator": "NumericalPropagator",
        "frame": "EME2000",
        "j2_frame": "ITRF(IERS_2010, simple_eop=True)",
        "force_models": ["J2OnlyPerturbation"],
        "orbit_type": "CARTESIAN",
        "position_angle_type": "TRUE",
        "integrator": "DormandPrince853Integrator",
        "integrator_min_step_s": 0.001,
        "integrator_max_step_s": 60.0,
        "integrator_initial_step_s": 60.0,
        "integrator_position_tolerance_m": 0.001,
        "units": "suite km/km_s converted to Orekit m/m_s",
    }
    final_state = trajectory.samples[-1].state
    assert final_state.position_km == pytest.approx((7000.0, 4500.0, 600.0))
    assert final_state.velocity_km_s == pytest.approx((0.0, 7.5, 1.0))


def test_propagate_orekit_high_fidelity_uses_numerical_force_model_with_fake_runtime() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml")).model_copy(
        update={
            "force_model": ForceModelConfig(
                gravity=ForceModelName.OREKIT_HIGH_FIDELITY,
            )
        }
    )

    trajectory = propagate_orekit(scenario, runtime_loader=_fake_runtime)

    assert trajectory.backend == "orekit"
    assert trajectory.force_model.gravity is ForceModelName.OREKIT_HIGH_FIDELITY
    assert trajectory.metadata["propagator"] == "NumericalPropagator"
    assert trajectory.metadata["force_models"] == ["J2OnlyPerturbation"]
    assert trajectory.metadata["gravity_model"] == "orekit_high_fidelity"
    assert trajectory.metadata["unsupported_force_model_flags"] == []


@pytest.mark.parametrize(
    "force_model_update",
    [
        {"atmospheric_drag": True},
        {"solar_radiation_pressure": True},
        {"third_body_gravity": True},
    ],
)
def test_propagate_orekit_reports_unsupported_high_fidelity_flags(
    force_model_update: dict[str, bool],
) -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml")).model_copy(
        update={
            "force_model": ForceModelConfig(
                gravity=ForceModelName.OREKIT_HIGH_FIDELITY,
                **force_model_update,
            )
        }
    )
    flag_name = next(iter(force_model_update))

    with pytest.raises(UnsupportedBackendError, match=flag_name):
        propagate_orekit(scenario, runtime_loader=_fake_runtime)


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


@pytest.mark.orekit_live
def test_live_orekit_j2_matches_local_reference_scale() -> None:
    if os.environ.get("ASTRO_RUN_OREKIT_LIVE") != "1":
        pytest.skip("set ASTRO_RUN_OREKIT_LIVE=1 to run live Orekit propagation")
    pytest.importorskip("orekit_jpype")

    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml")).model_copy(
        update={"force_model": ForceModelConfig(gravity=ForceModelName.J2)}
    )
    orekit_trajectory = propagate_orekit(scenario)
    local_trajectory = propagate_local(scenario)

    assert orekit_trajectory.backend == "orekit"
    assert orekit_trajectory.metadata["propagator"] == "NumericalPropagator"
    assert orekit_trajectory.metadata["force_models"] == ["J2OnlyPerturbation"]
    final_orekit = orekit_trajectory.samples[-1].state
    final_local = local_trajectory.samples[-1].state
    position_delta_km = final_orekit.position_array() - final_local.position_array()
    velocity_delta_km_s = final_orekit.velocity_array() - final_local.velocity_array()
    assert np.linalg.norm(position_delta_km) < 0.05
    assert np.linalg.norm(velocity_delta_km_s) < 1.0e-4


class _FakeFramesFactory:
    @staticmethod
    def getEME2000() -> str:
        return "EME2000"

    @staticmethod
    def getITRF(_conventions: str, _simple_eop: bool) -> str:
        return "ITRF"


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
            tzinfo=UTC,
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
        self._orbit: _FakeCartesianOrbit | None = None
        if isinstance(pv_coordinates, _FakeCartesianOrbit):
            self._orbit = pv_coordinates
            self._pv_coordinates = pv_coordinates.pv_coordinates
        else:
            self._pv_coordinates = pv_coordinates

    def getPVCoordinates(self, _frame: str) -> _FakePVCoordinates:
        return self._pv_coordinates

    @property
    def orbit(self) -> _FakeCartesianOrbit | None:
        return self._orbit


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


class _FakeDormandPrince853Integrator:
    def __init__(
        self,
        min_step_s: float,
        max_step_s: float,
        absolute_tolerances: list[float],
        relative_tolerances: list[float],
    ) -> None:
        self.min_step_s = min_step_s
        self.max_step_s = max_step_s
        self.absolute_tolerances = absolute_tolerances
        self.relative_tolerances = relative_tolerances
        self.initial_step_s: float | None = None

    def setInitialStepSize(self, initial_step_s: float) -> None:
        self.initial_step_s = initial_step_s


class _FakeJ2OnlyPerturbation:
    def __init__(self, mu: float, radius: float, j2: float, frame: str) -> None:
        self.mu = mu
        self.radius = radius
        self.j2 = j2
        self.frame = frame


class _FakeNumericalPropagator(_FakeKeplerianPropagator):
    def __init__(self, integrator: _FakeDormandPrince853Integrator) -> None:
        self.integrator = integrator
        self.orbit_type: str | None = None
        self.position_angle_type: str | None = None
        self.force_models: list[_FakeJ2OnlyPerturbation] = []
        self._orbit: _FakeCartesianOrbit | None = None

    @staticmethod
    def tolerances(
        _position_tolerance_m: float,
        _orbit: _FakeCartesianOrbit,
        _orbit_type: str,
    ) -> list[list[float]]:
        return [[1.0e-3] * 7, [1.0e-9] * 7]

    def setOrbitType(self, orbit_type: str) -> None:
        self.orbit_type = orbit_type

    def setPositionAngleType(self, position_angle_type: str) -> None:
        self.position_angle_type = position_angle_type

    def setInitialState(self, initial_state: _FakeSpacecraftState) -> None:
        self._orbit = initial_state.orbit

    def addForceModel(self, force_model: _FakeJ2OnlyPerturbation) -> None:
        self.force_models.append(force_model)

    def propagate(self, target_date: _FakeAbsoluteDate) -> _FakeSpacecraftState:
        if self._orbit is None:
            raise AssertionError("initial state was not configured")
        return _FakeKeplerianPropagator(self._orbit).propagate(target_date)


class _FakeOrbitType:
    CARTESIAN = "CARTESIAN"


class _FakePositionAngleType:
    TRUE = "TRUE"


class _FakeIERSConventions:
    IERS_2010 = "IERS_2010"


def _fake_runtime() -> OrekitRuntime:
    return OrekitRuntime(
        wrapper="orekit_jpype",
        wrapper_version="13.1.0",
        data_path="fake-orekit-data.zip",
        frames_factory=_FakeFramesFactory,
        time_scales_factory=_FakeTimeScalesFactory,
        absolute_date=_FakeAbsoluteDate,
        vector3d=_FakeVector3D,
        pv_coordinates=_FakePVCoordinates,
        cartesian_orbit=_FakeCartesianOrbit,
        keplerian_propagator=_FakeKeplerianPropagator,
        spacecraft_state=_FakeSpacecraftState,
        numerical_propagator=_FakeNumericalPropagator,
        dormand_prince_853_integrator=_FakeDormandPrince853Integrator,
        j2_only_perturbation=_FakeJ2OnlyPerturbation,
        orbit_type=_FakeOrbitType,
        position_angle_type=_FakePositionAngleType,
        iers_conventions=_FakeIERSConventions,
    )
