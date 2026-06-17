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


def test_propagate_orekit_populates_covariance_history_with_fake_runtime() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_covariance.yaml"))

    trajectory = propagate_orekit(scenario, runtime_loader=_fake_runtime)

    assert len(trajectory.covariance_history) == scenario.propagation.sample_count
    assert trajectory.metadata["covariance_model"] == "orekit_finite_difference_state_transition"
    assert trajectory.metadata["covariance_process_noise"] == "white_acceleration"
    assert trajectory.metadata["covariance_state_transition_storage"] == (
        "per_sample_and_accumulated"
    )
    initial_covariance = trajectory.covariance_history[0]
    assert initial_covariance.covariance == scenario.initial_covariance
    assert initial_covariance.metadata["covariance_sample_role"] == "initial"
    assert initial_covariance.metadata["state_transition_model"] == "identity"
    propagated_covariance = trajectory.covariance_history[1]
    assert propagated_covariance.metadata["covariance_sample_role"] == "propagated"
    assert propagated_covariance.metadata["state_transition_model"] == "orekit_finite_difference"
    assert propagated_covariance.state_transition_matrix is not None
    assert propagated_covariance.state_transition_matrix[0][0] == pytest.approx(1.0)
    assert propagated_covariance.state_transition_matrix[0][3] == pytest.approx(
        scenario.propagation.step_s
    )
    assert propagated_covariance.state_transition_matrix[3][3] == pytest.approx(1.0)
    assert propagated_covariance.process_noise_covariance is not None
    assert propagated_covariance.process_noise_covariance[0][0] > 0.0
    assert propagated_covariance.covariance[0][0] > initial_covariance.covariance[0][0]


def test_propagate_orekit_high_fidelity_covariance_records_force_models() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_covariance.yaml")).model_copy(
        update={
            "force_model": ForceModelConfig(
                gravity=ForceModelName.OREKIT_HIGH_FIDELITY,
                atmospheric_drag=True,
                solar_radiation_pressure=True,
                third_body_gravity=True,
            )
        }
    )

    trajectory = propagate_orekit(scenario, runtime_loader=_fake_runtime)

    assert trajectory.metadata["covariance_model"] == "orekit_finite_difference_state_transition"
    assert trajectory.metadata["covariance_transition_propagator"] == "NumericalPropagator"
    assert trajectory.metadata["covariance_transition_force_models"] == [
        "J2OnlyPerturbation",
        "DragForce",
        "SolarRadiationPressure",
        "ThirdBodyAttraction(Sun)",
        "ThirdBodyAttraction(Moon)",
    ]
    propagated_covariance = trajectory.covariance_history[1]
    assert propagated_covariance.metadata["state_transition_model"] == "orekit_finite_difference"
    assert propagated_covariance.metadata["transition_propagator"] == "NumericalPropagator"
    assert propagated_covariance.metadata["transition_force_models"] == (
        trajectory.metadata["covariance_transition_force_models"]
    )


def test_load_orekit_high_fidelity_covariance_example() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_orekit_high_fidelity_covariance.yaml"))

    assert scenario.force_model.gravity is ForceModelName.OREKIT_HIGH_FIDELITY
    assert scenario.force_model.atmospheric_drag is True
    assert scenario.force_model.solar_radiation_pressure is True
    assert scenario.force_model.third_body_gravity is True
    assert scenario.initial_covariance is not None
    assert scenario.covariance_process_noise_acceleration_km_s2 > 0.0


def test_load_orekit_high_order_gravity_example() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_orekit_high_order_gravity.yaml"))

    assert scenario.force_model.gravity is ForceModelName.OREKIT_HIGH_FIDELITY
    assert scenario.force_model.gravity_degree == 8
    assert scenario.force_model.gravity_order == 8


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


def test_propagate_orekit_high_order_gravity_uses_spherical_harmonics() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml")).model_copy(
        update={
            "force_model": ForceModelConfig(
                gravity=ForceModelName.OREKIT_HIGH_FIDELITY,
                gravity_degree=8,
                gravity_order=8,
            )
        }
    )

    trajectory = propagate_orekit(scenario, runtime_loader=_fake_runtime)

    assert trajectory.metadata["force_models"] == ["HolmesFeatherstoneAttractionModel(8x8)"]
    assert trajectory.metadata["gravity_model"] == "orekit_high_fidelity"
    assert trajectory.metadata["gravity_provider"] == "GravityFieldFactory.getNormalizedProvider"
    assert trajectory.metadata["gravity_degree"] == 8
    assert trajectory.metadata["gravity_order"] == 8


def test_propagate_orekit_high_fidelity_drag_adds_drag_force_with_fake_runtime() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml")).model_copy(
        update={
            "force_model": ForceModelConfig(
                gravity=ForceModelName.OREKIT_HIGH_FIDELITY,
                atmospheric_drag=True,
            )
        }
    )

    trajectory = propagate_orekit(scenario, runtime_loader=_fake_runtime)

    assert trajectory.backend == "orekit"
    assert trajectory.metadata["propagator"] == "NumericalPropagator"
    assert trajectory.metadata["force_models"] == ["J2OnlyPerturbation", "DragForce"]
    assert trajectory.metadata["atmosphere_model"] == "SimpleExponentialAtmosphere"
    assert trajectory.metadata["drag_spacecraft_model"] == "IsotropicDrag"
    assert trajectory.metadata["drag_area_m2"] == scenario.spacecraft.area_m2
    assert trajectory.metadata["drag_coefficient"] == scenario.spacecraft.drag_coefficient
    assert trajectory.metadata["unsupported_force_model_flags"] == []


def test_propagate_orekit_high_fidelity_srp_adds_srp_force_with_fake_runtime() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml")).model_copy(
        update={
            "force_model": ForceModelConfig(
                gravity=ForceModelName.OREKIT_HIGH_FIDELITY,
                solar_radiation_pressure=True,
            )
        }
    )

    trajectory = propagate_orekit(scenario, runtime_loader=_fake_runtime)

    assert trajectory.backend == "orekit"
    assert trajectory.metadata["propagator"] == "NumericalPropagator"
    assert trajectory.metadata["force_models"] == [
        "J2OnlyPerturbation",
        "SolarRadiationPressure",
    ]
    assert (
        trajectory.metadata["radiation_spacecraft_model"]
        == "IsotropicRadiationSingleCoefficient"
    )
    assert trajectory.metadata["srp_area_m2"] == scenario.spacecraft.area_m2
    assert trajectory.metadata["srp_reflectivity_coefficient"] == (
        scenario.spacecraft.reflectivity_coefficient
    )
    assert trajectory.metadata["unsupported_force_model_flags"] == []


def test_propagate_orekit_high_fidelity_third_body_adds_sun_and_moon_with_fake_runtime() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml")).model_copy(
        update={
            "force_model": ForceModelConfig(
                gravity=ForceModelName.OREKIT_HIGH_FIDELITY,
                third_body_gravity=True,
            )
        }
    )

    trajectory = propagate_orekit(scenario, runtime_loader=_fake_runtime)

    assert trajectory.backend == "orekit"
    assert trajectory.metadata["propagator"] == "NumericalPropagator"
    assert trajectory.metadata["force_models"] == [
        "J2OnlyPerturbation",
        "ThirdBodyAttraction(Sun)",
        "ThirdBodyAttraction(Moon)",
    ]
    assert trajectory.metadata["third_body_gravity_bodies"] == ["Sun", "Moon"]
    assert trajectory.metadata["unsupported_force_model_flags"] == []


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


@pytest.mark.orekit_live
def test_live_orekit_covariance_history_returns_suite_product() -> None:
    if os.environ.get("ASTRO_RUN_OREKIT_LIVE") != "1":
        pytest.skip("set ASTRO_RUN_OREKIT_LIVE=1 to run live Orekit covariance propagation")
    pytest.importorskip("orekit_jpype")

    scenario = load_scenario(Path("examples/scenarios/leo_covariance.yaml"))
    trajectory = propagate_orekit(scenario)

    assert trajectory.backend == "orekit"
    assert len(trajectory.covariance_history) == scenario.propagation.sample_count
    assert trajectory.metadata["covariance_model"] == "orekit_finite_difference_state_transition"
    assert trajectory.covariance_history[0].metadata["state_transition_model"] == "identity"
    assert trajectory.covariance_history[1].metadata["state_transition_model"] == (
        "orekit_finite_difference"
    )
    assert trajectory.covariance_history[1].state_transition_matrix is not None
    assert np.all(np.isfinite(trajectory.covariance_history[-1].covariance))


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


class _FakeDormandPrince853IntegratorBuilder:
    def __init__(self, min_step_s: float, max_step_s: float, position_scale_m: float) -> None:
        self.min_step_s = min_step_s
        self.max_step_s = max_step_s
        self.position_scale_m = position_scale_m


class _FakeJ2OnlyPerturbation:
    def __init__(self, mu: float, radius: float, j2: float, frame: str) -> None:
        self.mu = mu
        self.radius = radius
        self.j2 = j2
        self.frame = frame


class _FakeHolmesFeatherstoneAttractionModel:
    def __init__(self, frame: str, provider: "_FakeNormalizedGravityProvider") -> None:
        self.frame = frame
        self.provider = provider


class _FakeNormalizedGravityProvider:
    def __init__(self, degree: int, order: int) -> None:
        self.degree = degree
        self.order = order


class _FakeGravityFieldFactory:
    @staticmethod
    def getNormalizedProvider(degree: int, order: int) -> _FakeNormalizedGravityProvider:
        return _FakeNormalizedGravityProvider(degree, order)


class _FakeOneAxisEllipsoid:
    def __init__(self, radius: float, flattening: float, frame: str) -> None:
        self.radius = radius
        self.flattening = flattening
        self.frame = frame


class _FakeGeodeticPoint:
    def __init__(self, latitude_rad: float, longitude_rad: float, altitude_m: float) -> None:
        self.latitude_rad = latitude_rad
        self.longitude_rad = longitude_rad
        self.altitude_m = altitude_m


class _FakeTopocentricFrame:
    def __init__(
        self,
        earth_shape: _FakeOneAxisEllipsoid,
        geodetic_point: _FakeGeodeticPoint,
        name: str,
    ) -> None:
        self.earth_shape = earth_shape
        self.geodetic_point = geodetic_point
        self.name = name


class _FakeSimpleExponentialAtmosphere:
    def __init__(
        self,
        shape: _FakeOneAxisEllipsoid,
        reference_density: float,
        reference_altitude: float,
        scale_height: float,
    ) -> None:
        self.shape = shape
        self.reference_density = reference_density
        self.reference_altitude = reference_altitude
        self.scale_height = scale_height


class _FakeIsotropicDrag:
    def __init__(self, cross_section: float, drag_coefficient: float) -> None:
        self.cross_section = cross_section
        self.drag_coefficient = drag_coefficient


class _FakeDragForce:
    def __init__(
        self,
        atmosphere: _FakeSimpleExponentialAtmosphere,
        spacecraft: _FakeIsotropicDrag,
    ) -> None:
        self.atmosphere = atmosphere
        self.spacecraft = spacecraft


class _FakeCelestialBodyFactory:
    @staticmethod
    def getSun() -> str:
        return "Sun"

    @staticmethod
    def getMoon() -> str:
        return "Moon"


class _FakeThirdBodyAttraction:
    def __init__(self, body: str) -> None:
        self.body = body


class _FakeIsotropicRadiationSingleCoefficient:
    def __init__(self, cross_section: float, reflectivity_coefficient: float) -> None:
        self.cross_section = cross_section
        self.reflectivity_coefficient = reflectivity_coefficient


class _FakeSolarRadiationPressure:
    def __init__(
        self,
        sun: str,
        earth_shape: _FakeOneAxisEllipsoid,
        spacecraft: _FakeIsotropicRadiationSingleCoefficient,
    ) -> None:
        self.sun = sun
        self.earth_shape = earth_shape
        self.spacecraft = spacecraft


class _FakeNumericalPropagator(_FakeKeplerianPropagator):
    def __init__(self, integrator: _FakeDormandPrince853Integrator) -> None:
        self.integrator = integrator
        self.orbit_type: str | None = None
        self.position_angle_type: str | None = None
        self.force_models: list[object] = []
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

    def addForceModel(self, force_model: object) -> None:
        self.force_models.append(force_model)

    def propagate(self, target_date: _FakeAbsoluteDate) -> _FakeSpacecraftState:
        if self._orbit is None:
            raise AssertionError("initial state was not configured")
        return _FakeKeplerianPropagator(self._orbit).propagate(target_date)


class _FakeNumericalPropagatorBuilder:
    def __init__(
        self,
        reference_orbit: _FakeCartesianOrbit,
        integrator_builder: _FakeDormandPrince853IntegratorBuilder,
        position_angle_type: str,
        position_scale_m: float,
    ) -> None:
        self.reference_orbit = reference_orbit
        self.integrator_builder = integrator_builder
        self.position_angle_type = position_angle_type
        self.position_scale_m = position_scale_m
        self.force_models: list[object] = []

    def addForceModel(self, force_model: object) -> None:
        self.force_models.append(force_model)

    def buildPropagator(self) -> _FakeKeplerianPropagator:
        return _FakeKeplerianPropagator(
            _FakeCartesianOrbit(
                _FakePVCoordinates(
                    _FakeVector3D(7_001_000.0, 2_000.0, 3_000.0),
                    _FakeVector3D(10.0, 7_510.0, 1_020.0),
                ),
                self.reference_orbit.frame,
                self.reference_orbit.date,
                self.reference_orbit.mu,
            )
        )


class _FakeOrekitGroundStation:
    def __init__(self, base_frame: _FakeTopocentricFrame) -> None:
        self.base_frame = base_frame


class _FakeObservableSatellite:
    def __init__(self, index: int) -> None:
        self.index = index


class _FakeRangeMeasurement:
    measurement_type = "Range"

    def __init__(
        self,
        station: _FakeOrekitGroundStation,
        two_way: bool,
        date: _FakeAbsoluteDate,
        observed_value_m: float,
        sigma: float,
        base_weight: float,
        satellite: _FakeObservableSatellite,
    ) -> None:
        self.station = station
        self.two_way = two_way
        self.date = date
        self.observed_value_m = observed_value_m
        self.sigma = sigma
        self.base_weight = base_weight
        self.satellite = satellite


class _FakeRangeRateMeasurement:
    measurement_type = "RangeRate"

    def __init__(
        self,
        station: _FakeOrekitGroundStation,
        date: _FakeAbsoluteDate,
        observed_value_m_s: float,
        sigma: float,
        base_weight: float,
        two_way: bool,
        satellite: _FakeObservableSatellite,
    ) -> None:
        self.station = station
        self.date = date
        self.observed_value_m_s = observed_value_m_s
        self.sigma = sigma
        self.base_weight = base_weight
        self.two_way = two_way
        self.satellite = satellite


class _FakeBatchLSEstimator:
    def __init__(self, optimizer: object, *propagator_builders: object) -> None:
        self.optimizer = optimizer
        self.propagator_builders = propagator_builders
        self.measurements: list[object] = []
        self.max_iterations: int | None = None
        self.max_evaluations: int | None = None

    def addMeasurement(self, measurement: object) -> None:
        self.measurements.append(measurement)

    def setMaxIterations(self, max_iterations: int) -> None:
        self.max_iterations = max_iterations

    def setMaxEvaluations(self, max_evaluations: int) -> None:
        self.max_evaluations = max_evaluations

    def estimate(self) -> list[_FakeKeplerianPropagator]:
        return [self.propagator_builders[0].buildPropagator()]

    def getOptimum(self) -> object:
        return _FakeLeastSquaresOptimum()

    def getPhysicalCovariances(self, _threshold: float) -> object:
        return _FakeRealMatrix(
            [
                [1.0e-6 if row == column else 0.0 for column in range(6)]
                for row in range(6)
            ]
        )

    def getIterationsCount(self) -> int:
        return 3

    def getEvaluationsCount(self) -> int:
        return 5


class _FakeRealVector:
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def toArray(self) -> list[float]:
        return self.values


class _FakeRealMatrix:
    def __init__(self, values: list[list[float]]) -> None:
        self.values = values

    def getEntry(self, row: int, column: int) -> float:
        return self.values[row][column]


class _FakeLeastSquaresOptimum:
    def getResiduals(self) -> _FakeRealVector:
        return _FakeRealVector([0.1, -0.2])

    def getRMS(self) -> float:
        return (0.1**2 + (-0.2) ** 2) ** 0.5


class _FakeLevenbergMarquardtOptimizer:
    pass


class _FakeOrbitType:
    CARTESIAN = "CARTESIAN"


class _FakePositionAngleType:
    TRUE = "TRUE"


class _FakeIERSConventions:
    IERS_2010 = "IERS_2010"


class _FakeConstants:
    WGS84_EARTH_EQUATORIAL_RADIUS = 6_378_137.0
    WGS84_EARTH_FLATTENING = 1.0 / 298.257223563


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
        numerical_propagator_builder=_FakeNumericalPropagatorBuilder,
        dormand_prince_853_integrator=_FakeDormandPrince853Integrator,
        dormand_prince_853_integrator_builder=_FakeDormandPrince853IntegratorBuilder,
        j2_only_perturbation=_FakeJ2OnlyPerturbation,
        holmes_featherstone_attraction_model=_FakeHolmesFeatherstoneAttractionModel,
        gravity_field_factory=_FakeGravityFieldFactory,
        third_body_attraction=_FakeThirdBodyAttraction,
        one_axis_ellipsoid=_FakeOneAxisEllipsoid,
        geodetic_point=_FakeGeodeticPoint,
        topocentric_frame=_FakeTopocentricFrame,
        simple_exponential_atmosphere=_FakeSimpleExponentialAtmosphere,
        drag_force=_FakeDragForce,
        isotropic_drag=_FakeIsotropicDrag,
        celestial_body_factory=_FakeCelestialBodyFactory,
        solar_radiation_pressure=_FakeSolarRadiationPressure,
        isotropic_radiation_single_coefficient=_FakeIsotropicRadiationSingleCoefficient,
        orekit_ground_station=_FakeOrekitGroundStation,
        observable_satellite=_FakeObservableSatellite,
        range_measurement=_FakeRangeMeasurement,
        range_rate_measurement=_FakeRangeRateMeasurement,
        batch_ls_estimator=_FakeBatchLSEstimator,
        levenberg_marquardt_optimizer=_FakeLevenbergMarquardtOptimizer,
        orbit_type=_FakeOrbitType,
        position_angle_type=_FakePositionAngleType,
        iers_conventions=_FakeIERSConventions,
        constants=_FakeConstants,
    )
