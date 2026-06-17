from types import SimpleNamespace
from typing import Any

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


def test_propagate_tudat_reports_missing_propagation_api() -> None:
    scenario = load_scenario("examples/scenarios/leo_two_body.yaml")

    with pytest.raises(UnsupportedBackendError, match="TudatPy propagation API import failed"):
        propagate_tudat(scenario, runtime_loader=_fake_runtime)


def test_propagate_tudat_default_runner_rejects_unsupported_force_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = load_scenario("examples/scenarios/leo_orekit_drag.yaml")
    fake_modules = _FakeTudatModules()
    monkeypatch.setattr(
        "astro_backends.tudat.propagation.import_module",
        fake_modules.import_module,
    )

    with pytest.raises(UnsupportedBackendError, match="does not yet support high-fidelity"):
        propagate_tudat(scenario, runtime_loader=_fake_runtime)


def test_propagate_tudat_runs_default_two_body_with_fake_tudat_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = load_scenario("examples/scenarios/leo_two_body.yaml")
    fake_modules = _FakeTudatModules()
    monkeypatch.setattr(
        "astro_backends.tudat.propagation.import_module",
        fake_modules.import_module,
    )

    trajectory = propagate_tudat(scenario, runtime_loader=_fake_runtime)

    assert fake_modules.spice.standard_kernels_loaded is True
    assert trajectory.backend == "tudat"
    assert trajectory.metadata["adapter"] == "tudat"
    assert trajectory.metadata["source_backend"] == "tudat_native"
    assert trajectory.metadata["tudat_acceleration_models"] == {
        "AstroSuiteSpacecraft": {"Earth": ["point_mass_gravity"]}
    }
    assert trajectory.metadata["tudat_force_models"] == ["Earth point-mass gravity"]
    assert trajectory.metadata["tudat_integrator"] == "runge_kutta_fixed_step_rk4"
    assert trajectory.metadata["tudat_propagator"] == "cowell"
    assert len(trajectory.samples) == scenario.propagation.sample_count
    assert trajectory.samples[0].state.position_km == pytest.approx((7000.0, 0.0, 0.0))
    assert trajectory.samples[1].state.position_km == pytest.approx((7000.0, 450.0, 60.0))


def test_propagate_tudat_runs_default_j2_with_fake_tudat_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = load_scenario("examples/scenarios/leo_j2.yaml")
    fake_modules = _FakeTudatModules()
    monkeypatch.setattr(
        "astro_backends.tudat.propagation.import_module",
        fake_modules.import_module,
    )

    trajectory = propagate_tudat(scenario, runtime_loader=_fake_runtime)

    assert trajectory.backend == "tudat"
    assert trajectory.force_model.gravity.value == "j2"
    assert trajectory.metadata["tudat_runner"] == "native_j2"
    assert trajectory.metadata["tudat_acceleration_models"] == {
        "AstroSuiteSpacecraft": {"Earth": ["spherical_harmonic_gravity_degree_2_order_0"]}
    }
    assert trajectory.metadata["tudat_force_models"] == ["Earth spherical harmonic gravity 2x0"]
    assert len(trajectory.samples) == scenario.propagation.sample_count
    assert trajectory.samples[1].state.position_km == pytest.approx((7000.0, 450.0, 60.0))


def test_propagate_tudat_runs_third_body_with_fake_tudat_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = load_scenario("examples/scenarios/leo_orekit_third_body.yaml")
    fake_modules = _FakeTudatModules()
    monkeypatch.setattr(
        "astro_backends.tudat.propagation.import_module",
        fake_modules.import_module,
    )

    trajectory = propagate_tudat(scenario, runtime_loader=_fake_runtime)

    assert fake_modules.environment_setup.requested_bodies == ["Earth", "Sun", "Moon"]
    assert trajectory.backend == "tudat"
    assert trajectory.metadata["tudat_runner"] == "native_j2_third_body"
    assert trajectory.metadata["tudat_acceleration_models"] == {
        "AstroSuiteSpacecraft": {
            "Earth": ["spherical_harmonic_gravity_degree_2_order_0"],
            "Sun": ["point_mass_gravity"],
            "Moon": ["point_mass_gravity"],
        }
    }
    assert trajectory.metadata["tudat_force_models"] == [
        "Earth spherical harmonic gravity 2x0",
        "Sun point-mass third-body gravity",
        "Moon point-mass third-body gravity",
    ]
    assert len(trajectory.samples) == scenario.propagation.sample_count


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


class _FakeDateTime:
    def __init__(
        self,
        year: int,
        month: int,
        day: int,
        hour: int,
        minute: int,
        second: float,
    ) -> None:
        from datetime import UTC, datetime

        whole_seconds = int(second)
        microsecond = int(round((second - whole_seconds) * 1_000_000))
        self._datetime = datetime(
            year,
            month,
            day,
            hour,
            minute,
            whole_seconds,
            microsecond,
            tzinfo=UTC,
        )

    def to_epoch(self) -> float:
        return self._datetime.timestamp()


class _FakeSpice:
    def __init__(self) -> None:
        self.standard_kernels_loaded = False

    def load_standard_kernels(self) -> None:
        self.standard_kernels_loaded = True


class _FakeBody:
    def __init__(self) -> None:
        self.mass: float | None = None
        self.gravitational_parameter = 398600.4418e9


class _FakeBodies:
    def __init__(self) -> None:
        self._bodies = {"Earth": _FakeBody()}

    def create_empty_body(self, name: str) -> None:
        self._bodies[name] = _FakeBody()

    def get(self, name: str) -> _FakeBody:
        return self._bodies[name]


class _FakeEnvironmentSetup:
    requested_bodies: list[str] = []

    @classmethod
    def get_default_body_settings(
        cls,
        bodies_to_create: list[str],
        global_frame_origin: str,
        global_frame_orientation: str,
    ) -> dict[str, object]:
        cls.requested_bodies = bodies_to_create
        return {
            "bodies_to_create": bodies_to_create,
            "global_frame_origin": global_frame_origin,
            "global_frame_orientation": global_frame_orientation,
        }

    @staticmethod
    def create_system_of_bodies(_body_settings: dict[str, object]) -> _FakeBodies:
        return _FakeBodies()


class _FakeAcceleration:
    @staticmethod
    def point_mass_gravity() -> str:
        return "point_mass_gravity"

    @staticmethod
    def spherical_harmonic_gravity(degree: int, order: int) -> str:
        return f"spherical_harmonic_gravity_degree_{degree}_order_{order}"


class _FakeIntegrator:
    rk_4 = "rk_4"

    @staticmethod
    def runge_kutta_fixed_step(time_step: float, coefficient_set: str) -> dict[str, object]:
        return {
            "type": "runge_kutta_fixed_step",
            "time_step": time_step,
            "coefficient_set": coefficient_set,
        }


class _FakePropagator:
    cowell = "cowell"

    @staticmethod
    def time_termination(end_epoch: float) -> dict[str, object]:
        return {"end_epoch": end_epoch}

    @staticmethod
    def translational(
        central_bodies: list[str],
        acceleration_models: dict[str, object],
        bodies_to_propagate: list[str],
        initial_state: list[float],
        start_epoch: float,
        integrator_settings: dict[str, object],
        termination_settings: dict[str, object],
        *,
        propagator: str,
    ) -> dict[str, object]:
        return {
            "central_bodies": central_bodies,
            "acceleration_models": acceleration_models,
            "bodies_to_propagate": bodies_to_propagate,
            "initial_state": initial_state,
            "start_epoch": start_epoch,
            "integrator_settings": integrator_settings,
            "termination_settings": termination_settings,
            "propagator": propagator,
        }


class _FakePropagationSetup:
    acceleration = _FakeAcceleration
    integrator = _FakeIntegrator
    propagator = _FakePropagator

    @staticmethod
    def create_acceleration_models(
        _bodies: _FakeBodies,
        acceleration_settings: dict[str, object],
        _bodies_to_propagate: list[str],
        _central_bodies: list[str],
    ) -> dict[str, object]:
        return acceleration_settings


class _FakeDynamicsSimulator:
    def __init__(self, propagator_settings: dict[str, object]) -> None:
        initial_state = propagator_settings["initial_state"]
        if not isinstance(initial_state, list):
            raise AssertionError("initial_state should be a list")
        start_epoch = float(propagator_settings["start_epoch"])
        end_epoch = float(propagator_settings["termination_settings"]["end_epoch"])
        step_s = float(propagator_settings["integrator_settings"]["time_step"])
        self.state_history: dict[float, list[float]] = {}
        sample_count = int(round((end_epoch - start_epoch) / step_s)) + 1
        for index in range(sample_count):
            elapsed_s = index * step_s
            epoch = start_epoch + elapsed_s
            self.state_history[epoch] = [
                float(initial_state[0]) + float(initial_state[3]) * elapsed_s,
                float(initial_state[1]) + float(initial_state[4]) * elapsed_s,
                float(initial_state[2]) + float(initial_state[5]) * elapsed_s,
                float(initial_state[3]),
                float(initial_state[4]),
                float(initial_state[5]),
            ]


class _FakeSimulator:
    @staticmethod
    def create_dynamics_simulator(
        _bodies: _FakeBodies,
        propagator_settings: dict[str, object],
    ) -> _FakeDynamicsSimulator:
        return _FakeDynamicsSimulator(propagator_settings)


class _FakeTudatModules:
    def __init__(self) -> None:
        self.spice = _FakeSpice()
        self.environment_setup = _FakeEnvironmentSetup
        self.modules: dict[str, Any] = {
            "tudatpy.interface.spice": self.spice,
            "tudatpy.dynamics.environment_setup": self.environment_setup,
            "tudatpy.dynamics.propagation_setup": _FakePropagationSetup,
            "tudatpy.dynamics.simulator": _FakeSimulator,
            "tudatpy.astro.time_representation": SimpleNamespace(DateTime=_FakeDateTime),
        }

    def import_module(self, module_name: str) -> Any:
        return self.modules[module_name]
