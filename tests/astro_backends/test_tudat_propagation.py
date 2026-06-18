from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from astro_backends.tudat.propagation import propagate_tudat
from astro_backends.tudat.runtime import TudatRuntime
from astro_core.errors import UnsupportedBackendError
from astro_core.io import load_scenario
from astro_core.models import (
    CovarianceSample,
    ForceModelConfig,
    ForceModelName,
    Scenario,
    Trajectory,
)
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


def test_propagate_tudat_runs_drag_with_fake_tudat_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = load_scenario("examples/scenarios/leo_orekit_drag.yaml")
    fake_modules = _FakeTudatModules(include_variational_api=False)
    monkeypatch.setattr(
        "astro_backends.tudat.propagation.import_module",
        fake_modules.import_module,
    )

    trajectory = propagate_tudat(scenario, runtime_loader=_fake_runtime)

    assert fake_modules.environment_setup.aerodynamic_interfaces == {
        "AstroSuiteSpacecraft": {
            "reference_area": 2.5,
            "constant_force_coefficient": [2.2, 0.0, 0.0],
        }
    }
    assert trajectory.metadata["tudat_runner"] == "native_j2_drag"
    assert trajectory.metadata["tudat_acceleration_models"] == {
        "AstroSuiteSpacecraft": {
            "Earth": ["spherical_harmonic_gravity_degree_2_order_0", "aerodynamic"]
        }
    }
    assert trajectory.metadata["tudat_force_models"] == [
        "Earth spherical harmonic gravity 2x0",
        "Earth aerodynamic drag",
    ]


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


def test_load_tudat_high_order_gravity_example() -> None:
    scenario = load_scenario("examples/scenarios/leo_tudat_high_order_gravity.yaml")

    assert scenario.force_model.gravity is ForceModelName.OREKIT_HIGH_FIDELITY
    assert scenario.force_model.gravity_degree == 8
    assert scenario.force_model.gravity_order == 8
    assert scenario.force_model.atmospheric_drag is False
    assert scenario.force_model.solar_radiation_pressure is False
    assert scenario.force_model.third_body_gravity is False


def test_propagate_tudat_runs_high_order_gravity_with_fake_tudat_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_scenario = load_scenario("examples/scenarios/leo_orekit_high_fidelity.yaml")
    scenario = base_scenario.model_copy(
        update={
            "force_model": ForceModelConfig(
                gravity=ForceModelName.OREKIT_HIGH_FIDELITY,
                gravity_degree=8,
                gravity_order=8,
            )
        }
    )
    fake_modules = _FakeTudatModules()
    monkeypatch.setattr(
        "astro_backends.tudat.propagation.import_module",
        fake_modules.import_module,
    )

    trajectory = propagate_tudat(scenario, runtime_loader=_fake_runtime)

    assert trajectory.backend == "tudat"
    assert trajectory.metadata["tudat_runner"] == "native_spherical_harmonic_8x8"
    assert trajectory.metadata["tudat_acceleration_models"] == {
        "AstroSuiteSpacecraft": {"Earth": ["spherical_harmonic_gravity_degree_8_order_8"]}
    }
    assert trajectory.metadata["tudat_force_models"] == [
        "Earth spherical harmonic gravity 8x8"
    ]
    assert trajectory.metadata["tudat_gravity_degree"] == 8
    assert trajectory.metadata["tudat_gravity_order"] == 8


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


def test_propagate_tudat_runs_srp_with_fake_tudat_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = load_scenario("examples/scenarios/leo_orekit_srp.yaml")
    fake_modules = _FakeTudatModules()
    monkeypatch.setattr(
        "astro_backends.tudat.propagation.import_module",
        fake_modules.import_module,
    )

    trajectory = propagate_tudat(scenario, runtime_loader=_fake_runtime)

    assert fake_modules.environment_setup.requested_bodies == ["Earth", "Sun"]
    assert fake_modules.environment_setup.radiation_pressure_targets == {
        "AstroSuiteSpacecraft": {
            "reference_area": 2.5,
            "radiation_pressure_coefficient": 1.3,
            "per_source_occulting_bodies": {"Sun": ["Earth"]},
        }
    }
    assert trajectory.metadata["tudat_runner"] == "native_j2_srp"
    assert trajectory.metadata["tudat_acceleration_models"] == {
        "AstroSuiteSpacecraft": {
            "Earth": ["spherical_harmonic_gravity_degree_2_order_0"],
            "Sun": ["radiation_pressure"],
        }
    }
    assert trajectory.metadata["tudat_force_models"] == [
        "Earth spherical harmonic gravity 2x0",
        "Sun cannonball solar radiation pressure",
    ]


def test_propagate_tudat_populates_covariance_history_with_fake_tudat_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_scenario = load_scenario("examples/scenarios/leo_orekit_high_fidelity_covariance.yaml")
    scenario = base_scenario.model_copy(
        update={"propagation": base_scenario.propagation.model_copy(update={"duration_s": 120.0})}
    )
    fake_modules = _FakeTudatModules()
    monkeypatch.setattr(
        "astro_backends.tudat.propagation.import_module",
        fake_modules.import_module,
    )

    trajectory = propagate_tudat(scenario, runtime_loader=_fake_runtime)

    assert trajectory.backend == "tudat"
    assert len(trajectory.covariance_history) == scenario.propagation.sample_count
    assert trajectory.metadata["covariance_model"] == "tudat_finite_difference_state_transition"
    assert trajectory.metadata["covariance_transition_force_models"] == [
        "Earth spherical harmonic gravity 2x0",
        "Earth aerodynamic drag",
        "Sun cannonball solar radiation pressure",
        "Sun point-mass third-body gravity",
        "Moon point-mass third-body gravity",
    ]
    assert trajectory.covariance_history[0].metadata["state_transition_model"] == "identity"
    assert trajectory.covariance_history[1].metadata["state_transition_model"] == (
        "tudat_finite_difference"
    )
    assert trajectory.covariance_history[1].metadata["transition_step_s"] == (
        scenario.propagation.step_s
    )
    assert trajectory.covariance_history[1].metadata["transition_force_models"] == (
        trajectory.metadata["covariance_transition_force_models"]
    )
    final_covariance = np.array(trajectory.covariance_history[-1].covariance)
    final_transition = np.array(trajectory.covariance_history[-1].state_transition_matrix)
    final_process_noise = np.array(trajectory.covariance_history[-1].process_noise_covariance)
    np.testing.assert_allclose(final_covariance, final_covariance.T, rtol=0.0, atol=1.0e-10)
    assert final_transition.shape == (6, 6)
    assert not np.allclose(final_transition, np.eye(6))
    assert final_process_noise[0, 0] > 0.0


def test_propagate_tudat_requires_native_variational_runner_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_scenario = load_scenario("examples/scenarios/leo_orekit_high_fidelity_covariance.yaml")
    scenario = Scenario.model_validate(
        {
            **base_scenario.model_dump(mode="json"),
            "covariance_state_transition_model": "tudat_variational",
        }
    )
    fake_modules = _FakeTudatModules()
    monkeypatch.setattr(
        "astro_backends.tudat.propagation.import_module",
        fake_modules.import_module,
    )

    with pytest.raises(UnsupportedBackendError, match="TudatPy variational API import failed"):
        propagate_tudat(scenario, runtime_loader=_fake_runtime)


def test_propagate_tudat_uses_default_native_variational_runner_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_scenario = load_scenario("examples/scenarios/leo_orekit_high_fidelity_covariance.yaml")
    scenario = Scenario.model_validate(
        {
            **base_scenario.model_dump(mode="json"),
            "covariance_state_transition_model": "tudat_variational",
            "covariance_process_noise_acceleration_km_s2": 0.0,
            "propagation": {
                **base_scenario.propagation.model_dump(mode="json"),
                "duration_s": 120.0,
            },
        }
    )
    fake_modules = _FakeTudatModules(include_variational_api=True)
    monkeypatch.setattr(
        "astro_backends.tudat.propagation.import_module",
        fake_modules.import_module,
    )

    trajectory = propagate_tudat(scenario, runtime_loader=_fake_runtime)

    assert fake_modules.estimation_setup.parameter.initial_state_settings is not None
    assert fake_modules.estimation_setup.created_parameter_settings is not None
    assert fake_modules.simulator.variational_solver_created is True
    assert trajectory.metadata["covariance_model"] == "tudat_native_variational_equations"
    assert trajectory.metadata["covariance_native_variational_runner"] == "default_tudatpy"
    assert trajectory.metadata["native_variational_parameter_set"] == "initial_cartesian_state"
    assert trajectory.metadata["native_variational_solver"] == "create_variational_equations_solver"
    assert trajectory.covariance_history[0].metadata["state_transition_model"] == "identity"
    assert trajectory.covariance_history[1].metadata["state_transition_model"] == (
        "tudat_native_variational"
    )
    first_transition = np.array(trajectory.covariance_history[1].state_transition_matrix)
    assert first_transition[0, 0] == pytest.approx(1.01)


def test_propagate_tudat_uses_native_variational_runner_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_scenario = load_scenario("examples/scenarios/leo_orekit_high_fidelity_covariance.yaml")
    scenario = Scenario.model_validate(
        {
            **base_scenario.model_dump(mode="json"),
            "covariance_state_transition_model": "tudat_variational",
            "propagation": {
                **base_scenario.propagation.model_dump(mode="json"),
                "duration_s": 120.0,
            },
        }
    )
    fake_modules = _FakeTudatModules()
    monkeypatch.setattr(
        "astro_backends.tudat.propagation.import_module",
        fake_modules.import_module,
    )

    def fake_variational_runner(
        candidate: Scenario,
        runtime: TudatRuntime,
        trajectory: Trajectory,
    ) -> tuple[list[CovarianceSample], dict[str, object]]:
        assert candidate is scenario
        assert runtime.package_version == "1.0.0"
        covariance_history = []
        for sample_index, sample in enumerate(trajectory.samples):
            transition = np.eye(6) + sample_index * 0.01 * np.eye(6)
            covariance_history.append(
                CovarianceSample(
                    epoch=sample.epoch,
                    covariance=(transition @ np.eye(6) @ transition.T).tolist(),
                    state_transition_matrix=transition.tolist(),
                    accumulated_state_transition_matrix=transition.tolist(),
                    process_noise_covariance=np.zeros((6, 6)).tolist(),
                    metadata={
                        "state_transition_model": (
                            "identity" if sample_index == 0 else "tudat_native_variational"
                        ),
                    },
                )
            )
        return covariance_history, {"native_variational_parameter_set": "initial_cartesian_state"}

    trajectory = propagate_tudat(
        scenario,
        runtime_loader=_fake_runtime,
        tudat_variational_runner=fake_variational_runner,
    )

    assert trajectory.metadata["covariance_model"] == "tudat_native_variational_equations"
    assert trajectory.metadata["covariance_native_variational_runner"] == "provided"
    assert trajectory.metadata["native_variational_parameter_set"] == "initial_cartesian_state"
    assert trajectory.covariance_history[1].metadata["state_transition_model"] == (
        "tudat_native_variational"
    )


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
    aerodynamic_interfaces: dict[str, object] = {}
    radiation_pressure_targets: dict[str, object] = {}

    class aerodynamic_coefficients:
        @staticmethod
        def constant(
            *,
            reference_area: float,
            constant_force_coefficient: list[float],
        ) -> dict[str, object]:
            return {
                "reference_area": reference_area,
                "constant_force_coefficient": constant_force_coefficient,
            }

    class radiation_pressure:
        @staticmethod
        def cannonball_radiation_target(
            *,
            reference_area: float,
            radiation_pressure_coefficient: float,
            per_source_occulting_bodies: dict[str, list[str]],
        ) -> dict[str, object]:
            return {
                "reference_area": reference_area,
                "radiation_pressure_coefficient": radiation_pressure_coefficient,
                "per_source_occulting_bodies": per_source_occulting_bodies,
            }

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

    @classmethod
    def add_aerodynamic_coefficient_interface(
        cls,
        _bodies: _FakeBodies,
        body_name: str,
        aero_coefficient_settings: object,
    ) -> None:
        cls.aerodynamic_interfaces[body_name] = aero_coefficient_settings

    @classmethod
    def add_radiation_pressure_target_model(
        cls,
        _bodies: _FakeBodies,
        body_name: str,
        radiation_pressure_settings: object,
    ) -> None:
        cls.radiation_pressure_targets[body_name] = radiation_pressure_settings


class _FakeAcceleration:
    @staticmethod
    def point_mass_gravity() -> str:
        return "point_mass_gravity"

    @staticmethod
    def spherical_harmonic_gravity(degree: int, order: int) -> str:
        return f"spherical_harmonic_gravity_degree_{degree}_order_{order}"

    @staticmethod
    def aerodynamic() -> str:
        return "aerodynamic"

    @staticmethod
    def radiation_pressure() -> str:
        return "radiation_pressure"


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
        termination_settings = cast(
            dict[str, object],
            propagator_settings["termination_settings"],
        )
        integrator_settings = cast(
            dict[str, object],
            propagator_settings["integrator_settings"],
        )
        start_epoch = float(cast(float, propagator_settings["start_epoch"]))
        end_epoch = float(cast(float, termination_settings["end_epoch"]))
        step_s = float(cast(float, integrator_settings["time_step"]))
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
    variational_solver_created = False

    @classmethod
    def create_dynamics_simulator(
        cls,
        _bodies: _FakeBodies,
        propagator_settings: dict[str, object],
    ) -> _FakeDynamicsSimulator:
        return _FakeDynamicsSimulator(propagator_settings)

    @classmethod
    def create_variational_equations_solver(
        cls,
        _bodies: _FakeBodies,
        propagator_settings: dict[str, object],
        parameters_to_estimate: dict[str, object],
        *,
        simulate_dynamics_on_creation: bool,
    ) -> "_FakeVariationalEquationsSolver":
        cls.variational_solver_created = True
        return _FakeVariationalEquationsSolver(
            propagator_settings,
            parameters_to_estimate,
            simulate_dynamics_on_creation=simulate_dynamics_on_creation,
        )


class _FakeVariationalEquationsSolver:
    def __init__(
        self,
        propagator_settings: dict[str, object],
        parameters_to_estimate: dict[str, object],
        *,
        simulate_dynamics_on_creation: bool,
    ) -> None:
        parameter_settings = cast(
            dict[str, object],
            parameters_to_estimate["parameter_settings"],
        )
        if parameter_settings["type"] != "initial_states":
            raise AssertionError("expected initial state parameter settings")
        if not simulate_dynamics_on_creation:
            raise AssertionError("expected dynamics simulation on creation")
        termination_settings = cast(
            dict[str, object],
            propagator_settings["termination_settings"],
        )
        integrator_settings = cast(
            dict[str, object],
            propagator_settings["integrator_settings"],
        )
        start_epoch = float(cast(float, propagator_settings["start_epoch"]))
        end_epoch = float(cast(float, termination_settings["end_epoch"]))
        step_s = float(cast(float, integrator_settings["time_step"]))
        sample_count = int(round((end_epoch - start_epoch) / step_s)) + 1
        self.state_transition_matrix_history: dict[float, list[list[float]]] = {}
        for index in range(sample_count):
            transition = np.eye(6) + index * 0.01 * np.eye(6)
            self.state_transition_matrix_history[start_epoch + index * step_s] = (
                transition.tolist()
            )


class _FakeParameterSetup:
    initial_state_settings: dict[str, object] | None = None

    @classmethod
    def initial_states(
        cls,
        propagator_settings: dict[str, object],
        bodies: _FakeBodies,
    ) -> dict[str, object]:
        cls.initial_state_settings = {
            "type": "initial_states",
            "propagator": propagator_settings,
            "body_count": len(bodies._bodies),
        }
        return cls.initial_state_settings


class _FakeEstimationSetup:
    parameter = _FakeParameterSetup
    created_parameter_settings: dict[str, object] | None = None

    @classmethod
    def create_parameter_set(
        cls,
        parameter_settings: dict[str, object],
        bodies: _FakeBodies,
        propagator_settings: dict[str, object],
    ) -> dict[str, object]:
        cls.created_parameter_settings = parameter_settings
        return {
            "parameter_settings": parameter_settings,
            "body_count": len(bodies._bodies),
            "propagator": propagator_settings,
        }


class _FakeTudatModules:
    def __init__(self, *, include_variational_api: bool = False) -> None:
        self.spice = _FakeSpice()
        self.environment_setup = _FakeEnvironmentSetup
        self.environment_setup.requested_bodies = []
        self.environment_setup.aerodynamic_interfaces = {}
        self.environment_setup.radiation_pressure_targets = {}
        self.estimation_setup = _FakeEstimationSetup
        self.estimation_setup.parameter.initial_state_settings = None
        self.estimation_setup.created_parameter_settings = None
        self.simulator = _FakeSimulator
        self.simulator.variational_solver_created = False
        self.modules: dict[str, Any] = {
            "tudatpy.interface.spice": self.spice,
            "tudatpy.dynamics.environment_setup": self.environment_setup,
            "tudatpy.dynamics.propagation_setup": _FakePropagationSetup,
            "tudatpy.dynamics.simulator": self.simulator,
            "tudatpy.astro.time_representation": SimpleNamespace(DateTime=_FakeDateTime),
        }
        if include_variational_api:
            self.modules["tudatpy.dynamics.estimation_setup"] = self.estimation_setup

    def import_module(self, module_name: str) -> Any:
        return self.modules[module_name]
