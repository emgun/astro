from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from importlib import import_module
from typing import Any

from astro_backends.tudat.runtime import TudatRuntime, load_tudat_runtime
from astro_core.errors import UnsupportedBackendError
from astro_core.models import CartesianState, ForceModelName, Scenario, Trajectory, TrajectorySample

TudatRuntimeLoader = Callable[[], TudatRuntime]
TudatPropagationRunner = Callable[[Scenario, TudatRuntime], Trajectory]
_SPACECRAFT_NAME = "AstroSuiteSpacecraft"
_SUPPORTED_DEFAULT_TUDAT_FORCE_MODELS = (
    ForceModelName.TWO_BODY,
    ForceModelName.J2,
    ForceModelName.OREKIT_HIGH_FIDELITY,
)
_SUPPORTED_TUDAT_HIGH_FIDELITY_FLAGS = {
    "atmospheric_drag",
    "solar_radiation_pressure",
    "third_body_gravity",
}


def _load_tudat_propagation_api() -> dict[str, Any]:
    try:
        return {
            "spice": import_module("tudatpy.interface.spice"),
            "environment_setup": import_module("tudatpy.dynamics.environment_setup"),
            "propagation_setup": import_module("tudatpy.dynamics.propagation_setup"),
            "simulator": import_module("tudatpy.dynamics.simulator"),
            "time_representation": import_module("tudatpy.astro.time_representation"),
        }
    except ImportError as exc:
        raise UnsupportedBackendError(f"TudatPy propagation API import failed: {exc}") from exc


def _validate_default_tudat_scenario(scenario: Scenario) -> None:
    if scenario.force_model.gravity not in _SUPPORTED_DEFAULT_TUDAT_FORCE_MODELS:
        supported = ", ".join(
            force_model.value for force_model in _SUPPORTED_DEFAULT_TUDAT_FORCE_MODELS
        )
        raise UnsupportedBackendError(
            "Default Tudat propagation currently supports only "
            f"{supported}"
        )
    enabled_flags = set(scenario.force_model.enabled_high_fidelity_flags())
    unsupported_flags = enabled_flags - _SUPPORTED_TUDAT_HIGH_FIDELITY_FLAGS
    if unsupported_flags:
        raise UnsupportedBackendError(
            "Default Tudat propagation does not yet support high-fidelity force flags: "
            f"{', '.join(sorted(unsupported_flags))}"
        )
    if scenario.maneuvers:
        raise UnsupportedBackendError("Default Tudat propagation does not yet support maneuvers")


def _tudat_body_names(scenario: Scenario) -> list[str]:
    body_names = ["Earth"]
    if scenario.force_model.solar_radiation_pressure or scenario.force_model.third_body_gravity:
        body_names.append("Sun")
    if scenario.force_model.third_body_gravity:
        body_names.append("Moon")
    return body_names


def _configure_tudat_spacecraft_environment(
    scenario: Scenario,
    bodies: Any,
    environment_setup: Any,
) -> None:
    if scenario.force_model.atmospheric_drag:
        aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(
            reference_area=scenario.spacecraft.area_m2,
            constant_force_coefficient=[scenario.spacecraft.drag_coefficient, 0.0, 0.0],
        )
        environment_setup.add_aerodynamic_coefficient_interface(
            bodies,
            _SPACECRAFT_NAME,
            aero_coefficient_settings,
        )
    if scenario.force_model.solar_radiation_pressure:
        radiation_pressure_settings = (
            environment_setup.radiation_pressure.cannonball_radiation_target(
                reference_area=scenario.spacecraft.area_m2,
                radiation_pressure_coefficient=scenario.spacecraft.reflectivity_coefficient,
                per_source_occulting_bodies={"Sun": ["Earth"]},
            )
        )
        environment_setup.add_radiation_pressure_target_model(
            bodies,
            _SPACECRAFT_NAME,
            radiation_pressure_settings,
        )


def _tudat_gravity_acceleration_settings(
    scenario: Scenario,
    propagation_setup: Any,
) -> tuple[dict[str, list[Any]], dict[str, dict[str, list[str]]], list[str], str]:
    enabled_suffixes: list[str] = []
    if scenario.force_model.atmospheric_drag:
        enabled_suffixes.append("drag")
    if scenario.force_model.solar_radiation_pressure:
        enabled_suffixes.append("srp")
    if scenario.force_model.third_body_gravity:
        enabled_suffixes.append("third_body")
    runner_suffix = f"_{'_'.join(enabled_suffixes)}" if enabled_suffixes else ""

    if scenario.force_model.gravity is ForceModelName.TWO_BODY:
        spacecraft_accelerations = {"Earth": [propagation_setup.acceleration.point_mass_gravity()]}
        acceleration_metadata = {_SPACECRAFT_NAME: {"Earth": ["point_mass_gravity"]}}
        force_model_metadata = ["Earth point-mass gravity"]
        runner_name = f"native_two_body{runner_suffix}"
    elif scenario.force_model.gravity in {ForceModelName.J2, ForceModelName.OREKIT_HIGH_FIDELITY}:
        spacecraft_accelerations = {
            "Earth": [propagation_setup.acceleration.spherical_harmonic_gravity(2, 0)]
        }
        acceleration_metadata = {
            _SPACECRAFT_NAME: {"Earth": ["spherical_harmonic_gravity_degree_2_order_0"]}
        }
        force_model_metadata = ["Earth spherical harmonic gravity 2x0"]
        runner_name = f"native_j2{runner_suffix}"
    else:
        raise UnsupportedBackendError(
            f"Default Tudat propagation does not support {scenario.force_model.gravity.value}"
        )

    if scenario.force_model.atmospheric_drag:
        spacecraft_accelerations["Earth"].append(propagation_setup.acceleration.aerodynamic())
        acceleration_metadata[_SPACECRAFT_NAME]["Earth"].append("aerodynamic")
        force_model_metadata.append("Earth aerodynamic drag")

    if scenario.force_model.solar_radiation_pressure:
        spacecraft_accelerations["Sun"] = [propagation_setup.acceleration.radiation_pressure()]
        acceleration_metadata[_SPACECRAFT_NAME]["Sun"] = ["radiation_pressure"]
        force_model_metadata.append("Sun cannonball solar radiation pressure")

    if scenario.force_model.third_body_gravity:
        for body_name in ("Sun", "Moon"):
            spacecraft_accelerations.setdefault(body_name, []).append(
                propagation_setup.acceleration.point_mass_gravity()
            )
            acceleration_metadata[_SPACECRAFT_NAME].setdefault(body_name, []).append(
                "point_mass_gravity"
            )
            force_model_metadata.append(f"{body_name} point-mass third-body gravity")

    return spacecraft_accelerations, acceleration_metadata, force_model_metadata, runner_name


def _tudat_epoch(epoch: datetime, date_time: Any) -> float:
    epoch_utc = epoch.astimezone(UTC)
    seconds = epoch_utc.second + epoch_utc.microsecond / 1_000_000.0
    return float(
        date_time.DateTime(
            epoch_utc.year,
            epoch_utc.month,
            epoch_utc.day,
            epoch_utc.hour,
            epoch_utc.minute,
            seconds,
        ).to_epoch()
    )


def _initial_state_si(scenario: Scenario) -> list[float]:
    cartesian = scenario.initial_state.cartesian
    return [
        *(1000.0 * float(component) for component in cartesian.position_km),
        *(1000.0 * float(component) for component in cartesian.velocity_km_s),
    ]


def _trajectory_samples_from_tudat_history(
    scenario: Scenario,
    state_history: dict[float, list[float]],
    start_epoch_tudat: float,
) -> list[TrajectorySample]:
    samples: list[TrajectorySample] = []
    for step_index in range(scenario.propagation.sample_count):
        elapsed_s = step_index * scenario.propagation.step_s
        expected_tudat_epoch = start_epoch_tudat + elapsed_s
        if expected_tudat_epoch in state_history:
            state_vector = state_history[expected_tudat_epoch]
        else:
            closest_epoch = min(
                state_history,
                key=lambda candidate_epoch: abs(candidate_epoch - expected_tudat_epoch),
            )
            state_vector = state_history[closest_epoch]
        samples.append(
            TrajectorySample(
                epoch=scenario.initial_state.epoch + timedelta(seconds=elapsed_s),
                state=CartesianState(
                    position_km=(
                        float(state_vector[0]) / 1000.0,
                        float(state_vector[1]) / 1000.0,
                        float(state_vector[2]) / 1000.0,
                    ),
                    velocity_km_s=(
                        float(state_vector[3]) / 1000.0,
                        float(state_vector[4]) / 1000.0,
                        float(state_vector[5]) / 1000.0,
                    ),
                ),
            )
        )
    return samples


def _propagate_tudat_default(scenario: Scenario, runtime: TudatRuntime) -> Trajectory:
    _validate_default_tudat_scenario(scenario)
    api = _load_tudat_propagation_api()
    spice = api["spice"]
    environment_setup = api["environment_setup"]
    propagation_setup = api["propagation_setup"]
    simulator = api["simulator"]
    time_representation = api["time_representation"]

    spice.load_standard_kernels()
    global_frame_origin = "Earth"
    global_frame_orientation = "J2000"
    body_settings = environment_setup.get_default_body_settings(
        _tudat_body_names(scenario),
        global_frame_origin,
        global_frame_orientation,
    )
    bodies = environment_setup.create_system_of_bodies(body_settings)
    bodies.create_empty_body(_SPACECRAFT_NAME)
    bodies.get(_SPACECRAFT_NAME).mass = scenario.spacecraft.mass_kg
    _configure_tudat_spacecraft_environment(scenario, bodies, environment_setup)

    bodies_to_propagate = [_SPACECRAFT_NAME]
    central_bodies = ["Earth"]
    spacecraft_accelerations, acceleration_metadata, force_model_metadata, runner_name = (
        _tudat_gravity_acceleration_settings(scenario, propagation_setup)
    )
    acceleration_settings = {_SPACECRAFT_NAME: spacecraft_accelerations}
    acceleration_models = propagation_setup.create_acceleration_models(
        bodies,
        acceleration_settings,
        bodies_to_propagate,
        central_bodies,
    )
    start_epoch = _tudat_epoch(scenario.initial_state.epoch, time_representation)
    end_epoch = _tudat_epoch(
        scenario.initial_state.epoch + timedelta(seconds=scenario.propagation.duration_s),
        time_representation,
    )
    integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
        scenario.propagation.step_s,
        propagation_setup.integrator.rk_4,
    )
    termination_settings = propagation_setup.propagator.time_termination(end_epoch)
    propagator_settings = propagation_setup.propagator.translational(
        central_bodies,
        acceleration_models,
        bodies_to_propagate,
        _initial_state_si(scenario),
        start_epoch,
        integrator_settings,
        termination_settings,
        propagator=propagation_setup.propagator.cowell,
    )
    dynamics_simulator = simulator.create_dynamics_simulator(bodies, propagator_settings)
    samples = _trajectory_samples_from_tudat_history(
        scenario,
        dynamics_simulator.state_history,
        start_epoch,
    )
    return Trajectory(
        scenario_id=scenario.scenario_id,
        samples=samples,
        force_model=scenario.force_model,
        backend="tudat_native",
        metadata={
            "tudat_runner": runner_name,
            "tudat_version": runtime.package_version,
            "tudat_global_frame_origin": global_frame_origin,
            "tudat_global_frame_orientation": global_frame_orientation,
            "tudat_acceleration_models": acceleration_metadata,
            "tudat_force_models": force_model_metadata,
            "tudat_integrator": "runge_kutta_fixed_step_rk4",
            "tudat_step_s": scenario.propagation.step_s,
            "tudat_propagator": "cowell",
            "units": "suite km/km_s converted to Tudat SI m/m_s",
        },
    )


def _with_tudat_provenance(
    trajectory: Trajectory,
    runtime: TudatRuntime,
) -> Trajectory:
    metadata = {
        **trajectory.metadata,
        "adapter": "tudat",
        "source_backend": trajectory.backend,
        "tudat_version": runtime.package_version,
    }
    return trajectory.model_copy(update={"backend": "tudat", "metadata": metadata})


def propagate_tudat(
    scenario: Scenario,
    *,
    runtime_loader: TudatRuntimeLoader = load_tudat_runtime,
    tudat_runner: TudatPropagationRunner | None = None,
) -> Trajectory:
    runtime = runtime_loader()
    if tudat_runner is None:
        return _with_tudat_provenance(_propagate_tudat_default(scenario, runtime), runtime)

    trajectory = tudat_runner(scenario, runtime)
    return _with_tudat_provenance(trajectory, runtime)
