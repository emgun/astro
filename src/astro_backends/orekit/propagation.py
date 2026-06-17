from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from astro_backends.orekit.conversion import (
    absolute_date_from_datetime,
    km_s_to_m_s,
    km_to_m,
    m_s_to_km_s,
    m_to_km,
    validate_orekit_state_support,
)
from astro_backends.orekit.force_models import (
    build_atmospheric_drag_force_model,
    build_earth_shape,
    build_solar_radiation_pressure_force_model,
    build_third_body_gravity_force_models,
)
from astro_backends.orekit.runtime import OrekitRuntime, load_orekit_runtime
from astro_core.constants import J2_EARTH, MU_EARTH_KM3_S2, R_EARTH_KM
from astro_core.errors import UnsupportedBackendError
from astro_core.models import (
    CartesianState,
    CovarianceSample,
    ForceModelName,
    Scenario,
    Trajectory,
    TrajectorySample,
)

FloatArray = NDArray[np.float64]
RuntimeLoader = Callable[[], OrekitRuntime]
MU_EARTH_M3_S2 = MU_EARTH_KM3_S2 * 1.0e9
R_EARTH_M = R_EARTH_KM * 1.0e3
J2_POSITION_TOLERANCE_M = 0.001
NUMERICAL_MIN_STEP_S = 0.001
_COVARIANCE_FD_REL_STEP = 1.0e-6
_COVARIANCE_FD_ABS_STEP = 1.0e-8
_SUPPORTED_OREKIT_GRAVITY_MODELS = {
    ForceModelName.TWO_BODY,
    ForceModelName.J2,
    ForceModelName.OREKIT_HIGH_FIDELITY,
}


@dataclass(frozen=True)
class _OrekitPropagatorConfig:
    propagator: Any
    metadata: dict[str, object]


def _validate_orekit_scenario(scenario: Scenario) -> None:
    validate_orekit_state_support(scenario.initial_state)
    if scenario.force_model.gravity not in _SUPPORTED_OREKIT_GRAVITY_MODELS:
        raise UnsupportedBackendError(
            "Orekit propagation supports two_body, j2, and orekit_high_fidelity gravity"
        )
    if (
        scenario.force_model.gravity_degree is not None
        or scenario.force_model.gravity_order is not None
    ) and scenario.force_model.gravity is ForceModelName.OREKIT_HIGH_FIDELITY:
        raise UnsupportedBackendError(
            "Orekit propagation does not yet support configured high-order gravity; "
            "use the Tudat backend for spherical harmonic degree/order propagation"
        )


def _initial_orbit(scenario: Scenario, runtime: OrekitRuntime, frame: Any) -> Any:
    initial = scenario.initial_state.cartesian
    state_vector = np.array(
        [*initial.position_km, *initial.velocity_km_s],
        dtype=np.float64,
    )
    return _orbit_from_state_vector(
        scenario,
        runtime,
        frame,
        scenario.initial_state.epoch,
        state_vector,
    )


def _orbit_from_state_vector(
    scenario: Scenario,
    runtime: OrekitRuntime,
    frame: Any,
    epoch: datetime,
    state_vector: FloatArray,
) -> Any:
    initial_date = absolute_date_from_datetime(runtime, epoch)
    position = runtime.vector3d(
        km_to_m(float(state_vector[0])),
        km_to_m(float(state_vector[1])),
        km_to_m(float(state_vector[2])),
    )
    velocity = runtime.vector3d(
        km_s_to_m_s(float(state_vector[3])),
        km_s_to_m_s(float(state_vector[4])),
        km_s_to_m_s(float(state_vector[5])),
    )
    pv_coordinates = runtime.pv_coordinates(position, velocity)
    return runtime.cartesian_orbit(
        pv_coordinates,
        frame,
        initial_date,
        MU_EARTH_M3_S2,
    )


def _build_keplerian_propagator(runtime: OrekitRuntime, orbit: Any) -> _OrekitPropagatorConfig:
    return _OrekitPropagatorConfig(
        propagator=runtime.keplerian_propagator(orbit),
        metadata={
            "propagator": "KeplerianPropagator",
            "frame": "EME2000",
        },
    )


def _build_j2_numerical_propagator(
    scenario: Scenario,
    runtime: OrekitRuntime,
    orbit: Any,
) -> _OrekitPropagatorConfig:
    max_step_s = float(max(scenario.propagation.step_s, NUMERICAL_MIN_STEP_S))
    initial_step_s = float(min(scenario.propagation.step_s, max_step_s))
    orbit_type = runtime.orbit_type.CARTESIAN
    position_angle_type = runtime.position_angle_type.TRUE
    tolerances = runtime.numerical_propagator.tolerances(
        J2_POSITION_TOLERANCE_M,
        orbit,
        orbit_type,
    )
    integrator = runtime.dormand_prince_853_integrator(
        NUMERICAL_MIN_STEP_S,
        max_step_s,
        tolerances[0],
        tolerances[1],
    )
    integrator.setInitialStepSize(initial_step_s)

    propagator = runtime.numerical_propagator(integrator)
    propagator.setOrbitType(orbit_type)
    propagator.setPositionAngleType(position_angle_type)
    propagator.setInitialState(runtime.spacecraft_state(orbit))

    j2_frame = runtime.frames_factory.getITRF(runtime.iers_conventions.IERS_2010, True)
    earth_shape = build_earth_shape(runtime, j2_frame)
    force_model_names = ["J2OnlyPerturbation"]
    force_model_metadata: dict[str, object] = {}
    propagator.addForceModel(
        runtime.j2_only_perturbation(
            MU_EARTH_M3_S2,
            R_EARTH_M,
            J2_EARTH,
            j2_frame,
        )
    )
    if scenario.force_model.atmospheric_drag:
        drag_force_model = build_atmospheric_drag_force_model(scenario, runtime, earth_shape)
        propagator.addForceModel(drag_force_model.model)
        force_model_names.append(drag_force_model.name)
        force_model_metadata.update(drag_force_model.metadata)
    if scenario.force_model.solar_radiation_pressure:
        srp_force_model = build_solar_radiation_pressure_force_model(
            scenario,
            runtime,
            earth_shape,
        )
        propagator.addForceModel(srp_force_model.model)
        force_model_names.append(srp_force_model.name)
        force_model_metadata.update(srp_force_model.metadata)
    if scenario.force_model.third_body_gravity:
        third_body_force_models = build_third_body_gravity_force_models(runtime)
        for third_body_force_model in third_body_force_models:
            propagator.addForceModel(third_body_force_model.model)
            force_model_names.append(third_body_force_model.name)
        force_model_metadata["third_body_gravity_bodies"] = [
            "Sun",
            "Moon",
        ]

    return _OrekitPropagatorConfig(
        propagator=propagator,
        metadata={
            "propagator": "NumericalPropagator",
            "frame": "EME2000",
            "j2_frame": "ITRF(IERS_2010, simple_eop=True)",
            "force_models": force_model_names,
            "orbit_type": "CARTESIAN",
            "position_angle_type": "TRUE",
            "integrator": "DormandPrince853Integrator",
            "integrator_min_step_s": NUMERICAL_MIN_STEP_S,
            "integrator_max_step_s": max_step_s,
            "integrator_initial_step_s": initial_step_s,
            "integrator_position_tolerance_m": J2_POSITION_TOLERANCE_M,
            **force_model_metadata,
            **(
                {
                    "gravity_model": "orekit_high_fidelity",
                    "unsupported_force_model_flags": [],
                }
                if scenario.force_model.gravity is ForceModelName.OREKIT_HIGH_FIDELITY
                else {}
            ),
        },
    )


def _build_propagator_config(
    scenario: Scenario,
    runtime: OrekitRuntime,
    orbit: Any,
) -> _OrekitPropagatorConfig:
    if scenario.force_model.gravity is ForceModelName.TWO_BODY:
        return _build_keplerian_propagator(runtime, orbit)
    return _build_j2_numerical_propagator(scenario, runtime, orbit)


def _state_vector_from_spacecraft_state(
    runtime: OrekitRuntime,
    spacecraft_state: Any,
    frame: Any,
) -> FloatArray:
    pv_coordinates = spacecraft_state.getPVCoordinates(frame)
    position = pv_coordinates.getPosition()
    velocity = pv_coordinates.getVelocity()
    return cast(
        FloatArray,
        np.array(
            [
                m_to_km(float(position.getX())),
                m_to_km(float(position.getY())),
                m_to_km(float(position.getZ())),
                m_s_to_km_s(float(velocity.getX())),
                m_s_to_km_s(float(velocity.getY())),
                m_s_to_km_s(float(velocity.getZ())),
            ],
            dtype=np.float64,
        ),
    )


def _cartesian_from_state_vector(state_vector: FloatArray) -> CartesianState:
    return CartesianState(
        position_km=(
            float(state_vector[0]),
            float(state_vector[1]),
            float(state_vector[2]),
        ),
        velocity_km_s=(
            float(state_vector[3]),
            float(state_vector[4]),
            float(state_vector[5]),
        ),
    )


def _propagate_state_vector(
    scenario: Scenario,
    runtime: OrekitRuntime,
    frame: Any,
    state_vector: FloatArray,
    start_epoch: datetime,
    target_epoch: datetime,
) -> FloatArray:
    orbit = _orbit_from_state_vector(scenario, runtime, frame, start_epoch, state_vector)
    propagator_config = _build_propagator_config(scenario, runtime, orbit)
    target_date = absolute_date_from_datetime(runtime, target_epoch)
    spacecraft_state = propagator_config.propagator.propagate(target_date)
    return _state_vector_from_spacecraft_state(runtime, spacecraft_state, frame)


def _initial_covariance_matrix(scenario: Scenario) -> FloatArray | None:
    if scenario.initial_covariance is None:
        return None
    return cast(FloatArray, np.array(scenario.initial_covariance, dtype=np.float64))


def _matrix_to_nested_list(matrix: FloatArray) -> list[list[float]]:
    return [[float(component) for component in row] for row in matrix]


def _finite_difference_state_transition(
    state_vector: FloatArray,
    advance_state: Callable[[FloatArray], FloatArray],
) -> FloatArray:
    transition = np.zeros((6, 6), dtype=np.float64)
    for column in range(6):
        step = max(
            abs(float(state_vector[column])) * _COVARIANCE_FD_REL_STEP,
            _COVARIANCE_FD_ABS_STEP,
        )
        perturbation = np.zeros(6, dtype=np.float64)
        perturbation[column] = step
        plus = advance_state(state_vector + perturbation)
        minus = advance_state(state_vector - perturbation)
        transition[:, column] = (plus - minus) / (2.0 * step)
    return cast(FloatArray, transition)


def _covariance_process_noise_model(acceleration_sigma_km_s2: float) -> str:
    return "white_acceleration" if acceleration_sigma_km_s2 > 0.0 else "none"


def _process_noise_covariance(acceleration_sigma_km_s2: float, step_s: float) -> FloatArray:
    process_noise = np.zeros((6, 6), dtype=np.float64)
    if acceleration_sigma_km_s2 == 0.0:
        return cast(FloatArray, process_noise)

    acceleration_variance = acceleration_sigma_km_s2**2
    position_variance = 0.25 * acceleration_variance * step_s**4
    position_velocity_covariance = 0.5 * acceleration_variance * step_s**3
    velocity_variance = acceleration_variance * step_s**2
    for axis in range(3):
        velocity_axis = axis + 3
        process_noise[axis, axis] = position_variance
        process_noise[axis, velocity_axis] = position_velocity_covariance
        process_noise[velocity_axis, axis] = position_velocity_covariance
        process_noise[velocity_axis, velocity_axis] = velocity_variance
    return cast(FloatArray, process_noise)


def _propagate_covariance(
    covariance: FloatArray,
    transition: FloatArray,
    process_noise: FloatArray,
) -> FloatArray:
    propagated = transition @ covariance @ transition.T + process_noise
    return cast(FloatArray, 0.5 * (propagated + propagated.T))


def _covariance_sample(
    epoch: datetime,
    covariance: FloatArray,
    *,
    state_transition_matrix: FloatArray,
    accumulated_state_transition_matrix: FloatArray,
    process_noise_covariance: FloatArray,
    metadata: dict[str, object],
) -> CovarianceSample:
    return CovarianceSample(
        epoch=epoch,
        covariance=_matrix_to_nested_list(covariance),
        state_transition_matrix=_matrix_to_nested_list(state_transition_matrix),
        accumulated_state_transition_matrix=_matrix_to_nested_list(
            accumulated_state_transition_matrix
        ),
        process_noise_covariance=_matrix_to_nested_list(process_noise_covariance),
        metadata=metadata,
    )


def _covariance_transition_metadata(
    propagator_metadata: dict[str, object],
    *,
    prefix: str = "",
) -> dict[str, object]:
    metadata: dict[str, object] = {}
    if "propagator" in propagator_metadata:
        metadata[f"{prefix}propagator"] = propagator_metadata["propagator"]
    if "force_models" in propagator_metadata:
        force_models = propagator_metadata["force_models"]
        metadata[f"{prefix}force_models"] = (
            list(force_models) if isinstance(force_models, list) else force_models
        )
    if "gravity_model" in propagator_metadata:
        metadata[f"{prefix}gravity_model"] = propagator_metadata["gravity_model"]
    return metadata


def _covariance_metadata(
    scenario: Scenario,
    covariance: FloatArray | None,
    propagator_metadata: dict[str, object],
) -> dict[str, object]:
    if covariance is None:
        return {}
    process_noise_acceleration = scenario.covariance_process_noise_acceleration_km_s2
    return {
        "covariance_model": "orekit_finite_difference_state_transition",
        "covariance_state_transition_storage": "per_sample_and_accumulated",
        "covariance_process_noise": _covariance_process_noise_model(process_noise_acceleration),
        "covariance_process_noise_acceleration_km_s2": process_noise_acceleration,
        "covariance_process_noise_storage": "per_sample_matrix",
        "covariance_finite_difference_relative_step": _COVARIANCE_FD_REL_STEP,
        "covariance_finite_difference_absolute_step": _COVARIANCE_FD_ABS_STEP,
        **_covariance_transition_metadata(
            propagator_metadata,
            prefix="covariance_transition_",
        ),
    }


def propagate_orekit(
    scenario: Scenario,
    *,
    runtime_loader: RuntimeLoader = load_orekit_runtime,
) -> Trajectory:
    _validate_orekit_scenario(scenario)
    runtime = runtime_loader()
    frame = runtime.frames_factory.getEME2000()
    orbit = _initial_orbit(scenario, runtime, frame)
    propagator_config = _build_propagator_config(scenario, runtime, orbit)
    covariance_matrix = _initial_covariance_matrix(scenario)
    process_noise = _process_noise_covariance(
        scenario.covariance_process_noise_acceleration_km_s2,
        scenario.propagation.step_s,
    )
    zero_process_noise = np.zeros((6, 6), dtype=np.float64)
    identity_transition = np.eye(6, dtype=np.float64)
    accumulated_transition = identity_transition.copy()
    previous_transition = identity_transition.copy()
    previous_process_noise = zero_process_noise
    previous_transition_step_s = 0.0
    covariance_history: list[CovarianceSample] = []

    samples: list[TrajectorySample] = []
    for step_index in range(scenario.propagation.sample_count):
        epoch = scenario.initial_state.epoch + timedelta(
            seconds=step_index * scenario.propagation.step_s
        )
        target_date = absolute_date_from_datetime(runtime, epoch)
        spacecraft_state = propagator_config.propagator.propagate(target_date)
        state_vector = _state_vector_from_spacecraft_state(runtime, spacecraft_state, frame)
        samples.append(
            TrajectorySample(
                epoch=epoch,
                state=_cartesian_from_state_vector(state_vector),
            )
        )
        if covariance_matrix is not None:
            covariance_history.append(
                _covariance_sample(
                    epoch,
                    covariance_matrix,
                    state_transition_matrix=previous_transition,
                    accumulated_state_transition_matrix=accumulated_transition,
                    process_noise_covariance=previous_process_noise,
                    metadata={
                        "covariance_sample_role": (
                            "initial" if step_index == 0 else "propagated"
                        ),
                        "covariance_model": "orekit_finite_difference_state_transition",
                        "state_transition_model": (
                            "identity" if step_index == 0 else "orekit_finite_difference"
                        ),
                        "transition_step_s": previous_transition_step_s,
                        "process_noise_model": (
                            "none"
                            if step_index == 0
                            else _covariance_process_noise_model(
                                scenario.covariance_process_noise_acceleration_km_s2
                            )
                        ),
                        "process_noise_acceleration_km_s2": (
                            0.0
                            if step_index == 0
                            else scenario.covariance_process_noise_acceleration_km_s2
                        ),
                        "finite_difference_relative_step": _COVARIANCE_FD_REL_STEP,
                        "finite_difference_absolute_step": _COVARIANCE_FD_ABS_STEP,
                        **_covariance_transition_metadata(
                            propagator_config.metadata,
                            prefix="transition_",
                        ),
                    },
                )
            )

            if step_index < scenario.propagation.sample_count - 1:
                next_epoch = scenario.initial_state.epoch + timedelta(
                    seconds=(step_index + 1) * scenario.propagation.step_s
                )

                def advance_trial(
                    trial_state: FloatArray,
                    *,
                    start_epoch: datetime = epoch,
                    target_epoch: datetime = next_epoch,
                ) -> FloatArray:
                    return _propagate_state_vector(
                        scenario,
                        runtime,
                        frame,
                        trial_state,
                        start_epoch,
                        target_epoch,
                    )

                transition = _finite_difference_state_transition(state_vector, advance_trial)
                covariance_matrix = _propagate_covariance(
                    covariance_matrix,
                    transition,
                    process_noise,
                )
                accumulated_transition = cast(FloatArray, transition @ accumulated_transition)
                previous_transition = transition
                previous_process_noise = process_noise
                previous_transition_step_s = scenario.propagation.step_s

    return Trajectory(
        scenario_id=scenario.scenario_id,
        samples=samples,
        force_model=scenario.force_model,
        backend="orekit",
        covariance_history=covariance_history,
        metadata={
            "wrapper": runtime.wrapper,
            "wrapper_version": runtime.wrapper_version,
            "data_path": runtime.data_path,
            **propagator_config.metadata,
            "units": "suite km/km_s converted to Orekit m/m_s",
            **_covariance_metadata(
                scenario,
                _initial_covariance_matrix(scenario),
                propagator_config.metadata,
            ),
        },
    )
