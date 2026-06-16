from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from astro_backends.orekit.conversion import (
    absolute_date_from_datetime,
    km_s_to_m_s,
    km_to_m,
    m_s_to_km_s,
    m_to_km,
    validate_orekit_state_support,
)
from astro_backends.orekit.runtime import OrekitRuntime, load_orekit_runtime
from astro_core.constants import J2_EARTH, MU_EARTH_KM3_S2, R_EARTH_KM
from astro_core.errors import UnsupportedBackendError
from astro_core.models import (
    CartesianState,
    ForceModelName,
    Scenario,
    Trajectory,
    TrajectorySample,
)

RuntimeLoader = Callable[[], OrekitRuntime]
MU_EARTH_M3_S2 = MU_EARTH_KM3_S2 * 1.0e9
R_EARTH_M = R_EARTH_KM * 1.0e3
J2_POSITION_TOLERANCE_M = 0.001
NUMERICAL_MIN_STEP_S = 0.001
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
    enabled_flags = scenario.force_model.enabled_high_fidelity_flags()
    if enabled_flags:
        raise UnsupportedBackendError(
            "Orekit propagation does not yet support high-fidelity force model flags: "
            f"{', '.join(enabled_flags)}"
        )
    if scenario.force_model.gravity not in _SUPPORTED_OREKIT_GRAVITY_MODELS:
        raise UnsupportedBackendError(
            "Orekit propagation supports two_body, j2, and orekit_high_fidelity gravity"
        )


def _initial_orbit(scenario: Scenario, runtime: OrekitRuntime, frame: Any) -> Any:
    initial_date = absolute_date_from_datetime(runtime, scenario.initial_state.epoch)
    initial = scenario.initial_state.cartesian

    position = runtime.vector3d(
        km_to_m(initial.position_km[0]),
        km_to_m(initial.position_km[1]),
        km_to_m(initial.position_km[2]),
    )
    velocity = runtime.vector3d(
        km_s_to_m_s(initial.velocity_km_s[0]),
        km_s_to_m_s(initial.velocity_km_s[1]),
        km_s_to_m_s(initial.velocity_km_s[2]),
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
    propagator.addForceModel(
        runtime.j2_only_perturbation(
            MU_EARTH_M3_S2,
            R_EARTH_M,
            J2_EARTH,
            j2_frame,
        )
    )

    return _OrekitPropagatorConfig(
        propagator=propagator,
        metadata={
            "propagator": "NumericalPropagator",
            "frame": "EME2000",
            "j2_frame": "ITRF(IERS_2010, simple_eop=True)",
            "force_models": ["J2OnlyPerturbation"],
            "orbit_type": "CARTESIAN",
            "position_angle_type": "TRUE",
            "integrator": "DormandPrince853Integrator",
            "integrator_min_step_s": NUMERICAL_MIN_STEP_S,
            "integrator_max_step_s": max_step_s,
            "integrator_initial_step_s": initial_step_s,
            "integrator_position_tolerance_m": J2_POSITION_TOLERANCE_M,
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

    samples: list[TrajectorySample] = []
    for step_index in range(scenario.propagation.sample_count):
        epoch = scenario.initial_state.epoch + timedelta(
            seconds=step_index * scenario.propagation.step_s
        )
        target_date = absolute_date_from_datetime(runtime, epoch)
        spacecraft_state = propagator_config.propagator.propagate(target_date)
        propagated_pv = spacecraft_state.getPVCoordinates(frame)
        propagated_position = propagated_pv.getPosition()
        propagated_velocity = propagated_pv.getVelocity()
        samples.append(
            TrajectorySample(
                epoch=epoch,
                state=CartesianState(
                    position_km=(
                        m_to_km(float(propagated_position.getX())),
                        m_to_km(float(propagated_position.getY())),
                        m_to_km(float(propagated_position.getZ())),
                    ),
                    velocity_km_s=(
                        m_s_to_km_s(float(propagated_velocity.getX())),
                        m_s_to_km_s(float(propagated_velocity.getY())),
                        m_s_to_km_s(float(propagated_velocity.getZ())),
                    ),
                ),
            )
        )

    return Trajectory(
        scenario_id=scenario.scenario_id,
        samples=samples,
        force_model=scenario.force_model,
        backend="orekit",
        metadata={
            "wrapper": runtime.wrapper,
            "wrapper_version": runtime.wrapper_version,
            "data_path": runtime.data_path,
            **propagator_config.metadata,
            "units": "suite km/km_s converted to Orekit m/m_s",
        },
    )
