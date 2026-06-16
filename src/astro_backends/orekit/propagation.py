from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta

from astro_backends.orekit.conversion import (
    absolute_date_from_datetime,
    km_s_to_m_s,
    km_to_m,
    m_s_to_km_s,
    m_to_km,
    validate_orekit_state_support,
)
from astro_backends.orekit.runtime import OrekitRuntime, load_orekit_runtime
from astro_core.constants import MU_EARTH_KM3_S2
from astro_core.errors import UnsupportedBackendError
from astro_core.models import (
    CartesianState,
    ForceModelName,
    Scenario,
    Trajectory,
    TrajectorySample,
)

RuntimeLoader = Callable[[], OrekitRuntime]


def _validate_orekit_phase1_scenario(scenario: Scenario) -> None:
    validate_orekit_state_support(scenario.initial_state)
    if scenario.force_model.gravity is not ForceModelName.TWO_BODY:
        raise UnsupportedBackendError(
            "Orekit propagation phase 1 supports only two_body gravity; "
            "j2 and orekit_high_fidelity require the numerical force-model phase"
        )


def propagate_orekit(
    scenario: Scenario,
    *,
    runtime_loader: RuntimeLoader = load_orekit_runtime,
) -> Trajectory:
    _validate_orekit_phase1_scenario(scenario)
    runtime = runtime_loader()
    frame = runtime.frames_factory.getEME2000()
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
    orbit = runtime.cartesian_orbit(
        pv_coordinates,
        frame,
        initial_date,
        MU_EARTH_KM3_S2 * 1.0e9,
    )
    propagator = runtime.keplerian_propagator(orbit)

    samples: list[TrajectorySample] = []
    for step_index in range(scenario.propagation.sample_count):
        epoch = scenario.initial_state.epoch + timedelta(
            seconds=step_index * scenario.propagation.step_s
        )
        target_date = absolute_date_from_datetime(runtime, epoch)
        spacecraft_state = propagator.propagate(target_date)
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
            "propagator": "KeplerianPropagator",
            "frame": "EME2000",
            "units": "suite km/km_s converted to Orekit m/m_s",
        },
    )
