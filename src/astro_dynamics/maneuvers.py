from __future__ import annotations

from astro_core.models import CartesianState, Maneuver, OrbitState


def apply_impulsive_maneuver(state: OrbitState, maneuver: Maneuver) -> OrbitState:
    if state.frame != maneuver.frame:
        raise ValueError("Impulsive maneuver and orbit state must use the same frame")

    velocity = state.cartesian.velocity_km_s
    delta_v = maneuver.delta_v_km_s
    maneuvered_cartesian = CartesianState(
        position_km=state.cartesian.position_km,
        velocity_km_s=(
            velocity[0] + delta_v[0],
            velocity[1] + delta_v[1],
            velocity[2] + delta_v[2],
        ),
    )
    return state.model_copy(
        update={
            "epoch": maneuver.epoch,
            "cartesian": maneuvered_cartesian,
        }
    )
