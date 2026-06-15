from datetime import UTC, datetime

import pytest

from astro_core.models import (
    Body,
    CartesianState,
    Frame,
    Maneuver,
    OrbitRepresentation,
    OrbitState,
    TimeScale,
)
from astro_dynamics.maneuvers import apply_impulsive_maneuver


def test_apply_impulsive_maneuver_updates_velocity_and_epoch() -> None:
    state = _state()
    maneuver = Maneuver(
        name="trim",
        epoch=datetime(2026, 1, 1, 0, 5, tzinfo=UTC),
        frame=Frame.EME2000,
        delta_v_km_s=(0.0, 0.001, -0.002),
    )

    maneuvered_state = apply_impulsive_maneuver(state, maneuver)

    assert maneuvered_state.epoch == maneuver.epoch
    assert maneuvered_state.cartesian.position_km == state.cartesian.position_km
    assert maneuvered_state.cartesian.velocity_km_s == pytest.approx((0.0, 7.501, 0.998))


def test_apply_impulsive_maneuver_requires_matching_frame() -> None:
    state = _state()
    maneuver = Maneuver(
        name="trim",
        epoch=state.epoch,
        frame=Frame.EME2000,
        delta_v_km_s=(0.0, 0.001, 0.0),
    )
    state_with_bad_frame = state.model_copy(update={"frame": "BAD"})

    with pytest.raises(ValueError, match="same frame"):
        apply_impulsive_maneuver(state_with_bad_frame, maneuver)


def _state() -> OrbitState:
    return OrbitState(
        epoch=datetime(2026, 1, 1, tzinfo=UTC),
        time_scale=TimeScale.UTC,
        frame=Frame.EME2000,
        central_body=Body.EARTH,
        representation=OrbitRepresentation.CARTESIAN,
        cartesian=CartesianState(
            position_km=(7000.0, 0.0, 0.0),
            velocity_km_s=(0.0, 7.5, 1.0),
        ),
    )
