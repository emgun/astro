from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from astro_core.models import (
    Body,
    CartesianState,
    ForceModelConfig,
    ForceModelName,
    Frame,
    GroundStation,
    OrbitRepresentation,
    OrbitState,
    PropagationConfig,
    Scenario,
    Spacecraft,
    TimeScale,
)


def make_state() -> OrbitState:
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


def test_orbit_state_requires_finite_cartesian_values() -> None:
    with pytest.raises(ValidationError, match="finite"):
        CartesianState(position_km=(7000.0, float("nan"), 0.0), velocity_km_s=(0.0, 7.5, 0.0))


def test_spacecraft_requires_positive_mass_and_area() -> None:
    with pytest.raises(ValidationError):
        Spacecraft(
            name="bad",
            mass_kg=0.0,
            area_m2=3.0,
            drag_coefficient=2.2,
            reflectivity_coefficient=1.2,
        )

    with pytest.raises(ValidationError):
        Spacecraft(
            name="bad",
            mass_kg=100.0,
            area_m2=-1.0,
            drag_coefficient=2.2,
            reflectivity_coefficient=1.2,
        )


def test_scenario_accepts_minimal_valid_orbital_case() -> None:
    scenario = Scenario(
        scenario_id="leo-demo",
        description="LEO propagation demo",
        spacecraft=Spacecraft(
            name="demo",
            mass_kg=120.0,
            area_m2=2.5,
            drag_coefficient=2.2,
            reflectivity_coefficient=1.3,
        ),
        initial_state=make_state(),
        force_model=ForceModelConfig(gravity=ForceModelName.TWO_BODY),
        propagation=PropagationConfig(duration_s=600.0, step_s=60.0),
        ground_stations=[
            GroundStation(
                name="station-a",
                position_eci_km=(6378.1363, 0.0, 0.0),
                frame=Frame.EME2000,
                elevation_mask_deg=0.0,
            )
        ],
    )

    assert scenario.scenario_id == "leo-demo"
    assert scenario.propagation.sample_count == 11
