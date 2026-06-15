from datetime import UTC, datetime

from astro_core.models import Body
from astro_launch.models import (
    AtmosphereConfig,
    GuidanceConfig,
    LaunchEngine,
    LaunchPropagationConfig,
    LaunchScenario,
    LaunchSite,
    LaunchStage,
    LaunchVehicle,
    TargetOrbit,
)


def make_launch_scenario(**overrides: object) -> LaunchScenario:
    payload = {
        "scenario_id": "vertical-two-stage",
        "description": "Deterministic launch model test case.",
        "epoch": datetime(2026, 1, 1, tzinfo=UTC),
        "launch_site": LaunchSite(
            name="equator-pad",
            latitude_deg=0.0,
            longitude_deg=0.0,
            altitude_m=0.0,
            body=Body.EARTH,
        ),
        "vehicle": LaunchVehicle(
            name="mvp-two-stage",
            payload_mass_kg=500.0,
            stages=[
                LaunchStage(
                    name="stage-1",
                    dry_mass_kg=1200.0,
                    propellant_mass_kg=4500.0,
                    engine=LaunchEngine(
                        name="booster",
                        thrust_n=160000.0,
                        specific_impulse_s=280.0,
                    ),
                    burn_duration_s=70.0,
                    reference_area_m2=8.0,
                    drag_coefficient=0.35,
                ),
                LaunchStage(
                    name="stage-2",
                    dry_mass_kg=350.0,
                    propellant_mass_kg=900.0,
                    engine=LaunchEngine(
                        name="upper",
                        thrust_n=35000.0,
                        specific_impulse_s=315.0,
                    ),
                    burn_duration_s=50.0,
                    reference_area_m2=3.0,
                    drag_coefficient=0.25,
                ),
            ],
        ),
        "atmosphere": AtmosphereConfig(model="exponential"),
        "guidance": GuidanceConfig(mode="vertical"),
        "target_orbit": TargetOrbit(altitude_km=160.0, inclination_deg=0.0),
        "propagation": LaunchPropagationConfig(duration_s=140.0, step_s=10.0),
    }
    payload.update(overrides)
    return LaunchScenario(**payload)
