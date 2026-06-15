from __future__ import annotations

from astro_core.models import (
    ForceModelConfig,
    ForceModelName,
    PropagationConfig,
    Scenario,
    Spacecraft,
)
from astro_launch.models import LaunchTrajectory


def launch_trajectory_to_orbit_scenario(
    trajectory: LaunchTrajectory,
    *,
    duration_s: float,
    step_s: float,
    spacecraft_name: str = "launch-payload",
    spacecraft_mass_kg: float | None = None,
    area_m2: float = 2.5,
    drag_coefficient: float = 2.2,
    reflectivity_coefficient: float = 1.3,
    gravity: ForceModelName = ForceModelName.TWO_BODY,
    scenario_id: str | None = None,
    description: str | None = None,
) -> Scenario:
    resolved_scenario_id = scenario_id or f"{trajectory.scenario_id}-insertion"
    resolved_description = (
        description
        if description is not None
        else f"Orbital propagation scenario initialized from {trajectory.scenario_id} insertion."
    )
    resolved_spacecraft_mass_kg = (
        spacecraft_mass_kg if spacecraft_mass_kg is not None else trajectory.samples[-1].mass_kg
    )

    return Scenario(
        scenario_id=resolved_scenario_id,
        description=resolved_description,
        spacecraft=Spacecraft(
            name=spacecraft_name,
            mass_kg=resolved_spacecraft_mass_kg,
            area_m2=area_m2,
            drag_coefficient=drag_coefficient,
            reflectivity_coefficient=reflectivity_coefficient,
        ),
        initial_state=trajectory.insertion_state,
        force_model=ForceModelConfig(gravity=gravity),
        propagation=PropagationConfig(duration_s=duration_s, step_s=step_s),
        metadata={
            "workflow": "launch_orbit_handoff",
            "source_launch_scenario_id": trajectory.scenario_id,
            "source_launch_backend": trajectory.backend,
            "source_launch_sample_count": len(trajectory.samples),
            "source_launch_event_count": len(trajectory.events),
            "source_launch_target_miss": trajectory.target_miss,
        },
    )
