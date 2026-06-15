from __future__ import annotations

from collections.abc import Sequence
from math import isfinite

from astro_launch.local import propagate_launch_local
from astro_launch.models import (
    GuidanceConfig,
    LaunchPitchSweepCase,
    LaunchPitchSweepResult,
    LaunchScenario,
    PitchProgramPoint,
)


def _validated_weight(value: float, name: str) -> float:
    weight = float(value)
    if not isfinite(weight) or weight < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return weight


def _validated_pitch_values(pitch_values_deg: Sequence[float]) -> tuple[float, ...]:
    if not pitch_values_deg:
        raise ValueError("pitch_values_deg must include at least one pitch angle")

    values: list[float] = []
    for pitch_value_deg in pitch_values_deg:
        pitch_deg = float(pitch_value_deg)
        if not isfinite(pitch_deg) or not 0.0 <= pitch_deg <= 90.0:
            raise ValueError("pitch_values_deg entries must be finite numbers between 0 and 90")
        values.append(pitch_deg)
    return tuple(values)


def _validated_pitch_program(
    scenario: LaunchScenario,
    point_index: int,
) -> tuple[PitchProgramPoint, ...]:
    if scenario.guidance.mode != "pitch_program":
        raise ValueError("sweep_pitch_program requires pitch_program guidance")
    pitch_program = tuple(scenario.guidance.pitch_program)
    if point_index < 0 or point_index >= len(pitch_program):
        raise ValueError(
            f"point_index must select an existing pitch_program point: {point_index}"
        )
    if point_index == 0:
        raise ValueError("cannot sweep first pitch_program point at t=0")
    return pitch_program


def _scenario_with_pitch(
    scenario: LaunchScenario,
    *,
    pitch_program: tuple[PitchProgramPoint, ...],
    point_index: int,
    pitch_deg: float,
) -> LaunchScenario:
    swept_pitch_program = [
        point.model_copy(update={"pitch_deg": pitch_deg}) if index == point_index else point
        for index, point in enumerate(pitch_program)
    ]
    guidance = GuidanceConfig(mode="pitch_program", pitch_program=swept_pitch_program)
    return scenario.model_copy(update={"guidance": guidance})


def sweep_pitch_program(
    scenario: LaunchScenario,
    *,
    point_index: int,
    pitch_values_deg: Sequence[float],
    altitude_weight: float = 1.0,
    velocity_weight: float = 1.0,
) -> LaunchPitchSweepResult:
    """Sweep one pitch-program point and score final target miss."""
    pitch_program = _validated_pitch_program(scenario, point_index)
    pitch_values = _validated_pitch_values(pitch_values_deg)
    altitude_weight = _validated_weight(altitude_weight, "altitude_weight")
    velocity_weight = _validated_weight(velocity_weight, "velocity_weight")
    if altitude_weight == 0.0 and velocity_weight == 0.0:
        raise ValueError("at least one sweep score weight must be greater than zero")

    cases: list[LaunchPitchSweepCase] = []
    for pitch_deg in pitch_values:
        swept_scenario = _scenario_with_pitch(
            scenario,
            pitch_program=pitch_program,
            point_index=point_index,
            pitch_deg=pitch_deg,
        )
        trajectory = propagate_launch_local(swept_scenario)
        final_sample = trajectory.samples[-1]
        target_miss = {
            key: float(value) for key, value in trajectory.target_miss.items()
        }
        altitude_miss_km = target_miss["altitude_miss_km"]
        velocity_miss_km_s = target_miss["velocity_miss_km_s"]
        score = (
            abs(altitude_miss_km) * altitude_weight
            + abs(velocity_miss_km_s) * velocity_weight
        )
        cases.append(
            LaunchPitchSweepCase(
                pitch_deg=pitch_deg,
                score=score,
                altitude_miss_km=altitude_miss_km,
                velocity_miss_km_s=velocity_miss_km_s,
                final_altitude_km=final_sample.altitude_km,
                final_velocity_km_s=final_sample.velocity_km_s,
                final_radial_velocity_km_s=final_sample.radial_velocity_km_s,
                final_horizontal_velocity_km_s=final_sample.horizontal_velocity_km_s,
                final_downrange_km=final_sample.downrange_km,
                target_miss=target_miss,
            )
        )

    best_case = min(cases, key=lambda case: case.score)
    baseline_point = pitch_program[point_index]
    return LaunchPitchSweepResult(
        scenario_id=scenario.scenario_id,
        point_index=point_index,
        point_time_s=baseline_point.time_s,
        baseline_pitch_deg=baseline_point.pitch_deg,
        altitude_weight=altitude_weight,
        velocity_weight=velocity_weight,
        cases=cases,
        best_case=best_case,
        backend="local",
        metadata={
            "workflow": "pitch_program_sweep",
            "score": "abs(altitude_miss_km) * altitude_weight + "
            "abs(velocity_miss_km_s) * velocity_weight",
            "candidate_count": len(cases),
            "guidance_mode": scenario.guidance.mode,
        },
    )
