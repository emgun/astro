from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
from math import isfinite

from astro_launch.local import propagate_launch_local
from astro_launch.models import (
    GuidanceConfig,
    LaunchPitchSweepCase,
    LaunchPitchSweepResult,
    LaunchPitchTuningCase,
    LaunchPitchTuningIteration,
    LaunchPitchTuningPoint,
    LaunchPitchTuningResult,
    LaunchScenario,
    PitchProgramPoint,
)


@dataclass(frozen=True)
class _TargetingMetrics:
    score: float
    altitude_miss_km: float
    velocity_miss_km_s: float
    final_altitude_km: float
    final_velocity_km_s: float
    final_radial_velocity_km_s: float
    final_horizontal_velocity_km_s: float
    final_downrange_km: float
    target_miss: dict[str, float]


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


def _scenario_with_pitches(
    scenario: LaunchScenario,
    *,
    pitch_program: tuple[PitchProgramPoint, ...],
    pitch_deg_by_point_index: dict[int, float],
) -> LaunchScenario:
    swept_pitch_program = [
        point.model_copy(update={"pitch_deg": pitch_deg_by_point_index[index]})
        if index in pitch_deg_by_point_index
        else point
        for index, point in enumerate(pitch_program)
    ]
    guidance = GuidanceConfig(mode="pitch_program", pitch_program=swept_pitch_program)
    return scenario.model_copy(update={"guidance": guidance})


def _launch_targeting_metrics(
    scenario: LaunchScenario,
    *,
    altitude_weight: float,
    velocity_weight: float,
) -> _TargetingMetrics:
    trajectory = propagate_launch_local(scenario)
    final_sample = trajectory.samples[-1]
    target_miss = {key: float(value) for key, value in trajectory.target_miss.items()}
    altitude_miss_km = target_miss["altitude_miss_km"]
    velocity_miss_km_s = target_miss["velocity_miss_km_s"]
    score = (
        abs(altitude_miss_km) * altitude_weight
        + abs(velocity_miss_km_s) * velocity_weight
    )
    return _TargetingMetrics(
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


def _sweep_case_from_metrics(
    *,
    pitch_deg: float,
    metrics: _TargetingMetrics,
) -> LaunchPitchSweepCase:
    return LaunchPitchSweepCase(
        pitch_deg=pitch_deg,
        score=metrics.score,
        altitude_miss_km=metrics.altitude_miss_km,
        velocity_miss_km_s=metrics.velocity_miss_km_s,
        final_altitude_km=metrics.final_altitude_km,
        final_velocity_km_s=metrics.final_velocity_km_s,
        final_radial_velocity_km_s=metrics.final_radial_velocity_km_s,
        final_horizontal_velocity_km_s=metrics.final_horizontal_velocity_km_s,
        final_downrange_km=metrics.final_downrange_km,
        target_miss=metrics.target_miss,
    )


def _tuning_case_from_metrics(
    *,
    iteration: int,
    pitch_deg_by_point_index: dict[int, float],
    metrics: _TargetingMetrics,
) -> LaunchPitchTuningCase:
    return LaunchPitchTuningCase(
        iteration=iteration,
        pitch_deg_by_point_index={
            str(point_index): pitch_deg
            for point_index, pitch_deg in pitch_deg_by_point_index.items()
        },
        score=metrics.score,
        altitude_miss_km=metrics.altitude_miss_km,
        velocity_miss_km_s=metrics.velocity_miss_km_s,
        final_altitude_km=metrics.final_altitude_km,
        final_velocity_km_s=metrics.final_velocity_km_s,
        final_radial_velocity_km_s=metrics.final_radial_velocity_km_s,
        final_horizontal_velocity_km_s=metrics.final_horizontal_velocity_km_s,
        final_downrange_km=metrics.final_downrange_km,
        target_miss=metrics.target_miss,
    )


def _validate_score_weights(altitude_weight: float, velocity_weight: float) -> tuple[float, float]:
    altitude_weight = _validated_weight(altitude_weight, "altitude_weight")
    velocity_weight = _validated_weight(velocity_weight, "velocity_weight")
    if altitude_weight == 0.0 and velocity_weight == 0.0:
        raise ValueError("at least one sweep score weight must be greater than zero")
    return altitude_weight, velocity_weight


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
    altitude_weight, velocity_weight = _validate_score_weights(
        altitude_weight,
        velocity_weight,
    )

    cases: list[LaunchPitchSweepCase] = []
    for pitch_deg in pitch_values:
        swept_scenario = _scenario_with_pitch(
            scenario,
            pitch_program=pitch_program,
            point_index=point_index,
            pitch_deg=pitch_deg,
        )
        metrics = _launch_targeting_metrics(
            swept_scenario,
            altitude_weight=altitude_weight,
            velocity_weight=velocity_weight,
        )
        cases.append(_sweep_case_from_metrics(pitch_deg=pitch_deg, metrics=metrics))

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


def _validated_tuning_pitch_program(
    scenario: LaunchScenario,
    point_indices: Sequence[int],
) -> tuple[tuple[PitchProgramPoint, ...], tuple[int, int]]:
    if scenario.guidance.mode != "pitch_program":
        raise ValueError("tune_pitch_program requires pitch_program guidance")

    selected_indices = tuple(point_indices)
    if len(selected_indices) != 2:
        raise ValueError("point_indices must include exactly two pitch_program points")
    if len(set(selected_indices)) != 2:
        raise ValueError("point_indices must be distinct")

    pitch_program = tuple(scenario.guidance.pitch_program)
    for point_index in selected_indices:
        if point_index < 0 or point_index >= len(pitch_program):
            raise ValueError(
                f"point_indices must select existing pitch_program points: {point_index}"
            )
        if point_index == 0:
            raise ValueError("cannot tune first pitch_program point at t=0")

    return pitch_program, (selected_indices[0], selected_indices[1])


def _validated_positive_float(value: float, name: str) -> float:
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


def _validated_refinement_factor(refinement_factor: float) -> float:
    result = float(refinement_factor)
    if not isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError("refinement_factor must be a finite number between 0 and 1")
    return result


def _validated_iterations(iterations: int) -> int:
    if isinstance(iterations, bool) or iterations <= 0:
        raise ValueError("iterations must be a positive integer")
    return iterations


def _candidate_pitches(center_pitch_deg: float, span_deg: float) -> tuple[float, ...]:
    return tuple(
        sorted(
            {
                min(90.0, max(0.0, center_pitch_deg + offset_deg))
                for offset_deg in (-span_deg, 0.0, span_deg)
            }
        )
    )


def _case_pitch_map(case: LaunchPitchTuningCase) -> dict[int, float]:
    return {
        int(point_index): float(pitch_deg)
        for point_index, pitch_deg in case.pitch_deg_by_point_index.items()
    }


def _tuned_points(
    *,
    pitch_program: tuple[PitchProgramPoint, ...],
    point_indices: tuple[int, int],
    tuned_pitch_deg_by_point_index: dict[int, float],
) -> list[LaunchPitchTuningPoint]:
    return [
        LaunchPitchTuningPoint(
            point_index=point_index,
            time_s=pitch_program[point_index].time_s,
            baseline_pitch_deg=pitch_program[point_index].pitch_deg,
            tuned_pitch_deg=tuned_pitch_deg_by_point_index[point_index],
        )
        for point_index in point_indices
    ]


def tune_pitch_program(
    scenario: LaunchScenario,
    *,
    point_indices: Sequence[int],
    initial_span_deg: float = 10.0,
    iterations: int = 2,
    refinement_factor: float = 0.5,
    altitude_weight: float = 1.0,
    velocity_weight: float = 1.0,
) -> LaunchPitchTuningResult:
    """Coarse-to-fine tune two pitch-program points with a deterministic grid search."""
    pitch_program, point_indices = _validated_tuning_pitch_program(scenario, point_indices)
    initial_span_deg = _validated_positive_float(initial_span_deg, "initial_span_deg")
    iterations = _validated_iterations(iterations)
    refinement_factor = _validated_refinement_factor(refinement_factor)
    altitude_weight, velocity_weight = _validate_score_weights(
        altitude_weight,
        velocity_weight,
    )

    center_pitch_deg_by_point_index = {
        point_index: float(pitch_program[point_index].pitch_deg)
        for point_index in point_indices
    }
    span_deg = initial_span_deg
    tuning_iterations: list[LaunchPitchTuningIteration] = []
    best_case: LaunchPitchTuningCase | None = None

    for iteration in range(1, iterations + 1):
        candidate_pitches_by_point = [
            _candidate_pitches(center_pitch_deg_by_point_index[point_index], span_deg)
            for point_index in point_indices
        ]
        cases: list[LaunchPitchTuningCase] = []
        for candidate_pitches in product(*candidate_pitches_by_point):
            pitch_deg_by_point_index = dict(zip(point_indices, candidate_pitches, strict=True))
            candidate_scenario = _scenario_with_pitches(
                scenario,
                pitch_program=pitch_program,
                pitch_deg_by_point_index=pitch_deg_by_point_index,
            )
            metrics = _launch_targeting_metrics(
                candidate_scenario,
                altitude_weight=altitude_weight,
                velocity_weight=velocity_weight,
            )
            cases.append(
                _tuning_case_from_metrics(
                    iteration=iteration,
                    pitch_deg_by_point_index=pitch_deg_by_point_index,
                    metrics=metrics,
                )
            )

        iteration_best_case = min(cases, key=lambda case: case.score)
        tuning_iterations.append(
            LaunchPitchTuningIteration(
                iteration=iteration,
                span_deg=span_deg,
                cases=cases,
                best_case=iteration_best_case,
            )
        )
        if best_case is None or iteration_best_case.score < best_case.score:
            best_case = iteration_best_case
        center_pitch_deg_by_point_index = _case_pitch_map(iteration_best_case)
        span_deg *= refinement_factor

    if best_case is None:
        raise RuntimeError("pitch tuning did not evaluate any candidate cases")

    tuned_pitch_deg_by_point_index = _case_pitch_map(best_case)
    tuned_scenario = _scenario_with_pitches(
        scenario,
        pitch_program=pitch_program,
        pitch_deg_by_point_index=tuned_pitch_deg_by_point_index,
    )
    candidate_count = sum(
        len(tuning_iteration.cases) for tuning_iteration in tuning_iterations
    )
    return LaunchPitchTuningResult(
        scenario_id=scenario.scenario_id,
        point_indices=list(point_indices),
        tuned_points=_tuned_points(
            pitch_program=pitch_program,
            point_indices=point_indices,
            tuned_pitch_deg_by_point_index=tuned_pitch_deg_by_point_index,
        ),
        initial_span_deg=initial_span_deg,
        refinement_factor=refinement_factor,
        altitude_weight=altitude_weight,
        velocity_weight=velocity_weight,
        iterations=tuning_iterations,
        best_case=best_case,
        tuned_scenario=tuned_scenario,
        backend="local",
        metadata={
            "workflow": "pitch_program_tuning",
            "score": "abs(altitude_miss_km) * altitude_weight + "
            "abs(velocity_miss_km_s) * velocity_weight",
            "candidate_count": candidate_count,
            "grid": "coarse_to_fine_3x3",
            "guidance_mode": scenario.guidance.mode,
        },
    )
