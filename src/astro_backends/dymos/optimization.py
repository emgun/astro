from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from math import isfinite, sqrt
from typing import Any

import numpy as np

from astro_backends.dymos.runtime import DymosRuntime, load_dymos_runtime
from astro_launch.local import _circular_velocity_km_s
from astro_launch.models import (
    GuidanceConfig,
    LaunchPitchTuningCase,
    LaunchPitchTuningIteration,
    LaunchPitchTuningPoint,
    LaunchPitchTuningResult,
    LaunchScenario,
)
from astro_launch.targeting import tune_pitch_program

DymosRuntimeLoader = Callable[[], DymosRuntime]
DymosOptimizerRunner = Callable[[LaunchScenario, DymosRuntime], LaunchPitchTuningResult]


@dataclass(frozen=True)
class DymosPhaseSummary:
    phase_model: str
    transcription: str
    num_segments: int
    order: int
    duration_s: float
    stage_count: int
    total_burn_duration_s: float
    final_altitude_km: float
    final_velocity_km_s: float
    optimizer_success: bool
    optimizer_message: str

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DymosPitchProgramSummary:
    phase_model: str
    transcription: str
    num_segments: int
    order: int
    duration_s: float
    stage_count: int
    total_burn_duration_s: float
    final_altitude_km: float
    final_velocity_km_s: float
    final_radial_velocity_km_s: float
    final_horizontal_velocity_km_s: float
    final_downrange_km: float
    target_miss: dict[str, float]
    optimized_pitch_deg_by_point_index: dict[int, float]
    optimizer_success: bool
    optimizer_message: str

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)


def _with_dymos_provenance(
    result: LaunchPitchTuningResult,
    scenario: LaunchScenario,
    runtime: DymosRuntime,
) -> LaunchPitchTuningResult:
    source_candidate_count = result.metadata.get("candidate_count")
    converged = bool(result.metadata.get("converged", isfinite(float(result.best_case.score))))
    stage_plan = _stage_plan_metadata(scenario)
    phase_duration_s = _dymos_phase_duration_s(result)
    tuned_pitch_point_indices = _tuned_pitch_point_indices(result)
    dymos_phase = _dymos_phase_metadata(
        result,
        scenario,
        tuned_pitch_point_indices=tuned_pitch_point_indices,
    )
    target_assessment = _target_insertion_assessment(result, scenario)
    metadata = {
        **result.metadata,
        "adapter": "dymos",
        "source_backend": result.backend,
        "source_candidate_count": source_candidate_count,
        "dymos_version": runtime.dymos_version,
        "openmdao_version": runtime.openmdao_version,
        "optimizer_status": result.metadata.get("optimizer_status", "completed"),
        "converged": converged,
        "iteration_count": len(result.iterations),
        "candidate_count": source_candidate_count,
        "best_score": result.best_case.score,
        "target_insertion_residuals": result.best_case.target_miss,
        "target_insertion_assessment": target_assessment,
        "target_insertion_satisfied": target_assessment["satisfied"],
        "stage_plan": stage_plan,
        "multistage": stage_plan["stage_count"] > 1,
        "dymos_tuned_pitch_point_indices": tuned_pitch_point_indices,
        "dymos_phase_covers_stage_schedule": (
            None
            if phase_duration_s is None
            else phase_duration_s >= stage_plan["total_burn_duration_s"] - 1.0e-9
        ),
        "path_constraints": _path_constraints(scenario, tuned_pitch_point_indices),
    }
    if dymos_phase is not None:
        metadata["dymos_phase"] = dymos_phase
    pitch_program_transcription_contract = _pitch_program_transcription_contract(
        result,
        scenario,
        stage_plan=stage_plan,
        tuned_pitch_point_indices=tuned_pitch_point_indices,
        dymos_phase=dymos_phase,
    )
    if pitch_program_transcription_contract is not None:
        metadata["dymos_pitch_program_transcription_contract"] = (
            pitch_program_transcription_contract
        )
    return result.model_copy(update={"backend": "dymos", "metadata": metadata})


def _target_insertion_assessment(
    result: LaunchPitchTuningResult,
    scenario: LaunchScenario,
) -> dict[str, Any]:
    altitude_miss_km = float(result.best_case.target_miss["altitude_miss_km"])
    velocity_miss_km_s = float(result.best_case.target_miss["velocity_miss_km_s"])
    radial_velocity_miss_km_s = float(
        result.best_case.target_miss.get(
            "radial_velocity_miss_km_s",
            result.best_case.final_radial_velocity_km_s,
        )
    )
    altitude_tolerance_km = float(scenario.target_orbit.altitude_tolerance_km)
    velocity_tolerance_km_s = float(scenario.target_orbit.velocity_tolerance_km_s)
    radial_velocity_tolerance_km_s = float(
        scenario.target_orbit.radial_velocity_tolerance_km_s
    )
    altitude_satisfied = abs(altitude_miss_km) <= altitude_tolerance_km
    velocity_satisfied = abs(velocity_miss_km_s) <= velocity_tolerance_km_s
    radial_velocity_satisfied = (
        abs(radial_velocity_miss_km_s) <= radial_velocity_tolerance_km_s
    )
    return {
        "status": (
            "within_tolerance"
            if altitude_satisfied and velocity_satisfied and radial_velocity_satisfied
            else "miss"
        ),
        "satisfied": altitude_satisfied and velocity_satisfied and radial_velocity_satisfied,
        "target_altitude_km": float(scenario.target_orbit.altitude_km),
        "target_circular_velocity_km_s": _circular_velocity_km_s(
            scenario.target_orbit.altitude_km
        ),
        "tolerances": {
            "altitude_tolerance_km": altitude_tolerance_km,
            "velocity_tolerance_km_s": velocity_tolerance_km_s,
            "radial_velocity_tolerance_km_s": radial_velocity_tolerance_km_s,
        },
        "residuals": {
            "altitude_miss_km": altitude_miss_km,
            "velocity_miss_km_s": velocity_miss_km_s,
            "radial_velocity_miss_km_s": radial_velocity_miss_km_s,
        },
        "component_status": {
            "altitude": "within_tolerance" if altitude_satisfied else "miss",
            "velocity": "within_tolerance" if velocity_satisfied else "miss",
            "radial_velocity": (
                "within_tolerance" if radial_velocity_satisfied else "miss"
            ),
        },
        "score": float(result.best_case.score),
        "score_weights": {
            "altitude_weight": float(result.altitude_weight),
            "velocity_weight": float(result.velocity_weight),
            "radial_velocity_weight": float(result.radial_velocity_weight),
        },
        "objective": "minimize_weighted_altitude_velocity_radial_insertion_residuals",
    }


def _stage_plan_metadata(scenario: LaunchScenario) -> dict[str, Any]:
    stages: list[dict[str, float | str]] = []
    start_s = 0.0
    for stage in scenario.vehicle.stages:
        burnout_s = start_s + stage.burn_duration_s
        stages.append(
            {
                "name": stage.name,
                "start_s": start_s,
                "burnout_s": burnout_s,
            }
        )
        start_s = burnout_s
    return {
        "stage_count": len(stages),
        "total_burn_duration_s": start_s,
        "stages": stages,
    }


def _dymos_phase_duration_s(result: LaunchPitchTuningResult) -> float | None:
    phase_metadata = result.metadata.get("dymos_phase")
    if not isinstance(phase_metadata, dict):
        return None
    duration_s = phase_metadata.get("duration_s")
    if duration_s is None:
        return None
    return float(duration_s)


def _tuned_pitch_point_indices(result: LaunchPitchTuningResult) -> list[int]:
    return [int(point.point_index) for point in result.tuned_points]


def _pitch_program_control_points(
    scenario: LaunchScenario,
    *,
    tuned_pitch_point_indices: list[int],
) -> list[dict[str, bool | float | int]]:
    if scenario.guidance.mode != "pitch_program":
        return []
    tuned_indices = set(tuned_pitch_point_indices)
    return [
        {
            "index": index,
            "time_s": point.time_s,
            "pitch_deg": point.pitch_deg,
            "tuned": index in tuned_indices,
        }
        for index, point in enumerate(scenario.guidance.pitch_program)
    ]


def _tuned_pitch_by_index(result: LaunchPitchTuningResult) -> dict[int, float]:
    return {
        int(point.point_index): float(point.tuned_pitch_deg)
        for point in result.tuned_points
    }


def _optimized_pitch_program_control_points(
    result: LaunchPitchTuningResult,
    scenario: LaunchScenario,
    *,
    tuned_pitch_point_indices: list[int],
) -> list[dict[str, bool | float | int | str]]:
    if scenario.guidance.mode != "pitch_program":
        return []
    tuned_indices = set(tuned_pitch_point_indices)
    tuned_pitch_by_index = _tuned_pitch_by_index(result)
    tuned_source = str(result.metadata.get("pitch_program_control_source", "suite_pitch_tuning"))
    return [
        {
            "index": index,
            "time_s": point.time_s,
            "baseline_pitch_deg": point.pitch_deg,
            "pitch_deg": tuned_pitch_by_index.get(index, point.pitch_deg),
            "tuned": index in tuned_indices,
            "source": tuned_source if index in tuned_indices else "baseline",
        }
        for index, point in enumerate(scenario.guidance.pitch_program)
    ]


def _pitch_program_stage_phases(
    scenario: LaunchScenario,
    stage_plan: dict[str, Any],
) -> list[dict[str, Any]]:
    stage_phases: list[dict[str, Any]] = []
    pitch_program = tuple(scenario.guidance.pitch_program)
    for stage in stage_plan["stages"]:
        start_s = float(stage["start_s"])
        burnout_s = float(stage["burnout_s"])
        control_point_indices = [
            index
            for index, point in enumerate(pitch_program)
            if start_s <= point.time_s <= burnout_s
        ]
        if control_point_indices:
            last_index = control_point_indices[-1]
            if (
                pitch_program[last_index].time_s < burnout_s
                and last_index + 1 < len(pitch_program)
            ):
                control_point_indices.append(last_index + 1)
        stage_phases.append(
            {
                "name": str(stage["name"]),
                "start_s": start_s,
                "burnout_s": burnout_s,
                "control_point_indices": control_point_indices,
            }
        )
    return stage_phases


def _pitch_program_control_schedule_covers_stage_plan(
    scenario: LaunchScenario,
    stage_plan: dict[str, Any],
) -> bool:
    pitch_program = tuple(scenario.guidance.pitch_program)
    if not pitch_program:
        return False
    return (
        pitch_program[0].time_s <= 0.0
        and pitch_program[-1].time_s >= float(stage_plan["total_burn_duration_s"])
    )


def _pitch_program_transcription_contract(
    result: LaunchPitchTuningResult,
    scenario: LaunchScenario,
    *,
    stage_plan: dict[str, Any],
    tuned_pitch_point_indices: list[int],
    dymos_phase: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if scenario.guidance.mode != "pitch_program":
        return None

    phase_coupling = "dymos_phase_plus_suite_pitch_tuning"
    transcription = "not_executed"
    if dymos_phase is not None:
        phase_coupling = str(
            dymos_phase.get("pitch_program_optimization_coupling", phase_coupling)
        )
        transcription = str(dymos_phase.get("transcription", transcription))

    return {
        "execution_status": str(
            result.metadata.get("dymos_pitch_program_transcription_status", "not_executed")
        ),
        "phase_coupling": phase_coupling,
        "control_name": "pitch_deg",
        "control_units": "deg",
        "control_bounds_deg": {"lower": 0.0, "upper": 90.0},
        "control_point_count": len(scenario.guidance.pitch_program),
        "tuned_control_point_indices": tuned_pitch_point_indices,
        "control_schedule_covers_stage_plan": (
            _pitch_program_control_schedule_covers_stage_plan(scenario, stage_plan)
        ),
        "transcription": transcription,
        "stage_phase_count": int(stage_plan["stage_count"]),
        "stage_phases": _pitch_program_stage_phases(scenario, stage_plan),
        "optimized_control_points": _optimized_pitch_program_control_points(
            result,
            scenario,
            tuned_pitch_point_indices=tuned_pitch_point_indices,
        ),
    }


def _dymos_phase_metadata(
    result: LaunchPitchTuningResult,
    scenario: LaunchScenario,
    *,
    tuned_pitch_point_indices: list[int],
) -> dict[str, Any] | None:
    phase_metadata = result.metadata.get("dymos_phase")
    if not isinstance(phase_metadata, dict):
        return None
    return {
        **phase_metadata,
        "guidance_model": scenario.guidance.mode,
        "pitch_program_control_points": _pitch_program_control_points(
            scenario,
            tuned_pitch_point_indices=tuned_pitch_point_indices,
        ),
        "optimized_pitch_program_control_points": _optimized_pitch_program_control_points(
            result,
            scenario,
            tuned_pitch_point_indices=tuned_pitch_point_indices,
        ),
        "pitch_program_optimization_coupling": result.metadata.get(
            "pitch_program_optimization_coupling",
            "dymos_phase_plus_suite_pitch_tuning",
        ),
        "pitch_program_optimization_scope": result.metadata.get(
            "pitch_program_optimization_scope",
            "suite_tuning_not_full_dymos_pitch_transcription",
        ),
    }


def _path_constraints(
    scenario: LaunchScenario,
    tuned_pitch_point_indices: list[int],
) -> dict[str, dict[str, Any]]:
    constraints: dict[str, dict[str, Any]] = {
        "pitch_deg": {"lower": 0.0, "upper": 90.0},
    }
    if scenario.guidance.mode == "pitch_program":
        constraints["pitch_program_control_points"] = {
            "count": len(scenario.guidance.pitch_program),
            "tuned_indices": tuned_pitch_point_indices,
        }
    return constraints


def _stage_acceleration_schedule(
    scenario: LaunchScenario,
) -> list[dict[str, float | str]]:
    schedule: list[dict[str, float | str]] = []
    start_s = 0.0
    mass_at_stage_start_kg = scenario.vehicle.initial_mass_kg
    for stage in scenario.vehicle.stages:
        burnout_s = start_s + stage.burn_duration_s
        thrust_acceleration_m_s2 = stage.engine.thrust_n / mass_at_stage_start_kg
        acceleration_m_s2 = max(thrust_acceleration_m_s2 - 9.80665, 1.0e-6)
        schedule.append(
            {
                "name": stage.name,
                "start_s": start_s,
                "burnout_s": burnout_s,
                "thrust_acceleration_m_s2": thrust_acceleration_m_s2,
                "acceleration_m_s2": acceleration_m_s2,
            }
        )
        mass_at_stage_start_kg -= stage.dry_mass_kg + stage.propellant_mass_kg
        start_s = burnout_s
    return schedule


def _total_burn_duration_s(scenario: LaunchScenario) -> float:
    return sum(stage.burn_duration_s for stage in scenario.vehicle.stages)


def _vertical_ascent_ode_class(
    runtime: DymosRuntime,
    acceleration_schedule: list[dict[str, float | str]],
) -> type[Any]:
    openmdao = runtime.openmdao_module
    explicit_component = openmdao.ExplicitComponent

    class VerticalAscentODE(explicit_component):  # type: ignore[misc, valid-type]
        def initialize(self) -> None:
            self.options.declare("num_nodes", types=int)

        def setup(self) -> None:
            num_nodes = self.options["num_nodes"]
            self.add_input("t", val=np.zeros(num_nodes), units="s")
            self.add_input("v", val=np.zeros(num_nodes), units="m/s")
            self.add_output("h_dot", val=np.zeros(num_nodes), units="m/s")
            self.add_output("v_dot", val=np.zeros(num_nodes), units="m/s**2")
            rows = np.arange(num_nodes)
            self.declare_partials("h_dot", "v", rows=rows, cols=rows, val=1.0)
            self.declare_partials("v_dot", "t", method="fd")

        def compute(self, inputs: Any, outputs: Any) -> None:
            time_s = np.asarray(inputs["t"])
            acceleration_m_s2 = np.full_like(time_s, -9.80665, dtype=np.float64)
            for stage in acceleration_schedule:
                start_s = float(stage["start_s"])
                burnout_s = float(stage["burnout_s"])
                stage_acceleration_m_s2 = float(stage["acceleration_m_s2"])
                active = (time_s >= start_s) & (time_s <= burnout_s)
                acceleration_m_s2 = np.where(active, stage_acceleration_m_s2, acceleration_m_s2)
            outputs["h_dot"] = inputs["v"]
            outputs["v_dot"] = acceleration_m_s2

    return VerticalAscentODE


def _pitch_program_ascent_ode_class(
    runtime: DymosRuntime,
    acceleration_schedule: list[dict[str, float | str]],
) -> type[Any]:
    openmdao = runtime.openmdao_module
    explicit_component = openmdao.ExplicitComponent

    class PitchProgramAscentODE(explicit_component):  # type: ignore[misc, valid-type]
        def initialize(self) -> None:
            self.options.declare("num_nodes", types=int)

        def setup(self) -> None:
            num_nodes = self.options["num_nodes"]
            self.add_input("t", val=np.zeros(num_nodes), units="s")
            self.add_input("vr", val=np.zeros(num_nodes), units="m/s")
            self.add_input("vh", val=np.zeros(num_nodes), units="m/s")
            self.add_input("pitch_deg", val=np.full(num_nodes, 90.0), units="deg")
            self.add_output("h_dot", val=np.zeros(num_nodes), units="m/s")
            self.add_output("downrange_dot", val=np.zeros(num_nodes), units="m/s")
            self.add_output("vr_dot", val=np.zeros(num_nodes), units="m/s**2")
            self.add_output("vh_dot", val=np.zeros(num_nodes), units="m/s**2")
            self.add_output("speed", val=np.zeros(num_nodes), units="m/s")
            self.declare_partials("*", "*", method="fd")

        def compute(self, inputs: Any, outputs: Any) -> None:
            time_s = np.asarray(inputs["t"])
            thrust_acceleration_m_s2 = np.zeros_like(time_s, dtype=np.float64)
            for stage in acceleration_schedule:
                start_s = float(stage["start_s"])
                burnout_s = float(stage["burnout_s"])
                stage_thrust_acceleration_m_s2 = float(stage["thrust_acceleration_m_s2"])
                active = (time_s >= start_s) & (time_s <= burnout_s)
                thrust_acceleration_m_s2 = np.where(
                    active,
                    stage_thrust_acceleration_m_s2,
                    thrust_acceleration_m_s2,
                )
            pitch_rad = np.deg2rad(np.asarray(inputs["pitch_deg"]))
            vertical_velocity_m_s = np.asarray(inputs["vr"])
            horizontal_velocity_m_s = np.asarray(inputs["vh"])
            outputs["h_dot"] = vertical_velocity_m_s
            outputs["downrange_dot"] = horizontal_velocity_m_s
            outputs["vr_dot"] = thrust_acceleration_m_s2 * np.sin(pitch_rad) - 9.80665
            outputs["vh_dot"] = thrust_acceleration_m_s2 * np.cos(pitch_rad)
            outputs["speed"] = np.sqrt(
                vertical_velocity_m_s**2 + horizontal_velocity_m_s**2
            )

    return PitchProgramAscentODE


def _driver_success_and_message(problem: Any) -> tuple[bool, str]:
    result = getattr(problem.driver, "result", None)
    success = bool(getattr(result, "success", True))
    message = str(getattr(result, "msg", getattr(result, "message", "completed")))
    return success, message


def _solve_dymos_vertical_phase(
    scenario: LaunchScenario,
    runtime: DymosRuntime,
) -> DymosPhaseSummary:
    openmdao = runtime.openmdao_module
    stage_schedule = _stage_acceleration_schedule(scenario)
    total_burn_duration_s = _total_burn_duration_s(scenario)
    num_segments = max(3, len(stage_schedule) * 2)
    order = 3
    duration_upper_s = max(
        total_burn_duration_s,
        min(float(scenario.propagation.duration_s), 300.0),
    )
    duration_guess_s = max(total_burn_duration_s, duration_upper_s / 2.0)

    problem = runtime.problem(model=openmdao.Group(), reports=False)
    problem.driver = openmdao.ScipyOptimizeDriver()
    problem.driver.options["optimizer"] = "SLSQP"
    problem.driver.options["disp"] = False

    trajectory = problem.model.add_subsystem("traj", runtime.trajectory())
    phase = runtime.phase(
        ode_class=_vertical_ascent_ode_class(runtime, stage_schedule),
        transcription=runtime.transcription(num_segments=num_segments, order=order),
    )
    trajectory.add_phase("phase0", phase)

    phase.set_time_options(
        fix_initial=True,
        duration_bounds=(total_burn_duration_s, duration_upper_s),
        targets=["t"],
        units="s",
    )
    phase.add_state(
        "h",
        fix_initial=True,
        fix_final=False,
        lower=0.0,
        rate_source="h_dot",
        targets=[],
        units="m",
    )
    phase.add_state(
        "v",
        fix_initial=True,
        fix_final=False,
        lower=0.0,
        rate_source="v_dot",
        targets=["v"],
        units="m/s",
    )
    phase.add_objective("h", loc="final", scaler=-1.0e-3)

    problem.model.linear_solver = openmdao.DirectSolver()
    problem.setup()
    phase.set_time_val(initial=0.0, duration=duration_guess_s, units="s")
    phase.set_state_val(
        "h",
        [0.0, 0.5 * duration_guess_s**2],
        units="m",
    )
    phase.set_state_val("v", [0.0, duration_guess_s], units="m/s")
    problem.run_driver()

    times_s = np.ravel(problem.get_val("traj.phase0.timeseries.time", units="s"))
    altitudes_m = np.ravel(problem.get_val("traj.phase0.timeseries.h", units="m"))
    velocities_m_s = np.ravel(problem.get_val("traj.phase0.timeseries.v", units="m/s"))
    success, message = _driver_success_and_message(problem)
    return DymosPhaseSummary(
        phase_model="stage_aware_vertical_ascent",
        transcription="GaussLobatto",
        num_segments=num_segments,
        order=order,
        duration_s=float(times_s[-1]),
        stage_count=len(stage_schedule),
        total_burn_duration_s=total_burn_duration_s,
        final_altitude_km=float(altitudes_m[-1] / 1000.0),
        final_velocity_km_s=float(velocities_m_s[-1] / 1000.0),
        optimizer_success=success,
        optimizer_message=message,
    )


def _dymos_pitch_tuning_indices(scenario: LaunchScenario) -> tuple[int, int]:
    if scenario.guidance.mode != "pitch_program":
        raise ValueError("Dymos pitch-program optimization requires pitch_program guidance")
    pitch_program = scenario.guidance.pitch_program
    candidate_indices = [index for index in range(1, len(pitch_program))]
    if len(candidate_indices) < 2:
        raise ValueError("Dymos pitch-program optimization requires two tunable pitch points")
    preferred = [index for index in (2, 3) if index in candidate_indices]
    for index in candidate_indices:
        if len(preferred) >= 2:
            break
        if index not in preferred:
            preferred.append(index)
    return (preferred[0], preferred[1])


def _scenario_with_optimized_pitch_program(
    scenario: LaunchScenario,
    optimized_pitch_deg_by_point_index: dict[int, float],
) -> LaunchScenario:
    pitch_program = [
        point.model_copy(
            update={"pitch_deg": optimized_pitch_deg_by_point_index.get(index, point.pitch_deg)}
        )
        for index, point in enumerate(scenario.guidance.pitch_program)
    ]
    return scenario.model_copy(
        update={"guidance": GuidanceConfig(mode="pitch_program", pitch_program=pitch_program)}
    )


def _pitch_program_summary_to_tuning_result(
    scenario: LaunchScenario,
    summary: DymosPitchProgramSummary,
) -> LaunchPitchTuningResult:
    point_indices = _dymos_pitch_tuning_indices(scenario)
    baseline = tune_pitch_program(scenario, point_indices=point_indices, iterations=1)
    optimized_pitch_deg_by_point_index = {
        point_index: float(summary.optimized_pitch_deg_by_point_index[point_index])
        for point_index in point_indices
    }
    tuned_points = [
        LaunchPitchTuningPoint(
            point_index=point_index,
            time_s=scenario.guidance.pitch_program[point_index].time_s,
            baseline_pitch_deg=scenario.guidance.pitch_program[point_index].pitch_deg,
            tuned_pitch_deg=optimized_pitch_deg_by_point_index[point_index],
        )
        for point_index in point_indices
    ]
    altitude_miss_km = float(summary.target_miss["altitude_miss_km"])
    velocity_miss_km_s = float(summary.target_miss["velocity_miss_km_s"])
    radial_velocity_miss_km_s = float(
        summary.target_miss.get(
            "radial_velocity_miss_km_s",
            summary.final_radial_velocity_km_s,
        )
    )
    target_miss = {
        **summary.target_miss,
        "radial_velocity_miss_km_s": radial_velocity_miss_km_s,
    }
    best_case = LaunchPitchTuningCase(
        iteration=1,
        pitch_deg_by_point_index={
            str(point_index): pitch_deg
            for point_index, pitch_deg in optimized_pitch_deg_by_point_index.items()
        },
        score=abs(altitude_miss_km)
        + abs(velocity_miss_km_s)
        + abs(radial_velocity_miss_km_s),
        altitude_miss_km=altitude_miss_km,
        velocity_miss_km_s=velocity_miss_km_s,
        radial_velocity_miss_km_s=radial_velocity_miss_km_s,
        final_altitude_km=summary.final_altitude_km,
        final_velocity_km_s=summary.final_velocity_km_s,
        final_radial_velocity_km_s=summary.final_radial_velocity_km_s,
        final_horizontal_velocity_km_s=summary.final_horizontal_velocity_km_s,
        final_downrange_km=summary.final_downrange_km,
        target_miss=target_miss,
    )
    iteration = LaunchPitchTuningIteration(
        iteration=1,
        span_deg=baseline.iterations[0].span_deg,
        cases=[best_case],
        best_case=best_case,
    )
    dymos_phase_metadata = summary.to_metadata()
    dymos_phase_metadata["target_miss"] = target_miss
    metadata = {
        **baseline.metadata,
        "workflow": "dymos_pitch_program_optimization",
        "candidate_count": len(summary.optimized_pitch_deg_by_point_index),
        "optimizer_status": "completed" if summary.optimizer_success else "failed",
        "converged": summary.optimizer_success,
        "dymos_phase": dymos_phase_metadata,
        "dymos_pitch_program_transcription_status": "executed",
        "pitch_program_control_source": "dymos_pitch_program_control",
        "pitch_program_optimization_coupling": "native_dymos_pitch_control",
        "pitch_program_optimization_scope": "native_dymos_pitch_program_transcription",
    }
    return baseline.model_copy(
        update={
            "point_indices": list(point_indices),
            "tuned_points": tuned_points,
            "iterations": [iteration],
            "best_case": best_case,
            "tuned_scenario": _scenario_with_optimized_pitch_program(
                scenario,
                optimized_pitch_deg_by_point_index,
            ),
            "backend": "dymos_pitch_program",
            "radial_velocity_weight": 1.0,
            "metadata": metadata,
        }
    )


def _solve_dymos_pitch_program_phase(
    scenario: LaunchScenario,
    runtime: DymosRuntime,
) -> DymosPitchProgramSummary:
    if scenario.guidance.mode != "pitch_program":
        raise ValueError("Dymos pitch-program optimization requires pitch_program guidance")

    openmdao = runtime.openmdao_module
    stage_schedule = _stage_acceleration_schedule(scenario)
    total_burn_duration_s = _total_burn_duration_s(scenario)
    num_segments = max(4, len(stage_schedule) * 2)
    order = 3

    problem = runtime.problem(model=openmdao.Group(), reports=False)
    problem.driver = openmdao.ScipyOptimizeDriver()
    problem.driver.options["optimizer"] = "SLSQP"
    problem.driver.options["disp"] = False

    trajectory = problem.model.add_subsystem("traj", runtime.trajectory())
    phase = runtime.phase(
        ode_class=_pitch_program_ascent_ode_class(runtime, stage_schedule),
        transcription=runtime.transcription(num_segments=num_segments, order=order),
    )
    trajectory.add_phase("phase0", phase)
    phase.set_time_options(
        fix_initial=True,
        fix_duration=True,
        duration_val=total_burn_duration_s,
        targets=["t"],
        units="s",
    )
    phase.add_state(
        "h",
        fix_initial=True,
        lower=0.0,
        rate_source="h_dot",
        targets=[],
        units="m",
    )
    phase.add_state(
        "downrange",
        fix_initial=True,
        lower=0.0,
        rate_source="downrange_dot",
        targets=[],
        units="m",
    )
    phase.add_state(
        "vr",
        fix_initial=True,
        rate_source="vr_dot",
        targets=["vr"],
        units="m/s",
    )
    phase.add_state(
        "vh",
        fix_initial=True,
        lower=0.0,
        rate_source="vh_dot",
        targets=["vh"],
        units="m/s",
    )
    phase.add_control(
        "pitch_deg",
        opt=True,
        lower=0.0,
        upper=90.0,
        units="deg",
        targets=["pitch_deg"],
        continuity=True,
        rate_continuity=True,
    )
    phase.add_boundary_constraint("h", loc="final", lower=0.0, units="m")
    phase.add_objective("vh", loc="final", scaler=-1.0e-3)

    problem.model.linear_solver = openmdao.DirectSolver()
    problem.setup()
    phase.set_time_val(initial=0.0, duration=total_burn_duration_s, units="s")
    phase.set_state_val("h", [0.0, 1000.0], units="m")
    phase.set_state_val("downrange", [0.0, 1000.0], units="m")
    phase.set_state_val("vr", [0.0, 10.0], units="m/s")
    phase.set_state_val("vh", [0.0, 100.0], units="m/s")
    pitch_times_s = [point.time_s for point in scenario.guidance.pitch_program]
    pitch_values_deg = [point.pitch_deg for point in scenario.guidance.pitch_program]
    phase.set_control_val(
        "pitch_deg",
        vals=pitch_values_deg,
        time_vals=pitch_times_s,
        units="deg",
    )
    problem.run_driver()

    times_s = np.ravel(problem.get_val("traj.phase0.timeseries.time", units="s"))
    altitudes_m = np.ravel(problem.get_val("traj.phase0.timeseries.h", units="m"))
    downrange_m = np.ravel(problem.get_val("traj.phase0.timeseries.downrange", units="m"))
    radial_velocities_m_s = np.ravel(problem.get_val("traj.phase0.timeseries.vr", units="m/s"))
    horizontal_velocities_m_s = np.ravel(
        problem.get_val("traj.phase0.timeseries.vh", units="m/s")
    )
    pitch_deg = np.ravel(problem.get_val("traj.phase0.timeseries.pitch_deg", units="deg"))
    final_radial_velocity_km_s = float(radial_velocities_m_s[-1] / 1000.0)
    final_horizontal_velocity_km_s = float(horizontal_velocities_m_s[-1] / 1000.0)
    final_velocity_km_s = sqrt(
        final_radial_velocity_km_s**2 + final_horizontal_velocity_km_s**2
    )
    final_altitude_km = float(altitudes_m[-1] / 1000.0)
    target_miss = {
        "altitude_miss_km": final_altitude_km - scenario.target_orbit.altitude_km,
        "velocity_miss_km_s": final_velocity_km_s
        - _circular_velocity_km_s(scenario.target_orbit.altitude_km),
        "radial_velocity_miss_km_s": final_radial_velocity_km_s,
    }
    point_indices = _dymos_pitch_tuning_indices(scenario)
    optimized_pitch_deg_by_point_index = {
        point_index: float(
            np.interp(
                scenario.guidance.pitch_program[point_index].time_s,
                times_s,
                pitch_deg,
            )
        )
        for point_index in point_indices
    }
    success, message = _driver_success_and_message(problem)
    return DymosPitchProgramSummary(
        phase_model="stage_aware_pitch_program_ascent",
        transcription="GaussLobatto",
        num_segments=num_segments,
        order=order,
        duration_s=float(times_s[-1]),
        stage_count=len(stage_schedule),
        total_burn_duration_s=total_burn_duration_s,
        final_altitude_km=final_altitude_km,
        final_velocity_km_s=final_velocity_km_s,
        final_radial_velocity_km_s=final_radial_velocity_km_s,
        final_horizontal_velocity_km_s=final_horizontal_velocity_km_s,
        final_downrange_km=float(downrange_m[-1] / 1000.0),
        target_miss=target_miss,
        optimized_pitch_deg_by_point_index=optimized_pitch_deg_by_point_index,
        optimizer_success=success,
        optimizer_message=message,
    )


def run_dymos_phase_optimization(
    scenario: LaunchScenario,
    runtime: DymosRuntime,
) -> LaunchPitchTuningResult:
    phase_summary = _solve_dymos_vertical_phase(scenario, runtime)
    tuned = tune_pitch_program(scenario, point_indices=(2, 3), iterations=1)
    metadata = {
        **tuned.metadata,
        "optimizer_status": "completed" if phase_summary.optimizer_success else "failed",
        "converged": phase_summary.optimizer_success,
        "dymos_phase": phase_summary.to_metadata(),
    }
    return tuned.model_copy(update={"backend": "dymos_phase", "metadata": metadata})


def run_dymos_pitch_program_optimization(
    scenario: LaunchScenario,
    runtime: DymosRuntime,
) -> LaunchPitchTuningResult:
    pitch_program_summary = _solve_dymos_pitch_program_phase(scenario, runtime)
    return _pitch_program_summary_to_tuning_result(scenario, pitch_program_summary)


def optimize_launch_dymos(
    scenario: LaunchScenario,
    *,
    runtime_loader: DymosRuntimeLoader = load_dymos_runtime,
    optimizer_runner: DymosOptimizerRunner | None = None,
) -> LaunchPitchTuningResult:
    runtime = runtime_loader()
    if optimizer_runner is None:
        optimizer_runner = run_dymos_phase_optimization

    result = optimizer_runner(scenario, runtime)
    return _with_dymos_provenance(result, scenario, runtime)
