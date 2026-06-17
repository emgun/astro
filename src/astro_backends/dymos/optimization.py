from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any

import numpy as np

from astro_backends.dymos.runtime import DymosRuntime, load_dymos_runtime
from astro_launch.models import LaunchPitchTuningResult, LaunchScenario
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
    return result.model_copy(update={"backend": "dymos", "metadata": metadata})


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
        acceleration_m_s2 = max(
            stage.engine.thrust_n / mass_at_stage_start_kg - 9.80665,
            1.0e-6,
        )
        schedule.append(
            {
                "name": stage.name,
                "start_s": start_s,
                "burnout_s": burnout_s,
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
