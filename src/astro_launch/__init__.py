from astro_launch.handoff import launch_trajectory_to_orbit_scenario
from astro_launch.io import load_launch_scenario, load_launch_trajectory, load_tuned_launch_report
from astro_launch.local import propagate_launch_local
from astro_launch.models import (
    AtmosphereConfig,
    GuidanceConfig,
    LaunchEngine,
    LaunchEvent,
    LaunchPitchSweepCase,
    LaunchPitchSweepResult,
    LaunchPitchTuningCase,
    LaunchPitchTuningIteration,
    LaunchPitchTuningPoint,
    LaunchPitchTuningResult,
    LaunchPropagationConfig,
    LaunchReportAssessment,
    LaunchReportCheck,
    LaunchReportInsertionMetrics,
    LaunchReportMetricDelta,
    LaunchReportShortArcMetrics,
    LaunchRocketPyConfig,
    LaunchScenario,
    LaunchSite,
    LaunchStage,
    LaunchTrajectory,
    LaunchTrajectorySample,
    LaunchVehicle,
    PitchProgramPoint,
    TargetOrbit,
    TunedLaunchReport,
    TunedLaunchReportBatch,
    TunedLaunchReportBatchCase,
    TunedLaunchReportComparison,
)
from astro_launch.reporting import (
    compare_tuned_launch_reports,
    generate_tuned_launch_report,
    generate_tuned_launch_report_batch,
)
from astro_launch.targeting import sweep_pitch_program, tune_pitch_program


def propagate_launch_with_backend(
    scenario: LaunchScenario,
    backend: str,
) -> LaunchTrajectory:
    from astro_launch.backends import propagate_launch_with_backend as _propagate

    return _propagate(scenario, backend)

__all__ = [
    "AtmosphereConfig",
    "GuidanceConfig",
    "LaunchEngine",
    "LaunchEvent",
    "LaunchPitchSweepCase",
    "LaunchPitchSweepResult",
    "LaunchPitchTuningCase",
    "LaunchPitchTuningIteration",
    "LaunchPitchTuningPoint",
    "LaunchPitchTuningResult",
    "LaunchPropagationConfig",
    "LaunchReportAssessment",
    "LaunchReportCheck",
    "LaunchReportInsertionMetrics",
    "LaunchReportMetricDelta",
    "LaunchReportShortArcMetrics",
    "LaunchRocketPyConfig",
    "LaunchScenario",
    "LaunchSite",
    "LaunchStage",
    "LaunchTrajectory",
    "LaunchTrajectorySample",
    "LaunchVehicle",
    "PitchProgramPoint",
    "TargetOrbit",
    "TunedLaunchReport",
    "TunedLaunchReportBatch",
    "TunedLaunchReportBatchCase",
    "TunedLaunchReportComparison",
    "compare_tuned_launch_reports",
    "generate_tuned_launch_report",
    "generate_tuned_launch_report_batch",
    "launch_trajectory_to_orbit_scenario",
    "load_launch_scenario",
    "load_launch_trajectory",
    "load_tuned_launch_report",
    "propagate_launch_with_backend",
    "propagate_launch_local",
    "sweep_pitch_program",
    "tune_pitch_program",
]
