from importlib.metadata import entry_points

from typer import Typer
from typer.testing import CliRunner


def test_packages_import() -> None:
    import astro_backends
    import astro_backends.orekit
    import astro_cli
    import astro_core
    import astro_dynamics
    import astro_launch
    import astro_od

    expected_core_exports = {
        "AstroError",
        "CovarianceSample",
        "InvalidMeasurementFileError",
        "InvalidScenarioError",
        "J2_EARTH",
        "Maneuver",
        "MU_EARTH_KM3_S2",
        "NumericalConvergenceError",
        "R_EARTH_KM",
        "SECONDS_PER_DAY",
        "Scenario",
        "TrajectoryEvent",
        "UnsupportedBackendError",
        "load_trajectory",
    }
    expected_dynamics_exports = {
        "MonteCarloCase",
        "MonteCarloResult",
        "acceleration_km_s2",
        "apply_impulsive_maneuver",
        "derivative",
        "dump_trajectory_ephemeris_csv",
        "j2_acceleration_km_s2",
        "propagate_local",
        "propagate_with_backend",
        "rk4_step",
        "run_initial_state_monte_carlo",
        "two_body_acceleration_km_s2",
    }
    expected_od_exports = {
        "azimuth_deg",
        "declination_deg",
        "elevation_deg",
        "estimate_initial_state",
        "generate_synthetic_measurements",
        "load_measurements",
        "range_km",
        "range_rate_km_s",
        "right_ascension_deg",
    }
    expected_launch_exports = {
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
    }
    expected_orekit_exports = {
        "OrekitRuntime",
        "OrekitRuntimeUnavailable",
        "OrekitSmokeResult",
        "load_orekit_runtime",
        "propagate_orekit",
        "run_orekit_smoke",
    }

    assert expected_core_exports <= set(astro_core.__all__)
    assert astro_core.MU_EARTH_KM3_S2 == 398600.4418
    assert astro_core.R_EARTH_KM == 6378.1363
    assert astro_core.J2_EARTH == 1.08262668e-3
    assert astro_core.SECONDS_PER_DAY == 86400.0
    assert issubclass(astro_core.InvalidScenarioError, astro_core.AstroError)
    assert issubclass(astro_core.InvalidMeasurementFileError, astro_core.AstroError)
    assert issubclass(astro_core.UnsupportedBackendError, astro_core.AstroError)
    assert issubclass(astro_core.NumericalConvergenceError, astro_core.AstroError)
    assert set(astro_dynamics.__all__) == expected_dynamics_exports
    assert set(astro_od.__all__) == expected_od_exports
    assert set(astro_launch.__all__) == expected_launch_exports
    assert astro_backends.__all__ == []
    assert set(astro_backends.orekit.__all__) == expected_orekit_exports
    assert astro_cli.__all__ == ["app"]


def test_console_script_entry_point_loads() -> None:
    entry_point = next(
        entry_point
        for entry_point in entry_points(group="console_scripts")
        if entry_point.name == "astro" and entry_point.value == "astro_cli.main:app"
    )

    app = entry_point.load()

    assert isinstance(app, Typer)

    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Usage" in result.output
