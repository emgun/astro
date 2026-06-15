# Launch Report Assessments Implementation Plan

## Goal

Add explicit pass/fail assessment gates to tuned launch reports so the JSON product answers both
"what happened?" and "did it meet the configured target tolerances?".

## Orientation

The existing launch report already computes insertion metrics and short-arc orbital metrics. The
`TargetOrbit` model already carries `altitude_tolerance_km` and `velocity_tolerance_km_s`, so the
highest-signal path is to evaluate the report against those scenario-owned tolerances. This avoids a
second threshold configuration surface and keeps the launch report deterministic.

## Approach

1. Add report assessment models:
   - `LaunchReportCheck` for one measured value, tolerance, units, and boolean result.
   - `LaunchReportAssessment` for a group-level boolean and check list.
2. Extend `TunedLaunchReport` with:
   - `insertion_assessment`
   - `short_arc_assessment`
   - overall `passed`
3. Compute assessments in `astro_launch.reporting` from the existing metrics:
   - insertion altitude miss against `altitude_tolerance_km`
   - insertion velocity miss against `velocity_tolerance_km_s`
   - short-arc final altitude miss against `altitude_tolerance_km`
   - short-arc final velocity miss against `velocity_tolerance_km_s`
4. Expose the new models through `astro_launch.__all__`.
5. Update CLI/report tests and README wording.

## Verification

Use TDD for the behavior:
1. Add tests that assert report and CLI JSON assessment fields fail under the default example
   tolerances.
2. Add a loose-tolerance test that proves the same workflow can pass when configured tolerances
   admit the observed errors.
3. Run focused tests, tracked test-suite workaround, ruff on tracked files, mypy, and a CLI smoke
   check.

## Tradeoffs

This is intentionally a deterministic assessment of the current local report, not a new optimizer
or mission-success classifier. The report remains transparent: every pass/fail boolean is backed by
named numeric checks and explicit tolerances in the output.
