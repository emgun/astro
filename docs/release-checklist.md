# Release Checklist

Date: 2026-06-15

Use this checklist before tagging or publishing a release candidate.

## Required Local Gates

- [ ] `python -m pytest -q`
- [ ] `python -m ruff check .`
- [ ] `python -m mypy`
- [ ] `astro validate examples/scenarios/leo_two_body.yaml`
- [ ] `astro propagate examples/scenarios/leo_two_body.yaml --backend local --output /tmp/astro-local-trajectory.json`
- [ ] `astro propagate examples/scenarios/meo_two_body.yaml --backend local --output /tmp/astro-meo.json`
- [ ] `astro propagate examples/scenarios/geo_two_body.yaml --backend local --output /tmp/astro-geo.json`
- [ ] `astro propagate examples/scenarios/leo_finite_burn.yaml --backend local --output /tmp/astro-finite-burn.json`
- [ ] `astro propagate examples/scenarios/leo_radial_burn.yaml --backend local --output /tmp/astro-radial-burn.json`
- [ ] `astro propagate examples/scenarios/leo_covariance.yaml --backend local --output /tmp/astro-covariance.json`
- [ ] `astro export-trajectory /tmp/astro-local-trajectory.json --format csv --output /tmp/astro-local-trajectory.csv`
- [ ] `astro synth-measurements examples/scenarios/leo_two_station_od.yaml --backend local --output /tmp/astro-measurements.json`
- [ ] `astro synth-measurements examples/scenarios/leo_radiometric_media.yaml --backend local --output /tmp/astro-radiometric-media.json`
- [ ] `astro synth-measurements examples/scenarios/leo_two_station_angles.yaml --backend local --output /tmp/astro-angle-measurements.json`
- [ ] `astro synth-measurements examples/scenarios/leo_two_station_topocentric.yaml --backend local --output /tmp/astro-topocentric-measurements.json`
- [ ] `astro estimate-measurements examples/scenarios/leo_two_station_od.yaml examples/measurements/leo_two_station_od_measurements.json --backend local --output /tmp/astro-local-estimate.json`
- [ ] `astro launch examples/launch/pitch_program_two_stage.yaml --backend local --output /tmp/astro-launch.json`
- [ ] `python -m pytest tests/astro_launch/test_launch_io.py::test_load_rocketpy_configured_launch_scenario -q`
- [ ] `astro optimize-launch examples/launch/pitch_program_two_stage.yaml --backend local --point-indices 2,3 --iterations 1 --output /tmp/astro-optimized-launch.json`
- [ ] `astro research-propagate examples/scenarios/leo_two_body.yaml --backend local --cases 2 --position-sigma-km 0.01 --velocity-sigma-km-s 0.000001 --seed 7 --output /tmp/astro-research.json`

## Optional Backend Gates

Run when the matching runtime is expected to be present:

- [ ] `astro orekit-smoke`
- [ ] `ASTRO_RUN_OREKIT_LIVE=1 python -m pytest tests/astro_backends/test_orekit_propagation.py::test_live_orekit_two_body_matches_local_reference -v`
- [ ] `ASTRO_RUN_OREKIT_LIVE=1 python -m pytest tests/astro_backends/test_orekit_propagation.py::test_live_orekit_j2_matches_local_reference_scale -v`
- [ ] `astro propagate examples/scenarios/leo_orekit_high_fidelity.yaml --backend orekit --output /tmp/astro-orekit-high-fidelity.json`
- [ ] `astro propagate examples/scenarios/leo_orekit_drag.yaml --backend orekit --output /tmp/astro-orekit-drag.json`
- [ ] `astro propagate examples/scenarios/leo_orekit_srp.yaml --backend orekit --output /tmp/astro-orekit-srp.json`
- [ ] `astro propagate examples/scenarios/leo_orekit_third_body.yaml --backend orekit --output /tmp/astro-orekit-third-body.json`
- [ ] `astro propagate examples/scenarios/leo_covariance.yaml --backend orekit --output /tmp/astro-orekit-covariance.json`
- [ ] `ASTRO_RUN_OREKIT_LIVE=1 python -m pytest tests/astro_backends/test_orekit_propagation.py::test_live_orekit_covariance_history_returns_suite_product -q`
- [ ] `ASTRO_RUN_OREKIT_LIVE=1 python -m pytest tests/astro_backends/test_orekit_estimation.py::test_live_orekit_native_od_executes_batch_estimator -q`
- [ ] `astro estimate-measurements <geodetic-range-rate-scenario.yaml> <measurements.json> --estimator orekit-native --output /tmp/astro-orekit-native-estimate.json`
- [ ] `astro rocketpy-smoke`
- [ ] `astro launch examples/launch/rocketpy_configured_single_stage.yaml --backend rocketpy --output /tmp/astro-rocketpy-launch.json`
- [ ] `ASTRO_RUN_ROCKETPY_LIVE=1 python -m pytest tests/astro_backends/test_rocketpy_simulation.py::test_live_rocketpy_configured_launch_examples_return_suite_products -q`
- [ ] `astro dymos-smoke`
- [ ] `astro optimize-launch examples/launch/pitch_program_two_stage.yaml --backend dymos --output /tmp/astro-dymos-optimized-launch.json`
- [ ] `ASTRO_RUN_DYMOS_LIVE=1 python -m pytest tests/astro_backends/test_dymos_optimization.py::test_live_dymos_optimization_returns_suite_product -q`
- [ ] `astro tudat-smoke`
- [ ] `astro jax-smoke`
- [ ] `astro research-propagate examples/scenarios/leo_orekit_drag.yaml --backend jax --cases 1 --position-sigma-km 0 --velocity-sigma-km-s 0 --seed 7 --output /tmp/astro-jax-drag-research.json`
- [ ] `astro research-propagate examples/scenarios/leo_orekit_srp.yaml --backend jax --cases 1 --position-sigma-km 0 --velocity-sigma-km-s 0 --seed 7 --output /tmp/astro-jax-srp-research.json`

If an optional runtime is intentionally absent, capture the structured unavailable JSON and confirm
the message is actionable.

## Documentation Gates

- [ ] README current-scope and command list match implemented behavior.
- [ ] `docs/validation-matrix.md` reflects current command names and tolerances.
- [ ] `docs/backend-installation.md` documents every optional extra and non-pip install caveat.
- [ ] Roadmap plan statuses distinguish implemented product boundaries from live backend work that
  still requires external configuration.

## Packaging Gate

- [ ] Build succeeds with `python -m build` when build tooling is installed.
- [ ] Wheel metadata includes optional extras: `dev`, `orekit`, `launch`, `optimization`, and
  `research`.
