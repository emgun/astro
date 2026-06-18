# Release Checklist

Date: 2026-06-15

Use this checklist before tagging or publishing a release candidate.

## Required Local Gates

- [ ] `python -m pytest -q`
- [ ] `python -m ruff check .`
- [ ] `python -m mypy`
- [ ] `astro validate examples/scenarios/leo_two_body.yaml`
- [ ] `astro propagate examples/scenarios/leo_two_body.yaml --backend local --output /tmp/astro-local-trajectory.json`
- [ ] `astro propagate examples/scenarios/leo_eccentric_two_body.yaml --backend local --output /tmp/astro-eccentric-trajectory.json`
- [ ] `astro propagate examples/scenarios/meo_two_body.yaml --backend local --output /tmp/astro-meo.json`
- [ ] `astro propagate examples/scenarios/geo_two_body.yaml --backend local --output /tmp/astro-geo.json`
- [ ] `astro propagate examples/scenarios/leo_finite_burn.yaml --backend local --output /tmp/astro-finite-burn.json`
- [ ] `astro propagate examples/scenarios/leo_velocity_aligned_burn.yaml --backend local --output /tmp/astro-velocity-aligned-burn.json`
- [ ] `astro propagate examples/scenarios/leo_radial_burn.yaml --backend local --output /tmp/astro-radial-burn.json`
- [ ] `astro propagate examples/scenarios/leo_covariance.yaml --backend local --output /tmp/astro-covariance.json`
- [ ] `astro propagate examples/scenarios/leo_variational_covariance.yaml --backend local --output /tmp/astro-variational-covariance.json`
- [ ] `astro propagate examples/scenarios/leo_j2_variational_covariance.yaml --backend local --output /tmp/astro-j2-variational-covariance.json`
- [ ] `astro export-trajectory /tmp/astro-local-trajectory.json --format csv --output /tmp/astro-local-trajectory.csv`
- [ ] `astro export-trajectory /tmp/astro-local-trajectory.json --format oem --output /tmp/astro-local-trajectory.oem`
- [ ] `astro export-trajectory /tmp/astro-local-trajectory.json --format opm --output /tmp/astro-local-state.opm`
- [ ] `astro import-trajectory examples/trajectories/leo_initial_state.opm --format opm --scenario examples/scenarios/leo_two_body.yaml --output /tmp/astro-local-state-from-opm.json`
- [ ] `astro propagate examples/scenarios/leo_velocity_aligned_burn.yaml --backend local --output /tmp/astro-velocity-aligned-burn.json`
- [ ] `astro export-trajectory /tmp/astro-velocity-aligned-burn.json --format aem --output /tmp/astro-attitude.aem`
- [ ] `astro import-trajectory /tmp/astro-attitude.aem --format aem --scenario examples/scenarios/leo_velocity_aligned_burn.yaml --state-trajectory /tmp/astro-velocity-aligned-burn.json --output /tmp/astro-attitude-from-aem.json`
- [ ] `astro propagate-attitude examples/attitude/rigid_body_torque.yaml --output /tmp/astro-attitude-dynamics.json`
- [ ] `astro propagate-attitude examples/attitude/closed_loop_pd.yaml --output /tmp/astro-attitude-control.json`
- [ ] `astro propagate-attitude examples/attitude/closed_loop_sensor_actuator.yaml --output /tmp/astro-attitude-sensor-actuator.json`
- [ ] `astro screen-conjunction /tmp/astro-covariance.json /tmp/astro-covariance.json --threshold-km 1.0 --hard-body-radius-km 0.02 --probability-method integrated --output /tmp/astro-conjunction-screening.json`
- [ ] `astro assess-conjunction /tmp/astro-conjunction-screening.json --output /tmp/astro-conjunction-assessment.json`
- [ ] `astro synth-measurements examples/scenarios/leo_two_station_od.yaml --backend local --output /tmp/astro-measurements.json`
- [ ] `astro synth-measurements examples/scenarios/leo_doppler.yaml --backend local --output /tmp/astro-doppler-measurements.json`
- [ ] `astro export-measurements /tmp/astro-doppler-measurements.json --format tdm --output /tmp/astro-doppler-measurements.tdm`
- [ ] `astro synth-measurements examples/scenarios/leo_radiometric_media.yaml --backend local --output /tmp/astro-radiometric-media.json`
- [ ] `astro synth-measurements examples/scenarios/leo_radiometric_weather_frequency.yaml --backend local --output /tmp/astro-radiometric-weather-frequency.json`
- [ ] `astro dsn-calibration examples/scenarios/leo_radiometric_weather_frequency.yaml --backend local --output /tmp/astro-dsn-calibration.json`
- [ ] `astro export-measurements /tmp/astro-radiometric-weather-frequency.json --format tdm --output /tmp/astro-radiometric-weather-frequency.tdm`
- [ ] `astro dsn-calibration examples/scenarios/leo_radiometric_weather_frequency.yaml --measurements /tmp/astro-radiometric-weather-frequency.tdm --format tdm --output /tmp/astro-dsn-calibration-from-tdm.json`
- [ ] `astro import-dsn-tracking examples/measurements/dsn_tracking_normalized.csv --output /tmp/astro-dsn-tracking-measurements.json`
- [ ] `astro import-dsn-kvn-tracking examples/measurements/dsn_tracking_kvn.txt --output /tmp/astro-dsn-kvn-tracking-measurements.json`
- [ ] `python -m pytest tests/astro_od/test_dsn_tracking.py::test_load_dsn_binary_tracking_measurements_maps_fixed_records tests/astro_cli/test_cli.py::test_import_dsn_binary_tracking_command_writes_measurement_json -q`
- [ ] `astro station-calibration examples/scenarios/leo_two_station_od.yaml examples/measurements/leo_two_station_od_measurements.json --output /tmp/astro-station-calibration.json`
- [ ] `astro synth-measurements examples/scenarios/leo_two_station_angles.yaml --backend local --output /tmp/astro-angle-measurements.json`
- [ ] `astro synth-measurements examples/scenarios/leo_two_station_topocentric.yaml --backend local --output /tmp/astro-topocentric-measurements.json`
- [ ] `astro synth-measurements examples/scenarios/leo_geodetic_precession_nutation_topocentric.yaml --backend local --output /tmp/astro-geodetic-precession-nutation-measurements.json`
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
- [ ] `astro propagate examples/scenarios/leo_orekit_high_order_gravity.yaml --backend orekit --output /tmp/astro-orekit-high-order-gravity.json`
- [ ] `astro propagate examples/scenarios/leo_covariance.yaml --backend orekit --output /tmp/astro-orekit-covariance.json`
- [ ] `astro propagate examples/scenarios/leo_orekit_high_fidelity_covariance.yaml --backend orekit --output /tmp/astro-orekit-high-fidelity-covariance.json`
- [ ] `ASTRO_RUN_OREKIT_LIVE=1 python -m pytest tests/astro_backends/test_orekit_propagation.py::test_live_orekit_covariance_history_returns_suite_product -q`
- [ ] `ASTRO_RUN_OREKIT_LIVE=1 python -m pytest tests/astro_backends/test_orekit_estimation.py::test_live_orekit_native_od_executes_batch_estimator -q`
- [ ] `astro estimate-measurements <geodetic-range-rate-scenario.yaml> <measurements.json> --estimator orekit-native --output /tmp/astro-orekit-native-estimate.json`
- [ ] `astro rocketpy-smoke`
- [ ] `astro launch examples/launch/rocketpy_configured_single_stage.yaml --backend rocketpy --output /tmp/astro-rocketpy-launch.json`
- [ ] `ASTRO_RUN_ROCKETPY_LIVE=1 python -m pytest tests/astro_backends/test_rocketpy_simulation.py::test_live_rocketpy_configured_launch_examples_return_suite_products -q`
- [ ] `astro dymos-smoke`
- [ ] `astro optimize-launch examples/launch/pitch_program_two_stage.yaml --backend dymos --output /tmp/astro-dymos-optimized-launch.json`
- [ ] `astro optimize-launch examples/launch/pitch_program_two_stage.yaml --backend dymos --dymos-mode pitch-program --output /tmp/astro-dymos-pitch-program-launch.json`
- [ ] `ASTRO_RUN_DYMOS_LIVE=1 python -m pytest tests/astro_backends/test_dymos_optimization.py::test_live_dymos_optimization_returns_suite_product tests/astro_backends/test_dymos_optimization.py::test_live_dymos_pitch_program_optimization_executes_native_transcription -q`
- [ ] `astro tudat-smoke`
- [ ] `astro propagate examples/scenarios/leo_two_body.yaml --backend tudat --output /tmp/astro-tudat-two-body.json`
- [ ] `astro propagate examples/scenarios/leo_j2.yaml --backend tudat --output /tmp/astro-tudat-j2.json`
- [ ] `astro propagate examples/scenarios/leo_orekit_drag.yaml --backend tudat --output /tmp/astro-tudat-drag.json`
- [ ] `astro propagate examples/scenarios/leo_orekit_srp.yaml --backend tudat --output /tmp/astro-tudat-srp.json`
- [ ] `astro propagate examples/scenarios/leo_orekit_third_body.yaml --backend tudat --output /tmp/astro-tudat-third-body.json`
- [ ] `astro propagate examples/scenarios/leo_tudat_high_order_gravity.yaml --backend tudat --output /tmp/astro-tudat-high-order-gravity.json`
- [ ] `astro propagate examples/scenarios/leo_orekit_high_fidelity_covariance.yaml --backend tudat --output /tmp/astro-tudat-high-fidelity-covariance.json`
- [ ] `astro propagate examples/scenarios/leo_tudat_variational_covariance.yaml --backend tudat --output /tmp/astro-tudat-variational-covariance.json`
- [ ] `python -m pytest tests/astro_backends/test_tudat_propagation.py::test_propagate_tudat_uses_default_native_variational_runner_when_requested -q`
- [ ] `python -m pytest tests/astro_backends/test_tudat_propagation.py::test_propagate_tudat_uses_native_variational_runner_when_requested -q`
- [ ] `astro compare-tudat-reference examples/scenarios/leo_two_body.yaml --reference-backend local --position-tolerance-km 0.001 --velocity-tolerance-km-s 0.000001 --output /tmp/astro-tudat-reference-comparison.json`
- [ ] `astro compare-tudat-campaign examples/scenarios/leo_two_body.yaml examples/scenarios/leo_j2.yaml --reference-backend local --position-tolerance-km 0.001 --velocity-tolerance-km-s 0.000001 --output /tmp/astro-tudat-reference-campaign.json`
- [ ] `astro jax-smoke`
- [ ] `astro research-estimate examples/scenarios/leo_two_station_od.yaml examples/measurements/leo_two_station_od_measurements.json --backend jax --max-iterations 5 --output /tmp/astro-jax-research-estimate.json`
- [ ] `astro synth-measurements examples/scenarios/leo_two_station_angles.yaml --backend local --output /tmp/astro-angle-measurements.json`
- [ ] `astro research-od-sensitivity examples/scenarios/leo_two_station_angles.yaml /tmp/astro-angle-measurements.json --backend jax --output /tmp/astro-jax-angle-sensitivity.json`
- [ ] `astro synth-measurements examples/scenarios/leo_two_station_topocentric.yaml --backend local --output /tmp/astro-topocentric-measurements.json`
- [ ] `astro research-od-sensitivity examples/scenarios/leo_two_station_topocentric.yaml /tmp/astro-topocentric-measurements.json --backend jax --output /tmp/astro-jax-topocentric-sensitivity.json`
- [ ] `astro research-estimate examples/scenarios/leo_two_station_topocentric.yaml /tmp/astro-topocentric-measurements.json --backend jax --max-iterations 8 --output /tmp/astro-jax-topocentric-estimate.json`
- [ ] `astro research-propagate examples/scenarios/leo_orekit_drag.yaml --backend jax --cases 1 --position-sigma-km 0 --velocity-sigma-km-s 0 --seed 7 --output /tmp/astro-jax-drag-research.json`
- [ ] `astro research-propagate examples/scenarios/leo_orekit_srp.yaml --backend jax --cases 1 --position-sigma-km 0 --velocity-sigma-km-s 0 --seed 7 --output /tmp/astro-jax-srp-research.json`
- [ ] `astro research-propagate examples/scenarios/leo_jax_high_order_gravity_research.yaml --backend jax --cases 1 --position-sigma-km 0 --velocity-sigma-km-s 0 --seed 7 --output /tmp/astro-jax-high-order-research.json`
- [ ] `astro research-propagate examples/scenarios/leo_jax_third_body_research.yaml --backend jax --cases 1 --position-sigma-km 0 --velocity-sigma-km-s 0 --seed 7 --output /tmp/astro-jax-third-body-research.json`
- [ ] `astro research-propagate examples/scenarios/leo_jax_third_body_ephemeris_research.yaml --backend jax --cases 1 --position-sigma-km 0 --velocity-sigma-km-s 0 --seed 7 --output /tmp/astro-jax-third-body-ephemeris-research.json`

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
