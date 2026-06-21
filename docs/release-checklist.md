# Release Checklist

Date: 2026-06-20

Use this checklist before tagging or publishing a release candidate.

Latest merged-main evidence:

- 2026-06-20: `main` fast-forwarded to `0fb9a87`.
- 2026-06-20: Release evidence recorded on `main` at `870cb13`.
- 2026-06-20: `python -m pytest -q` passed with 505 passed, 11 skipped.
- 2026-06-20: `python -m ruff check .`, `python -m mypy`, `git diff --check`, and
  `python -m build` passed.
- 2026-06-20: Required local CLI checklist passed 42 command gates on merged `main`.
- 2026-06-20 17:50 PDT: Optional backend smoke refresh passed on current checkout for Orekit,
  RocketPy, Dymos/OpenMDAO, TudatPy, and JAX. These smoke checks do not replace optional live
  propagation, launch, optimization, covariance, or OD campaign gates.

## Required Local Gates

- [x] `python -m pytest -q`
- [x] `python -m ruff check .`
- [x] `python -m mypy`
- [x] `astro validate examples/scenarios/leo_two_body.yaml`
- [x] `astro propagate examples/scenarios/leo_two_body.yaml --backend local --output /tmp/astro-local-trajectory.json`
- [x] `astro propagate examples/scenarios/leo_eccentric_two_body.yaml --backend local --output /tmp/astro-eccentric-trajectory.json`
- [x] `astro propagate examples/scenarios/meo_two_body.yaml --backend local --output /tmp/astro-meo.json`
- [x] `astro propagate examples/scenarios/geo_two_body.yaml --backend local --output /tmp/astro-geo.json`
- [x] `astro propagate examples/scenarios/leo_finite_burn.yaml --backend local --output /tmp/astro-finite-burn.json`
- [x] `astro propagate examples/scenarios/leo_velocity_aligned_burn.yaml --backend local --output /tmp/astro-velocity-aligned-burn.json`
- [x] `astro propagate examples/scenarios/leo_radial_burn.yaml --backend local --output /tmp/astro-radial-burn.json`
- [x] `astro propagate examples/scenarios/leo_covariance.yaml --backend local --output /tmp/astro-covariance.json`
- [x] `astro propagate examples/scenarios/leo_variational_covariance.yaml --backend local --output /tmp/astro-variational-covariance.json`
- [x] `astro propagate examples/scenarios/leo_j2_variational_covariance.yaml --backend local --output /tmp/astro-j2-variational-covariance.json`
- [x] `astro export-trajectory /tmp/astro-local-trajectory.json --format csv --output /tmp/astro-local-trajectory.csv`
- [x] `astro export-trajectory /tmp/astro-local-trajectory.json --format oem --output /tmp/astro-local-trajectory.oem`
- [x] `astro export-trajectory /tmp/astro-local-trajectory.json --format opm --output /tmp/astro-local-state.opm`
- [x] `astro import-trajectory examples/trajectories/leo_initial_state.opm --format opm --scenario examples/scenarios/leo_two_body.yaml --output /tmp/astro-local-state-from-opm.json`
- [x] `astro propagate examples/scenarios/leo_velocity_aligned_burn.yaml --backend local --output /tmp/astro-velocity-aligned-burn.json`
- [x] `astro export-trajectory /tmp/astro-velocity-aligned-burn.json --format aem --output /tmp/astro-attitude.aem`
- [x] `astro import-trajectory /tmp/astro-attitude.aem --format aem --scenario examples/scenarios/leo_velocity_aligned_burn.yaml --state-trajectory /tmp/astro-velocity-aligned-burn.json --output /tmp/astro-attitude-from-aem.json`
- [x] `astro propagate-attitude examples/attitude/rigid_body_torque.yaml --output /tmp/astro-attitude-dynamics.json`
- [x] `astro propagate-attitude examples/attitude/closed_loop_pd.yaml --output /tmp/astro-attitude-control.json`
- [x] `astro propagate-attitude examples/attitude/closed_loop_sensor_actuator.yaml --output /tmp/astro-attitude-sensor-actuator.json`
- [x] `astro screen-conjunction /tmp/astro-covariance.json /tmp/astro-covariance.json --threshold-km 1.0 --hard-body-radius-km 0.02 --probability-method integrated --output /tmp/astro-conjunction-screening.json`
- [x] `astro assess-conjunction /tmp/astro-conjunction-screening.json --output /tmp/astro-conjunction-assessment.json`
- [x] `astro synth-measurements examples/scenarios/leo_two_station_od.yaml --backend local --output /tmp/astro-measurements.json`
- [x] `astro synth-measurements examples/scenarios/leo_doppler.yaml --backend local --output /tmp/astro-doppler-measurements.json`
- [x] `astro export-measurements /tmp/astro-doppler-measurements.json --format tdm --output /tmp/astro-doppler-measurements.tdm`
- [x] `astro synth-measurements examples/scenarios/leo_radiometric_media.yaml --backend local --output /tmp/astro-radiometric-media.json`
- [x] `astro synth-measurements examples/scenarios/leo_radiometric_weather_frequency.yaml --backend local --output /tmp/astro-radiometric-weather-frequency.json`
- [x] `astro dsn-calibration examples/scenarios/leo_radiometric_weather_frequency.yaml --backend local --output /tmp/astro-dsn-calibration.json`
- [x] `astro export-measurements /tmp/astro-radiometric-weather-frequency.json --format tdm --output /tmp/astro-radiometric-weather-frequency.tdm`
- [x] `astro dsn-calibration examples/scenarios/leo_radiometric_weather_frequency.yaml --measurements /tmp/astro-radiometric-weather-frequency.tdm --format tdm --output /tmp/astro-dsn-calibration-from-tdm.json`
- [x] `astro import-dsn-tracking examples/measurements/dsn_tracking_normalized.csv --output /tmp/astro-dsn-tracking-measurements.json`
- [x] `astro import-dsn-kvn-tracking examples/measurements/dsn_tracking_kvn.txt --output /tmp/astro-dsn-kvn-tracking-measurements.json`
- [x] `python -m pytest tests/astro_od/test_dsn_tracking.py::test_load_dsn_binary_tracking_measurements_maps_fixed_records tests/astro_cli/test_cli.py::test_import_dsn_binary_tracking_command_writes_measurement_json -q`
- [x] `astro station-calibration examples/scenarios/leo_two_station_od.yaml examples/measurements/leo_two_station_od_measurements.json --output /tmp/astro-station-calibration.json`
- [x] `astro synth-measurements examples/scenarios/leo_two_station_angles.yaml --backend local --output /tmp/astro-angle-measurements.json`
- [x] `astro synth-measurements examples/scenarios/leo_two_station_topocentric.yaml --backend local --output /tmp/astro-topocentric-measurements.json`
- [x] `astro synth-measurements examples/scenarios/leo_geodetic_precession_nutation_topocentric.yaml --backend local --output /tmp/astro-geodetic-precession-nutation-measurements.json`
- [x] `astro estimate-measurements examples/scenarios/leo_two_station_od.yaml examples/measurements/leo_two_station_od_measurements.json --backend local --output /tmp/astro-local-estimate.json`
- [x] `astro launch examples/launch/pitch_program_two_stage.yaml --backend local --output /tmp/astro-launch.json`
- [x] `python -m pytest tests/astro_launch/test_launch_io.py::test_load_rocketpy_configured_launch_scenario -q`
- [x] `astro optimize-launch examples/launch/pitch_program_two_stage.yaml --backend local --point-indices 2,3 --iterations 1 --radial-velocity-weight 1 --output /tmp/astro-optimized-launch.json`
- [x] `astro research-propagate examples/scenarios/leo_two_body.yaml --backend local --cases 2 --position-sigma-km 0.01 --velocity-sigma-km-s 0.000001 --seed 7 --output /tmp/astro-research.json`

## Optional Backend Gates

Run when the matching runtime is expected to be present:

If a backend runtime is intentionally absent, capture the smoke command's structured unavailable
JSON and treat the gate as not-run rather than failed or complete.

- [x] Record the optional backend campaign outcome in `docs/validation/live-backend-campaigns.md`
  before promoting any optional live gate.
- [x] `astro orekit-smoke`
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
- [ ] `ASTRO_RUN_OREKIT_LIVE=1 python -m pytest tests/astro_backends/test_orekit_propagation.py::test_live_orekit_high_fidelity_covariance_records_force_models -q`
- [ ] `ASTRO_RUN_OREKIT_LIVE=1 python -m pytest tests/astro_backends/test_orekit_estimation.py::test_live_orekit_native_od_executes_batch_estimator -q`
- [ ] `astro estimate-measurements <geodetic-range-rate-scenario.yaml> <measurements.json> --estimator orekit-native --output /tmp/astro-orekit-native-estimate.json`
- [x] `astro rocketpy-smoke`
- [ ] `astro launch examples/launch/rocketpy_configured_single_stage.yaml --backend rocketpy --output /tmp/astro-rocketpy-launch.json`
- [ ] `python -m pytest tests/astro_backends/test_rocketpy_simulation.py::test_propagate_launch_rocketpy_rejects_additional_motors_until_backend_supports_them -q`
- [ ] `ASTRO_RUN_ROCKETPY_LIVE=1 python -m pytest tests/astro_backends/test_rocketpy_simulation.py::test_live_rocketpy_configured_launch_examples_return_suite_products -q`
- [x] `astro dymos-smoke`
- [ ] `astro optimize-launch examples/launch/pitch_program_two_stage.yaml --backend dymos --output /tmp/astro-dymos-optimized-launch.json`
- [ ] `astro optimize-launch examples/launch/pitch_program_two_stage.yaml --backend dymos --dymos-mode pitch-program --output /tmp/astro-dymos-pitch-program-launch.json`
- [ ] `astro optimize-launch examples/launch/pitch_program_two_stage.yaml --backend dymos --dymos-mode multistage-pitch-program --output /tmp/astro-dymos-multistage-pitch-program-launch.json`
- [ ] `ASTRO_RUN_DYMOS_LIVE=1 python -m pytest tests/astro_backends/test_dymos_optimization.py::test_live_dymos_optimization_returns_suite_product tests/astro_backends/test_dymos_optimization.py::test_live_dymos_pitch_program_optimization_executes_native_transcription tests/astro_backends/test_dymos_optimization.py::test_live_dymos_multistage_pitch_program_executes_native_multiphase -q`
- [x] `astro tudat-smoke`
- [ ] `astro propagate examples/scenarios/leo_two_body.yaml --backend tudat --output /tmp/astro-tudat-two-body.json`
- [ ] `astro propagate examples/scenarios/leo_j2.yaml --backend tudat --output /tmp/astro-tudat-j2.json`
- [ ] `astro propagate examples/scenarios/leo_orekit_drag.yaml --backend tudat --output /tmp/astro-tudat-drag.json`
- [ ] `astro propagate examples/scenarios/leo_orekit_srp.yaml --backend tudat --output /tmp/astro-tudat-srp.json`
- [ ] `astro propagate examples/scenarios/leo_orekit_third_body.yaml --backend tudat --output /tmp/astro-tudat-third-body.json`
- [ ] `astro propagate examples/scenarios/leo_tudat_high_order_gravity.yaml --backend tudat --output /tmp/astro-tudat-high-order-gravity.json`
- [ ] `astro propagate examples/scenarios/leo_orekit_high_fidelity_covariance.yaml --backend tudat --output /tmp/astro-tudat-high-fidelity-covariance.json`
- [ ] `ASTRO_RUN_TUDAT_LIVE=1 python -m pytest tests/astro_backends/test_tudat_propagation.py::test_live_tudat_high_fidelity_covariance_records_force_models -q`
- [ ] `astro propagate examples/scenarios/leo_tudat_variational_covariance.yaml --backend tudat --output /tmp/astro-tudat-variational-covariance.json`
- [ ] `ASTRO_RUN_TUDAT_LIVE=1 python -m pytest tests/astro_backends/test_tudat_propagation.py::test_live_tudat_native_variational_covariance_records_force_models -q`
- [ ] `python -m pytest tests/astro_backends/test_tudat_propagation.py::test_propagate_tudat_uses_default_native_variational_runner_when_requested -q`
- [ ] `python -m pytest tests/astro_backends/test_tudat_propagation.py::test_propagate_tudat_uses_native_variational_runner_when_requested -q`
- [ ] `astro compare-tudat-reference examples/scenarios/leo_two_body.yaml --reference-backend local --position-tolerance-km 0.001 --velocity-tolerance-km-s 0.000001 --output /tmp/astro-tudat-reference-comparison.json`
- [ ] `astro compare-tudat-campaign examples/scenarios/leo_two_body.yaml examples/scenarios/leo_j2.yaml --reference-backend local --position-tolerance-km 0.01 --velocity-tolerance-km-s 0.00003 --output /tmp/astro-tudat-reference-campaign-calibrated.json`
- [x] `astro jax-smoke`
- [ ] `astro research-estimate examples/scenarios/leo_two_station_od.yaml examples/measurements/leo_two_station_od_measurements.json --backend jax --max-iterations 5 --output /tmp/astro-jax-research-estimate.json`
- [ ] `astro synth-measurements examples/scenarios/leo_two_station_angles.yaml --backend local --output /tmp/astro-angle-measurements.json`
- [ ] `astro research-od-sensitivity examples/scenarios/leo_two_station_angles.yaml /tmp/astro-angle-measurements.json --backend jax --output /tmp/astro-jax-angle-sensitivity.json`
- [ ] `astro research-estimate examples/scenarios/leo_two_station_angles.yaml /tmp/astro-angle-measurements.json --backend jax --max-iterations 8 --output /tmp/astro-jax-angle-estimate.json`
- [ ] `astro synth-measurements examples/scenarios/leo_two_station_topocentric.yaml --backend local --output /tmp/astro-topocentric-measurements.json`
- [ ] `astro research-od-sensitivity examples/scenarios/leo_two_station_topocentric.yaml /tmp/astro-topocentric-measurements.json --backend jax --output /tmp/astro-jax-topocentric-sensitivity.json`
- [ ] `astro research-estimate examples/scenarios/leo_two_station_topocentric.yaml /tmp/astro-topocentric-measurements.json --backend jax --max-iterations 8 --output /tmp/astro-jax-topocentric-estimate.json`
- [ ] `astro research-propagate examples/scenarios/leo_orekit_drag.yaml --backend jax --cases 1 --position-sigma-km 0 --velocity-sigma-km-s 0 --seed 7 --output /tmp/astro-jax-drag-research.json`
- [ ] `astro research-propagate examples/scenarios/leo_covariance.yaml --backend jax --cases 1 --position-sigma-km 0 --velocity-sigma-km-s 0 --seed 7 --include-sensitivities --output /tmp/astro-jax-covariance-sensitivity.json`
- [ ] `astro research-propagate examples/scenarios/leo_orekit_srp.yaml --backend jax --cases 1 --position-sigma-km 0 --velocity-sigma-km-s 0 --seed 7 --output /tmp/astro-jax-srp-research.json`
- [ ] `astro research-propagate examples/scenarios/leo_jax_high_order_gravity_research.yaml --backend jax --cases 1 --position-sigma-km 0 --velocity-sigma-km-s 0 --seed 7 --output /tmp/astro-jax-high-order-research.json`
- [ ] `astro research-propagate examples/scenarios/leo_jax_third_body_research.yaml --backend jax --cases 1 --position-sigma-km 0 --velocity-sigma-km-s 0 --seed 7 --output /tmp/astro-jax-third-body-research.json`
- [ ] `astro research-propagate examples/scenarios/leo_jax_third_body_ephemeris_research.yaml --backend jax --cases 1 --position-sigma-km 0 --velocity-sigma-km-s 0 --seed 7 --output /tmp/astro-jax-third-body-ephemeris-research.json`

If an optional runtime is intentionally absent, capture the structured unavailable JSON and confirm
the message is actionable.

## Documentation Gates

- [x] README current-scope and command list match implemented behavior.
- [x] `docs/validation-matrix.md` reflects current command names and tolerances.
- [x] `docs/backend-installation.md` documents every optional extra and non-pip install caveat.
- [x] Roadmap plan statuses distinguish implemented product boundaries from live backend work that
  still requires external configuration.

## Packaging Gate

- [x] Build succeeds with `python -m build` when build tooling is installed.
- [x] Wheel metadata includes optional extras: `dev`, `orekit`, `launch`, `optimization`, and
  `research`.
