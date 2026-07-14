# Release Checklist

Date: 2026-07-10

Use this checklist before tagging or publishing a release candidate.

Latest merged-main evidence:

- 2026-07-10: annotated final tag `v0.1.0` was published at `6cc7d2f`, exactly matching the reviewed
  `v0.1.0-rc.2` commit. A detached release worktree repeated Ruff, strict MyPy across 97 source
  files, `686 passed, 11 skipped`, the public mission lifecycle command and artifact bundle, both
  distribution builds, wheel-content inspection, `git diff --check`, and clean-worktree review
  before promotion.
- 2026-07-10: annotated tag `v0.1.0-rc.2` was published at `6cc7d2f`, including the merged Mission
  Lifecycle Workflow and its recorded verification state.
- 2026-07-10: annotated tag `v0.1.0-rc.1` was published at release baseline `1182493`, before the
  Mission Lifecycle Workflow branch.
- 2026-06-20: `main` fast-forwarded to `0fb9a87`.
- 2026-06-20: Release evidence recorded on `main` at `870cb13`.
- 2026-06-20: `python -m pytest -q` passed with 505 passed, 11 skipped.
- 2026-06-20: `python -m ruff check .`, `python -m mypy`, `git diff --check`, and
  `python -m build` passed.
- 2026-06-20: Required local CLI checklist passed 42 command gates on merged `main`.
- 2026-06-20 17:50 PDT: Optional backend smoke refresh passed on current checkout for Orekit,
  RocketPy, Dymos/OpenMDAO, TudatPy, and JAX. These smoke checks do not replace optional live
  propagation, launch, optimization, covariance, or OD campaign gates.

Latest post-launch mission-assurance integration evidence:

- 2026-07-13: `codex/post-launch-mission-assurance` added the suite-owned launch insertion,
  simulated tracking, OD, bounded candidate correction, estimate/truth replay, corrected digital
  twin, continuity, margin, digest-manifest, and artifact-bundle workflow.
- 2026-07-13: the checked six-site geodetic reference retained 130 above-mask measurements from
  1,452 deterministic candidates, converged OD, used a `0.004065448 km/s` candidate impulse, and
  reduced truth position error from `8.558551 km` to `0.568642 km`. All embedded twin margins pass,
  and correction propellant depletion is reflected in the returned twin.
  These are simulation and design-screening results, not RF acquisition, operational navigation,
  or flight-command authority.
- 2026-07-13: review hardening made below-mask tracking ineligible, promoted embedded twin failures,
  bound four source inputs and 16 emitted artifacts by digest, added public verification and
  tamper rejection, rejected input drift during execution and manifest omissions, and made 17-file
  bundle publication atomic for exclusive Astro writers with stale/concurrent-destination refusal.
- 2026-07-13: focused assurance and packaging tests reported `21 passed`; the full suite reported
  `930 passed, 11 skipped`; Ruff, strict MyPy across 127 source files, `git diff --check`, the fresh
  public CLI run and verifier, wheel-content inspection, and sdist/wheel builds passed.
- 2026-07-13: GitHub CI passed in `1m25s`; PR #17 merged the reviewed scope to `main` at
  `0d73d5b`.
- 2026-07-13: mission-assurance uncertainty integration passed the focused assurance/UQ/campaign
  gate (`207 passed`), the full suite (`935 passed, 11 skipped`), Ruff, strict MyPy across 128 source
  files, `git diff --check`, a fresh two-worker 8-case public campaign, and sdist/wheel builds. The
  reference reports all-completed-case design-space frequencies under illustrative bounds and
  fixed-seed common random numbers, not calibrated operational probabilities.
- 2026-07-13: GitHub CI passed in `1m51s`; PR #18 merged the mission-assurance uncertainty scope to
  `main` at `c10268e`.
- 2026-07-13: paired mission-assurance validation added an explicit one-hour protocol with 30
  minutes of causal pre-decision tracking and 30 minutes of truth verification. The corrected
  eight-pair run completed 8/8 matched profiles passing and 0/8 J2-truth/two-body-estimator
  profiles passing. Counts remain unpooled simulation design-space evidence, not probability or
  operational authority.
- 2026-07-13: independent review hardening made paired verification recompute profile metrics,
  force roles, pass dispositions, deltas, reversals, coordinates, and summaries from bound sources
  and embedded cases; source drift aborts the campaign, coordinate seeds must be unique, and both
  commanded and executed component/total authority limits are preserved separately from the wider
  diagnostic solver envelope.
- 2026-07-13: the corrected public campaign ran from outside the repository and its 14.3 MB result
  passed the strengthened verifier. Focused assurance/OD tests, `951 passed, 11 skipped`, Ruff,
  strict MyPy across 132 source files, `git diff --check`, 6 packaging tests, and sdist/wheel builds
  passed locally.
- 2026-07-13: final rereview closed coordinate substitution and profile-slot swap gaps by binding
  each embedded case to its protocol identity, resolved overrides, insertion dispersion, timing,
  diagnostic limits, tracking duration, seed, and fixed profile slot. The CLI also rejects aliased
  primary and summary output paths before writing either artifact.
- 2026-07-13: the dependent eight-case mission-assurance uncertainty campaign was refreshed after
  the causal OD cutoff. It completed 8/8 with all configured requirements passing, 64 pre-decision
  measurements per case, recovery-position q05/q95 of `0.4151/0.6741 km`, and unchanged bounded
  design-space rather than operational-probability interpretation.

Latest optional validation refresh:

- 2026-07-09: `codex/orekit-validation-refresh` refreshed optional backend smoke on the current
  machine. Orekit, RocketPy, Dymos/OpenMDAO, and JAX were available; TudatPy was not installed in the
  base environment, and the historical isolated `/tmp/astro-tudat-live-env` path no longer existed.
- 2026-07-09: Orekit high-fidelity covariance validation passed with
  `ASTRO_RUN_OREKIT_LIVE=1 python -m pytest tests/astro_backends/test_orekit_propagation.py::test_live_orekit_covariance_history_returns_suite_product tests/astro_backends/test_orekit_propagation.py::test_live_orekit_high_fidelity_covariance_records_force_models -q`
  reporting `2 passed in 9.22s`.
- 2026-07-09: Orekit wrote `/tmp/astro-orekit-validation-high-fidelity-covariance-20260709.json`,
  `/tmp/astro-orekit-validation-drag-20260709.json`,
  `/tmp/astro-orekit-validation-srp-20260709.json`, and
  `/tmp/astro-orekit-validation-third-body-20260709.json`. The covariance artifact records 11
  covariance samples with J2, drag, SRP, Sun, and Moon transition force models. This is optional
  machine-scoped backend evidence, not production covariance certification.
- 2026-07-09: PR #7 merged the optional validation refresh to `main` at `b668e1f`.

Latest reentry integration evidence:

- 2026-07-10: `codex/reentry-suite` added suite-owned ballistic, prescribed-bank lifting, and
  target-tracking entry products plus local guidance optimization and trajectory handoff.
- 2026-07-10: the checked ballistic, lifting, and guided public CLI cases completed. The guided
  case produced 18.431 km target miss; bounded bank-schedule optimization converged in 13
  iterations and reduced miss to 6.231 km.
- 2026-07-10: internal-step convergence, monotonic heat-load, model/IO/guidance/margin,
  optimization, handoff, CLI, strict typing, packaging, and full regression gates passed on the
  branch and were repeated on merged `main`: `676 passed, 11 skipped`, with `33 passed` in the
  focused reentry/CLI slice. GitHub CI passed and PR #8 merged at `3c73f8b`. This remains
  deterministic screening evidence within the claim boundary in `docs/reentry.md`.

Latest mission lifecycle branch evidence:

- 2026-07-10: `codex/mission-lifecycle` added the checked launch, operations, digital twin,
  deorbit, and reentry phase chain with suite-owned products, eight continuity checks, cross-phase
  margins, and a nine-file artifact bundle.
- 2026-07-10: the public command passed with all margins passing; focused lifecycle tests reported
  `8 passed`; full local tests reported `686 passed, 11 skipped`; Ruff, strict MyPy across 97 source
  files, `git diff --check`, wheel inspection, and both distribution builds passed. GitHub CI
  passed and PR #9 merged at `d37b9a6`.

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
- [x] `astro simulate-reentry examples/reentry/ballistic_capsule.yaml --output /tmp/astro-ballistic-reentry.json --summary-output /tmp/astro-ballistic-reentry.txt`
- [x] `astro simulate-reentry examples/reentry/lifting_bank_schedule.yaml --output /tmp/astro-lifting-reentry.json --summary-output /tmp/astro-lifting-reentry.txt`
- [x] `astro simulate-reentry examples/reentry/guided_lifting_body.yaml --output /tmp/astro-guided-reentry.json --summary-output /tmp/astro-guided-reentry.txt`
- [x] `astro optimize-reentry examples/reentry/guided_lifting_body.yaml --maximum-iterations 20 --output /tmp/astro-reentry-optimization.json --tuned-scenario-output /tmp/astro-reentry-tuned.yaml --summary-output /tmp/astro-reentry-optimization.txt`
- [x] `astro handoff-reentry /tmp/astro-local-trajectory.json examples/reentry/ballistic_capsule.yaml --sample-index 0 --output /tmp/astro-reentry-handoff.yaml`
- [x] `astro run-mission-lifecycle examples/lifecycle/leo_round_trip.yaml --output /tmp/astro-mission-lifecycle.json --summary-output /tmp/astro-mission-lifecycle.txt --artifacts-dir /tmp/astro-mission-lifecycle-artifacts`
- [x] `python -m pytest tests/astro_mission -q`
- [x] `astro run-mission-assurance examples/assurance/post_launch_orbit_acquisition.yaml --output /tmp/astro-mission-assurance.json --summary-output /tmp/astro-mission-assurance.txt --artifacts-dir /tmp/astro-mission-assurance-artifacts`
- [x] `astro verify-mission-assurance /tmp/astro-mission-assurance-artifacts`
- [x] `python -m pytest tests/astro_assurance tests/test_packaging.py -q`
- [x] `python -m pytest tests/astro_reentry tests/astro_cli/test_reentry_cli.py -q`
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
- [x] `astro propagate examples/scenarios/leo_orekit_drag.yaml --backend orekit --output /tmp/astro-orekit-drag.json`
- [x] `astro propagate examples/scenarios/leo_orekit_srp.yaml --backend orekit --output /tmp/astro-orekit-srp.json`
- [x] `astro propagate examples/scenarios/leo_orekit_third_body.yaml --backend orekit --output /tmp/astro-orekit-third-body.json`
- [ ] `astro propagate examples/scenarios/leo_orekit_high_order_gravity.yaml --backend orekit --output /tmp/astro-orekit-high-order-gravity.json`
- [ ] `astro propagate examples/scenarios/leo_covariance.yaml --backend orekit --output /tmp/astro-orekit-covariance.json`
- [x] `astro propagate examples/scenarios/leo_orekit_high_fidelity_covariance.yaml --backend orekit --output /tmp/astro-orekit-high-fidelity-covariance.json`
- [x] `ASTRO_RUN_OREKIT_LIVE=1 python -m pytest tests/astro_backends/test_orekit_propagation.py::test_live_orekit_covariance_history_returns_suite_product -q`
- [x] `ASTRO_RUN_OREKIT_LIVE=1 python -m pytest tests/astro_backends/test_orekit_propagation.py::test_live_orekit_high_fidelity_covariance_records_force_models -q`
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
- [x] `astro tudat-smoke` captured the 2026-07-09 unavailable diagnostic; live Tudat gates were not
  refreshed.
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
- [x] Wheel contents include the suite-owned `astro_reentry` package.
- [x] Wheel contents include the suite-owned `astro_mission` package.
- [x] Wheel contents include the suite-owned `astro_assurance` package.
