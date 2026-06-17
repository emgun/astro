# Astro Suite

Astro Suite is a Python flight dynamics project for scenario validation, local reference
propagation, launch/ascent sanity cases, synthetic orbit-determination measurements, batch OD,
and backend adapters.

## Current Scope

The current implementation slice covers:

- Pydantic scenario validation from YAML.
- Local two-body and J2 reference propagation with deterministic provenance metadata.
- Flight-dynamics trajectory product fields for events, impulsive and finite-burn maneuvers, and
  covariance history, plus local commanded-attitude samples for maneuvered trajectories, local
  finite-difference covariance propagation with optional acceleration process noise, per-sample
  state-transition/process-noise products, CSV and CCSDS OEM ephemeris export, and seeded
  initial-state Monte Carlo propagation.
- Local launch/ascent reference propagation with vertical and pitch-program guidance, staged mass
  depletion, drag, events, and launch-to-orbit insertion handoff.
- Launch pitch-program sweep, two-knot tuning, and tuned launch-to-orbit reporting over repeated
  local ascent/orbit propagations, batch ranking, and report-to-report comparison.
- Synthetic range, range-rate, one-way Doppler, iterative linearized two-way/three-way range and
  range-rate, inertial right ascension, declination, azimuth, and elevation measurement generation.
- Local SciPy batch least-squares orbit determination with rank and convergence checks.
- CLI workflows for validation, propagation, launch, launch-to-orbit handoff, synthetic
  measurements, synthetic OD, measurement-file OD ingest/export, and research propagation.
- Optional backend smoke gates and product boundaries for Orekit, RocketPy, Dymos/OpenMDAO, TudatPy,
  and JAX.

Launch/ascent includes deliberately simple local vertical and pitch-program baselines plus a
RocketPy direct-simulation path for explicitly configured single-stage solid rockets. Dymos/OpenMDAO
is recognized as an optimization adapter boundary; its live phase transcription is currently a
stage-aware vertical-ascent model rather than a full multistage optimal-control ascent solver.

## Setup

```bash
python -m pip install -e '.[dev]'
```

Optional Orekit wrapper smoke and propagation support:

```bash
python -m pip install -e '.[orekit]'
export JAVA_HOME=/opt/homebrew/opt/openjdk/libexec/openjdk.jdk/Contents/Home
export PATH="/opt/homebrew/opt/openjdk/bin:$PATH"
export ASTRO_OREKIT_DATA_PATH="$HOME/.orekit/orekit-data.zip"
astro orekit-smoke
astro propagate examples/scenarios/leo_two_body.yaml --backend orekit --output orekit_trajectory.json
```

If `orekit-jpype`, Java, or Orekit data are not configured, `astro orekit-smoke` exits nonzero with
structured JSON explaining the missing piece. The runtime checks `ASTRO_OREKIT_DATA_PATH`,
`OREKIT_DATA_PATH`, then `~/.orekit/orekit-data.zip` for Orekit data. `astro propagate --backend
orekit` currently supports two-body propagation through Orekit's Keplerian propagator, J2 through
Orekit's numerical propagator with `J2OnlyPerturbation`, and `orekit_high_fidelity` as the numerical
propagation expansion path. Atmospheric drag is available through Orekit `DragForce` with
`SimpleExponentialAtmosphere` and `IsotropicDrag`; solar radiation pressure is available through
Orekit `SolarRadiationPressure` and `IsotropicRadiationSingleCoefficient`; third-body gravity is
available through Orekit `ThirdBodyAttraction` for the Sun and Moon.

Optional launch backend smoke checks:

```bash
python -m pip install -e '.[launch,optimization]'
astro rocketpy-smoke
astro dymos-smoke
```

These extras are pinned to NumPy-1-compatible backend lines: RocketPy `>=1.11,<1.12`,
Dymos `>=1.13.1,<1.14`, and OpenMDAO `>=3.41,<3.42`.

RocketPy and Dymos/OpenMDAO are behind explicit adapter gates. The current `rocketpy` launch path
loads the optional runtime, requires explicit `rocketpy` vehicle/motor/flight configuration on the
launch scenario, runs configured solid-rocket flights through RocketPy, preserves the
`LaunchTrajectory` product boundary, and can annotate multistage suite scenarios with stage
events/samples reached by a single configured RocketPy flight, including metadata for whether the
RocketPy solution covered the full suite stage schedule. That multistage path is an adapter
composition layer, not a validated multi-motor RocketPy staging solver. The `dymos` launch
optimization path runs a stage-aware Dymos/OpenMDAO vertical-ascent phase transcription and returns
the existing `LaunchPitchTuningResult` product with explicit phase diagnostics, suite stage-plan
metadata, and a flag showing that the Dymos phase duration covers the configured burn schedule;
full pitch-program multistage Dymos ascent optimization remains future work.

Optional research backend smoke checks:

```bash
python -m pip install -e '.[research]'
astro tudat-smoke
astro jax-smoke
```

Tudat and JAX are optional research/cross-check boundaries. TudatPy is not currently assumed to be
available from PyPI on every platform, so its smoke command reports installation state without
promising a pip-only install path. JAX research propagation returns suite `MonteCarloResult`
products, can optionally include a final-state transition sensitivity matrix, and supports
differentiable screening approximations for `orekit_high_fidelity`, atmospheric drag, solar
radiation pressure, and analytic circular Sun/Moon third-body gravity flags. Those JAX force flags
are explicitly research products, not operational ephemerides; ephemeris-backed third-body gravity
remains an Orekit/Tudat-grade integration task.

## Commands

```bash
astro validate examples/scenarios/leo_two_body.yaml
astro propagate examples/scenarios/leo_two_body.yaml --backend local --output trajectory.json
astro propagate examples/scenarios/leo_eccentric_two_body.yaml --backend local --output eccentric_trajectory.json
astro propagate examples/scenarios/meo_two_body.yaml --backend local --output meo_trajectory.json
astro propagate examples/scenarios/geo_two_body.yaml --backend local --output geo_trajectory.json
astro propagate examples/scenarios/leo_j2.yaml --backend local --output j2_trajectory.json
astro propagate examples/scenarios/leo_finite_burn.yaml --backend local --output finite_burn_trajectory.json
astro propagate examples/scenarios/leo_velocity_aligned_burn.yaml --backend local --output velocity_aligned_burn_trajectory.json
astro propagate examples/scenarios/leo_radial_burn.yaml --backend local --output radial_burn_trajectory.json
astro propagate examples/scenarios/leo_covariance.yaml --backend local --output covariance_trajectory.json
astro propagate examples/scenarios/leo_variational_covariance.yaml --backend local --output variational_covariance_trajectory.json
astro propagate examples/scenarios/leo_j2_variational_covariance.yaml --backend local --output j2_variational_covariance_trajectory.json
astro propagate examples/scenarios/leo_two_body.yaml --backend orekit --output orekit_trajectory.json
astro propagate examples/scenarios/leo_j2.yaml --backend orekit --output orekit_j2_trajectory.json
astro propagate examples/scenarios/leo_orekit_high_fidelity.yaml --backend orekit --output orekit_high_fidelity_trajectory.json
astro propagate examples/scenarios/leo_orekit_drag.yaml --backend orekit --output orekit_drag_trajectory.json
astro propagate examples/scenarios/leo_orekit_srp.yaml --backend orekit --output orekit_srp_trajectory.json
astro propagate examples/scenarios/leo_orekit_third_body.yaml --backend orekit --output orekit_third_body_trajectory.json
astro propagate examples/scenarios/leo_orekit_high_fidelity_covariance.yaml --backend orekit --output orekit_high_fidelity_covariance.json
astro export-trajectory trajectory.json --format csv --output trajectory.csv
astro export-trajectory trajectory.json --format oem --output trajectory.oem
astro import-trajectory trajectory.oem --format oem --scenario examples/scenarios/leo_two_body.yaml --output imported_trajectory.json
astro screen-conjunction primary_trajectory.json secondary_trajectory.json --threshold-km 1.0 --hard-body-radius-km 0.02 --probability-method integrated --output conjunction_screening.json
astro monte-carlo examples/scenarios/leo_two_body.yaml --cases 4 --position-sigma-km 0.01 --velocity-sigma-km-s 0.000001 --seed 7 --backend local --output monte_carlo.json
astro rocketpy-smoke
astro dymos-smoke
astro tudat-smoke
astro jax-smoke
astro launch examples/launch/vertical_two_stage.yaml --backend local --output launch.json
astro launch examples/launch/pitch_program_two_stage.yaml --backend local --output pitch_launch.json
astro sweep-launch-pitch examples/launch/pitch_program_two_stage.yaml --point-index 3 --pitch-deg-values 10,20,30 --output pitch_sweep.json
astro tune-launch-pitch examples/launch/pitch_program_two_stage.yaml --point-indices 2,3 --initial-span-deg 10 --iterations 2 --output pitch_tuning.json --tuned-scenario-output tuned_pitch_program.yaml
astro optimize-launch examples/launch/pitch_program_two_stage.yaml --backend local --point-indices 2,3 --iterations 2 --output launch_optimization.json
astro report-tuned-launch examples/launch/pitch_program_two_stage.yaml --point-indices 2,3 --initial-span-deg 10 --iterations 2 --orbit-duration-s 600 --orbit-step-s 60 --output tuned_launch_report.json
astro batch-report-tuned-launch examples/launch/pitch_program_two_stage.yaml --point-indices 2,3 --iterations-values 1,2,3 --initial-span-deg 10 --orbit-duration-s 600 --orbit-step-s 60 --output tuned_launch_batch.json
astro compare-tuned-launch-reports tuned_launch_report_baseline.json tuned_launch_report_candidate.json --output tuned_launch_comparison.json
astro handoff-launch launch.json --output insertion.yaml --duration-s 600 --step-s 60
astro propagate insertion.yaml --backend local --output insertion_trajectory.json
astro import-earth-orientation examples/eop/finals2000A_excerpt.txt --format iers-finals --source finals2000A-example --output earth_orientation.json
astro synth-measurements examples/scenarios/leo_two_station_od.yaml --backend local --output measurements.json
astro synth-measurements examples/scenarios/leo_two_station_od.yaml --backend orekit --output orekit_measurements.json
astro synth-measurements examples/scenarios/leo_two_station_angles.yaml --backend local --output angle_measurements.json
astro synth-measurements examples/scenarios/leo_two_station_topocentric.yaml --backend local --output topocentric_measurements.json
astro synth-measurements examples/scenarios/leo_geodetic_topocentric.yaml --backend local --output geodetic_topocentric_measurements.json
astro synth-measurements examples/scenarios/leo_geodetic_eop_topocentric.yaml --backend local --output geodetic_eop_topocentric_measurements.json
astro synth-measurements examples/scenarios/leo_geodetic_eop_table_topocentric.yaml --backend local --output geodetic_eop_table_topocentric_measurements.json
astro synth-measurements examples/scenarios/leo_doppler.yaml --backend local --output doppler_measurements.json
astro export-measurements measurements.json --format csv --output measurements.csv
astro export-measurements measurements.json --format tdm --output measurements.tdm
astro estimate examples/scenarios/leo_two_body.yaml --backend local --output estimate.json
astro estimate examples/scenarios/leo_two_body.yaml --backend orekit --output orekit_estimate.json
astro estimate-measurements examples/scenarios/leo_two_station_od.yaml measurements.json --backend local --output estimate.json
astro estimate-measurements examples/scenarios/leo_two_station_od.yaml measurements.json --backend orekit --output orekit_estimate.json
astro estimate-measurements examples/scenarios/leo_two_station_od.yaml measurements.json --estimator orekit-native --output orekit_native_estimate.json
astro research-propagate examples/scenarios/leo_two_body.yaml --backend local --cases 4 --position-sigma-km 0.01 --velocity-sigma-km-s 0.000001 --seed 7 --output research_propagation.json
astro research-propagate examples/scenarios/leo_orekit_drag.yaml --backend jax --cases 1 --position-sigma-km 0 --velocity-sigma-km-s 0 --seed 7 --output jax_drag_research.json
astro research-propagate examples/scenarios/leo_orekit_srp.yaml --backend jax --cases 1 --position-sigma-km 0 --velocity-sigma-km-s 0 --seed 7 --output jax_srp_research.json
astro research-propagate examples/scenarios/leo_jax_third_body_research.yaml --backend jax --cases 1 --position-sigma-km 0 --velocity-sigma-km-s 0 --seed 7 --output jax_third_body_research.json
astro research-od-sensitivity examples/scenarios/leo_two_station_od.yaml measurements.json --backend jax --output od_sensitivity.json
astro research-od-sensitivity examples/scenarios/leo_two_station_angles.yaml angle_measurements.json --backend jax --output angle_od_sensitivity.json
astro research-od-sensitivity examples/scenarios/leo_two_station_topocentric.yaml topocentric_measurements.json --backend jax --output topocentric_od_sensitivity.json
astro research-estimate examples/scenarios/leo_two_station_od.yaml measurements.json --backend jax --max-iterations 5 --output research_estimate.json
astro orekit-smoke
```

`astro estimate` is an MVP synthetic demonstration workflow. It keeps the source scenario unchanged,
adds in-memory demo geometry for observability, generates synthetic measurements, perturbs the
initial state as an estimate seed, and records that provenance in the output metadata. The `--backend`
option selects the propagation backend used for synthetic truth generation and residual propagation.
With `--backend orekit`, the suite estimator uses Orekit-backed propagation when the optional Orekit
runtime is installed. The Orekit backend also includes a native OD construction bridge that maps
suite geodetic range/range-rate records into Orekit `Range`/`RangeRate` measurements and a
`BatchLSEstimator` object. The bridge can execute the native estimator and map the estimated state,
residuals, RMS, covariance, and iteration diagnostics into the suite `EstimateResult` model. Use
`astro estimate-measurements --estimator orekit-native` to select that bridge explicitly. It remains
limited to geodetic range/range-rate records and still depends on a live Java/Orekit runtime and
data context. Orekit covariance extraction can be singular on short or weakly observed arcs; when
that happens the suite returns a zero covariance fallback with `covariance_status = "unavailable"`
and the backend error recorded in metadata instead of treating the fallback as a valid covariance.

`astro export-trajectory` converts suite trajectory JSON into either a CSV ephemeris table or a
CCSDS OEM KVN text product containing UTC epochs plus Cartesian position and velocity samples in km
and km/s. `astro import-trajectory --format oem` converts a CCSDS OEM KVN text product back into a
suite `Trajectory`; it requires `--scenario` because OEM does not encode the suite force model.
The importer is intentionally strict: UTC time system, EME2000 reference frame, and Earth center are
required. `astro monte-carlo` runs a seeded initial-state ensemble by perturbing the scenario's
Cartesian state and propagating each case through the selected backend. This is a repeatable product
workflow for uncertainty screening; it is not yet production conjunction analysis.

Local covariance propagation accepts an optional `initial_covariance` and
`covariance_process_noise_acceleration_km_s2`. The default `covariance_state_transition_model` is
`finite_difference`, preserving the existing force-model and maneuver compatibility. Two-body
scenarios without maneuvers may opt into `two_body_variational`, which integrates the state
transition matrix with the analytic two-body acceleration Jacobian. J2 scenarios without maneuvers
may opt into `j2_variational`, which integrates the state transition matrix with a finite-difference
J2 acceleration Jacobian. Both variational paths store per-step plus accumulated state-transition
matrices in the trajectory product.

Local orbital propagation accepts an optional `maneuvers` schedule on `Scenario`. Impulsive
maneuvers apply their full `delta_v_km_s` at the maneuver epoch; finite burns apply the configured
total delta-v as constant inertial acceleration over `duration_s`, or use optional
`thrust_vector_n` plus `specific_impulse_s` to integrate thrust-vector acceleration with mass
depletion. Trajectory samples include `mass_kg`, and maneuver start/end events preserve maneuver
metadata. Thrust-vector finite burns default to inertial direction; setting
`thrust_direction_mode` to `velocity_aligned` rotates the thrust magnitude along the instantaneous
velocity direction. `radial_outward` and `radial_inward` rotate the thrust magnitude along the
instantaneous local radial direction. Maneuvered local trajectories also include per-sample
`AttitudeState` products with a body-to-inertial unit quaternion that points the spacecraft body +X
axis along the commanded thrust direction during active thrust-vector burns. These are commanded
pointing products, not full torque-level attitude-control simulations. Local orbital propagation
also annotates periapsis/apoapsis `TrajectoryEvent` records. For no-maneuver trajectories, apsides
are root-located between propagation samples through radial-velocity bisection and include bracket,
elapsed time, radius, and radial-velocity metadata. Maneuvered trajectories keep sample-safe apsis
annotation because thrust discontinuities need maneuver-aware event isolation.

Local and Orekit propagation also accept an optional `initial_covariance` 6x6 matrix. When present,
the backend emits a `covariance_history` sample at each trajectory epoch using a finite-difference
state transition. The local backend computes finite differences through its deterministic RK4
reference dynamics; the Orekit backend computes finite differences by rebuilding and propagating
perturbed Orekit states through the selected Orekit propagator and force model, including the
high-fidelity drag/SRP/third-body flags when they are enabled. Orekit covariance metadata records
the transition propagator and force-model list so high-fidelity covariance products remain auditable.
Each covariance sample can carry the per-step `state_transition_matrix`, the
`accumulated_state_transition_matrix` from the initial epoch, the `process_noise_covariance` applied
for that step, and metadata naming the model and step size. The optional
`covariance_process_noise_acceleration_km_s2` field adds a simple per-axis white-acceleration
process-noise term over each propagation sample interval. This is useful for product wiring and
first-order sensitivity screening; analytic variational-equation covariance dynamics, drag/SRP
variational equations, and production conjunction analysis remain future work.

`astro screen-conjunction` compares two trajectory products at common sample epochs and writes a
deterministic first-order screening result with time of closest approach, miss distance, relative
speed, threshold status, and relative-state metadata. When both trajectories carry covariance
history at TCA and `--hard-body-radius-km` is supplied, it also writes a bounded encounter-plane
probability estimate. The default `integrated` method numerically integrates the 2D Gaussian over
the hard-body disk with Gauss-Legendre polar quadrature; `--probability-method density` preserves the
faster local-density-times-area screening approximation. This is useful for covariance-aware
screening, but it is not a full production conjunction assessment.

`astro launch` is the launch/ascent MVP workflow. It loads a launch scenario, runs the local
vertical or pitch-program baseline, and writes a launch trajectory product with samples, stage
events, dynamic pressure, acceleration, downrange, target miss metrics, and an `insertion_state`
compatible with the shared `OrbitState` product. This local backend is a deterministic data-flow
baseline, not a production launch simulator.

`astro sweep-launch-pitch` is the first launch targeting workflow. It varies one pitch-program knot,
runs the local launch propagator for each candidate pitch angle, and writes a JSON product with
altitude miss, velocity miss, weighted score, final downrange, and the best case. It is a transparent
grid sweep rather than an optimizer; that keeps the target-miss contract clear before adding Dymos,
OpenMDAO, or RocketPy-backed targeting.

`astro tune-launch-pitch` is the first multi-knot targeting workflow. It varies two pitch-program
knots on a deterministic 3x3 grid, shrinks the search span each iteration, writes a JSON trace of
every evaluated candidate, and can write the best tuned `LaunchScenario` back to YAML. This is still
a coarse-to-fine targeting analysis tool, not a production optimizer.

`astro optimize-launch` is the neutral launch optimization entry point. With `--backend local`, it
uses the current pitch-program tuner and writes the same `LaunchPitchTuningResult` product. With
`--backend dymos`, it loads the optional Dymos/OpenMDAO runtime and requires a validated Dymos phase
runner; the current suite does not fake an optimal-control solve from the local aggregate schema.

`astro report-tuned-launch` runs the current local end-to-end launch analysis: tune two pitch knots,
propagate the tuned ascent, hand off insertion to an orbit scenario, propagate a short orbital arc,
and write one JSON product with the component products plus insertion and short-arc target metrics.
The report also includes pass/fail assessment gates using the target orbit's configured altitude
and velocity tolerances, with named checks for insertion and short-arc misses.
It is a deterministic report over local baselines, not a substitute for high-fidelity ascent design.

`astro batch-report-tuned-launch` runs the tuned launch report workflow for multiple iteration
counts and ranks the resulting reports by normalized assessment error: the sum of absolute check
misses divided by their configured tolerances. It is a deterministic tuning-depth comparison table,
not a new optimizer.

`astro compare-tuned-launch-reports` compares two saved tuned launch report JSON products without
rerunning analysis. It writes signed deltas and absolute-error improvement for insertion and
short-arc target misses, plus pass/fail changes between the baseline and candidate reports.

`astro handoff-launch` converts a launch trajectory product into a normal orbital propagation
scenario initialized from `LaunchTrajectory.insertion_state`. The generated YAML is intentionally
plain `Scenario` input, so the next step is the existing `astro propagate` command rather than a
special launch-aware propagation path.

`astro synth-measurements` and `astro estimate-measurements` both accept `--backend local` or
`--backend orekit`. The explicit ingest workflow loads a scenario plus a JSON, CSV, or CCSDS
Tracking Data Message (TDM) measurement file, then estimates from the caller-provided station
geometry and measurement records without adding demo geometry. JSON and CSV can carry the suite's
range, range-rate, one-way Doppler in Hz, iterative linearized two-way/three-way range and range-rate,
inertial right-ascension/declination, and local-horizon azimuth/elevation records. Doppler uses the
scenario's `doppler_transmit_frequency_hz` to convert line-of-sight range rate into a
received-frequency shift. Two-way and three-way range-like observables use iterative vacuum
light-time over a linearized spacecraft state and carry `participant_path`/`transmitter` metadata,
uplink/downlink light-time diagnostics, transmit/reflection/receive offsets, and an explicit
media-corrections marker. Scenarios may set `radiometric_media_uplink_delay_km` and
`radiometric_media_downlink_delay_km` for configured constant range-delay corrections, or set
`radiometric_media_model: weather_frequency` to apply a configured surface-weather troposphere
delay plus first-order TEC/frequency ionosphere group delay with per-leg elevation mapping.
`radiometric_media_source` and the component delays are preserved in measurement metadata. These
are configurable product and estimator primitives, not full DSN calibration pipelines. Angle
records use degrees.
Ground stations can be supplied either as fixed
`position_eci_km` vectors or as WGS-84 geodetic `latitude_deg`, `longitude_deg`, and `altitude_km`
coordinates. Geodetic stations are rotated into the inertial measurement frame at each measurement
epoch using a deterministic UTC sidereal-time model by default. Scenarios may also provide fixed
`earth_orientation` values with `ut1_minus_utc_s`, `polar_motion_x_arcsec`,
`polar_motion_y_arcsec`, and a `source` label, or a `samples` table with timestamped values that
are linearly interpolated per measurement epoch. These values drive an approximate EOP-aware
Earth-fixed to inertial correction for geodetic station pointing. This supports explicit fixed EOP
values, simple tabulated interpolation, and `astro import-earth-orientation --format iers-finals`
conversion from IERS finals/finals2000A-style rows into suite Earth-orientation JSON. It is not a
full precession/nutation reduction. JSON inputs
match the output of `astro synth-measurements`; CSV and TDM inputs are auto-detected by `.csv` and
`.tdm` extensions or can be forced with `--format csv` / `--format tdm`.

`astro export-measurements` converts suite JSON measurement files into JSON, CSV, or TDM products.
JSON and CSV preserve all supported suite measurement types, including one-way Doppler in Hz and
two-way/three-way radiometric metadata. TDM export supports range, range-rate, explicit
two-way/three-way range/range-rate, one-way Hz Doppler through a suite
`ASTRO_MEASUREMENT_TYPE = doppler_hz` metadata extension, right-ascension/declination, and
azimuth/elevation records. The example files under
`examples/measurements/` are generated from `leo_two_station_od.yaml` and cover all three
range/range-rate ingest/export formats.

`astro research-propagate` is the research backend entry point for seeded propagation ensembles.
With `--backend local`, it runs the deterministic Monte Carlo workflow. With `--backend jax`, it
loads the optional JAX runtime and runs vectorized RK4 ensembles for two-body, J2, and
screening-only `orekit_high_fidelity` scenarios, including research approximations for atmospheric
drag, solar radiation pressure, and analytic circular Sun/Moon third-body gravity. With
`--include-sensitivities`, the JAX path adds a nominal final-state transition matrix computed
through JAX autodiff to the `MonteCarloResult` metadata. Third-body JAX products record
`third_body_ephemeris_model = "analytic_circular_sun_moon_screening"` so callers do not mistake the
screening approximation for operational ephemeris-backed Orekit/Tudat semantics.

`astro research-od-sensitivity --backend jax` loads an explicit measurement file and writes an
`OdSensitivityResult` containing normalized OD residuals and the residual Jacobian with respect to
the initial Cartesian state. This is the first differentiable OD primitive for range/range-rate,
inertial right-ascension/declination, and local-horizon azimuth/elevation workflows; it currently
supports two-body/J2 force models and measurement epochs aligned with propagation samples. Right
ascension and azimuth residuals use wrapped degrees so 0/360 degree crossings stay continuous.
Topocentric angular sensitivity metadata records a horizontal-norm floor because azimuth is
geometrically undefined and elevation derivatives are singular at exact zenith/nadir passes.
`astro research-estimate --backend jax` runs a research Gauss-Newton
correction loop over the same JAX residual/Jacobian model and writes the suite `EstimateResult`
product with normalized-residual metadata. It is a differentiable OD workflow for screening and
method development, not a replacement for the deterministic local estimator or native Orekit OD.

CSV inputs use one row per measurement with these required columns:

```csv
scenario_id,measurement_type,epoch,observer,observed_object,value,sigma,units
```

The optional `metadata_json` column can carry a JSON object for row-level metadata. Valid units are
`km`, `km/s`, `Hz`, and `deg`, depending on measurement type. The
`leo_two_station_od.yaml` example includes two stations because the one-station propagation example
is intentionally under-observed for six-state OD.
`examples/scenarios/leo_radiometric_links.yaml` demonstrates two-way and three-way radiometric
measurement synthesis with explicit uplink/downlink station metadata and vacuum light-time
diagnostics. `examples/scenarios/leo_radiometric_media.yaml` adds configured constant media range
delays to the same product family. `examples/scenarios/leo_radiometric_weather_frequency.yaml`
adds configured pressure, temperature, humidity, TEC, carrier frequency, and elevation mapping
metadata for weather/frequency-dependent media corrections.

TDM ingest currently supports KVN-formatted sequential segments with `TIME_SYSTEM = UTC`,
`PARTICIPANT_n`, `PATH`, `RANGE` in `km`, `DOPPLER_INSTANTANEOUS` or `DOPPLER_INTEGRATED` mapped
to range-rate measurements in `km/s`, and `ANGLE_1`/`ANGLE_2` records in `deg`. Segments with
`ASTRO_MEASUREMENT_TYPE = two_way` or `three_way` preserve the suite's explicit multi-leg
radiometric types without reinterpreting legacy TDM files that omit the extension. Segments with
`ASTRO_MEASUREMENT_TYPE = doppler_hz` preserve suite one-way Doppler records in `Hz` with
`DOPPLER_UNITS = Hz` and optional `DOPPLER_SIGMA_HZ`. Angle segments use
`ANGLE_TYPE = RADEC` for right ascension/declination and `ANGLE_TYPE = AZEL` for
azimuth/elevation. TDM does not provide the suite's scenario identifier or estimator sigmas
directly, so an optional segment-level `SCENARIO_ID` extension is checked when present, and the
parser uses default sigmas of `0.01 km` for range, `1e-5 km/s` for range-rate, `0.1 Hz` for
suite Doppler, and `0.001 deg` for angles.

## Verification

```bash
python -m pytest -v
python -m ruff check .
python -m mypy
```

Additional release and backend documentation:

- [Validation matrix](docs/validation-matrix.md)
- [Backend installation guide](docs/backend-installation.md)
- [Release checklist](docs/release-checklist.md)
