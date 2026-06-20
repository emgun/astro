# Live Backend Campaign Ledger

Last local smoke run: 2026-06-19 17:29:40 PDT on branch `codex/orbit-fd-od-mvp`
at commit `9c7affb`.

Last available live campaign run: 2026-06-20 11:30 PDT on branch `codex/orbit-fd-od-mvp`
in the working tree recorded by this ledger update.

This ledger records optional backend campaign evidence. A passing smoke command means the local
runtime can be imported and the minimal API gate passed. It does not by itself complete live
propagation, OD, launch, optimization, covariance, or research validation. Unavailable smoke JSON is
recorded as not-run live evidence, not as a failed required local release gate.

## Campaign Summary

| Backend | Smoke status | Live gate status | Roadmap implication |
| --- | --- | --- | --- |
| Orekit | Available with explicit Homebrew OpenJDK environment | Passed propagation, generic/high-fidelity covariance, and native OD live gates | Orekit live propagation, covariance, and native OD claims are promoted for this machine only when the Java/data environment is configured. |
| RocketPy | Available | Passed configured-example live gate; multi-motor config fails closed | RocketPy configured launch examples passed live validation on this machine, and additional configured motors are rejected because RocketPy 1.11 overwrites earlier motors. |
| Dymos/OpenMDAO | Available | Passed live optimization gates | Dymos default phase and target-seeking pitch-program transcription live tests passed on this machine, without promoting a full multistage ascent design optimizer. |
| TudatPy | Available in isolated conda env | Propagation/high-fidelity covariance/native-variational gates passed; strict multi-scenario comparison found a calibrated J2 tolerance boundary | Tudat live force-model products are promoted only with the recorded comparison tolerances and remain cross-check products, not the operational authority. |
| JAX/JAXLIB | Available | Passed research promotion checklist | JAX research propagation, OD sensitivity, and research-estimate gates passed on this machine, but remain research workflows, not operational OD services. |

## Orekit

Backend: Orekit

Required runtime: `orekit-jpype`, a Java runtime, and Orekit data context through
`ASTRO_OREKIT_DATA_PATH`, `OREKIT_DATA_PATH`, or `~/.orekit/orekit-data.zip`.

Smoke command: `astro orekit-smoke`

Live validation command:
`ASTRO_RUN_OREKIT_LIVE=1 python -m pytest tests/astro_backends/test_orekit_propagation.py::test_live_orekit_two_body_matches_local_reference tests/astro_backends/test_orekit_propagation.py::test_live_orekit_j2_matches_local_reference_scale tests/astro_backends/test_orekit_propagation.py::test_live_orekit_covariance_history_returns_suite_product tests/astro_backends/test_orekit_estimation.py::test_live_orekit_native_od_executes_batch_estimator -q`

Current local status: available when launched with the Homebrew OpenJDK environment and the existing
`~/.orekit/orekit-data.zip` data context. Without that Java environment, the smoke command remains
an actionable unavailable diagnostic rather than a failed release gate.

Validated smoke command:

```bash
JAVA_HOME=/opt/homebrew/opt/openjdk/libexec/openjdk.jdk/Contents/Home PATH="/opt/homebrew/opt/openjdk/bin:$PATH" astro orekit-smoke
```

Smoke output:

```json
{
  "available": true,
  "wrapper": "orekit_jpype",
  "version": "13.1.5.0",
  "message": "Orekit JPype VM, EME2000 frame, and UTC time scale are available."
}
```

Run note: OpenJDK 26 emitted JPype restricted-native-access warnings during JVM startup; the smoke
and live tests still exited successfully.

Roadmap claim allowed: this machine completed the optional Orekit smoke, propagation,
generic/high-fidelity covariance, and native OD live gates with the explicit Java/data environment.

Live validation results:

```text
ASTRO_RUN_OREKIT_LIVE=1 python -m pytest tests/astro_backends/test_orekit_propagation.py::test_live_orekit_two_body_matches_local_reference tests/astro_backends/test_orekit_propagation.py::test_live_orekit_j2_matches_local_reference_scale tests/astro_backends/test_orekit_propagation.py::test_live_orekit_covariance_history_returns_suite_product -q
3 passed in 4.93s

JAVA_HOME=/opt/homebrew/opt/openjdk/libexec/openjdk.jdk/Contents/Home PATH="/opt/homebrew/opt/openjdk/bin:$PATH" ASTRO_RUN_OREKIT_LIVE=1 python -m pytest tests/astro_backends/test_orekit_propagation.py::test_live_orekit_high_fidelity_covariance_records_force_models -q
1 passed in 5.90s

ASTRO_RUN_OREKIT_LIVE=1 python -m pytest tests/astro_backends/test_orekit_estimation.py::test_live_orekit_native_od_executes_batch_estimator -q
1 passed in 5.51s

astro propagate examples/scenarios/leo_orekit_high_fidelity.yaml --backend orekit --output /tmp/astro-orekit-high-fidelity.json
astro propagate examples/scenarios/leo_orekit_drag.yaml --backend orekit --output /tmp/astro-orekit-drag.json
astro propagate examples/scenarios/leo_orekit_srp.yaml --backend orekit --output /tmp/astro-orekit-srp.json
astro propagate examples/scenarios/leo_orekit_third_body.yaml --backend orekit --output /tmp/astro-orekit-third-body.json
astro propagate examples/scenarios/leo_orekit_high_order_gravity.yaml --backend orekit --output /tmp/astro-orekit-high-order-gravity.json
astro propagate examples/scenarios/leo_covariance.yaml --backend orekit --output /tmp/astro-orekit-covariance.json
astro propagate examples/scenarios/leo_orekit_high_fidelity_covariance.yaml --backend orekit --output /tmp/astro-orekit-high-fidelity-covariance.json
astro estimate-measurements /tmp/astro-orekit-native-od-scenario.yaml /tmp/astro-orekit-native-measurements.json --estimator orekit-native --output /tmp/astro-orekit-native-estimate.json
# all commands completed and wrote the listed /tmp/astro-orekit-*.json products
```

Roadmap claim not allowed: the Orekit gates remain optional and environment-dependent; this does
not make default CI depend on Java/Orekit data or broaden native OD beyond the checked
geodetic range/range-rate boundary.

## RocketPy

Backend: RocketPy

Required runtime: RocketPy `>=1.11,<1.12` from the `launch` extra.

Smoke command: `astro rocketpy-smoke`

Live validation command:
`ASTRO_RUN_ROCKETPY_LIVE=1 python -m pytest tests/astro_backends/test_rocketpy_simulation.py::test_live_rocketpy_configured_launch_examples_return_suite_products -q`

Current local status: available. The smoke command exited `0`.

Smoke output:

```json
{
  "available": true,
  "package": "rocketpy",
  "version": "1.11.0",
  "message": "RocketPy Environment, SolidMotor, Rocket, and Flight APIs are available."
}
```

Roadmap claim allowed: configured single-motor RocketPy launch examples pass the suite live adapter
gate on this machine. The suite also has a fail-closed guard for additional configured RocketPy
motors.

Live validation result:

```text
ASTRO_RUN_ROCKETPY_LIVE=1 python -m pytest tests/astro_backends/test_rocketpy_simulation.py::test_live_rocketpy_configured_launch_examples_return_suite_products -q
1 passed in 1.62s
```

Additional-motor guard result:

```text
python -m pytest tests/astro_backends/test_rocketpy_simulation.py::test_propagate_launch_rocketpy_rejects_additional_motors_until_backend_supports_them -q
1 passed in 0.28s

astro launch examples/launch/rocketpy_configured_multimotor_unsupported.yaml --backend rocketpy --output /tmp/astro-rocketpy-multimotor-launch.json
RocketPy launch simulation supports only one motor per rocket in the validated adapter; RocketPy 1.11 overwrites earlier motors when add_motor is called more than once. Remove scenario.rocketpy.additional_motors (strap-on) or use the local/suite model until a validated native multi-motor RocketPy API is available.
exit=2
```

Roadmap claim not allowed: this does not promote native RocketPy staged separation, dropped dry mass,
coast/reignite stage transitions, changing vehicle geometry, or production launch certification
beyond the checked configured-example adapter boundary. It also does not promote native RocketPy
multi-motor direct flight on RocketPy 1.11, because the installed API reports that only one motor per
rocket is supported and later `add_motor` calls overwrite earlier motors.

## Dymos/OpenMDAO

Backend: Dymos/OpenMDAO

Required runtime: Dymos `>=1.13.1,<1.14` and OpenMDAO `>=3.41,<3.42` from the `optimization`
extra.

Smoke command: `astro dymos-smoke`

Live validation command:
`ASTRO_RUN_DYMOS_LIVE=1 python -m pytest tests/astro_backends/test_dymos_optimization.py::test_live_dymos_optimization_returns_suite_product tests/astro_backends/test_dymos_optimization.py::test_live_dymos_pitch_program_optimization_executes_native_transcription -q`

Current local status: available. The smoke command exited `0`.

Smoke output:

```json
{
  "available": true,
  "package": "dymos",
  "version": "1.13.1",
  "openmdao_version": "3.41.0",
  "message": "Dymos Trajectory/Phase and OpenMDAO Problem APIs are available."
}
```

Roadmap claim allowed: Dymos/OpenMDAO default phase and native pitch-program transcription live
tests pass on this machine. The native pitch-program path now minimizes a normalized final
target-insertion error and records target-score metadata in the suite product.

Live validation result:

```text
ASTRO_RUN_DYMOS_LIVE=1 python -m pytest tests/astro_backends/test_dymos_optimization.py::test_live_dymos_optimization_returns_suite_product tests/astro_backends/test_dymos_optimization.py::test_live_dymos_pitch_program_optimization_executes_native_transcription -q
2 passed, 2 OpenMDAO warnings in 2.23s
```

CLI target-seeking product check:

```text
astro optimize-launch examples/launch/pitch_program_two_stage.yaml --backend dymos --dymos-mode pitch-program --output /tmp/astro-dymos-target-seeking-launch.json
wrote launch optimization: /tmp/astro-dymos-target-seeking-launch.json
target_objective: minimize_final_normalized_target_insertion_error
target_score: 1.1572999135341908
target_score_terms: altitude=0.4847465458597893, velocity=0.6716775476699085, radial_velocity=0.0008758200044930763
```

Roadmap claim not allowed: this does not promote the current Dymos products to a full multistage
ascent design optimizer beyond the bounded pitch-program target objective.

## TudatPy

Backend: TudatPy

Required runtime: TudatPy installed through its platform-supported distribution channel.

Smoke command: `astro tudat-smoke`

Live validation command:
`astro compare-tudat-campaign examples/scenarios/leo_two_body.yaml examples/scenarios/leo_j2.yaml --reference-backend local --position-tolerance-km 0.01 --velocity-tolerance-km-s 0.00003 --output /tmp/astro-tudat-reference-campaign-calibrated.json`

Current local status: available in the isolated conda environment
`/tmp/astro-tudat-live-env` with Python 3.12.13 and TudatPy 1.0.0. A direct base-environment conda
dry run would have replaced/downgraded unrelated packages, so the live campaign used an isolated
environment instead.

Validated smoke command:

```bash
conda run -p /tmp/astro-tudat-live-env astro tudat-smoke
```

Smoke output:

```json
{
  "available": true,
  "package": "tudatpy",
  "version": "1.0.0",
  "message": "TudatPy module is available."
}
```

Roadmap claim allowed: this machine completed Tudat native propagation, high-fidelity
finite-difference covariance, high-fidelity native variational covariance, and calibrated
comparison campaign execution in the isolated TudatPy 1.0.0 environment.

Live validation results:

```text
conda run -p /tmp/astro-tudat-live-env astro propagate examples/scenarios/leo_two_body.yaml --backend tudat --output /tmp/astro-tudat-two-body.json
conda run -p /tmp/astro-tudat-live-env astro propagate examples/scenarios/leo_j2.yaml --backend tudat --output /tmp/astro-tudat-j2.json
conda run -p /tmp/astro-tudat-live-env astro propagate examples/scenarios/leo_orekit_drag.yaml --backend tudat --output /tmp/astro-tudat-drag.json
conda run -p /tmp/astro-tudat-live-env astro propagate examples/scenarios/leo_orekit_srp.yaml --backend tudat --output /tmp/astro-tudat-srp.json
conda run -p /tmp/astro-tudat-live-env astro propagate examples/scenarios/leo_orekit_third_body.yaml --backend tudat --output /tmp/astro-tudat-third-body.json
conda run -p /tmp/astro-tudat-live-env astro propagate examples/scenarios/leo_tudat_high_order_gravity.yaml --backend tudat --output /tmp/astro-tudat-high-order-gravity.json
conda run -p /tmp/astro-tudat-live-env astro propagate examples/scenarios/leo_orekit_high_fidelity_covariance.yaml --backend tudat --output /tmp/astro-tudat-high-fidelity-covariance.json
conda run -p /tmp/astro-tudat-live-env astro propagate examples/scenarios/leo_tudat_variational_covariance.yaml --backend tudat --output /tmp/astro-tudat-variational-covariance.json
# all commands completed and wrote suite trajectory/covariance products

ASTRO_RUN_TUDAT_LIVE=1 conda run -p /tmp/astro-tudat-live-env python -m pytest tests/astro_backends/test_tudat_propagation.py::test_live_tudat_high_fidelity_covariance_records_force_models -q
1 passed in 6.72s

ASTRO_RUN_TUDAT_LIVE=1 conda run -p /tmp/astro-tudat-live-env python -m pytest tests/astro_backends/test_tudat_propagation.py::test_live_tudat_native_variational_covariance_records_force_models -q
1 passed in 3.04s

conda run -p /tmp/astro-tudat-live-env astro compare-tudat-reference examples/scenarios/leo_two_body.yaml --reference-backend local --position-tolerance-km 0.001 --velocity-tolerance-km-s 0.000001 --output /tmp/astro-tudat-reference-comparison.json
# passed true; max position delta 0.0006172472620229077 km; max velocity delta 7.936198258437524e-07 km/s

conda run -p /tmp/astro-tudat-live-env astro compare-tudat-campaign examples/scenarios/leo_two_body.yaml examples/scenarios/leo_j2.yaml --reference-backend local --position-tolerance-km 0.001 --velocity-tolerance-km-s 0.000001 --output /tmp/astro-tudat-reference-campaign.json
# passed false; two-body passed, J2 exceeded the strict tolerance at 0.009359857626803657 km and 2.8680360453393343e-05 km/s

conda run -p /tmp/astro-tudat-live-env astro compare-tudat-campaign examples/scenarios/leo_two_body.yaml examples/scenarios/leo_j2.yaml --reference-backend local --position-tolerance-km 0.01 --velocity-tolerance-km-s 0.00003 --output /tmp/astro-tudat-reference-campaign-calibrated.json
# passed true; 2 scenarios passed, 0 failed
```

Roadmap claim not allowed: the strict two-scenario Tudat-vs-local comparison is not green at
1 meter / 1e-6 km/s because the J2 reference case currently needs the documented 10 meter /
3e-5 km/s calibrated tolerance. Tudat remains a live cross-check backend, not the operational
authority for standards-grade ephemerides.

## JAX/JAXLIB

Backend: JAX/JAXLIB

Required runtime: JAX and JAXLIB from the `research` extra.

Smoke command: `astro jax-smoke`

Representative live validation command:
`astro research-propagate examples/scenarios/leo_covariance.yaml --backend jax --cases 1 --position-sigma-km 0 --velocity-sigma-km-s 0 --seed 7 --include-sensitivities --output /tmp/astro-jax-covariance-sensitivity.json`

See `docs/release-checklist.md` for the full JAX promotion checklist, including research
propagation, OD sensitivity, and research-estimate gates.

Current local status: available. The smoke command exited `0`.

Smoke output:

```json
{
  "available": true,
  "package": "jax",
  "version": "0.7.1",
  "jaxlib_version": "0.7.1",
  "message": "JAX and jax.numpy modules are available."
}
```

Roadmap claim allowed: JAX/JAXLIB research propagation, OD sensitivity, and research-estimate
campaigns pass the release-checklist research gates on this machine.

Live validation result:

```text
astro research-estimate examples/scenarios/leo_two_station_od.yaml examples/measurements/leo_two_station_od_measurements.json --backend jax --max-iterations 5 --output /tmp/astro-jax-research-estimate.json
astro synth-measurements examples/scenarios/leo_two_station_angles.yaml --backend local --output /tmp/astro-angle-measurements.json
astro research-od-sensitivity examples/scenarios/leo_two_station_angles.yaml /tmp/astro-angle-measurements.json --backend jax --output /tmp/astro-jax-angle-sensitivity.json
astro research-estimate examples/scenarios/leo_two_station_angles.yaml /tmp/astro-angle-measurements.json --backend jax --max-iterations 8 --output /tmp/astro-jax-angle-estimate.json
astro synth-measurements examples/scenarios/leo_two_station_topocentric.yaml --backend local --output /tmp/astro-topocentric-measurements.json
astro research-od-sensitivity examples/scenarios/leo_two_station_topocentric.yaml /tmp/astro-topocentric-measurements.json --backend jax --output /tmp/astro-jax-topocentric-sensitivity.json
astro research-estimate examples/scenarios/leo_two_station_topocentric.yaml /tmp/astro-topocentric-measurements.json --backend jax --max-iterations 8 --output /tmp/astro-jax-topocentric-estimate.json
astro research-propagate examples/scenarios/leo_orekit_drag.yaml --backend jax --cases 1 --position-sigma-km 0 --velocity-sigma-km-s 0 --seed 7 --output /tmp/astro-jax-drag-research.json
astro research-propagate examples/scenarios/leo_covariance.yaml --backend jax --cases 1 --position-sigma-km 0 --velocity-sigma-km-s 0 --seed 7 --include-sensitivities --output /tmp/astro-jax-covariance-sensitivity.json
astro research-propagate examples/scenarios/leo_orekit_srp.yaml --backend jax --cases 1 --position-sigma-km 0 --velocity-sigma-km-s 0 --seed 7 --output /tmp/astro-jax-srp-research.json
astro research-propagate examples/scenarios/leo_jax_high_order_gravity_research.yaml --backend jax --cases 1 --position-sigma-km 0 --velocity-sigma-km-s 0 --seed 7 --output /tmp/astro-jax-high-order-research.json
astro research-propagate examples/scenarios/leo_jax_third_body_research.yaml --backend jax --cases 1 --position-sigma-km 0 --velocity-sigma-km-s 0 --seed 7 --output /tmp/astro-jax-third-body-research.json
astro research-propagate examples/scenarios/leo_jax_third_body_ephemeris_research.yaml --backend jax --cases 1 --position-sigma-km 0 --velocity-sigma-km-s 0 --seed 7 --output /tmp/astro-jax-third-body-ephemeris-research.json
```

All commands completed successfully and wrote the listed `/tmp/astro-jax-*.json` products.

Roadmap claim not allowed: JAX products remain research workflows and should not be described as
operational differentiable OD services.
