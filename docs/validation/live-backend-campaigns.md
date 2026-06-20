# Live Backend Campaign Ledger

Last local smoke run: 2026-06-19 17:29:40 PDT on branch `codex/orbit-fd-od-mvp`
at commit `9c7affb`.

This ledger records optional backend campaign evidence. A passing smoke command means the local
runtime can be imported and the minimal API gate passed. It does not by itself complete live
propagation, OD, launch, optimization, covariance, or research validation. Unavailable smoke JSON is
recorded as not-run live evidence, not as a failed required local release gate.

## Campaign Summary

| Backend | Smoke status | Live gate status | Roadmap implication |
| --- | --- | --- | --- |
| Orekit | Unavailable: wrapper installed, Java runtime missing | Not run | Keep Orekit live propagation, covariance, and native OD claims behind the optional live gate. |
| RocketPy | Available | Ready to attempt | RocketPy runtime is available for attempting live launch validation, but the campaign is not complete until live launch tests/CLI examples are executed. |
| Dymos/OpenMDAO | Available | Ready to attempt | Dymos/OpenMDAO runtime is available for attempting live optimization validation, but the campaign is not complete until live optimization tests/CLI examples are executed. |
| TudatPy | Unavailable: package missing | Not run | Keep Tudat live propagation/cross-check/covariance claims behind the optional live gate. |
| JAX/JAXLIB | Available | Ready to attempt | JAX/JAXLIB runtime is available for attempting research validation campaigns, but the products remain research workflows, not operational OD services. |

## Orekit

Backend: Orekit

Required runtime: `orekit-jpype`, a Java runtime, and Orekit data context through
`ASTRO_OREKIT_DATA_PATH`, `OREKIT_DATA_PATH`, or `~/.orekit/orekit-data.zip`.

Smoke command: `astro orekit-smoke`

Live validation command:
`ASTRO_RUN_OREKIT_LIVE=1 python -m pytest tests/astro_backends/test_orekit_propagation.py::test_live_orekit_two_body_matches_local_reference tests/astro_backends/test_orekit_propagation.py::test_live_orekit_j2_matches_local_reference_scale tests/astro_backends/test_orekit_propagation.py::test_live_orekit_covariance_history_returns_suite_product tests/astro_backends/test_orekit_estimation.py::test_live_orekit_native_od_executes_batch_estimator -q`

Current local status: unavailable. The smoke command exited `1`.

Run note: macOS `/usr/libexec/java_home` printed a Java Runtime notice before the structured JSON.

Unavailable diagnostic:

```json
{
  "available": false,
  "wrapper": "orekit_jpype",
  "version": "13.1.5.0",
  "message": "Orekit backend unavailable: JVM, Orekit imports, or data context failed: Command '['/usr/libexec/java_home']' returned non-zero exit status 1."
}
```

Roadmap claim allowed: the suite has an optional Orekit adapter and structured unavailable
diagnostics on this machine.

Roadmap claim not allowed: this machine has not completed live Orekit propagation, covariance, or
native OD validation because Java is not available.

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

Roadmap claim allowed: the RocketPy runtime is available on this machine, so the live launch
campaign is eligible to attempt.

Roadmap claim not allowed: RocketPy live launch validation is not complete until the live test or
equivalent configured RocketPy CLI campaign is executed and recorded.

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

Roadmap claim allowed: the Dymos/OpenMDAO runtime is available on this machine, so the live
launch-optimization campaign is eligible to attempt.

Roadmap claim not allowed: Dymos live optimization validation is not complete until the live tests
or equivalent configured Dymos CLI campaigns are executed and recorded.

## TudatPy

Backend: TudatPy

Required runtime: TudatPy installed through its platform-supported distribution channel.

Smoke command: `astro tudat-smoke`

Live validation command:
`astro compare-tudat-campaign examples/scenarios/leo_two_body.yaml examples/scenarios/leo_j2.yaml --reference-backend local --position-tolerance-km 0.001 --velocity-tolerance-km-s 0.000001 --output /tmp/astro-tudat-reference-campaign.json`

Current local status: unavailable. The smoke command exited `1`.

Unavailable diagnostic:

```json
{
  "available": false,
  "package": "tudatpy",
  "version": null,
  "message": "TudatPy is not installed."
}
```

Roadmap claim allowed: the suite has a TudatPy adapter boundary and structured unavailable
diagnostics on this machine.

Roadmap claim not allowed: this machine has not completed live Tudat propagation, cross-check, or
covariance validation.

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

Roadmap claim allowed: the JAX/JAXLIB runtime is available on this machine, so research
propagation, sensitivity, and research OD campaigns are eligible to attempt.

Roadmap claim not allowed: JAX products remain research workflows and should not be described as
operational differentiable OD services.
