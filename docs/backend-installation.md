# Backend Installation Guide

Date: 2026-06-15

Astro Suite installs a local deterministic baseline by default. External engines are optional and
loaded behind smoke gates so missing runtimes produce actionable diagnostics instead of import-time
failures.

## Base And Development

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

## Orekit

```bash
python -m pip install -e '.[orekit]'
astro orekit-smoke
```

The Orekit adapter uses `orekit-jpype`. The legacy JCC wrapper still exists, but `orekit-jpype` is
the pip-friendly wrapper recommended for new projects by current Orekit Python wrapper guidance.
A working Java runtime and Orekit data context are also required for live propagation. If the
wrapper, JVM, or data setup is unavailable, `astro orekit-smoke` and `astro propagate --backend
orekit` fail with structured diagnostics.

On macOS with Homebrew OpenJDK, a local shell can use:

```bash
export JAVA_HOME=/opt/homebrew/opt/openjdk/libexec/openjdk.jdk/Contents/Home
export PATH="/opt/homebrew/opt/openjdk/bin:$PATH"
export ASTRO_OREKIT_DATA_PATH="$HOME/.orekit/orekit-data.zip"
```

`ASTRO_OREKIT_DATA_PATH` may point to an Orekit data zip or directory. If it is not set, the suite
also checks `OREKIT_DATA_PATH`, then `~/.orekit/orekit-data.zip`.

The Orekit propagation adapter currently supports suite `two_body` scenarios through
`KeplerianPropagator`, suite `j2` scenarios through `NumericalPropagator` plus
`J2OnlyPerturbation`, and suite `orekit_high_fidelity` scenarios through the same numerical
propagation expansion path. Atmospheric drag is implemented with Orekit `DragForce`,
`SimpleExponentialAtmosphere`, and `IsotropicDrag`. Solar radiation pressure is implemented with
Orekit `SolarRadiationPressure` and `IsotropicRadiationSingleCoefficient`. Third-body gravity is
implemented with Orekit `ThirdBodyAttraction` for the Sun and Moon. Native Orekit OD is available
through `astro estimate-measurements --estimator orekit-native` for validated geodetic
range/range-rate records when the live Java/Orekit runtime and data context are installed.

## Launch Backends

```bash
python -m pip install -e '.[launch,optimization]'
astro rocketpy-smoke
astro dymos-smoke
```

The launch and optimization extras are pinned to NumPy-1-compatible backend lines:
RocketPy `>=1.11,<1.12`, Dymos `>=1.13.1,<1.14`, and OpenMDAO `>=3.41,<3.42`.
Newer backend releases may require NumPy 2.x and can break the suite's currently validated
SciPy/Matplotlib binary stack in shared Anaconda environments.

RocketPy and Dymos/OpenMDAO are optional adapter boundaries. The current local launch schema remains
an aggregate point-mass model, while RocketPy-specific vehicle/motor/flight fields live under the
optional `rocketpy` launch-scenario section. Live RocketPy simulation currently supports explicitly
configured solid rockets and can annotate multistage suite scenarios with stage events/samples
reached by one configured RocketPy flight, with metadata for whether the RocketPy solution covered
the full suite stage schedule. That multistage path is a composition adapter, not a validated
multi-motor RocketPy staging solver. Live Dymos optimization currently supports a stage-aware
vertical-ascent phase transcription wrapped in the suite launch-tuning product, with suite
stage-plan metadata, pitch-program control-point metadata, tuned point indices, path constraints,
and a coverage flag showing that the phase duration spans the configured burn schedule. Full
pitch-program multistage Dymos optimization still requires a validated backend runner before it
should be promoted beyond the adapter gate.

## Research Backends

```bash
python -m pip install -e '.[research]'
astro jax-smoke
```

The `research` extra installs JAX/JAXLIB for differentiable or accelerated research workflows. The
current JAX research runner supports vectorized two-body and J2 RK4 ensembles and an opt-in
final-state transition sensitivity matrix via `astro research-propagate --backend jax
--include-sensitivities`. JAX is not the operational frame/time authority; compare research results
against local/Orekit references. `astro research-od-sensitivity --backend jax` writes an
`OdSensitivityResult` with normalized range/range-rate, inertial right-ascension/declination, or
local-horizon azimuth/elevation residuals and a JAX-derived residual Jacobian with respect to the
initial Cartesian state for differentiable OD research workflows. Topocentric angular sensitivity
products record a horizontal-norm regularization floor for exact zenith/nadir geometry, where
azimuth and elevation derivatives are singular.

TudatPy is intentionally not listed as a pip extra because it may not be available from PyPI for this
environment. Install TudatPy through its supported distribution channel for your platform, then run:

```bash
astro tudat-smoke
astro propagate examples/scenarios/leo_two_body.yaml --backend tudat --output /tmp/astro-tudat-two-body.json
astro propagate examples/scenarios/leo_j2.yaml --backend tudat --output /tmp/astro-tudat-j2.json
astro propagate examples/scenarios/leo_orekit_drag.yaml --backend tudat --output /tmp/astro-tudat-drag.json
astro propagate examples/scenarios/leo_orekit_srp.yaml --backend tudat --output /tmp/astro-tudat-srp.json
astro propagate examples/scenarios/leo_orekit_third_body.yaml --backend tudat --output /tmp/astro-tudat-third-body.json
```

## Smoke Command Semantics

Smoke commands return JSON and use these exit codes:

- `0`: runtime is available and the minimal API gate passed.
- `1`: runtime is missing or failed to import; read `message` for the next action.
- `2`: workflow command input or backend dispatch error.

## Optional Backend Principle

Optional backends are loaded only inside adapter modules. Importing `astro_core`, `astro_dynamics`,
`astro_launch`, `astro_od`, or `astro_cli` must not require optional engines to be installed.
