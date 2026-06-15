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

The Orekit adapter uses `orekit-jpype`. A working Java runtime and Orekit data context are also
required for live propagation. If the wrapper, JVM, or data setup is unavailable, `astro orekit-smoke`
and `astro propagate --backend orekit` fail with structured diagnostics.

## Launch Backends

```bash
python -m pip install -e '.[launch,optimization]'
astro rocketpy-smoke
astro dymos-smoke
```

RocketPy and Dymos/OpenMDAO are optional adapter boundaries. The current local launch schema is an
aggregate point-mass model; live RocketPy simulation and Dymos optimization require backend-specific
vehicle/motor or phase-model configuration before they should be promoted beyond the adapter gates.

## Research Backends

```bash
python -m pip install -e '.[research]'
astro jax-smoke
```

The `research` extra installs JAX/JAXLIB for differentiable or accelerated research workflows. JAX is
not the operational frame/time authority; compare research results against local/Orekit references.

TudatPy is intentionally not listed as a pip extra because it may not be available from PyPI for this
environment. Install TudatPy through its supported distribution channel for your platform, then run:

```bash
astro tudat-smoke
```

## Smoke Command Semantics

Smoke commands return JSON and use these exit codes:

- `0`: runtime is available and the minimal API gate passed.
- `1`: runtime is missing or failed to import; read `message` for the next action.
- `2`: workflow command input or backend dispatch error.

## Optional Backend Principle

Optional backends are loaded only inside adapter modules. Importing `astro_core`, `astro_dynamics`,
`astro_launch`, `astro_od`, or `astro_cli` must not require optional engines to be installed.
