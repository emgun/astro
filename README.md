# Astro Suite

Astro Suite is a Python flight-dynamics sandbox for verifiable mission-analysis workflows. It
combines deterministic astrodynamics tools with a bounded assistant layer that turns natural-language
intent into typed workflow plans, allow-listed CLI commands, declared artifacts, and verification
traces.

The important idea is not "an LLM doing astrodynamics." The model stays at the interface layer:

```text
natural language intent -> typed plan -> allow-listed commands -> deterministic artifacts -> checks
```

## What It Does

- Validates YAML mission scenarios with Pydantic models.
- Propagates local two-body, J2, maneuvered, covariance, conjunction, and attitude workflows.
- Generates, imports, exports, and estimates orbit-determination measurements.
- Runs local least-squares OD with rank and convergence checks.
- Runs deterministic launch/ascent baselines, launch pitch tuning, and launch-to-orbit handoff.
- Provides optional adapter boundaries for Orekit, RocketPy, Dymos/OpenMDAO, TudatPy, and JAX.
- Exposes a constrained assistant workflow for scenario-bound local OD requests.

Astro Suite owns the public product boundaries: scenarios, trajectories, measurements, estimates,
launch reports, backend metadata, and assistant traces. External engines are adapters, not hidden
sources of truth.

## Install

```bash
python -m pip install -e '.[dev]'
```

Optional backend extras are installed separately:

```bash
python -m pip install -e '.[orekit]'
python -m pip install -e '.[launch,optimization]'
python -m pip install -e '.[research]'
```

Backend runtimes may also need system setup, data files, or platform-specific installs. See
[Backend Installation](docs/backend-installation.md).

## Quick Start

Validate and propagate a local orbit scenario:

```bash
astro validate examples/scenarios/leo_two_body.yaml
astro propagate examples/scenarios/leo_two_body.yaml --backend local --output /tmp/astro-trajectory.json
```

Generate measurements and run local OD:

```bash
astro synth-measurements examples/scenarios/leo_two_station_od.yaml \
  --backend local \
  --output /tmp/astro-measurements.json

astro estimate-measurements examples/scenarios/leo_two_station_od.yaml \
  /tmp/astro-measurements.json \
  --backend local \
  --output /tmp/astro-estimate.json
```

Run a launch/ascent baseline and hand off to orbit propagation:

```bash
astro launch examples/launch/pitch_program_two_stage.yaml \
  --backend local \
  --output /tmp/astro-launch.json

astro handoff-launch /tmp/astro-launch.json \
  --output /tmp/astro-insertion.yaml \
  --duration-s 600 \
  --step-s 60

astro propagate /tmp/astro-insertion.yaml \
  --backend local \
  --output /tmp/astro-insertion-trajectory.json
```

## Assistant Workflow

The assistant layer is deliberately constrained. It compiles supported local OD requests into a
typed plan, checks scenario support, keeps paths inside the allowed example boundary, emits
structured support classifications, and fails closed when a prompt or scenario is unsupported.

Check whether a request is supported:

```bash
astro verify-assistant "Run local OD on leo_two_station_topocentric.yaml"
```

Preview a plan without executing it:

```bash
astro ask "Run local orbit determination on examples/scenarios/leo_two_station_angles.yaml and export TDM." \
  --dry-run
```

Execute only with explicit approval:

```bash
astro ask "Run local orbit determination on examples/scenarios/leo_two_station_angles.yaml and export TDM." \
  --execute \
  --approved \
  --trace-output /tmp/astro-assistant/leo_two_station_angles/trace.json
```

See [Assistant Workflows](docs/assistant-workflows.md) for the support codes, policy gates, and
trace contract.

## Optional Backends

The default examples use deterministic local implementations. Optional backends are available for
cross-checks, research workflows, and adapter experiments:

- Orekit: operational-style propagation and OD adapter boundary.
- RocketPy: explicitly configured launch simulations.
- Dymos/OpenMDAO: bounded launch optimization transcriptions.
- TudatPy: propagation and covariance cross-checks.
- JAX: differentiable research propagation and OD sensitivity workflows.

Smoke checks:

```bash
astro orekit-smoke
astro rocketpy-smoke
astro dymos-smoke
astro tudat-smoke
astro jax-smoke
```

These are optional runtime gates, not claims of flight qualification.

## Repository Map

- `src/astro_core`: shared scenario, state, trajectory, and error models.
- `src/astro_dynamics`: local propagation, covariance, attitude, maneuvers, and conjunction tools.
- `src/astro_od`: measurement generation, import/export, calibration, and estimation.
- `src/astro_launch`: launch/ascent models, local propagation, handoff, tuning, and reporting.
- `src/astro_backends`: optional engine adapters and smoke checks.
- `src/astro_assistant`: typed assistant plans, policy, verification, and artifact validation.
- `src/astro_cli`: the `astro` command line interface.
- `examples/`: runnable scenarios, launch cases, measurements, and assistant prompts.
- `docs/`: validation, backend, release, assistant, and research notes.

## Verification

```bash
python -m ruff check .
python -m mypy
python -m pytest -q
python -m build
```

Useful docs:

- [Validation Matrix](docs/validation-matrix.md)
- [Assistant Workflows](docs/assistant-workflows.md)
- [Assistant MCP Contract](docs/assistant-mcp-contract.md)
- [Backend Installation](docs/backend-installation.md)
- [Current State](docs/current-state.md)
- [Release Checklist](docs/release-checklist.md)

## Safety Scope

Astro Suite is a deterministic mission-analysis and research codebase. It is not flight software,
does not claim flight qualification, and should not be used as spacecraft command authority. The
assistant interface is intentionally constrained to typed plans, allow-listed commands, approval
gates, and artifact validators.

## License

Astro Suite is licensed under the Apache License 2.0. See [LICENSE](LICENSE).
