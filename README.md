# Astro Suite

Astro Suite is a Python flight-dynamics toolkit for verifiable mission-analysis workflows. It
combines deterministic astrodynamics tools with an assistant layer that turns natural-language
requests into typed workflow plans, allow-listed CLI commands, declared artifacts, and verification
traces.

```text
request -> typed plan -> allow-listed commands -> deterministic artifacts -> verification trace
```

## What It Does

- Validates YAML mission scenarios with Pydantic models.
- Propagates local two-body, J2, maneuvered, covariance, conjunction, and attitude workflows.
- Generates, imports, exports, and estimates orbit-determination measurements.
- Runs local least-squares OD with rank and convergence checks.
- Runs deterministic launch/ascent baselines, launch pitch tuning, and launch-to-orbit handoff.
- Simulates ballistic, prescribed-bank lifting, and target-tracking guided reentry with load,
  aerothermal, target-miss, margin, and guidance-optimization products.
- Runs an integrated single-spacecraft digital twin screening workflow for orbit geometry, power,
  thermal, ADCS, coverage, link budget, itemized mass rollups, and design margins.
- Runs a checked launch-to-reentry mission lifecycle workflow with explicit state, epoch, mass,
  propellant-reserve, entry-interface, provenance, and margin evidence across every phase.
- Provides optional adapter boundaries for Orekit, RocketPy, Dymos/OpenMDAO, TudatPy, and JAX.
- Exposes assistant workflows for scenario-bound local OD requests.

Astro Suite owns the public product boundaries: scenarios, trajectories, measurements, estimates,
launch and reentry reports, digital-twin results, backend metadata, and assistant traces. External
engines are integrated through explicit adapter boundaries.

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

Run the integrated digital twin screening workflow:

```bash
astro run-twin examples/twin/leo_observer.yaml \
  --output /tmp/astro-twin-result.json \
  --summary-output /tmp/astro-twin-summary.txt
```

Run ballistic and guided reentry workflows:

```bash
astro simulate-reentry examples/reentry/ballistic_capsule.yaml \
  --output /tmp/astro-ballistic-reentry.json \
  --summary-output /tmp/astro-ballistic-reentry.txt

astro optimize-reentry examples/reentry/guided_lifting_body.yaml \
  --output /tmp/astro-reentry-optimization.json \
  --tuned-scenario-output /tmp/astro-reentry-tuned.yaml
```

Run the integrated launch-to-reentry mission lifecycle:

```bash
astro run-mission-lifecycle examples/lifecycle/leo_round_trip.yaml \
  --output /tmp/astro-mission-lifecycle.json \
  --summary-output /tmp/astro-mission-lifecycle.txt \
  --artifacts-dir /tmp/astro-mission-lifecycle-artifacts
```

## Assistant Workflow

The assistant layer compiles supported local OD requests into typed plans, validates scenario
support, keeps paths within supported examples, emits structured support classifications,
and requires explicit approval before execution.

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
  --trace-output /tmp/astro-assistant/leo_two_station_angles/trace.json \
  --report-output /tmp/astro-assistant/leo_two_station_angles/report.json \
  --report-summary-output /tmp/astro-assistant/leo_two_station_angles/report.txt
```

See [Assistant Workflows](docs/assistant-workflows.md) for the support codes, policy gates, and
trace contract. The local OD workflow pack manifest is in
`examples/workflows/local_od/manifest.yaml`; its checked-in golden prompt fixtures are in
`examples/workflows/local_od/golden_prompts.yaml`.

## Optional Backends

The default examples use deterministic local implementations. Optional backends are available for
cross-checks, research workflows, and adapter experiments:

- Orekit: operational-style propagation and OD adapter boundary.
- RocketPy: explicitly configured launch simulations.
- Dymos/OpenMDAO: launch optimization transcriptions.
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

## Repository Map

- `src/astro_core`: shared scenario, state, trajectory, and error models.
- `src/astro_dynamics`: local propagation, covariance, attitude, maneuvers, and conjunction tools.
- `src/astro_od`: measurement generation, import/export, calibration, and estimation.
- `src/astro_launch`: launch/ascent models, local propagation, handoff, tuning, and reporting.
- `src/astro_reentry`: entry scenarios, 3-DOF simulation, guidance, aerothermal loads, margins,
  optimization, and orbit handoff.
- `src/astro_backends`: optional engine adapters and smoke checks.
- `src/astro_assistant`: typed assistant plans, policy, verification, and artifact validation.
- `src/astro_twin`: integrated single-spacecraft digital twin screening workflow.
- `src/astro_mission`: checked orchestration across launch, operations, twin, deorbit, and reentry.
- `src/astro_cli`: the `astro` command line interface.
- `examples/`: runnable scenarios, launch cases, measurements, and assistant prompts.
- `docs/`: validation, backend, assistant, and research notes.

## Verification

```bash
python -m ruff check .
python -m mypy
python -m pytest -q
python -m build
```

Useful docs:

- [Validation Matrix](docs/validation-matrix.md)
- [Digital Twin](docs/digital-twin.md)
- [Reentry Modeling And Simulation](docs/reentry.md)
- [Mission Lifecycle Workflow](docs/mission-lifecycle.md)
- [Assistant Workflows](docs/assistant-workflows.md)
- [Assistant MCP Contract](docs/assistant-mcp-contract.md)
- [Backend Installation](docs/backend-installation.md)

## License

Astro Suite is licensed under the Apache License 2.0. See [LICENSE](LICENSE).
