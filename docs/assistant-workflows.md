# Assistant Workflows

Astro Suite explores verifiable AI-assisted mission workflows: natural-language intent is compiled
into typed flight-dynamics and orbit-determination plans, executed by deterministic backends, and
checked through reproducible artifact validators.

Astro Suite's assistant layer compiles natural-language mission-analysis requests into typed,
reviewable workflow plans. The assistant does not perform flight dynamics itself. Astro Suite CLI
commands generate and validate the artifacts.

## First Supported Workflow

The first workflow is scenario-parameterized local orbit determination:

```bash
astro verify-assistant "Run local OD on leo_two_station_topocentric.yaml"
astro ask "Run the local OD demo" --dry-run
astro ask "Run local orbit determination on examples/scenarios/leo_two_station_angles.yaml and export TDM." --dry-run
astro ask "Run local orbit determination on examples/scenarios/leo_two_station_angles.yaml and export TDM." --execute --approved --trace-output /tmp/astro-assistant/leo_two_station_angles/trace.json --report-output /tmp/astro-assistant/leo_two_station_angles/report.json --report-summary-output /tmp/astro-assistant/leo_two_station_angles/report.txt
```

The default demo validates `examples/scenarios/leo_two_station_od.yaml`. Explicit supported
scenario paths or aliases bind the same workflow to the requested scenario, synthesize local
measurements, export TDM, estimate the initial state, and record a trace under a scenario-specific
artifact directory. Execution can also write a JSON workflow report and concise text summary
covering the verified plan, executed steps, declared artifacts, measurement count, TDM line count,
estimate convergence, iteration count, RMS, Jacobian rank, residual count, and max absolute
residual.

The workflow-pack manifest lives at `examples/workflows/local_od/manifest.yaml`. It records the
supported scenarios, required step order, artifact kinds, report metrics, policy boundaries, and
golden prompt fixture path for this first supported workflow. The prompt fixtures live at
`examples/workflows/local_od/golden_prompts.yaml` and are replayed in tests so every supported
scenario keeps a stable prompt-to-plan binding.

Supported local OD scenario paths:

- `examples/scenarios/leo_doppler.yaml`
- `examples/scenarios/leo_geodetic_eop_table_topocentric.yaml`
- `examples/scenarios/leo_geodetic_eop_topocentric.yaml`
- `examples/scenarios/leo_geodetic_precession_nutation_topocentric.yaml`
- `examples/scenarios/leo_geodetic_topocentric.yaml`
- `examples/scenarios/leo_radiometric_links.yaml`
- `examples/scenarios/leo_radiometric_media.yaml`
- `examples/scenarios/leo_radiometric_weather_frequency.yaml`
- `examples/scenarios/leo_two_station_angles.yaml`
- `examples/scenarios/leo_two_station_od.yaml`
- `examples/scenarios/leo_two_station_topocentric.yaml`

Unsupported scenarios fail closed. The planner must not silently substitute a different scenario
than the one requested in the prompt.

`astro verify-assistant` emits a JSON support report without executing workflow commands. It is the
fastest way to see which scenario was resolved, where artifacts would be written, and which
deterministic verification diagnostics would block execution.

Unsupported-scenario reports classify the blocker instead of returning only a generic rejection.
Current classifier codes include:

- `path_policy`: the requested path is outside the allowed example-scenario boundary.
- `optional_backend`: the scenario requires an optional or high-fidelity backend outside local OD.
- `missing_measurements`: the scenario cannot generate measurement records for local OD.
- `rank_deficient_geometry`: the generated OD geometry cannot estimate a full six-state solution.
- `unsupported_local_model`: local propagation or estimation rejected the scenario configuration.
- `unsupported_prompt`: the prompt is outside the current local OD workflow.

## Safety Boundaries

- Plans are typed Pydantic models.
- Commands are generated from an allow-listed registry.
- Deterministic verification checks scenario binding, step order, local backends, output paths,
  export format, and declared artifacts before execution.
- Execution defaults to dry-run.
- Artifact-writing execution requires `--approved`.
- `--report-output` writes a product-level JSON workflow report. `--report-summary-output` writes a
  concise text view over the same report. Dry-run reports do not read stale artifacts from previous
  executions.
- Optional backends are blocked in this first assistant slice.
- A future agentic verifier may generate extra challenge checks, but deterministic validators remain
  the execution authority.
- Arbitrary shell commands are not supported.

## Paired Assurance Review Plan

`astro_assistant.planner.assurance_review_plan` compiles a fixed two-step workflow: verify one paired
assurance result, then write its deterministic review. Both steps must use the same source path and
the declared review output must match the command input. Verification is read-only; review output
requires explicit approval. The assistant does not derive findings, alter calibration status, or
interpret profile counts as probability; those rules remain in `astro_assurance`.

## Assurance Review Comparison Plan

`astro_assistant.planner.assurance_review_comparison_plan` compiles one fixed, approval-gated
comparison step. Baseline and candidate review paths must differ, and the declared output must match
the command input. The command re-verifies both reviews and their bound paired evidence before
writing a comparison. Its evidence recommendations remain non-executing decision support.
