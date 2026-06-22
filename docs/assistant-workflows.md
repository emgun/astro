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
astro ask "Run local orbit determination on examples/scenarios/leo_two_station_angles.yaml and export TDM." --execute --approved --trace-output /tmp/astro-assistant/leo_two_station_angles/trace.json
```

The default demo validates `examples/scenarios/leo_two_station_od.yaml`. Explicit supported
scenario paths or aliases bind the same workflow to the requested scenario, synthesize local
measurements, export TDM, estimate the initial state, and record a trace under a scenario-specific
artifact directory.

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

## Safety Boundaries

- Plans are typed Pydantic models.
- Commands are generated from an allow-listed registry.
- Deterministic verification checks scenario binding, step order, local backends, output paths,
  export format, and declared artifacts before execution.
- Execution defaults to dry-run.
- Artifact-writing execution requires `--approved`.
- Optional backends are blocked in this first assistant slice.
- A future agentic verifier may generate extra challenge checks, but deterministic validators remain
  the execution authority.
- Arbitrary shell commands are not supported.
