# Assistant Workflows

Astro Suite explores verifiable AI-assisted mission workflows: natural-language intent is compiled
into typed flight-dynamics and orbit-determination plans, executed by deterministic backends, and
checked through reproducible artifact validators.

Astro Suite's assistant layer compiles natural-language mission-analysis requests into typed,
reviewable workflow plans. The assistant does not perform flight dynamics itself. Astro Suite CLI
commands generate and validate the artifacts.

## First Supported Workflow

The first workflow is the local orbit-determination demo:

```bash
astro ask "Run the local OD demo" --dry-run
astro ask "Run the local OD demo" --execute --approved --trace-output /tmp/astro-assistant/trace.json
```

The generated plan validates `examples/scenarios/leo_two_station_od.yaml`, synthesizes local
measurements, exports TDM, estimates the initial state, and records a trace.

## Safety Boundaries

- Plans are typed Pydantic models.
- Commands are generated from an allow-listed registry.
- Execution defaults to dry-run.
- Artifact-writing execution requires `--approved`.
- Optional backends are blocked in this first assistant slice.
- Arbitrary shell commands are not supported.
