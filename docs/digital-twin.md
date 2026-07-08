# Digital Twin

Astro Suite's digital twin workflow runs a deterministic single-spacecraft mission screening model.
It combines local orbit propagation with power, thermal, ADCS, coverage, link budget, mass, and
design-margin products from one checked-in scenario.

```bash
astro run-twin examples/twin/leo_observer.yaml \
  --output /tmp/astro-twin-result.json \
  --summary-output /tmp/astro-twin-summary.txt
```

The command writes a suite-owned `DigitalTwinResult` JSON product and, optionally, a concise text
summary. The result includes:

- orbit-derived geometry samples with altitude and sunlight state
- power generation, load, and battery state-of-charge samples
- lumped thermal node temperatures
- ADCS pointing and torque screening margins
- ground-site access windows
- link budget windows with Eb/N0 margin and data-volume estimates
- aggregate mass, power, thermal, ADCS, and link design margins

The v1 twin is design-screening evidence. It is not flight qualification, thermal certification,
RF certification, operational readiness evidence, or constellation coverage authority. Coverage
geometry uses spherical Earth and uniform Earth-rotation assumptions, and orbit propagation uses the
local deterministic backend.

## Scenario Shape

Digital twin scenarios are YAML or JSON files that reference a normal Astro Suite orbit scenario and
add spacecraft, subsystem, ground-site, link, and mission-mode configuration. The checked-in example
is `examples/twin/leo_observer.yaml`.

The orbit scenario remains a normal suite scenario:

```yaml
scenario_id: leo-observer
orbit_scenario: examples/scenarios/leo_two_body.yaml
```

Subsystem sections define the screening assumptions:

- `spacecraft`: dry, payload, and propellant mass plus required mass margin
- `power`: solar array, battery, load, and minimum state-of-charge assumptions
- `thermal_nodes`: lumped thermal nodes with radiator, absorptivity, emissivity, and temperature
  limits
- `adcs`: pointing and torque screening requirements
- `ground_sites`: latitude, longitude, altitude, and elevation mask
- `links`: RF link assumptions tied to configured ground sites
- `mode_schedule`: optional elapsed-time mission modes that drive power load

## Validation

The required local gate is:

```bash
astro run-twin examples/twin/leo_observer.yaml \
  --output /tmp/astro-twin-result.json \
  --summary-output /tmp/astro-twin-summary.txt
python -m pytest tests/astro_twin -q
```

The gate validates the package-level models and deterministic subsystem products without requiring
Orekit, RocketPy, Dymos/OpenMDAO, TudatPy, JAX, or any external provider.
