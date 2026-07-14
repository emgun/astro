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
- power generation, scheduled load, battery energy/state-of-charge, interval battery-energy change,
  unmet load/energy, and curtailed surplus-energy samples
- lumped thermal node temperatures and per-node heat-balance screening values
- ADCS pointing, torque, slew-rate, and actuator-utilization screening margins
- ground-site access windows
- link budget windows with Eb/N0 margin and data-volume estimates
- itemized mass-budget rollup evidence
- aggregate mass, power, thermal, ADCS, link, and mass-budget design margins

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
- `spacecraft.mass_budget_items`: optional itemized mass entries with category and contingency
- `power`: solar array, battery, load, minimum state-of-charge, and battery efficiency assumptions
- `thermal_nodes`: lumped thermal nodes with radiator, absorptivity, emissivity, and temperature
  limits, optional albedo/planet-IR flux terms, and optional mode heat scaling
- `adcs`: pointing, torque, slew-rate, and actuator-utilization screening requirements
- `ground_sites`: latitude, longitude, altitude, and elevation mask
- `links`: RF link assumptions tied to configured ground sites
- `mode_schedule`: optional elapsed-time mission modes that drive power load
- `power_loads`: optional named elapsed-time load additions layered on top of mode loads

The subsystem fidelity pack keeps the model deterministic and low-order. Scheduled loads are summed
with mode loads; battery energy applies charge/discharge efficiency. When the battery cannot serve
an interval's deficit, the result records unmet bus load and energy instead of hiding the deficit
behind the zero-energy clamp. Surplus energy that a full battery cannot accept is recorded as
curtailed energy. These are passive accounting products, not an active load-shedding policy.
Thermal nodes remain lumped
nodes but include direct solar, albedo, planetary IR, internal heat, and radiated heat in the
reported heat balance. ADCS samples report fixed pointing, torque, slew-rate, and utilization
screening margins from the scenario assumptions. Mass-budget rollups compare itemized
contingency-adjusted dry-plus-payload mass against the configured dry-plus-payload reference.

## Validation

The required local gate is:

```bash
astro run-twin examples/twin/leo_observer.yaml \
  --output /tmp/astro-twin-result.json \
  --summary-output /tmp/astro-twin-summary.txt
python -m pytest tests/astro_twin -q
```

The checked full-orbit power/thermal uncertainty gate is:

```bash
astro validate-campaign examples/campaigns/leo_full_orbit_power_thermal.yaml
astro run-campaign examples/campaigns/leo_full_orbit_power_thermal.yaml \
  --output-dir /tmp/astro-full-orbit-power-thermal --workers 4
astro analyze-campaign-sensitivity /tmp/astro-full-orbit-power-thermal \
  --metric total_unmet_energy \
  --requirement-margin battery_soc \
  --requirement-margin no_unmet_energy \
  --requirement-margin bus_hot \
  --output /tmp/astro-full-orbit-power-thermal-sensitivity.json
```

The orbit reference spans 6,000 seconds at 60-second resolution. Eclipse duration and sunlit
fraction therefore remain sampled local-geometry screening quantities, not event-resolved shadow
transitions or operational energy predictions.

The gate validates the package-level models and deterministic subsystem products without requiring
Orekit, RocketPy, Dymos/OpenMDAO, TudatPy, JAX, or any external provider.

## Constellation Twin

The constellation twin workflow runs multiple single-spacecraft twin scenarios and aggregates fleet
access, revisit, link-budget, data-volume, target-grid sensor coverage, and margin evidence.

```bash
astro run-constellation-twin examples/twin/constellation_leo_observers.yaml \
  --output /tmp/astro-constellation-twin.json \
  --summary-output /tmp/astro-constellation-twin.txt
```

The checked-in reference runs two LEO observer members against `equator-eci`, embeds each member's
suite-owned `DigitalTwinResult`, and returns fleet access/link summaries, coverage-map summaries,
member link totals, and fleet margin evidence.

Constellation scenarios can include `coverage_maps`. A coverage map defines a nadir-pointed sensor
cone and a set of latitude/longitude target points:

```yaml
coverage_maps:
  - name: equatorial-targets
    sensor:
      name: nadir-imager
      field_of_view_half_angle_deg: 45.0
      minimum_elevation_deg: 0.0
    targets:
      - name: prime-meridian
        latitude_deg: 0.0
        longitude_deg: 0.0
      - name: east-equator
        latitude_deg: 0.0
        longitude_deg: 4.0
    minimum_target_coverage_fraction: 0.2
    maximum_target_revisit_gap_s: 600.0
```

The result records per-target coverage duration, coverage fraction, longest gap, mean gap, and
simultaneous-spacecraft count, then rolls those into map-level mean coverage, minimum target
coverage, maximum target gap, and design margins.

The v1 constellation product is deterministic design-screening evidence. It is not operational
coverage authority, contact scheduling, spectrum coordination, collision avoidance, certified sensor
performance, or constellation optimization. Coverage maps use spherical Earth geometry, uniform
Earth rotation, local sampled propagation, target elevation, optional range, and nadir off-boresight
cone checks.
