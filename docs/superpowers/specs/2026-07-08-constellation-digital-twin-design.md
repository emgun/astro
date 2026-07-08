# Constellation Digital Twin Design

Date: 2026-07-08
Status: Draft for user review

## North Star

Extend Astro Suite's integrated digital twin from one spacecraft to a suite-owned constellation
screening workflow. The workflow should answer early design questions that the single-spacecraft
twin cannot answer: fleet contact coverage, revisit gaps, simultaneous access, downlink capacity,
per-spacecraft margin outliers, and constellation-level readiness signals.

The product remains deterministic design-screening evidence. It is not coverage certification,
regulatory spectrum analysis, operational contact scheduling, autonomous mission planning, or
flight qualification.

## Selected Approach

Build a thin fleet aggregation layer over the existing single-spacecraft `astro_twin` workflow.

Each constellation member references a normal `DigitalTwinScenario`. The constellation runner calls
`run_digital_twin` for each member, then aggregates access windows, link windows, data volume, and
design margins into a new suite-owned `ConstellationTwinResult`. This keeps subsystem physics in
one place, avoids duplicating the digital-twin runner, and lets the first constellation slice ship
without optional providers or optimizer dependencies.

## Alternatives Considered

### Approach A: Thin Aggregation Over Single-Spacecraft Twins

This is the recommended path. It is small, locally verifiable, and uses the reviewed v1 twin as a
stable unit. The tradeoff is that members are propagated independently, so no inter-satellite links,
crosslink scheduling, collision avoidance, or coupled fleet operations are modeled in v1.

### Approach B: Native Multi-Spacecraft Propagation Core

This would load one constellation scenario and propagate every spacecraft through a new shared
propagation layer before subsystem modeling. It could support shared clocks, phasing, and later
inter-satellite geometry more naturally, but it would duplicate orchestration and make the first
fleet slice larger than necessary.

### Approach C: Coverage-Only Tool

This would ignore power, thermal, ADCS, mass, and link-budget margins and only compute access/revisit
metrics. It would be fast, but it would undercut the integrated twin's value: the fleet answer would
not show whether the spacecraft providing coverage still has credible design margins.

## Version 1 Scope

Included:

- A constellation scenario model that names members and references existing twin scenario files.
- A deterministic runner that returns one `DigitalTwinResult` per member plus fleet-level summary
  products.
- Fleet access aggregation by ground site.
- Revisit-gap metrics by ground site:
  - total visible duration
  - longest access gap
  - mean access gap
  - maximum simultaneous visible spacecraft count
  - percentage of the analysis window with at least one spacecraft visible
- Fleet link aggregation:
  - total data volume by ground site
  - total data volume by spacecraft
  - worst link margin by spacecraft and ground site
- Fleet margin aggregation:
  - per-spacecraft limiting margin
  - worst fleet margin
  - list of member scenarios with warning or failing margins
- CLI command:
  - `astro run-constellation-twin examples/twin/constellation_leo_observers.yaml --output /tmp/astro-constellation-twin.json --summary-output /tmp/astro-constellation-twin.txt`
- A checked-in two-member LEO reference constellation that produces overlapping and non-overlapping
  access windows against the same ground site.

Deferred:

- Inter-satellite links and crosslink routing.
- Operational contact scheduling or conflict resolution.
- Constellation station-keeping, collision risk, and relative-navigation products.
- Target-area coverage grids, swath geometry, sensor field-of-view, and latitude/longitude revisit
  maps.
- Optimizing constellation phasing, number of planes, or ground-station placement.
- Optional backend propagation or live provider data.

## Product Models

Add these models to `astro_twin.models` or a new `astro_twin.constellation_models` module if the
single file becomes too large.

```python
class ConstellationMemberConfig(AstroModel):
    name: str
    twin_scenario: str


class ConstellationCoverageRequirement(AstroModel):
    ground_site: str
    minimum_coverage_fraction: float = 0.0
    maximum_revisit_gap_s: float | None = None


class ConstellationTwinScenario(AstroModel):
    scenario_id: str
    members: tuple[ConstellationMemberConfig, ...]
    coverage_requirements: tuple[ConstellationCoverageRequirement, ...] = ()


class FleetAccessSummary(AstroModel):
    ground_site: str
    total_access_duration_s: float
    longest_gap_s: float
    mean_gap_s: float
    max_simultaneous_spacecraft: int
    coverage_fraction: float


class FleetLinkSummary(AstroModel):
    ground_site: str
    total_data_volume_mbit: float
    worst_ebn0_margin_db: float | None


class MemberLinkSummary(AstroModel):
    member_name: str
    total_data_volume_mbit: float
    worst_ebn0_margin_db: float | None


class MemberTwinResult(AstroModel):
    member_name: str
    result: DigitalTwinResult


class ConstellationTwinResult(AstroModel):
    scenario_id: str
    workflow: str = "constellation_digital_twin_v1"
    members: tuple[MemberTwinResult, ...]
    access_summaries: tuple[FleetAccessSummary, ...]
    link_summaries: tuple[FleetLinkSummary, ...]
    member_link_summaries: tuple[MemberLinkSummary, ...]
    fleet_margin_report: DesignMarginReport
    metadata: dict[str, Any]
    warnings: list[str]
```

The result should keep embedded member results as suite-owned `DigitalTwinResult`s. No backend-native
objects should appear in public outputs.

## Aggregation Rules

### Analysis Window

The v1 aggregation window is the overlap of all member geometry timelines:

- `start_s = max(first elapsed_s for each member)`
- `end_s = min(last elapsed_s for each member)`

If members have no overlapping timeline, the runner raises `InvalidScenarioError`.

This avoids overstating coverage from members that were not propagated over the same time span.

### Access Union

For each configured or otherwise represented ground site, collect all member access windows and clip
them to the common analysis window. Configured sites with no access windows remain present with zero
coverage and an analysis-window-length revisit gap. Compute a timeline of access-change events:

- `+1` when a member access starts
- `-1` when a member access ends

From that event list, compute total time with count greater than zero, longest gap, mean gap, and
maximum simultaneous count. Edge gaps from analysis-window start to first access and last access to
analysis-window end count as revisit gaps.

### Link Summary

For each configured or otherwise represented ground site, sum every member link-window data volume
clipped to the common analysis window. The worst link margin is the minimum `worst_ebn0_margin_db`
among available windows. If no link windows exist for a ground site, the worst link margin is `None`
and the fleet margin report adds a failing `fleet_link_margin_db` margin.

### Fleet Margins

The fleet margin report includes:

- `fleet_coverage_fraction_<site>` with the configured threshold from
  `coverage_requirements[*].minimum_coverage_fraction`, defaulting to `0.0`.
- `fleet_longest_gap_s_<site>` when `coverage_requirements[*].maximum_revisit_gap_s` is provided.
- `fleet_link_margin_db_<site>`.
- One `member_<name>_<limiting_margin_name>` margin for each member's limiting margin.

Limiting fleet margin uses the same normalized status-aware ranking as the single-spacecraft margin
report.

## File Boundaries

`src/astro_twin/constellation.py`
: Runner and aggregation logic: load member twin scenarios, call `run_digital_twin`, compute fleet
  summaries, and return `ConstellationTwinResult`.

`src/astro_twin/constellation_models.py`
: Constellation scenario/result models if `models.py` would become too broad.

`src/astro_twin/constellation_io.py`
: Load/write/summary functions for constellation scenarios and results. If the functions stay
  small, they may live in `astro_twin.io` instead.

`examples/twin/constellation_leo_observers.yaml`
: Public reference scenario with two members.

`examples/twin/leo_observer_plane_a.yaml` and `examples/twin/leo_observer_plane_b.yaml`
: Member twin scenarios. These may reference existing orbit scenarios or new phased LEO scenarios.

`tests/astro_twin/test_constellation_models.py`
: Scenario validation and invalid duplicate member names.

`tests/astro_twin/test_constellation_aggregation.py`
: Access-union, gap, simultaneous-count, link-sum, and fleet-margin aggregation tests using small
  synthetic windows.

`tests/astro_twin/test_constellation_runner.py`
: End-to-end runner test using the checked-in two-member example.

`tests/astro_cli/test_cli.py`
: CLI test for `astro run-constellation-twin`.

`docs/digital-twin.md` and `docs/validation-matrix.md`
: Public docs and required local gate updates.

## Error Handling

The runner should raise `InvalidScenarioError` for:

- constellation files that do not parse or validate
- duplicate member names
- duplicate coverage requirements for the same ground site
- member scenario load failures
- coverage requirements that name no ground site configured by any member scenario
- no common analysis window
- a member result with no geometry samples

The CLI should catch `InvalidScenarioError`, print the message to stderr, and exit code `2`,
matching the existing `astro` CLI conventions.

## Validation Strategy

Required local gates:

```bash
python -m pytest tests/astro_twin/test_constellation_models.py \
  tests/astro_twin/test_constellation_aggregation.py \
  tests/astro_twin/test_constellation_runner.py \
  tests/astro_cli/test_cli.py::test_run_constellation_twin_command_writes_json_and_summary -q

astro run-constellation-twin examples/twin/constellation_leo_observers.yaml \
  --output /tmp/astro-constellation-twin.json \
  --summary-output /tmp/astro-constellation-twin.txt

python -m ruff check .
python -m mypy
python -m pytest -q
git diff --check
python -m pytest tests/test_packaging.py -q
python -m build
```

The reference CLI output should include at least:

- member count
- common analysis window
- one fleet access summary
- total data volume
- maximum simultaneous visible spacecraft count
- limiting fleet margin
- warnings that v1 is deterministic design-screening evidence, not operational constellation
  coverage authority

## Acceptance Criteria

The scope is complete when:

- `astro run-constellation-twin` writes JSON and text artifacts from the checked-in reference
  scenario without optional backend dependencies.
- The result embeds member `DigitalTwinResult` objects and adds fleet access, link, and margin
  summaries.
- The fleet access summary correctly measures total visible duration, longest gap, mean gap,
  simultaneous count, and coverage fraction over the common analysis window.
- Documentation clearly states that v1 is a deterministic screening product and does not claim
  operational constellation coverage authority.
- Full local verification passes on the branch.

## Roadmap Implication

This scope makes the digital twin more novel and product-like without broadening physics claims. It
turns the single-spacecraft twin into a fleet trade-study surface, which is a better next step than
immediately deepening thermal, ADCS, or RF fidelity: fleet aggregation tests the workflow's value
while keeping assumptions explicit and locally verifiable.
