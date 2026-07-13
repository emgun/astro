# Uncertainty Campaign Baseline

**Protocol:** `uncertainty-campaign-baseline-v1`
**Manifest:** `examples/campaigns/benchmark-manifest.yaml`
**Claim:** Local, machine-scoped engineering benchmark evidence. These measurements do not establish operational performance, flight certification, or fidelity outside the recorded configurations.

## Decision

The first surrogate candidate is a bounded residual correction from local or JAX orbit propagation to an explicitly configured high-fidelity in-orbit teacher. It is the expected best fit for a stable input/output contract and authoritative replay.

The challenger is the coupled digital-twin evaluator, limited to declared subsystem margin and event outputs. It may be more expensive in campaigns, but its heterogeneous outputs and discontinuities make it a harder surrogate target.

Do not start acceleration work unless profiling shows that one candidate evaluator dominates end-to-end campaign cost and simpler vectorization, batching, caching, or analytical changes cannot remove the bottleneck more safely. Kill surrogate acceleration work if neither candidate clears that gate, or if bounded-domain, OOD, fallback, and authoritative-replay policies cannot be made deterministic.

## Representative Workflows

The manifest fixes one checked example for each required workflow:

| Workflow | Checked example | Expected primary artifact |
| --- | --- | --- |
| Orbit | `examples/scenarios/leo_j2.yaml` | `orbit.json` |
| Digital twin | `examples/twin/leo_observer.yaml` | `digital-twin.json` |
| Constellation | `examples/twin/constellation_leo_observers.yaml` | `constellation.json` |
| Reentry | `examples/reentry/guided_lifting_body.yaml` | `reentry.json` |
| Mission lifecycle | `examples/lifecycle/leo_round_trip.yaml` | `lifecycle.json` plus phase artifacts |

Commands are argument arrays rather than shell strings. Replace `{output_dir}` with a fresh absolute directory for each repetition so artifact writes and filesystem state are isolated.

## Measurement Protocol

1. Record UTC timestamp, hostname, operating system, architecture, CPU, logical CPU count, physical memory, Python and Astro versions, Git commit and dirty state, and relevant dependency versions.
2. Use a quiet machine with fixed power settings. Record any unavailable metadata rather than inferring it.
3. Run one unmeasured warm-up for every entry, then five measured repetitions in manifest order. Use a fresh output directory for every run.
4. Measure repeated wall time around the full command and peak RSS at process scope. Record evaluator, serialization, and metric-extraction time from workflow instrumentation; use `not_instrumented` rather than deriving those values from wall time.
5. Sum bytes for the expected primary artifact and all declared companion artifacts after each successful run. Keep failures in the record with status, exit code, and stderr summary.
6. Report median, minimum, maximum, and median absolute deviation for timings; report every raw repetition in a machine-local result artifact. Do not commit generated bulk artifacts.

The warm-up count, repetitions, required fields, commands, artifacts, and workflow claim boundaries are schema-tested in `tests/benchmarks/test_campaign_benchmark_manifest.py`.

## Baseline Summary

The initial local run completed on 2026-07-12 with one warm-up and five measured repetitions per workflow. The machine was an Apple M4 MacBook Air with 10 logical cores and 16 GB memory, running macOS 26.0.1 (Darwin 25.0.0), Python 3.12.7, Astro 0.1.0, NumPy 1.26.4, SciPy 1.13.1, and Pydantic 2.11.5 at Git revision `43ddfb03b1f4`. The worktree was dirty because the benchmark protocol and concurrent documentation work were uncommitted.

Wall times include process startup and full command execution. Peak RSS was captured with macOS `/usr/bin/time -l`; artifact bytes are the sum of files written under each isolated output directory. Component timings remain `not_instrumented` until the campaign evaluator exposes phase timings; that absence is evidence against making a surrogate decision from end-to-end wall time alone.

| Workflow | Wall time, median (s) | Peak RSS (bytes) | Artifact bytes | Evaluator / serialization / extraction | Result |
| --- | ---: | ---: | ---: | --- | --- |
| Orbit | 0.6075 | 94,519,296 | 5,096 | `not_instrumented` | pass |
| Digital twin | 0.6222 | 94,437,376 | 15,136 | `not_instrumented` | pass |
| Constellation | 0.6190 | 94,601,216 | 39,814 | `not_instrumented` | pass |
| Reentry | 0.6281 | 95,666,176 | 87,100 | `not_instrumented` | pass |
| Mission lifecycle | 0.6681 | 98,107,392 | 871,367 | `not_instrumented` | pass |

Wall-time ranges and median absolute deviations were: orbit 0.5968-0.6424 s (MAD 0.0107 s), digital twin 0.6038-0.6638 s (MAD 0.0071 s), constellation 0.6002-0.6425 s (MAD 0.0045 s), reentry 0.5920-0.6983 s (MAD 0.0361 s), and mission lifecycle 0.6461-0.7046 s (MAD 0.0125 s).

No workflow currently demonstrates an evaluator-dominated campaign cost. The first candidate and challenger remain preregistered, but the present decision is **do not begin surrogate acceleration work**. Add evaluator phase timing first, then test vectorization, batching, caching, and analytical simplification before reopening the gate.

Optional Orekit and Tudat teacher runs are separate machine-scoped evidence. Their results apply only to the recorded backend version, model configuration, input domain, machine, and runtime. A local-teacher result must not be described as Orekit or Tudat fidelity.

## Interpretation

Choose the residual orbit candidate only if its authoritative teacher evaluation dominates campaign cost after simpler fixes and the teacher/baseline force models, horizon, regime, parameters, features, targets, and units can be frozen. Choose the digital-twin challenger only if its bounded outputs dominate cost and requirement-boundary behavior can be validated.

Otherwise, stop. The uncertainty campaign product remains valuable without a surrogate, and replacing cheap two-body or J2 propagation with a learned model is not an objective.

## Campaign Timing Gate Refresh

On 2026-07-13, the campaign boundary gained explicit metric-extraction timing and the derived
`astro profile-campaign` product. Profiling remains separate from deterministic campaign statistics
and is explicitly machine-scoped. Legacy cases without extraction timing remain marked as
incompletely instrumented rather than being interpreted as zero-time measurements.

The gate was preregistered in `examples/campaigns/benchmark-manifest.yaml`: one warm-up, at least
five measured cases, complete phase timing, evaluation share of instrumented time of at least 0.80, and median evaluator
time of at least 0.050 seconds. Clearing those timing thresholds would still require a follow-up
vectorization, batching, caching, and analytical challenger before surrogate work could proceed.

The checked orbit candidate and digital-twin challenger each ran one warm-up followed by seven
measured Latin-hypercube cases on `Mac.lan`, Darwin 25.0.0 arm64, Python 3.12.7, from Git revision
`48c011d` with the timing implementation uncommitted. Generated case and profile artifacts remained
under `/tmp` and are not committed.

| Campaign | Complete cases | Evaluation median (s) | Evaluation MAD (s) | Evaluation share of instrumented time | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Local J2 orbit | 7/7 | 0.000553 | 0.000004 | 0.9737 | fail: absolute cost |
| Integrated digital twin | 7/7 | 0.001431 | 0.000070 | 0.9918 | fail: absolute cost |

Both evaluators dominate their measured instrumented phase time, but both are more than an order of magnitude
below the 0.050-second absolute-cost floor. The decision remains **do not begin surrogate
acceleration work** for these local evaluators. The next legitimate reopening condition is a
measured optional teacher or future high-fidelity evaluator with materially larger absolute cost,
not a learned replacement for the current inexpensive local physics.
