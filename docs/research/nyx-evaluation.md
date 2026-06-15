# Nyx And ANISE Evaluation Gate

Date: 2026-06-15

## Decision

Defer a production Nyx adapter. Keep Nyx and ANISE in evaluation-only status until the suite has a
clear Rust/Python integration path, a licensing decision, and reference-case parity against the
existing local and Orekit product surfaces.

## Current Read

Nyx Space presents Nyx as an open-source flight dynamics toolkit for mission design, orbit
determination, propagation, Monte Carlo, and spacecraft navigation. ANISE is positioned as a modern
SPICE-like toolkit for attitude, navigation, instrument, spacecraft, and ephemeris computations, with
Python and Rust installation paths.

References:

- Nyx project: https://github.com/nyx-space/nyx
- Nyx Space overview: https://nyxspace.com/
- ANISE docs: https://nyxspace.com/anise/
- ANISE repository: https://github.com/nyx-space/anise
- Rust API docs: https://docs.rs/nyx-space/

## Why Not Add It As A Dependency Now

- Product boundary risk: Astro Suite currently standardizes on `Scenario`, `Trajectory`,
  `EstimateResult`, `LaunchTrajectory`, and `MonteCarloResult`. A Nyx adapter should return those
  products, not leak Rust-side objects or a parallel scenario model.
- Licensing risk: the Nyx repository describes the core as AGPLv3. That may be incompatible with a
  permissively licensed Python package unless the adapter is optional, process-isolated, or covered by
  a clear legal decision.
- Integration maturity risk: the current operational backbone is Python with optional mature engines
  behind adapters. A Rust-first backend needs packaging, data, error handling, and reproducibility
  gates before it belongs in the default roadmap.
- Redundancy risk: Orekit is already the primary operational backend, Tudat is the cross-check
  candidate, and JAX is the differentiable research path. Nyx must prove a distinct advantage.

## Promotion Criteria

Promote Nyx from evaluation-only to adapter-planning only if all are true:

- Licensing: legal/project decision confirms whether AGPLv3 is acceptable, whether only ANISE MPL-2.0
  functionality is in scope, or whether the adapter must be process-isolated.
- Packaging: repeatable install path exists for macOS and Linux CI without manual Rust toolchain
  state leaking into normal Python installs.
- Product mapping: a prototype converts a controlled LEO scenario into a suite `Trajectory` with
  backend provenance and no Nyx-native objects in public outputs.
- Validation: final state agrees with local two-body or Orekit two-body within an explicit tolerance
  on `examples/scenarios/leo_two_body.yaml`.
- Differentiator: Nyx proves at least one meaningful advantage over Orekit/Tudat/JAX for this suite,
  such as faster OD/Monte Carlo, stronger covariance tooling, or better SPICE/ANISE geometry
  integration.

## Minimal Spike

If revisited, run a contained spike with these outputs:

1. Install notes for the selected Nyx or ANISE package path.
2. A one-file prototype outside `src/` that loads or constructs the LEO reference case.
3. A JSON comparison of final position/velocity against local and Orekit reference products.
4. A license note describing whether the prototype can be promoted into package code.
5. A yes/no recommendation.

## Current Recommendation

Do not implement a production Nyx adapter in the current roadmap pass. Keep the research document as
the evaluation artifact, continue with Tudat/JAX adapter boundaries, and revisit Nyx only after the
operational backend matrix has live Tudat or JAX evidence.
