# Verifiable AI Workflow Interfaces For Space Operations

Date: 2026-06-20

## Purpose

Capture the current brainstorm around an AI/NLP interface for Astro Suite, then ground it in the
space-domain literature and operational signals around verifiable, deterministic, and assured AI.

## Working Thesis

The strongest public-worthy direction is not "LLM astrodynamics." It is:

```text
natural language intent
-> typed, reviewable workflow plan
-> constrained space-domain tools
-> deterministic execution
-> artifact validation
-> replayable audit trace
```

The LLM translates intent and explains results. Flight-dynamics, orbit-determination, launch,
mission-planning, and anomaly-analysis tools remain the authority for computation and validation.

## Notes From Prior Brainstorming

- The public showcase should focus on one credible workflow first rather than presenting the whole
  suite as equally production-ready.
- The healthiest Astro Suite surface for a showcase is orbit determination: scenario validation,
  synthetic measurement generation, JSON/CSV/TDM export, measurement ingest, least-squares estimate,
  residual/rank/convergence diagnostics, and provenance.
- Launch-to-orbit handoff is intuitive and visually compelling, but current local launch is a
  deterministic baseline, not a production launch simulator. It should be secondary until its scope
  is framed precisely.
- The backend adapter story is architecturally strong: suite-owned schemas and products sit in front
  of optional Orekit, Tudat, RocketPy, Dymos, and JAX backends. It is a differentiator, but optional
  runtime setup creates public-demo friction.
- Combining an intent-to-workflow assistant with an MCP/tool server makes sense if both share one
  workflow-planning core. The CLI is the approachable showcase; MCP is the agent-native integration
  surface.
- The reusable unit should be a controlled workflow pack, not arbitrary agent execution:

```text
workflow-pack/
  manifest.yaml
  schemas.py/json
  tools.py
  validators.py
  examples/
  policies.yaml
```

- Generalization should happen through a common workflow kernel plus domain packs. Verifiability
  should come from typed plans, policy gates, deterministic adapters, artifact validators, and audit
  traces.
- The best one-line positioning is: "AI translates intent. Deterministic workflows compute.
  Validators decide what is true."

## General Verifiable AI Patterns Already Researched

- **Structured generation:** JSON Schema, strict tool schemas, constrained decoding, and typed
  outputs reduce malformed model output but do not prove intent correctness.
- **Schema-gated orchestration:** models can converse freely, but execution authority belongs to
  schema-valid plans and tool calls.
- **Cognitive/executive separation:** the model proposes; deterministic policy, approval, and
  execution layers decide what can run.
- **Durable workflow graphs:** checkpoints, replay, event logs, and artifact lineage make agent
  behavior inspectable and comparable.
- **Formal or probabilistic verification:** the highest assurance layer is a mathematical model,
  temporal-logic property, model checker, optimizer constraint, or proof assistant that can reject
  unsafe plans.
- **Compiled AI:** use the model to generate a bounded artifact once, then validate and execute
  normal software thereafter.

## Space-Domain Landscape

### 1. Space autonomy already has an assurance culture

JPL's autonomy-assurance literature separates automation from autonomy and stresses that autonomy
must be trusted to keep expensive mission assets safe while pursuing mission objectives. The key
lesson for Astro is that mission assurance is not a bolt-on UI feature. It is the product boundary.

JPL's DS1 Remote Agent Experiment is still a useful design precedent. It combined onboard planning,
closed-loop execution, and model-based fault diagnosis, then earned trust through integration,
layered testing, formal validation objectives, spacecraft testbeds, and operational readiness tests.
The lesson is directly relevant: a planner that can synthesize new actions needs a stronger evidence
trail than a static script.

JPL's later formal-methods work reinforces the same direction: scenario-based testing is necessary
but insufficient, while model checking, static analysis, logical domain specifications, flight-rule
checks, code review, and safety analysis form a layered assurance stack. The useful pattern is not
"formal methods everywhere"; it is selective formalism where plan/action spaces are consequential.

### 2. The credible current frontier is AI-assisted operations, not unchecked autonomy

ESA's A2I Roadmap is the closest operational mirror to the proposed Astro interface. ESOC frames AI
around mission operations domains: health monitoring, decision recommendation and planning,
simulation, user interaction with data, and data governance. ESA already reports:

- OCAI, an operations companion for query, retrieval, correlation, and analysis across heterogeneous
  mission systems.
- A short-term telemetry forecasting tool operationalized on one ESA mission.
- An LLM-powered root-cause anomaly assistant operationalized on two ESA missions.
- A public satellite telemetry anomaly dataset to benchmark anomaly-detection models.

This is important because ESA is not framing AI as free-form spacecraft control. It is putting AI
inside operational assistance, data access, anomaly triage, forecasting, and benchmarked model
evaluation.

DLR/GSOC's LLM evaluation for space operations is another strong cautionary source. Their use case
was local, non-cloud LLM support for information retrieval and operations documentation because
mission-control data is sensitive. Results were promising, but they explicitly called out
hallucination, below-50% answer accuracy in one study, and poor comparability/repeatability as
barriers for operations use. This supports a retrieval-plus-citation assistant only when it is paired
with source grounding and validation.

### 3. Real flight demonstrations are moving toward validated AI-generated plans

NASA/JPL's 2026 Perseverance report is the most relevant current example of generative AI touching
real operations. JPL used vision-language models to generate Mars rover waypoints from the same
orbital imagery and terrain data human planners use. The key assurance detail is that commands were
then processed through JPL's digital twin, verifying more than 500,000 telemetry variables before
commands were sent to Mars.

This is a clean pattern for Astro:

```text
LLM/VLM proposes path or workflow
-> domain simulator/digital twin checks compatibility
-> validated command/product emitted
-> operational system executes
```

### 4. Benchmarks and open datasets are becoming the credibility layer

ESA's anomaly dataset and ESA-ADB benchmark are important because they move the field from anecdotes
to measurable baselines. The benchmark includes years of telemetry from three spacecraft, hundreds
of channels/control signals, annotated events, operational requirements, and evaluation metrics.
The reported conclusion is sober: common anomaly-detection algorithms are not yet suitable for
effective deployment under the benchmark.

For Astro, this argues for golden prompts, golden workflow plans, reference scenario artifacts,
backend-comparison tolerances, and regression gates. A natural-language interface should be evaluated
like any other mission tool: with scenarios, expected plans, artifact checks, and failure modes.

### 5. New space-AI research is translating intent into formal optimization

Two recent lines matter for this repo:

- **Verifiable mission planning for space operations:** finite-horizon MDPs optimize mission reward
  while enforcing probabilistic safety constraints. The GRACE-FO case study treats uncertainty from
  environment and dynamics explicitly.
- **Semantic constraint synthesis for trajectory optimization:** LLMs translate natural-language
  mission requirements into executable trajectory-optimization code and mathematical formulations
  for rendezvous scenarios.

These match the desired Astro pattern: let language specify intent and constraints, but compile it
into a formal planning or optimization representation that deterministic tooling can solve and
inspect.

### 6. Pure LLM spacecraft operators are interesting but not the assurance path

The KSP Differential Games work shows LLM agents can compete in simulated spacecraft-control tasks.
It is useful as evidence that LLMs can participate in space-control research loops. It is not the
right public north star for Astro because direct LLM action selection is harder to verify than
LLM-to-plan-to-validator execution.

## Implications For Astro Suite

### Product Architecture

Recommended split:

```text
assistant_core/
  planner interfaces
  workflow plan schema
  policy/risk classification
  dry-run and approval model
  execution trace model
  provider abstraction

astro_workflow_pack/
  allowed Astro tools
  AstroWorkflowPlan schema
  scenario/resource resolvers
  artifact validators
  golden prompts
  reference products

interfaces/
  astro plan / astro ask CLI
  MCP server exposing the same tools/resources
```

### First Workflow Pack

Start with `astro_od_workflow`:

```text
validate scenario
-> synthesize measurements
-> export measurements
-> estimate initial state
-> inspect residuals/rank/RMS/convergence
-> emit report and trace
```

This path is already aligned with Astro's current validation surfaces and avoids optional backend
friction.

### Verifiability Requirements

- Plans are Pydantic/JSON-schema models, not free text.
- Every step names its tool, inputs, outputs, risk level, expected artifact type, and validator.
- Execution defaults to dry-run.
- Side effects, optional backends, and non-local paths require approval.
- Tool registry is allow-listed. No arbitrary shell execution belongs in the first public slice.
- Artifact validators check product schema, backend provenance, expected counts/ranges, convergence
  metadata, and file existence.
- Each run writes an audit trace with prompt, model/provider metadata, parsed plan, accepted plan,
  tool calls, outputs, validation status, and warnings.
- Tests use deterministic/fake LLM adapters plus golden prompt-plan pairs.
- Real-provider tests are opt-in and never required for default CI.

### Space-Specific Safety Boundaries

- Do not claim operational flight readiness.
- Do not let the model select arbitrary maneuvers outside a bounded operation schema.
- Do not let generated plans bypass scenario validation or product validators.
- Distinguish recommendation, simulation, and command generation.
- Treat optional backends as explicit capability gates with structured unavailable diagnostics.
- Require model output to cite source data when it summarizes procedures, telemetry, or mission docs.

## Recommended Public Framing

Use the space-domain language:

> Astro Suite explores verifiable AI-assisted mission workflows: natural-language intent is compiled
> into typed flight-dynamics and orbit-determination plans, executed by deterministic backends, and
> checked through reproducible artifact validators.

Avoid:

- "autonomous spacecraft operator"
- "LLM flight dynamics"
- "AI mission control"
- "production-grade launch optimization"

Prefer:

- "AI-assisted workflow planning"
- "validated mission-analysis artifacts"
- "typed, auditable execution"
- "deterministic backend products"
- "replayable operation traces"

## Source Notes

- NASA/JPL Perseverance AI-planned drive:
  https://www.jpl.nasa.gov/news/nasas-perseverance-rover-completes-first-ai-planned-drive-on-mars/
- NASA Perseverance AI-planned drive mirror:
  https://www.nasa.gov/missions/mars-2020-perseverance/perseverance-rover/nasas-perseverance-rover-completes-first-ai-planned-drive-on-mars/
- JPL visualization of Perseverance AI-planned drive:
  https://www.jpl.nasa.gov/images/pia26646-visualizing-perseverances-ai-planned-drive-on-mars/
- ESA A2I Roadmap:
  https://esoc.esa.int/a2i-roadmap-0
- ESA anomaly dataset release:
  https://esoc.esa.int/esa-releases-building-block-open-database-satellite-anomalies
- ESA-ADB benchmark:
  https://openreview.net/forum?id=FYEGPuUrpo
- DLR/GSOC LLMs for space operations:
  https://elib.dlr.de/210113/1/evaluating_large_language_models_for_space_operations.pdf
- JPL assurance for autonomy:
  https://arxiv.org/abs/2305.11902
- JPL formal methods for trusted space autonomy:
  https://ai.jpl.nasa.gov/public/documents/papers/chien-nfm-2022.pdf
- DS1 Remote Agent validation:
  https://ai.jpl.nasa.gov/public/documents/papers/rax-results-isairas99.pdf
- JPL Autonomous Sciencecraft Experiment:
  https://ai.jpl.nasa.gov/public/projects/ase/
- JPL FAME:
  https://ai.jpl.nasa.gov/public/projects/fame/
- Verifiable mission planning for space operations:
  https://arxiv.org/abs/2504.11631
- Semantic constraint synthesis for adaptive trajectory optimization via LLMs:
  https://arxiv.org/abs/2606.04123
- LLMs as autonomous spacecraft operators in Kerbal Space Program:
  https://arxiv.org/abs/2505.19896
- NASA AI-enabled autonomous systems presentation:
  https://ntrs.nasa.gov/api/citations/20240002420/downloads/IAPG_2024_Final.pdf
