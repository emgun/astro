# Mission Knowledge Graph

Astro's mission knowledge graph is a deterministic, rebuildable read model over verified mission
artifacts. It is not a graph database, an embedding index, or a global fact store. The captured
Director, operator, and conditional-campaign bundles remain authoritative.

The first slice connects an explicitly declared mission grouping to typed run, intent, objective,
requirement, candidate, assessment, evidence, assertion, conflict, claim, decision, baseline,
verification-episode, and tool/version records. Record nodes retain the typed fields needed for the
relationship and query; the captured source bundle remains the complete authoritative record.
Edges are emitted only from explicit IDs and typed relationships; the reducer does not extract
relationships from prose.

## Checked Cross-Run Workflow

From the repository root:

```bash
astro run-mission-design-director \
  examples/design/leo_mission_design_director.yaml \
  --reasoner-replay examples/operator/leo_lifecycle_trade_study_replay.yaml \
  --output-dir build/mission-knowledge-director

astro run-mission-operator \
  examples/operator/post_launch_recovery_review.yaml \
  --reasoner-replay examples/operator/post_launch_recovery_review_replay.yaml \
  --mission-design-context build/mission-knowledge-director \
  --output-dir build/mission-knowledge-post-launch

astro run-mission-design-conditional-campaign \
  build/mission-knowledge-director \
  examples/design/leo_mission_design_conditional_campaign.yaml \
  --output-dir build/mission-knowledge-verification

astro build-mission-knowledge-graph \
  examples/knowledge/leo_mission_knowledge_graph.yaml \
  --output-dir build/mission-knowledge-graph

astro verify-mission-knowledge-graph build/mission-knowledge-graph

astro trace-mission-baseline build/mission-knowledge-graph \
  --baseline-id leo-mission-design:baseline

astro evaluate-mission-orchestration build/mission-knowledge-graph \
  --baseline-id leo-mission-design:baseline \
  --operator-objective-id post-launch-recovery-review \
  --claim-id post-launch-review-ready \
  --manual-review-gate-predicate-id manual-review-required

astro evaluate-mission-design-verification build/mission-knowledge-graph \
  --episode-id leo-mission-design-conditional-verification \
  --baseline-id leo-mission-design:baseline
```

The trace answers a bounded decision question: which selected candidate, requirements,
assessments, evidence records, and exact capability version justify the baseline? It also returns
the baseline's claim boundary and binds the answer to the graph digest.

The graph bundle is self-contained. It captures both complete source directories under
`sources/<source-id>`, inventories every byte by path, size, role, and SHA-256, and stores the
derived graph separately. Offline verification:

1. rejects extra, missing, altered, escaping, or symbolic-link artifacts;
2. invokes the native Director, operator, and conditional-campaign verifiers on the captured
   sources;
3. re-derives operational assertions and the Director decision through those verifiers;
4. reconstructs the canonical graph from the verified typed records; and
5. compares the reconstruction to the stored graph.

Refreshing the graph and outer manifest digests therefore cannot make a forged derived node valid.
The bundle proves internal consistency, not external origin authenticity; deployments that require
origin authentication must pin the bundle digest or add an external signature/transparency anchor.

## Boundaries And Next Handoff

The graph preserves conflicts instead of selecting a winner, and preserves epistemic kind, scope,
valid time, applicability, and any uncertainty fields present in source records. Current
simulation-screening observations are not labeled as operational outcomes. Capability
`qualification_sha256` values remain identities because the current Director contract does not
carry the qualification evidence bytes.

Operator schema `1.4` adds an explicit mission/baseline context: mission identity, exact Director
run digest, baseline identity/version/digest, and operational configuration. Graph schema `1.1`
emits `operates_against` only when that context resolves to exactly one selected, fully checked
Director baseline. Names or prose never create the edge, and mismatched context fails graph
construction. `--mission-design-context` verifies the freshly generated Director bundle and
resolves its exact run and baseline digests into the operator journal. This avoids treating a
platform-specific example digest as the identity of a newly generated run while retaining the
declared baseline ID, version, mission, and operational configuration.

The orchestration reducer then checks the linked operator run, its non-command authority, the
target claim, every cited assertion and predicate, configuration applicability, conflicts, and a
named exact-value manual-review gate. Its dispositions are deliberately asymmetric:

- `abstain` for missing or ambiguous bindings, operational authority, invalid checks, or failed
  applicability;
- `hold` for qualified/disputed claims, readiness conflicts, or failed readiness predicates;
- `continue` only when every check passes.

This first disposition scope is exactly `manual_review_readiness`. `continue` routes the verified
evidence package into manual review; it does not approve a maneuver, execute a command, or confer
operational authority. The decision is digest-bound to the verified graph and typed query.

Graph schema `1.2` also ingests a natively verified conditional-campaign bundle as a typed
verification episode. It binds the episode to exactly one Director run, conditional decision,
candidate, capability, and historical baseline by digest. Definition, sample, case, and statistics
digests support each gate assessment. The resulting baseline relation is
`supports_retention_of`, `requests_revision_of`, or `leaves_unresolved`; none mutates baseline
state.

The mission-design verification reducer routes one exact episode:

- `support_retention_within_declared_scope` records the campaign definition digest and claim
  boundary alongside its scoped support for retention;
- `open_new_director_decision` emits a digest-bound handoff containing the prior run, decision,
  baseline, failed requirements, capability allow-list envelope, prior analysis-cost ceiling,
  consumed design and completed verification costs, and prior authority identity;
- `abstain` covers inconclusive evidence and missing, ambiguous, or invalid bindings.

The revision handoff is not a new design decision, does not select a replacement, and exposes the
old cost and authority values only as provenance. A later Director invocation must supply fresh
inputs, authority, and budget and prove they remain inside the recorded envelope.
