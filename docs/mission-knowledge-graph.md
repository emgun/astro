# Mission Knowledge Graph

Astro's mission knowledge graph is a deterministic, rebuildable read model over verified mission
run artifacts. It is not a graph database, an embedding index, or a global fact store. The
captured Director and operator bundles remain authoritative.

The first slice connects an explicitly declared mission grouping to typed run, intent, objective,
requirement, candidate, assessment, evidence, assertion, conflict, claim, decision, baseline, and
tool/version records. Record nodes retain the typed fields needed for the relationship and query;
the captured source bundle remains the complete authoritative record. Edges are emitted only from
explicit IDs and typed relationships; the reducer does not extract relationships from prose.

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
  --output-dir build/mission-knowledge-post-launch

astro build-mission-knowledge-graph \
  examples/knowledge/leo_mission_knowledge_graph.yaml \
  --output-dir build/mission-knowledge-graph

astro verify-mission-knowledge-graph build/mission-knowledge-graph

astro trace-mission-baseline build/mission-knowledge-graph \
  --baseline-id leo-mission-design:baseline
```

The trace answers a bounded decision question: which selected candidate, requirements,
assessments, evidence records, and exact capability version justify the baseline? It also returns
the baseline's claim boundary and binds the answer to the graph digest.

The graph bundle is self-contained. It captures both complete source directories under
`sources/<source-id>`, inventories every byte by path, size, role, and SHA-256, and stores the
derived graph separately. Offline verification:

1. rejects extra, missing, altered, escaping, or symbolic-link artifacts;
2. invokes the native Director and operator verifiers on the captured sources;
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

The post-launch review and design baseline intentionally remain separate episodes under the
declared mission node. The current operator contract has no typed baseline ID or digest, so the
graph does not infer a handoff from phrases such as “baseline v1.” The next architecture slice
should add a versioned mission/baseline context to operator inputs, resolve it against exactly one
verified baseline, and then reduce checked claim predicates into typed `continue`, `hold`, or
`abstain` planning branches.
