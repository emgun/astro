from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from pydantic import JsonValue

from astro_operator.evidence_tools import EvidenceToolRegistry
from astro_operator.models import (
    AcquisitionStatus,
    ApplicabilityPredicate,
    ClaimDisposition,
    ConclusionClaim,
    EpistemicKind,
    EvidenceAcquisitionResult,
    EvidenceAssertion,
    EvidenceReference,
    EvidenceRequest,
    EvidenceToolSpec,
    ExactValuePredicate,
    FreshnessPredicate,
    NumericComparisonOperator,
    NumericThresholdPredicate,
)
from astro_operator.world_state import (
    assertion_digest,
    reduce_world_state,
    validate_conclusion_claims,
    world_state_digest,
)

SHA = "0" * 64


def _assertion(assertion_id: str, value: JsonValue) -> EvidenceAssertion:
    return EvidenceAssertion(
        assertion_id=assertion_id,
        subject="spacecraft-a",
        predicate="dry_mass_kg",
        value=value,
        epistemic_kind=EpistemicKind.OBSERVED,
        scope="mission-a",
        source_evidence_ids=(f"evidence-{assertion_id}",),
        producer_tool_id="mass-tool",
        producer_tool_version="1.0",
    )


def _spec() -> EvidenceToolSpec:
    return EvidenceToolSpec(
        tool_id="mass-tool",
        version="1.0",
        request_kind="measure_mass",
        parameter_schema_sha256=SHA,
        output_assertion_kinds=("dry_mass_kg",),
    )


def _request(**updates: str) -> EvidenceRequest:
    values: dict[str, Any] = {
        "request_id": "request-1",
        "tool_id": "mass-tool",
        "tool_version": "1.0",
        "request_kind": "measure_mass",
        "parameters": {},
    }
    values.update(updates)
    return EvidenceRequest(**values)


@dataclass
class _Tool:
    spec: EvidenceToolSpec
    calls: int = 0

    def acquire(
        self, request: EvidenceRequest, world_state: object
    ) -> EvidenceAcquisitionResult:
        del world_state
        self.calls += 1
        evidence = EvidenceReference(
            evidence_id="evidence-a",
            kind="measurement",
            epistemic_kind=EpistemicKind.OBSERVED,
            claim_scope="mission-a",
            path="artifact.json",
            sha256=SHA,
        )
        return EvidenceAcquisitionResult(
            request=request,
            tool=self.spec,
            status=AcquisitionStatus.SUCCEEDED,
            evidence=(evidence,),
            assertions=(_assertion("a", 100.0),),
        )


def test_registry_dispatches_only_exact_tool_contract() -> None:
    tool = _Tool(_spec())
    registry = EvidenceToolRegistry((tool,))
    state = reduce_world_state(())

    result = registry.acquire(_request(), state)

    assert result.tool == tool.spec
    assert tool.calls == 1
    with pytest.raises(ValueError, match="version mismatch"):
        registry.acquire(_request(tool_version="2.0"), state)
    with pytest.raises(ValueError, match="request kind"):
        registry.acquire(_request(request_kind="estimate_mass"), state)
    with pytest.raises(ValueError, match="not registered"):
        registry.acquire(_request(tool_id="missing"), state)
    assert tool.calls == 1


def test_registry_rejects_duplicate_registration() -> None:
    tool = _Tool(_spec())
    with pytest.raises(ValueError, match="already registered"):
        EvidenceToolRegistry((tool, tool))


def test_reducer_and_digests_are_deterministic_and_preserve_conflicts() -> None:
    first = _assertion("a", 100.0)
    second = _assertion("b", 101.0)

    forward = reduce_world_state((first, second))
    reverse = reduce_world_state((second, first))

    assert forward == reverse
    assert tuple(item.assertion_id for item in forward.assertions) == ("a", "b")
    assert len(forward.conflicts) == 1
    assert forward.conflicts[0].assertion_ids == ("a", "b")
    assert all(item.assertion_sha256 == assertion_digest(item) for item in forward.assertions)
    assert forward.state_sha256 == world_state_digest(forward)


def test_reducer_rejects_tampered_assertion_digest() -> None:
    assertion = _assertion("a", 100.0).model_copy(update={"assertion_sha256": SHA})

    with pytest.raises(ValueError, match="digest does not match"):
        reduce_world_state((assertion,))


def test_claim_validation_rejects_unknown_tampered_and_supported_conflicts() -> None:
    state = reduce_world_state((_assertion("a", 100.0), _assertion("b", 101.0)))
    qualified = ConclusionClaim(
        claim_id="qualified",
        statement="Mass measurements disagree.",
        conclusion_sha256=SHA,
        disposition=ClaimDisposition.QUALIFIED,
        assertion_ids=("a", "b"),
        qualification="Two valid measurements conflict.",
    )
    validate_conclusion_claims((qualified,), state)

    supported = ConclusionClaim(
        claim_id="supported",
        statement="Mass is exactly 100 kg.",
        conclusion_sha256=SHA,
        disposition=ClaimDisposition.SUPPORTED,
        assertion_ids=("a",),
    )
    with pytest.raises(ValueError, match="unresolved conflicting"):
        validate_conclusion_claims((supported,), state)

    unknown = qualified.model_copy(update={"assertion_ids": ("missing",)})
    with pytest.raises(ValueError, match="unknown assertions"):
        validate_conclusion_claims((unknown,), state)

    tampered = state.model_copy(update={"state_sha256": SHA})
    with pytest.raises(ValueError, match="state digest"):
        validate_conclusion_claims((qualified,), tampered)


def test_supported_claim_accepts_non_conflicting_assertion() -> None:
    state = reduce_world_state((_assertion("a", 100.0),))
    claim = ConclusionClaim(
        claim_id="supported",
        statement="The observed mass is 100 kg.",
        conclusion_sha256=SHA,
        disposition=ClaimDisposition.SUPPORTED,
        assertion_ids=("a",),
    )

    validate_conclusion_claims((claim,), state)


def test_supported_claim_recomputes_threshold_freshness_and_applicability() -> None:
    as_of = datetime(2026, 1, 1, 0, 30, 30, tzinfo=UTC)
    assertions = (
        _assertion("sigma", 0.042).model_copy(
            update={"predicate": "position_sigma_km", "valid_at": as_of}
        ),
        _assertion("limit", 0.1).model_copy(
            update={
                "predicate": "maximum_position_sigma_km",
                "subject": "spacecraft-a",
                "scope": "procedure-v1",
            }
        ),
        _assertion("age", 60.0).model_copy(
            update={"predicate": "maximum_age_s"}
        ),
        _assertion("actual-config", "baseline-v1").model_copy(
            update={
                "predicate": "asset_configuration_id",
                "valid_at": as_of - timedelta(seconds=30),
            }
        ),
        _assertion("required-config", "baseline-v1").model_copy(
            update={
                "predicate": "applicable_configuration_id",
                "subject": "spacecraft-a",
                "scope": "procedure-v1",
            }
        ),
        _assertion("manual", True).model_copy(
            update={"predicate": "manual_review_required"}
        ),
        _assertion("decision", as_of.isoformat()).model_copy(
            update={"predicate": "simulated_decision_time", "valid_at": as_of}
        ),
    )
    state = reduce_world_state(assertions)
    claim = ConclusionClaim(
        claim_id="checked",
        statement="The evidence satisfies the procedure.",
        conclusion_sha256=SHA,
        disposition=ClaimDisposition.SUPPORTED,
        assertion_ids=tuple(item.assertion_id for item in assertions),
        predicates=(
            NumericThresholdPredicate(
                predicate_id="threshold",
                assertion_id="sigma",
                assertion_predicate="position_sigma_km",
                operator=NumericComparisonOperator.LESS_THAN_OR_EQUAL,
                threshold_assertion_id="limit",
                threshold_assertion_predicate="maximum_position_sigma_km",
            ),
            FreshnessPredicate(
                predicate_id="fresh",
                assertion_id="actual-config",
                assertion_predicate="asset_configuration_id",
                reference_assertion_id="decision",
                reference_assertion_predicate="simulated_decision_time",
                max_age_assertion_id="age",
            ),
            ApplicabilityPredicate(
                predicate_id="applicable",
                actual_assertion_id="actual-config",
                actual_assertion_predicate="asset_configuration_id",
                required_assertion_id="required-config",
                required_assertion_predicate="applicable_configuration_id",
                expected_subject="spacecraft-a",
                expected_scope="procedure-v1",
            ),
            ExactValuePredicate(
                predicate_id="manual",
                assertion_id="manual",
                assertion_predicate="manual_review_required",
                expected_value=True,
            ),
        ),
    )

    validate_conclusion_claims((claim,), state)

    stale_state = reduce_world_state(
        tuple(
            item.model_copy(
                update={"value": (as_of + timedelta(seconds=31)).isoformat()}
            )
            if item.assertion_id == "decision"
            else item
            for item in assertions
        )
    )
    with pytest.raises(ValueError, match="unsatisfied freshness"):
        validate_conclusion_claims((claim,), stale_state)

    mismatched = reduce_world_state(
        tuple(
            item.model_copy(update={"value": "other-baseline"})
            if item.assertion_id == "actual-config"
            else item
            for item in assertions
        )
    )
    with pytest.raises(ValueError, match="unsatisfied applicability"):
        validate_conclusion_claims((claim,), mismatched)


def test_numeric_and_exact_predicates_reject_bool_number_equivalence() -> None:
    state = reduce_world_state(
        (
            _assertion("bool", True).model_copy(
                update={"predicate": "manual_review_required"}
            ),
        )
    )
    numeric = ConclusionClaim(
        claim_id="numeric",
        statement="Boolean is not a number.",
        conclusion_sha256=SHA,
        disposition=ClaimDisposition.SUPPORTED,
        assertion_ids=("bool",),
        predicates=(
            NumericThresholdPredicate(
                predicate_id="numeric",
                assertion_id="bool",
                assertion_predicate="manual_review_required",
                operator=NumericComparisonOperator.EQUAL,
                threshold_value=1.0,
            ),
        ),
    )
    with pytest.raises(ValueError, match="not numeric"):
        validate_conclusion_claims((numeric,), state)

    exact = numeric.model_copy(
        update={
            "claim_id": "exact",
            "predicates": (
                ExactValuePredicate(
                    predicate_id="exact",
                    assertion_id="bool",
                    assertion_predicate="manual_review_required",
                    expected_value=1,
                ),
            ),
        }
    )
    with pytest.raises(ValueError, match="unsatisfied exact_value"):
        validate_conclusion_claims((exact,), state)
