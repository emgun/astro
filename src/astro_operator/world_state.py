"""Pure, deterministic reduction and validation for evidence-backed world state."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime
from hashlib import sha256

from astro_operator.models import (
    ApplicabilityPredicate,
    AssertionConflict,
    ClaimDisposition,
    ConclusionClaim,
    EvidenceAssertion,
    ExactValuePredicate,
    FreshnessPredicate,
    NumericComparisonOperator,
    NumericThresholdPredicate,
    WorldState,
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def assertion_digest(assertion: EvidenceAssertion) -> str:
    """Bind an assertion to all semantic and provenance fields except its digest."""

    payload = assertion.model_dump(mode="json", exclude={"assertion_sha256"})
    return sha256(_canonical_json(payload)).hexdigest()


def _validated_assertion(assertion: EvidenceAssertion) -> EvidenceAssertion:
    digest = assertion_digest(assertion)
    if assertion.assertion_sha256 is not None and assertion.assertion_sha256 != digest:
        raise ValueError(f"assertion {assertion.assertion_id!r} digest does not match its content")
    if assertion.assertion_sha256 == digest:
        return assertion
    return assertion.model_copy(update={"assertion_sha256": digest})


def _valid_at_key(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _conflict_id(
    key: tuple[str, str, str, datetime | None], assertion_ids: tuple[str, ...]
) -> str:
    subject, predicate, scope, valid_at = key
    return sha256(
        _canonical_json(
            {
                "subject": subject,
                "predicate": predicate,
                "scope": scope,
                "valid_at": _valid_at_key(valid_at),
                "assertion_ids": assertion_ids,
            }
        )
    ).hexdigest()


def reduce_world_state(assertions: Iterable[EvidenceAssertion]) -> WorldState:
    """Reduce assertions without discarding contradictory evidence.

    Input order does not affect the resulting assertion order, conflicts, or state digest.
    Assertion IDs are globally unique; duplicates fail even when their content is equivalent.
    """

    by_id: dict[str, EvidenceAssertion] = {}
    for candidate in assertions:
        assertion = _validated_assertion(candidate)
        existing = by_id.get(assertion.assertion_id)
        if existing is not None:
            raise ValueError(f"assertion ID {assertion.assertion_id!r} is duplicated")
        by_id[assertion.assertion_id] = assertion

    ordered = tuple(by_id[assertion_id] for assertion_id in sorted(by_id))
    groups: dict[
        tuple[str, str, str, datetime | None], list[EvidenceAssertion]
    ] = defaultdict(list)
    for assertion in ordered:
        key = (assertion.subject, assertion.predicate, assertion.scope, assertion.valid_at)
        groups[key].append(assertion)

    conflicts: list[AssertionConflict] = []
    for key, group in groups.items():
        distinct_values = {_canonical_json(item.value) for item in group}
        if len(distinct_values) < 2:
            continue
        assertion_ids = tuple(sorted(item.assertion_id for item in group))
        conflicts.append(
            AssertionConflict(
                conflict_id=_conflict_id(key, assertion_ids),
                subject=key[0],
                predicate=key[1],
                scope=key[2],
                valid_at=key[3],
                assertion_ids=assertion_ids,
            )
        )
    ordered_conflicts = tuple(sorted(conflicts, key=lambda item: item.conflict_id))
    digest = _world_state_content_digest(ordered, ordered_conflicts)
    return WorldState(assertions=ordered, conflicts=ordered_conflicts, state_sha256=digest)


def _world_state_content_digest(
    assertions: tuple[EvidenceAssertion, ...], conflicts: tuple[AssertionConflict, ...]
) -> str:
    return sha256(
        _canonical_json(
            {
                "assertions": [item.model_dump(mode="json") for item in assertions],
                "conflicts": [item.model_dump(mode="json") for item in conflicts],
            }
        )
    ).hexdigest()


def world_state_digest(state: WorldState) -> str:
    """Return the digest of state content, independently of the stored digest field."""

    return _world_state_content_digest(state.assertions, state.conflicts)


def validate_conclusion_claims(
    claims: Iterable[ConclusionClaim], state: WorldState
) -> None:
    """Fail closed unless every claim cites intact state evidence consistently."""

    _validate_state_integrity(state)
    assertions = {item.assertion_id: item for item in state.assertions}
    conflicted_ids = {
        assertion_id for conflict in state.conflicts for assertion_id in conflict.assertion_ids
    }
    seen_claim_ids: set[str] = set()
    for claim in claims:
        if claim.claim_id in seen_claim_ids:
            raise ValueError(f"conclusion claim ID {claim.claim_id!r} is duplicated")
        seen_claim_ids.add(claim.claim_id)
        unknown = set(claim.assertion_ids) - assertions.keys()
        if unknown:
            raise ValueError(
                "conclusion claim cites unknown assertions: " + ", ".join(sorted(unknown))
            )
        unresolved = set(claim.assertion_ids).intersection(conflicted_ids)
        if claim.disposition == ClaimDisposition.SUPPORTED and unresolved:
            raise ValueError(
                f"supported claim {claim.claim_id!r} cites unresolved conflicting assertions: "
                + ", ".join(sorted(unresolved))
            )
        if claim.disposition == ClaimDisposition.SUPPORTED:
            for predicate in claim.predicates:
                if not _evaluate_claim_predicate(predicate, assertions):
                    raise ValueError(
                        f"supported claim {claim.claim_id!r} has an unsatisfied "
                        f"{predicate.kind} predicate"
                    )


def _numeric_assertion(
    assertion_id: str,
    assertions: dict[str, EvidenceAssertion],
    *,
    expected_predicate: str,
) -> float:
    assertion = assertions[assertion_id]
    if assertion.predicate != expected_predicate:
        raise ValueError(
            f"assertion {assertion_id!r} predicate {assertion.predicate!r} does not "
            f"match {expected_predicate!r}"
        )
    value = assertion.value
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"assertion {assertion_id!r} is not numeric")
    return float(value)


def _evaluate_claim_predicate(
    predicate: (
        NumericThresholdPredicate
        | FreshnessPredicate
        | ApplicabilityPredicate
        | ExactValuePredicate
    ),
    assertions: dict[str, EvidenceAssertion],
) -> bool:
    if isinstance(predicate, NumericThresholdPredicate):
        value = _numeric_assertion(
            predicate.assertion_id,
            assertions,
            expected_predicate=predicate.assertion_predicate,
        )
        threshold = (
            float(predicate.threshold_value)
            if predicate.threshold_value is not None
            else _numeric_assertion(
                predicate.threshold_assertion_id or "",
                assertions,
                expected_predicate=predicate.threshold_assertion_predicate or "",
            )
        )
        comparisons = {
            NumericComparisonOperator.LESS_THAN: value < threshold,
            NumericComparisonOperator.LESS_THAN_OR_EQUAL: value <= threshold,
            NumericComparisonOperator.GREATER_THAN: value > threshold,
            NumericComparisonOperator.GREATER_THAN_OR_EQUAL: value >= threshold,
            NumericComparisonOperator.EQUAL: value == threshold,
        }
        return comparisons[predicate.operator]
    if isinstance(predicate, FreshnessPredicate):
        assertion = assertions[predicate.assertion_id]
        if assertion.predicate != predicate.assertion_predicate:
            raise ValueError(
                f"assertion {predicate.assertion_id!r} predicate does not match "
                f"{predicate.assertion_predicate!r}"
            )
        if assertion.valid_at is None:
            raise ValueError(
                f"assertion {predicate.assertion_id!r} has no valid_at for freshness"
            )
        reference_assertion = assertions[predicate.reference_assertion_id]
        if reference_assertion.predicate != predicate.reference_assertion_predicate:
            raise ValueError(
                f"assertion {predicate.reference_assertion_id!r} predicate does not match "
                f"{predicate.reference_assertion_predicate!r}"
            )
        reference_value = reference_assertion.value
        if not isinstance(reference_value, str):
            raise ValueError(
                f"assertion {predicate.reference_assertion_id!r} is not an ISO datetime"
            )
        try:
            reference_time = datetime.fromisoformat(reference_value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(
                f"assertion {predicate.reference_assertion_id!r} is not an ISO datetime"
            ) from exc
        if reference_time.tzinfo is None or reference_time.utcoffset() is None:
            raise ValueError(
                f"assertion {predicate.reference_assertion_id!r} datetime is not aware"
            )
        max_age_s = (
            float(predicate.max_age_s)
            if predicate.max_age_s is not None
            else _numeric_assertion(
                predicate.max_age_assertion_id or "",
                assertions,
                expected_predicate=predicate.max_age_assertion_predicate,
            )
        )
        age_s = (reference_time - assertion.valid_at).total_seconds()
        return 0.0 <= age_s <= max_age_s
    if isinstance(predicate, ApplicabilityPredicate):
        actual = assertions[predicate.actual_assertion_id]
        required = assertions[predicate.required_assertion_id]
        return (
            actual.predicate == predicate.actual_assertion_predicate
            and required.predicate == predicate.required_assertion_predicate
            and actual.subject == predicate.expected_subject
            and required.subject == predicate.expected_subject
            and required.scope == predicate.expected_scope
            and type(actual.value) is type(required.value)
            and actual.value == required.value
        )
    assertion = assertions[predicate.assertion_id]
    return (
        assertion.predicate == predicate.assertion_predicate
        and type(assertion.value) is type(predicate.expected_value)
        and assertion.value == predicate.expected_value
    )


def _validate_state_integrity(state: WorldState) -> None:
    for assertion in state.assertions:
        if assertion.assertion_sha256 != assertion_digest(assertion):
            raise ValueError(
                f"assertion {assertion.assertion_id!r} digest does not match its content"
            )
    expected = reduce_world_state(state.assertions)
    if state.assertions != expected.assertions:
        raise ValueError("world state assertions are not in canonical order")
    if state.conflicts != expected.conflicts:
        raise ValueError("world state conflicts do not match its assertions")
    if state.state_sha256 != world_state_digest(state):
        raise ValueError("world state digest does not match its content")
