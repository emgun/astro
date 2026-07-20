"""Pure, deterministic reduction and validation for evidence-backed world state."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime
from hashlib import sha256

from astro_operator.models import (
    AssertionConflict,
    ClaimDisposition,
    ConclusionClaim,
    EvidenceAssertion,
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
