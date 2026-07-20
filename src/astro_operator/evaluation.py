"""Offline, provider-neutral evaluation of untrusted mission reasoner actions."""

from __future__ import annotations

import json
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, ValidationError, model_validator

from astro_core.errors import InvalidScenarioError
from astro_core.models import AstroModel
from astro_operator.errors import OperatorPolicyError
from astro_operator.models import (
    OperatorAction,
    OperatorActionKind,
    OperatorState,
)
from astro_operator.policy import validate_action_against_state, validate_operator_state


class DecisionDisposition(StrEnum):
    ACCEPTED = "accepted"
    SCHEMA_REJECTED = "schema_rejected"
    POLICY_REJECTED = "policy_rejected"


_REQUIRED_CASE_DIGESTS = {
    "valid-finish": "3b5ab009dcc433b536932e2f6d86b529f913517b854b1c1b6dadf743590e5fc3",
    "null-finish-conclusion": "c40063baf65a2effccb84162f8e50a33c38a75d5969d0e3404eea16f07d1391b",
    "payload-kind-mismatch": "a696c4bf712c110ec927e54f391b07a5934264c0140a025f82616cf8b59072bc",
    "tool-call-only-shape": "710392b812868eff7855d51410a842f48baf03cba9521817e12074993d4a9f50",
    "out-of-envelope-candidate": "41e1f42f58cdc971498cb887ac78b0f992d9eaaa212db2c2c75d9e0b06a4e5aa",
    "unknown-evidence-citation": "687e56a0713a36ceacf0066ead09915e970d2cee9bc698baeff1f48a0f9ee10e",
    "premature-selection": "8941500e7449bf51e01b4dd47b2d7a8c1e2a69357aa4e6d0c2246791ff859c2b",
    "exhausted-evaluation-budget": (
        "f2e673af2f0f091589bfb494e88fe3e92fec6514e186584a1341aa2ab4eb1593"
    ),
}


class AdversarialDecisionCase(AstroModel):
    case_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
    description: str = Field(min_length=1)
    tags: tuple[str, ...] = ()
    state: OperatorState
    response: dict[str, Any]
    expected_disposition: DecisionDisposition
    expected_action_kind: OperatorActionKind | None = None

    @model_validator(mode="after")
    def accepted_case_must_name_action_kind(self) -> AdversarialDecisionCase:
        try:
            validate_operator_state(self.state)
        except OperatorPolicyError as exc:
            raise ValueError(f"adversarial case has an invalid operator state: {exc}") from exc
        if (
            self.expected_disposition == DecisionDisposition.ACCEPTED
            and self.expected_action_kind is None
        ):
            raise ValueError("accepted cases must declare expected_action_kind")
        return self


class AdversarialDecisionCorpus(AstroModel):
    schema_version: str = Field(pattern=r"^1\.0$")
    corpus_id: str = Field(min_length=1)
    cases: tuple[AdversarialDecisionCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def case_ids_must_be_unique(self) -> AdversarialDecisionCorpus:
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("adversarial corpus case IDs must be unique")
        return self


class AdversarialCaseResult(AstroModel):
    case_id: str
    expected_disposition: DecisionDisposition
    actual_disposition: DecisionDisposition
    expected_action_kind: OperatorActionKind | None = None
    actual_action_kind: OperatorActionKind | None = None
    matched: bool
    safety_critical: bool
    diagnostic: str | None = None


class AdversarialCorpusScore(AstroModel):
    schema_version: str = Field(pattern=r"^1\.0$")
    corpus_id: str
    total_cases: int = Field(ge=1)
    matched_cases: int = Field(ge=0)
    safety_critical_cases: int = Field(ge=0)
    safety_critical_matched: int = Field(ge=0)
    coverage_complete: bool
    promoted: bool
    results: tuple[AdversarialCaseResult, ...]


def load_adversarial_corpus(path: Path | str) -> AdversarialDecisionCorpus:
    corpus_path = Path(path)
    try:
        raw: Any = yaml.safe_load(corpus_path.read_text(encoding="utf-8"))
        return AdversarialDecisionCorpus.model_validate(raw)
    except (OSError, UnicodeError, yaml.YAMLError, ValidationError) as exc:
        raise InvalidScenarioError(
            f"Could not load adversarial reasoner corpus {corpus_path}: {exc}"
        ) from exc


def score_adversarial_corpus(corpus: AdversarialDecisionCorpus) -> AdversarialCorpusScore:
    results = tuple(_score_case(case) for case in corpus.cases)
    safety_results = tuple(result for result in results if result.safety_critical)
    matched = sum(result.matched for result in results)
    safety_matched = sum(result.matched for result in safety_results)
    cases_by_id = {case.case_id: case for case in corpus.cases}
    results_by_id = {result.case_id: result for result in results}
    coverage_complete = all(
        (result := results_by_id.get(case_id)) is not None
        and result.matched
        and (case := cases_by_id.get(case_id)) is not None
        and _case_digest(case) == expected_digest
        for case_id, expected_digest in _REQUIRED_CASE_DIGESTS.items()
    )
    return AdversarialCorpusScore(
        schema_version="1.0",
        corpus_id=corpus.corpus_id,
        total_cases=len(results),
        matched_cases=matched,
        safety_critical_cases=len(safety_results),
        safety_critical_matched=safety_matched,
        coverage_complete=coverage_complete,
        promoted=matched == len(results) and coverage_complete,
        results=results,
    )


def _score_case(case: AdversarialDecisionCase) -> AdversarialCaseResult:
    action: OperatorAction | None = None
    diagnostic: str | None = None
    try:
        action = OperatorAction.model_validate(case.response)
    except ValidationError as exc:
        actual = DecisionDisposition.SCHEMA_REJECTED
        diagnostic = str(exc).splitlines()[0]
    else:
        try:
            validate_action_against_state(action, case.state)
        except OperatorPolicyError as exc:
            actual = DecisionDisposition.POLICY_REJECTED
            diagnostic = str(exc)
        else:
            actual = DecisionDisposition.ACCEPTED
    actual_kind = action.kind if action is not None else None
    matched = actual == case.expected_disposition and (
        case.expected_action_kind is None or case.expected_action_kind == actual_kind
    )
    return AdversarialCaseResult(
        case_id=case.case_id,
        expected_disposition=case.expected_disposition,
        actual_disposition=actual,
        expected_action_kind=case.expected_action_kind,
        actual_action_kind=actual_kind,
        matched=matched,
        safety_critical="safety-critical" in case.tags,
        diagnostic=diagnostic,
    )


def _case_digest(case: AdversarialDecisionCase) -> str:
    data = case.model_dump(mode="json", exclude={"description", "tags"})
    _strip_post_v1_defaults(data)
    payload = json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _strip_post_v1_defaults(value: object) -> None:
    """Keep the locked schema-1.0 corpus digest stable across additive model fields."""

    if isinstance(value, dict):
        for key, default in (
            ("base_assertions", []),
            ("allowed_evidence_tools", []),
            ("command_envelopes", []),
            ("max_evidence_acquisitions", 0),
            ("valid_from", None),
            ("expires_at", None),
            ("world_state", None),
            ("remaining_evidence_acquisitions", 0),
            ("conclusion_claims", []),
            ("acquisition_result", None),
            ("command_execution", None),
            ("command_execution_record", None),
        ):
            if value.get(key) == default:
                value.pop(key, None)
        for item in value.values():
            _strip_post_v1_defaults(item)
    elif isinstance(value, list):
        for item in value:
            _strip_post_v1_defaults(item)
