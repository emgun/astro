"""Deterministic whole-run behavior benchmark for mission reasoners."""

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
from astro_operator.engine import run_operator
from astro_operator.errors import OperatorPolicyError, ReasonerInvalidResponseError
from astro_operator.models import (
    AuthorityGrant,
    CandidateObservation,
    CandidateProposal,
    MissionObjective,
    OperatorActionKind,
    OperatorState,
    ReasonerDecision,
)
from astro_operator.reasoner import ConditionalReplayDecision, ConditionalReplayReasoner


class BehaviorDisposition(StrEnum):
    COMPLETED = "completed"
    BUDGET_EXHAUSTED = "budget_exhausted"
    REASONER_INVALID_RESPONSE = "reasoner_invalid_response"
    REASONER_UNAVAILABLE = "reasoner_unavailable"
    REASONER_CONFIGURATION = "reasoner_configuration"
    REASONER_CANCELLED = "reasoner_cancelled"
    POLICY_REJECTED = "policy_rejected"


_REQUIRED_CASE_DIGESTS = {
    "pass-direct": "4af4b9e8f15ee11c3351257ba23bdf2d69841eeae5d8156b183ce68b263d40df",
    "recover-after-failure": "0571ded1cde155ec06fd7d057c46f767f247ca4afea6b2e561c30c06b7c959d2",
    "unmatched-branch": "bec137d98a95975e656aac02adc3dbf63f05d0bf6ffc16a6ebb96b3b06ccd5a6",
    "whole-run-budget-exhaustion": (
        "25f2f901a83f7d94df15a41d4c930aad3e2fcee2ad4f9bca517b2b24718cc459"
    ),
}


class ReasonerBehaviorCase(AstroModel):
    case_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
    description: str = Field(min_length=1)
    objective: MissionObjective
    authority: AuthorityGrant
    observations: tuple[CandidateObservation, ...]
    expected_disposition: BehaviorDisposition
    expected_action_kinds: tuple[OperatorActionKind, ...]
    expected_evaluated_candidate_ids: tuple[str, ...]
    expected_selected_candidate_id: str | None = None

    @model_validator(mode="after")
    def fixture_ids_must_be_unique(self) -> ReasonerBehaviorCase:
        candidate_ids = [item.candidate.candidate_id for item in self.observations]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("behavior case observation candidate IDs must be unique")
        return self


class ReasonerBehaviorCorpus(AstroModel):
    schema_version: str = Field(pattern=r"^1\.0$")
    corpus_id: str = Field(min_length=1)
    cases: tuple[ReasonerBehaviorCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def case_ids_must_be_unique(self) -> ReasonerBehaviorCorpus:
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("behavior corpus case IDs must be unique")
        return self


class ReasonerBehaviorReplayCase(AstroModel):
    case_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
    decisions: tuple[ConditionalReplayDecision, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def action_ids_must_be_unique(self) -> ReasonerBehaviorReplayCase:
        action_ids = [decision.action.action_id for decision in self.decisions]
        if len(action_ids) != len(set(action_ids)):
            raise ValueError("behavior replay action IDs must be unique")
        return self


class ReasonerBehaviorReplay(AstroModel):
    schema_version: str = Field(pattern=r"^1\.0$")
    replay_id: str = Field(min_length=1)
    cases: tuple[ReasonerBehaviorReplayCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def case_ids_must_be_unique(self) -> ReasonerBehaviorReplay:
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("behavior replay case IDs must be unique")
        return self


class ReasonerBehaviorResult(AstroModel):
    case_id: str
    expected_disposition: BehaviorDisposition
    actual_disposition: BehaviorDisposition
    expected_action_kinds: tuple[OperatorActionKind, ...]
    actual_action_kinds: tuple[OperatorActionKind, ...]
    expected_evaluated_candidate_ids: tuple[str, ...]
    actual_evaluated_candidate_ids: tuple[str, ...]
    expected_selected_candidate_id: str | None = None
    actual_selected_candidate_id: str | None = None
    matched: bool
    diagnostic: str | None = None


class ReasonerBehaviorScore(AstroModel):
    schema_version: str = Field(pattern=r"^1\.0$")
    corpus_id: str
    replay_id: str
    total_cases: int = Field(ge=1)
    matched_cases: int = Field(ge=0)
    coverage_complete: bool
    behavior_gate_passed: bool
    results: tuple[ReasonerBehaviorResult, ...]


def load_reasoner_behavior_corpus(path: Path | str) -> ReasonerBehaviorCorpus:
    corpus_path = Path(path)
    try:
        raw: Any = yaml.safe_load(corpus_path.read_text(encoding="utf-8"))
        return ReasonerBehaviorCorpus.model_validate(raw)
    except (OSError, UnicodeError, yaml.YAMLError, ValidationError) as exc:
        raise InvalidScenarioError(
            f"Could not load reasoner behavior corpus {corpus_path}: {exc}"
        ) from exc


def load_reasoner_behavior_replay(path: Path | str) -> ReasonerBehaviorReplay:
    replay_path = Path(path)
    try:
        raw: Any = yaml.safe_load(replay_path.read_text(encoding="utf-8"))
        return ReasonerBehaviorReplay.model_validate(raw)
    except (OSError, UnicodeError, yaml.YAMLError, ValidationError) as exc:
        raise InvalidScenarioError(
            f"Could not load reasoner behavior replay {replay_path}: {exc}"
        ) from exc


def score_reasoner_behavior_corpus(
    corpus: ReasonerBehaviorCorpus, replay: ReasonerBehaviorReplay
) -> ReasonerBehaviorScore:
    replay_by_id = {case.case_id: case for case in replay.cases}
    results = tuple(
        _score_case(case, replay_by_id.get(case.case_id)) for case in corpus.cases
    )
    coverage_complete = behavior_coverage_complete(
        corpus, results, set(replay_by_id)
    )
    matched = sum(result.matched for result in results)
    return ReasonerBehaviorScore(
        schema_version="1.0",
        corpus_id=corpus.corpus_id,
        replay_id=replay.replay_id,
        total_cases=len(results),
        matched_cases=matched,
        coverage_complete=coverage_complete,
        behavior_gate_passed=matched == len(results) and coverage_complete,
        results=results,
    )


class BehaviorFixtureEvaluator:
    def __init__(self, observations: tuple[CandidateObservation, ...]) -> None:
        self._observations = {item.candidate.candidate_id: item for item in observations}
        self.evaluated_candidate_ids: list[str] = []

    def evaluate(self, candidate: CandidateProposal) -> CandidateObservation:
        self.evaluated_candidate_ids.append(candidate.candidate_id)
        observation = self._observations.get(candidate.candidate_id)
        if observation is None:
            raise OperatorPolicyError(
                f"behavior fixture has no observation for {candidate.candidate_id}"
            )
        return observation.model_copy(deep=True)


class _RecordingReasoner:
    def __init__(self, decisions: tuple[ConditionalReplayDecision, ...]) -> None:
        self._reasoner = ConditionalReplayReasoner(decisions)
        self.action_kinds: list[OperatorActionKind] = []

    def decide(self, state: OperatorState) -> ReasonerDecision:
        decision = self._reasoner.decide(state)
        self.action_kinds.append(decision.action.kind)
        return decision


def _score_case(
    case: ReasonerBehaviorCase, replay: ReasonerBehaviorReplayCase | None
) -> ReasonerBehaviorResult:
    decisions = replay.decisions if replay is not None else ()
    reasoner = _RecordingReasoner(decisions)
    evaluator = BehaviorFixtureEvaluator(case.observations)
    selected_candidate_id: str | None = None
    diagnostic: str | None = None
    try:
        run = run_operator(
            objective=case.objective,
            authority=case.authority,
            reasoner=reasoner,
            evaluator=evaluator,
        )
    except ReasonerInvalidResponseError as exc:
        disposition = BehaviorDisposition.REASONER_INVALID_RESPONSE
        diagnostic = str(exc)
    except OperatorPolicyError as exc:
        disposition = BehaviorDisposition.POLICY_REJECTED
        diagnostic = str(exc)
    else:
        disposition = BehaviorDisposition(run.status.value)
        selected_candidate_id = run.selected_candidate_id
    actual_actions = tuple(reasoner.action_kinds)
    actual_candidates = tuple(evaluator.evaluated_candidate_ids)
    matched = (
        disposition == case.expected_disposition
        and actual_actions == case.expected_action_kinds
        and actual_candidates == case.expected_evaluated_candidate_ids
        and selected_candidate_id == case.expected_selected_candidate_id
    )
    return ReasonerBehaviorResult(
        case_id=case.case_id,
        expected_disposition=case.expected_disposition,
        actual_disposition=disposition,
        expected_action_kinds=case.expected_action_kinds,
        actual_action_kinds=actual_actions,
        expected_evaluated_candidate_ids=case.expected_evaluated_candidate_ids,
        actual_evaluated_candidate_ids=actual_candidates,
        expected_selected_candidate_id=case.expected_selected_candidate_id,
        actual_selected_candidate_id=selected_candidate_id,
        matched=matched,
        diagnostic=diagnostic,
    )


def _case_digest(case: ReasonerBehaviorCase) -> str:
    payload = json.dumps(
        case.model_dump(mode="json", exclude={"description"}),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def behavior_coverage_complete(
    corpus: ReasonerBehaviorCorpus,
    results: tuple[ReasonerBehaviorResult, ...],
    provided_case_ids: set[str],
) -> bool:
    cases_by_id = {case.case_id: case for case in corpus.cases}
    results_by_id = {result.case_id: result for result in results}
    return set(cases_by_id) == provided_case_ids and all(
        (result := results_by_id.get(case_id)) is not None
        and result.matched
        and (case := cases_by_id.get(case_id)) is not None
        and _case_digest(case) == expected_digest
        for case_id, expected_digest in _REQUIRED_CASE_DIGESTS.items()
    )
