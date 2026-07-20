from __future__ import annotations

import hashlib
import json
import warnings
from collections.abc import Sequence
from typing import Any, Protocol

from pydantic import Field, StrictBool, field_validator

from astro_core.models import AstroModel
from astro_operator.errors import ReasonerError, ReasonerInvalidResponseError
from astro_operator.models import (
    OperatorAction,
    OperatorState,
    ReasonerDecision,
    ReasonerInvocation,
)


class MissionReasoner(Protocol):
    """Provider-neutral decision interface for an adaptive mission operator."""

    def decide(self, state: OperatorState) -> ReasonerDecision: ...


class InvocationRecordingReasoner:
    """Capture successful provider-neutral decisions from another reasoner."""

    def __init__(self, reasoner: MissionReasoner) -> None:
        self._reasoner = reasoner
        self._decisions: list[ReasonerDecision] = []
        self._states: list[OperatorState] = []
        self._failure_state: OperatorState | None = None
        self._failure: ReasonerError | None = None

    @property
    def decisions(self) -> tuple[ReasonerDecision, ...]:
        return tuple(decision.model_copy(deep=True) for decision in self._decisions)

    @property
    def states(self) -> tuple[OperatorState, ...]:
        return tuple(state.model_copy(deep=True) for state in self._states)

    @property
    def failure_state(self) -> OperatorState | None:
        return (
            self._failure_state.model_copy(deep=True)
            if self._failure_state is not None
            else None
        )

    @property
    def failure(self) -> ReasonerError | None:
        return self._failure

    def decide(self, state: OperatorState) -> ReasonerDecision:
        recorded_state = state.model_copy(deep=True)
        try:
            decision = validate_reasoner_decision(
                recorded_state, self._reasoner.decide(state)
            )
        except ReasonerError as exc:
            self._failure_state = recorded_state
            self._failure = exc
            raise
        self._states.append(recorded_state)
        self._decisions.append(decision.model_copy(deep=True))
        return decision


class BoundDecisionReasoner:
    """Replay exact invocation-bound decisions for engine verification."""

    def __init__(self, decisions: Sequence[ReasonerDecision]) -> None:
        self._decisions = tuple(decision.model_copy(deep=True) for decision in decisions)
        self._cursor = 0

    def decide(self, state: OperatorState) -> ReasonerDecision:
        del state
        if self._cursor >= len(self._decisions):
            raise ReasonerInvalidResponseError(
                "invocation-bound recording exhausted before finishing"
            )
        decision = self._decisions[self._cursor].model_copy(deep=True)
        self._cursor += 1
        return decision


class ScriptedReasoner:
    """Deterministic replay harness for testing the reasoner boundary."""

    def __init__(self, actions: Sequence[OperatorAction]) -> None:
        self._actions = tuple(actions)
        self._cursor = 0

    def decide(self, state: OperatorState) -> ReasonerDecision:
        if self._cursor >= len(self._actions):
            raise ReasonerInvalidResponseError(
                "scripted reasoner exhausted before finishing"
            )
        action = self._actions[self._cursor]
        self._cursor += 1
        return _replay_decision("scripted", state, action)


class ReplayCondition(AstroModel):
    step_count: int | None = Field(default=None, ge=0)
    last_candidate_id: str | None = None
    last_evaluation_status: str | None = None
    last_candidate_passed: StrictBool | None = None
    evidence_ids_present: tuple[str, ...] = ()

    @field_validator("step_count", mode="before")
    @classmethod
    def step_count_must_be_strict(cls, value: Any) -> Any:
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise ValueError("replay step_count must be an integer")
        return value


class ConditionalReplayDecision(AstroModel):
    when: ReplayCondition
    action: OperatorAction


class ConditionalReplayReasoner:
    """Checked branching replay that selects actions from the typed operator state."""

    def __init__(self, decisions: Sequence[ConditionalReplayDecision]) -> None:
        self._decisions = tuple(decisions)
        self._consumed_action_ids: set[str] = set()

    def decide(self, state: OperatorState) -> ReasonerDecision:
        for decision in self._decisions:
            if decision.action.action_id in self._consumed_action_ids:
                continue
            if _condition_matches(decision.when, state):
                self._consumed_action_ids.add(decision.action.action_id)
                return _replay_decision("conditional-replay", state, decision.action)
        raise ReasonerInvalidResponseError(
            "conditional replay has no action matching the current state"
        )


def _replay_decision(
    adapter: str, state: OperatorState, action: OperatorAction
) -> ReasonerDecision:
    invocation = ReasonerInvocation(
        adapter=adapter,
        provider="deterministic-replay",
        model="checked-fixture",
        input_sha256=model_digest(state),
        output_sha256=model_digest(action),
    )
    invocation = invocation.model_copy(
        update={"record_sha256": invocation_digest(invocation)}
    )
    return ReasonerDecision(action=action, invocation=invocation)


def model_digest(model: AstroModel) -> str:
    """Return the canonical digest used to bind reasoner inputs and outputs."""

    payload = json.dumps(
        model.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def invocation_digest(invocation: ReasonerInvocation) -> str:
    """Bind the portable invocation metadata independently of action identity."""

    payload = json.dumps(
        invocation.model_dump(mode="json", exclude={"record_sha256"}),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_reasoner_decision(
    state: OperatorState, raw_decision: object
) -> ReasonerDecision:
    """Normalize and verify one untrusted decision against its exact input state."""

    if not isinstance(raw_decision, ReasonerDecision):
        raise ReasonerInvalidResponseError(
            "mission reasoner must return a ReasonerDecision"
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            decision = ReasonerDecision.model_validate(
                raw_decision.model_dump(mode="python", round_trip=True)
            )
    except Exception as exc:
        raise ReasonerInvalidResponseError(
            "mission reasoner returned an invalid ReasonerDecision"
        ) from exc
    if decision.invocation.input_sha256 != model_digest(state):
        raise ReasonerInvalidResponseError(
            "reasoner invocation input digest does not match operator state"
        )
    if decision.invocation.output_sha256 != model_digest(decision.action):
        raise ReasonerInvalidResponseError(
            "reasoner invocation output digest does not match action"
        )
    if (
        decision.invocation.record_sha256 is None
        or decision.invocation.record_sha256 != invocation_digest(decision.invocation)
    ):
        raise ReasonerInvalidResponseError(
            "reasoner invocation record digest does not match provenance"
        )
    return decision


def _condition_matches(condition: ReplayCondition, state: OperatorState) -> bool:
    if condition.step_count is not None and condition.step_count != len(state.steps):
        return False
    known_ids = {item.evidence_id for item in state.known_evidence}
    if not set(condition.evidence_ids_present).issubset(known_ids):
        return False
    observations = [step.observation for step in state.steps if step.observation is not None]
    needs_observation = any(
        value is not None
        for value in (
            condition.last_candidate_id,
            condition.last_evaluation_status,
            condition.last_candidate_passed,
        )
    )
    if not observations:
        return not needs_observation
    last = observations[-1]
    assert last is not None
    return (
        (
            condition.last_candidate_id is None
            or condition.last_candidate_id == last.candidate.candidate_id
        )
        and (
            condition.last_evaluation_status is None
            or condition.last_evaluation_status == last.evaluation_status
        )
        and (
            condition.last_candidate_passed is None
            or condition.last_candidate_passed == last.passed
        )
    )
