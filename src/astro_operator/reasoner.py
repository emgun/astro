from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from pydantic import Field, StrictBool, field_validator

from astro_core.models import AstroModel
from astro_operator.errors import OperatorPolicyError
from astro_operator.models import OperatorAction, OperatorState


class MissionReasoner(Protocol):
    """Provider-neutral decision interface for an adaptive mission operator."""

    def decide(self, state: OperatorState) -> OperatorAction: ...


class ScriptedReasoner:
    """Deterministic replay harness for testing the reasoner boundary."""

    def __init__(self, actions: Sequence[OperatorAction]) -> None:
        self._actions = tuple(actions)
        self._cursor = 0

    def decide(self, state: OperatorState) -> OperatorAction:
        del state
        if self._cursor >= len(self._actions):
            raise OperatorPolicyError("scripted reasoner exhausted before finishing")
        action = self._actions[self._cursor]
        self._cursor += 1
        return action


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

    def decide(self, state: OperatorState) -> OperatorAction:
        for decision in self._decisions:
            if decision.action.action_id in self._consumed_action_ids:
                continue
            if _condition_matches(decision.when, state):
                self._consumed_action_ids.add(decision.action.action_id)
                return decision.action
        raise OperatorPolicyError("conditional replay has no action matching the current state")


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
