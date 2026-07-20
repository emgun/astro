from __future__ import annotations

from typing import Protocol

from astro_operator.errors import OperatorPolicyError
from astro_operator.models import (
    AuthorityGrant,
    CandidateObservation,
    CandidateProposal,
    CommandRequest,
    EvidenceReference,
    EvidenceRequest,
    MissionObjective,
    OperatorActionKind,
    OperatorRun,
    OperatorRunStatus,
    OperatorState,
    OperatorStep,
)
from astro_operator.policy import (
    validate_action_against_state,
    validate_candidate_against_objective,
    validate_observation_against_objective,
    validate_operator_run_policy,
)
from astro_operator.reasoner import MissionReasoner, validate_reasoner_decision


class CandidateEvaluator(Protocol):
    def evaluate(self, candidate: CandidateProposal) -> CandidateObservation: ...


class EvidenceProvider(Protocol):
    def acquire(self, request: EvidenceRequest) -> tuple[EvidenceReference, ...]: ...


class AuthorityMonitor(Protocol):
    def check(self, authority: AuthorityGrant) -> None: ...


def run_operator(
    *,
    objective: MissionObjective,
    authority: AuthorityGrant,
    reasoner: MissionReasoner,
    evaluator: CandidateEvaluator,
    evidence_provider: EvidenceProvider | None = None,
    authority_monitor: AuthorityMonitor | None = None,
) -> OperatorRun:
    _check_authority(authority, authority_monitor)

    steps: list[OperatorStep] = []
    known_evidence = list(objective.base_evidence)
    known_evidence_ids = {item.evidence_id for item in known_evidence}
    evaluated_candidates: set[str] = set()
    proposed_commands: dict[str, CommandRequest] = {}
    evaluation_count = 0

    while len(steps) < authority.max_steps:
        _check_authority(authority, authority_monitor)
        state = OperatorState(
            objective=objective.model_copy(deep=True),
            authority=authority.model_copy(deep=True),
            steps=tuple(step.model_copy(deep=True) for step in steps),
            known_evidence=tuple(item.model_copy(deep=True) for item in known_evidence),
            remaining_steps=authority.max_steps - len(steps),
            remaining_candidate_evaluations=(
                authority.max_candidate_evaluations - evaluation_count
            ),
        )
        decision_state = state.model_copy(deep=True)
        decision = validate_reasoner_decision(decision_state, reasoner.decide(state))
        action = decision.action
        _check_authority(authority, authority_monitor)
        validate_action_against_state(action, decision_state)
        if action.kind == OperatorActionKind.EVALUATE_CANDIDATE:
            if evaluation_count >= authority.max_candidate_evaluations:
                raise OperatorPolicyError("candidate evaluation budget exhausted")
            assert action.candidate is not None
            if action.candidate.candidate_id in evaluated_candidates:
                raise OperatorPolicyError(
                    f"candidate {action.candidate.candidate_id} has already been evaluated"
                )
            validate_candidate_against_objective(action.candidate, objective)
            observation = evaluator.evaluate(action.candidate)
            if observation.candidate != action.candidate:
                raise OperatorPolicyError("evaluator returned an observation for another candidate")
            validate_observation_against_objective(observation, objective)
            steps.append(
                OperatorStep(
                    sequence=len(steps) + 1,
                    action=action,
                    reasoner_invocation=decision.invocation,
                    observation=observation,
                )
            )
            evaluated_candidates.add(action.candidate.candidate_id)
            evaluation_count += 1
            _add_evidence(observation.evidence, known_evidence, known_evidence_ids)
            continue

        if action.kind == OperatorActionKind.REQUEST_EVIDENCE:
            if evidence_provider is None:
                raise OperatorPolicyError("evidence request requires an evidence provider")
            assert action.evidence_request is not None
            acquired = evidence_provider.acquire(action.evidence_request)
            _add_evidence(acquired, known_evidence, known_evidence_ids)
            steps.append(
                OperatorStep(
                    sequence=len(steps) + 1,
                    action=action,
                    reasoner_invocation=decision.invocation,
                    acquired_evidence=acquired,
                )
            )
            continue

        if action.kind == OperatorActionKind.PROPOSE_COMMAND:
            assert action.command is not None
            if action.command.command_id in proposed_commands:
                raise OperatorPolicyError(
                    f"command {action.command.command_id} has already been proposed"
                )
            proposed_commands[action.command.command_id] = action.command
            steps.append(
                OperatorStep(
                    sequence=len(steps) + 1,
                    action=action,
                    reasoner_invocation=decision.invocation,
                )
            )
            continue

        if action.kind == OperatorActionKind.EXECUTE_COMMAND:
            raise OperatorPolicyError(
                "operator command contract stages authority but does not support command commit; "
                "add prepare/commit journaling and idempotency before enabling execution"
            )

        assert action.kind == OperatorActionKind.FINISH
        if (
            action.selected_candidate_id is not None
            and action.selected_candidate_id not in evaluated_candidates
        ):
            raise OperatorPolicyError("selected candidate must have been evaluated")
        steps.append(
            OperatorStep(
                sequence=len(steps) + 1,
                action=action,
                reasoner_invocation=decision.invocation,
            )
        )
        assert action.conclusion is not None
        run = OperatorRun(
            schema_version="1.1",
            objective=objective,
            authority=authority,
            status=OperatorRunStatus.COMPLETED,
            steps=tuple(steps),
            known_evidence=tuple(known_evidence),
            selected_candidate_id=action.selected_candidate_id,
            conclusion=action.conclusion,
        )
        validate_operator_run_policy(run)
        return run

    run = OperatorRun(
        schema_version="1.1",
        objective=objective,
        authority=authority,
        status=OperatorRunStatus.BUDGET_EXHAUSTED,
        steps=tuple(steps),
        known_evidence=tuple(known_evidence),
        conclusion="The operator reached its step budget without a finish action.",
    )
    validate_operator_run_policy(run)
    return run


def _add_evidence(
    additions: tuple[EvidenceReference, ...],
    known_evidence: list[EvidenceReference],
    known_ids: set[str],
) -> None:
    for evidence in additions:
        if evidence.evidence_id in known_ids:
            raise OperatorPolicyError(f"evidence ID {evidence.evidence_id} is not unique")
        known_evidence.append(evidence)
        known_ids.add(evidence.evidence_id)


def _check_authority(
    authority: AuthorityGrant, authority_monitor: AuthorityMonitor | None
) -> None:
    if authority.revoked:
        raise OperatorPolicyError(f"authority grant {authority.grant_id} is revoked")
    if authority_monitor is not None:
        authority_monitor.check(authority)
