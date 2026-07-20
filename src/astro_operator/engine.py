from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol

from astro_operator.errors import OperatorPolicyError
from astro_operator.models import (
    AuthorityGrant,
    CandidateObservation,
    CandidateProposal,
    CommandExecutionRecord,
    CommandResult,
    EvidenceAcquisitionResult,
    EvidenceAssertion,
    EvidenceReference,
    EvidenceRequest,
    MissionObjective,
    OperatorAction,
    OperatorActionKind,
    OperatorRun,
    OperatorRunStatus,
    OperatorState,
    OperatorStep,
    WorldState,
)
from astro_operator.policy import (
    validate_action_against_state,
    validate_candidate_against_objective,
    validate_observation_against_objective,
    validate_operator_run_policy,
)
from astro_operator.reasoner import MissionReasoner, validate_reasoner_decision
from astro_operator.world_state import reduce_world_state, validate_conclusion_claims


class CandidateEvaluator(Protocol):
    def evaluate(self, candidate: CandidateProposal) -> CandidateObservation: ...


class EvidenceProvider(Protocol):
    def acquire(
        self, request: EvidenceRequest, world_state: WorldState
    ) -> EvidenceAcquisitionResult: ...


class AuthorityMonitor(Protocol):
    def check(self, authority: AuthorityGrant) -> None: ...


class CommandExecutor(Protocol):
    def execute(
        self,
        *,
        action: OperatorAction,
        proposal_action: OperatorAction,
        authority: AuthorityGrant,
        world_state: WorldState,
    ) -> tuple[CommandResult, CommandExecutionRecord]: ...


def run_operator(
    *,
    objective: MissionObjective,
    authority: AuthorityGrant,
    reasoner: MissionReasoner,
    evaluator: CandidateEvaluator,
    evidence_provider: EvidenceProvider | None = None,
    command_executor: CommandExecutor | None = None,
    authority_monitor: AuthorityMonitor | None = None,
) -> OperatorRun:
    _check_authority(authority, authority_monitor)

    steps: list[OperatorStep] = []
    known_evidence = list(objective.base_evidence)
    known_evidence_ids = {item.evidence_id for item in known_evidence}
    assertions = list(objective.base_assertions)
    assertion_ids = {item.assertion_id for item in assertions}
    world_state = reduce_world_state(tuple(assertions))
    if tuple(assertions) != world_state.assertions:
        assertions = list(world_state.assertions)
        objective = objective.model_copy(update={"base_assertions": world_state.assertions})
    evaluated_candidates: set[str] = set()
    proposed_commands: dict[str, OperatorAction] = {}
    evaluation_count = 0
    acquisition_count = 0
    schema_1_2_enabled = bool(
        assertions
        or authority.max_evidence_acquisitions
        or authority.allowed_evidence_tools
        or OperatorActionKind.EXECUTE_COMMAND in authority.allowed_actions
    )

    while len(steps) < authority.max_steps:
        _check_authority(authority, authority_monitor)
        state = OperatorState(
            objective=objective.model_copy(deep=True),
            authority=authority.model_copy(deep=True),
            steps=tuple(step.model_copy(deep=True) for step in steps),
            known_evidence=tuple(item.model_copy(deep=True) for item in known_evidence),
            world_state=(
                world_state.model_copy(deep=True) if schema_1_2_enabled else None
            ),
            remaining_steps=authority.max_steps - len(steps),
            remaining_candidate_evaluations=(
                authority.max_candidate_evaluations - evaluation_count
            ),
            remaining_evidence_acquisitions=(
                authority.max_evidence_acquisitions - acquisition_count
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
            if acquisition_count >= authority.max_evidence_acquisitions:
                raise OperatorPolicyError("evidence acquisition budget exhausted")
            acquired = evidence_provider.acquire(
                action.evidence_request, world_state.model_copy(deep=True)
            )
            acquired = EvidenceAcquisitionResult.model_validate(
                acquired.model_dump(mode="python")
            )
            if acquired.request != action.evidence_request:
                raise OperatorPolicyError("evidence provider returned a result for another request")
            bound_assertions = reduce_world_state(acquired.assertions).assertions
            if acquired.assertions != bound_assertions:
                acquired = acquired.model_copy(update={"assertions": bound_assertions})
            _add_evidence(acquired.evidence, known_evidence, known_evidence_ids)
            _add_assertions(
                acquired.assertions,
                assertions,
                assertion_ids,
                known_evidence_ids,
            )
            world_state = reduce_world_state(tuple(assertions))
            steps.append(
                OperatorStep(
                    sequence=len(steps) + 1,
                    action=action,
                    reasoner_invocation=decision.invocation,
                    acquisition_result=acquired,
                )
            )
            acquisition_count += 1
            continue

        if action.kind == OperatorActionKind.PROPOSE_COMMAND:
            assert action.command is not None
            if action.command.command_id in proposed_commands:
                raise OperatorPolicyError(
                    f"command {action.command.command_id} has already been proposed"
                )
            proposed_commands[action.command.command_id] = action.model_copy(deep=True)
            steps.append(
                OperatorStep(
                    sequence=len(steps) + 1,
                    action=action,
                    reasoner_invocation=decision.invocation,
                )
            )
            continue

        if action.kind == OperatorActionKind.EXECUTE_COMMAND:
            if command_executor is None or action.command_execution is None:
                raise OperatorPolicyError(
                    "command execution requires the transactional command coordinator"
                )
            proposal = proposed_commands.get(action.command_execution.command_id)
            if proposal is None:
                raise OperatorPolicyError("command execution has no matching proposal")
            result, execution_record = command_executor.execute(
                action=action.model_copy(deep=True),
                proposal_action=proposal.model_copy(deep=True),
                authority=authority.model_copy(deep=True),
                world_state=world_state.model_copy(deep=True),
            )
            _check_authority(authority, authority_monitor)
            if result.command != proposal.command:
                raise OperatorPolicyError("command result does not match the proposed command")
            _add_evidence(result.evidence, known_evidence, known_evidence_ids)
            bound_assertions = reduce_world_state(result.assertions).assertions
            if result.assertions != bound_assertions:
                result = result.model_copy(update={"assertions": bound_assertions})
            _add_assertions(
                result.assertions,
                assertions,
                assertion_ids,
                known_evidence_ids,
            )
            world_state = reduce_world_state(tuple(assertions))
            steps.append(
                OperatorStep(
                    sequence=len(steps) + 1,
                    action=action,
                    reasoner_invocation=decision.invocation,
                    command_result=result,
                    command_execution_record=execution_record,
                )
            )
            continue

        assert action.kind == OperatorActionKind.FINISH
        if (
            action.selected_candidate_id is not None
            and action.selected_candidate_id not in evaluated_candidates
        ):
            raise OperatorPolicyError("selected candidate must have been evaluated")
        if assertions and not action.conclusion_claims:
            raise OperatorPolicyError(
                "a conclusion over typed assertions requires claim-backed support"
            )
        try:
            validate_conclusion_claims(action.conclusion_claims, world_state)
        except ValueError as exc:
            raise OperatorPolicyError(f"invalid conclusion claims: {exc}") from exc
        steps.append(
            OperatorStep(
                sequence=len(steps) + 1,
                action=action,
                reasoner_invocation=decision.invocation,
            )
        )
        assert action.conclusion is not None
        run = OperatorRun(
            schema_version=(
                "1.2" if schema_1_2_enabled or action.conclusion_claims else "1.1"
            ),
            objective=objective,
            authority=authority,
            status=OperatorRunStatus.COMPLETED,
            steps=tuple(steps),
            known_evidence=tuple(known_evidence),
            world_state=world_state if schema_1_2_enabled else None,
            selected_candidate_id=action.selected_candidate_id,
            conclusion=action.conclusion,
        )
        validate_operator_run_policy(run)
        return run

    run = OperatorRun(
        schema_version="1.2" if schema_1_2_enabled else "1.1",
        objective=objective,
        authority=authority,
        status=OperatorRunStatus.BUDGET_EXHAUSTED,
        steps=tuple(steps),
        known_evidence=tuple(known_evidence),
        world_state=world_state if schema_1_2_enabled else None,
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


def _add_assertions(
    additions: tuple[EvidenceAssertion, ...],
    assertions: list[EvidenceAssertion],
    known_ids: set[str],
    known_evidence_ids: set[str],
) -> None:
    for assertion in additions:
        if assertion.assertion_id in known_ids:
            raise OperatorPolicyError(f"assertion ID {assertion.assertion_id} is not unique")
        if not set(assertion.source_evidence_ids).issubset(known_evidence_ids):
            raise OperatorPolicyError(
                f"assertion {assertion.assertion_id} cites unavailable evidence"
            )
        assertions.append(assertion)
        known_ids.add(assertion.assertion_id)


def _check_authority(
    authority: AuthorityGrant, authority_monitor: AuthorityMonitor | None
) -> None:
    if authority.revoked:
        raise OperatorPolicyError(f"authority grant {authority.grant_id} is revoked")
    now = datetime.now(UTC)
    if authority.valid_from is not None and now < authority.valid_from:
        raise OperatorPolicyError(f"authority grant {authority.grant_id} is not yet valid")
    if authority.expires_at is not None and now >= authority.expires_at:
        raise OperatorPolicyError(f"authority grant {authority.grant_id} has expired")
    if authority_monitor is not None:
        authority_monitor.check(authority)
