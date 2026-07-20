from __future__ import annotations

import json
from hashlib import sha256

from astro_operator.errors import OperatorPolicyError
from astro_operator.models import (
    AuthorityGrant,
    CandidateObservation,
    CandidateProposal,
    MissionObjective,
    OperatorAction,
    OperatorActionKind,
    OperatorRun,
    OperatorRunStatus,
    OperatorState,
    OperatorStep,
)
from astro_operator.reasoner import invocation_digest, model_digest


def action_digest(action: OperatorAction) -> str:
    payload = json.dumps(
        action.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def validate_action_against_grant(
    action: OperatorAction, authority: AuthorityGrant
) -> None:
    if authority.revoked:
        raise OperatorPolicyError(f"authority grant {authority.grant_id} is revoked")
    if action.kind not in authority.allowed_actions:
        raise OperatorPolicyError(
            f"action {action.kind.value} is outside grant {authority.grant_id}"
        )
    if action.kind in authority.approval_required_for:
        digest = action_digest(action)
        if not any(
            approval.action_id == action.action_id
            and approval.grant_version == authority.grant_version
            and approval.action_sha256 == digest
            for approval in authority.approvals
        ):
            raise OperatorPolicyError(
                f"action {action.action_id} requires approval bound to its content under this grant"
            )
    if (
        action.command is not None
        and action.command.command_type not in authority.allowed_command_types
    ):
        raise OperatorPolicyError(
            f"command type {action.command.command_type} is outside grant {authority.grant_id}"
        )


def validate_candidate_against_objective(
    candidate: CandidateProposal, objective: MissionObjective
) -> None:
    variables = {item.variable_id: item for item in objective.design_variables}
    unknown = set(candidate.assignments) - set(variables)
    if unknown:
        raise OperatorPolicyError(
            f"candidate uses unknown design variables: {', '.join(sorted(unknown))}"
        )
    for variable_id, value in candidate.assignments.items():
        variable = variables[variable_id]
        if not variable.lower_bound <= value <= variable.upper_bound:
            raise OperatorPolicyError(
                f"candidate {candidate.candidate_id} sets {variable_id} outside "
                f"[{variable.lower_bound}, {variable.upper_bound}] {variable.unit}"
            )


def validate_action_against_state(action: OperatorAction, state: OperatorState) -> None:
    """Validate one proposed action without executing it or calling a provider."""

    if state.remaining_steps == 0:
        raise OperatorPolicyError("operator step budget is exhausted")
    if action.action_id in {step.action.action_id for step in state.steps}:
        raise OperatorPolicyError(f"action ID {action.action_id} has already been used")
    validate_action_against_grant(action, state.authority)
    known_evidence_ids = {evidence.evidence_id for evidence in state.known_evidence}
    unknown_evidence = set(action.evidence_ids) - known_evidence_ids
    if unknown_evidence:
        raise OperatorPolicyError(
            f"action cites unknown evidence: {', '.join(sorted(unknown_evidence))}"
        )
    evaluated = {
        step.observation.candidate.candidate_id
        for step in state.steps
        if step.observation is not None
    }
    if action.kind == OperatorActionKind.EVALUATE_CANDIDATE:
        if state.remaining_candidate_evaluations == 0:
            raise OperatorPolicyError("candidate evaluation budget exhausted")
        assert action.candidate is not None
        if action.candidate.candidate_id in evaluated:
            raise OperatorPolicyError(
                f"candidate {action.candidate.candidate_id} has already been evaluated"
            )
        validate_candidate_against_objective(action.candidate, state.objective)
    elif action.kind == OperatorActionKind.PROPOSE_COMMAND:
        assert action.command is not None
        proposed = {
            step.action.command.command_id
            for step in state.steps
            if step.action.kind == OperatorActionKind.PROPOSE_COMMAND
            and step.action.command is not None
        }
        if action.command.command_id in proposed:
            raise OperatorPolicyError(
                f"command {action.command.command_id} has already been proposed"
            )
    elif action.kind == OperatorActionKind.EXECUTE_COMMAND:
        raise OperatorPolicyError(
            "operator command contract stages authority but does not support command commit"
        )
    elif (
        action.kind == OperatorActionKind.FINISH
        and action.selected_candidate_id is not None
        and action.selected_candidate_id not in evaluated
    ):
        raise OperatorPolicyError("selected candidate must have been evaluated")


def validate_operator_state(state: OperatorState) -> None:
    """Reject states whose counters or evidence cannot be reconstructed from history."""

    if len(state.steps) > state.authority.max_steps:
        raise OperatorPolicyError("operator state exceeds its step budget")
    expected_remaining_steps = state.authority.max_steps - len(state.steps)
    if state.remaining_steps != expected_remaining_steps:
        raise OperatorPolicyError("operator state remaining step budget is inconsistent")
    evaluation_count = sum(
        step.action.kind == OperatorActionKind.EVALUATE_CANDIDATE for step in state.steps
    )
    if evaluation_count > state.authority.max_candidate_evaluations:
        raise OperatorPolicyError("operator state exceeds its candidate evaluation budget")
    if (
        state.remaining_candidate_evaluations
        != state.authority.max_candidate_evaluations - evaluation_count
    ):
        raise OperatorPolicyError("operator state remaining evaluation budget is inconsistent")
    if [step.sequence for step in state.steps] != list(range(1, len(state.steps) + 1)):
        raise OperatorPolicyError("operator state step sequences must be contiguous and one-based")
    known = list(state.objective.base_evidence)
    known_ids = {evidence.evidence_id for evidence in known}
    action_ids: set[str] = set()
    evaluated: set[str] = set()
    proposed_commands: set[str] = set()
    for step in state.steps:
        if step.action.kind == OperatorActionKind.FINISH:
            raise OperatorPolicyError("operator state cannot continue after a finish action")
        if step.action.action_id in action_ids:
            raise OperatorPolicyError("operator state action IDs must be unique")
        action_ids.add(step.action.action_id)
        validate_action_against_grant(step.action, state.authority)
        if not set(step.action.evidence_ids).issubset(known_ids):
            raise OperatorPolicyError("operator state action cites unavailable evidence")
        if step.action.kind == OperatorActionKind.EVALUATE_CANDIDATE:
            assert step.action.candidate is not None
            assert step.observation is not None
            candidate_id = step.action.candidate.candidate_id
            if candidate_id in evaluated:
                raise OperatorPolicyError("operator state evaluates a candidate more than once")
            if step.observation.candidate != step.action.candidate:
                raise OperatorPolicyError("operator state observation does not match its action")
            validate_candidate_against_objective(step.action.candidate, state.objective)
            validate_observation_against_objective(step.observation, state.objective)
            evaluated.add(candidate_id)
        elif step.action.kind == OperatorActionKind.PROPOSE_COMMAND:
            assert step.action.command is not None
            command_id = step.action.command.command_id
            if command_id in proposed_commands:
                raise OperatorPolicyError("operator state proposes a command more than once")
            proposed_commands.add(command_id)
        elif step.action.kind == OperatorActionKind.EXECUTE_COMMAND:
            raise OperatorPolicyError(
                "operator command contract stages authority but does not support command commit"
            )
        additions = step.acquired_evidence
        if step.observation is not None:
            additions += step.observation.evidence
        if step.command_result is not None:
            additions += step.command_result.evidence
        for evidence in additions:
            if evidence.evidence_id in known_ids:
                raise OperatorPolicyError("operator state evidence IDs must be unique")
            known.append(evidence)
            known_ids.add(evidence.evidence_id)
    if tuple(known) != state.known_evidence:
        raise OperatorPolicyError("operator state evidence inventory is inconsistent")


def validate_observation_against_objective(
    observation: CandidateObservation, objective: MissionObjective
) -> None:
    metrics = {item.metric_id: item for item in observation.metrics}
    if observation.evaluation_status == "evaluated":
        missing = {goal.metric_id for goal in objective.metric_goals} - set(metrics)
        if missing:
            raise OperatorPolicyError(
                f"evaluator omitted objective metrics: {', '.join(sorted(missing))}"
            )
    for goal in objective.metric_goals:
        metric = metrics.get(goal.metric_id)
        if metric is not None and metric.unit != goal.unit:
            raise OperatorPolicyError(
                f"evaluator metric {goal.metric_id} uses {metric.unit}, expected {goal.unit}"
            )


def validate_operator_run_policy(run: OperatorRun) -> None:
    authority = run.authority
    if authority.revoked:
        raise OperatorPolicyError(f"authority grant {authority.grant_id} is revoked")
    if len(run.steps) > authority.max_steps:
        raise OperatorPolicyError("operator journal exceeds its step budget")

    evaluated: set[str] = set()
    proposed_commands: dict[str, object] = {}
    evaluation_count = 0
    known_evidence = list(run.objective.base_evidence)
    prior_steps: list[OperatorStep] = []
    for step_index, step in enumerate(run.steps):
        action = step.action
        if run.schema_version == "1.1":
            invocation = step.reasoner_invocation
            assert invocation is not None
            state = OperatorState(
                objective=run.objective,
                authority=run.authority,
                steps=tuple(prior_steps),
                known_evidence=tuple(known_evidence),
                remaining_steps=authority.max_steps - step_index,
                remaining_candidate_evaluations=(
                    authority.max_candidate_evaluations - evaluation_count
                ),
            )
            if invocation.input_sha256 != model_digest(state):
                raise OperatorPolicyError(
                    "reasoner invocation input digest does not match journal state"
                )
            if invocation.output_sha256 != model_digest(action):
                raise OperatorPolicyError(
                    "reasoner invocation output digest does not match journal action"
                )
            if (
                invocation.record_sha256 is None
                or invocation.record_sha256 != invocation_digest(invocation)
            ):
                raise OperatorPolicyError(
                    "reasoner invocation record digest does not match journal provenance"
                )
        validate_action_against_grant(action, authority)
        if action.kind == OperatorActionKind.EVALUATE_CANDIDATE:
            assert action.candidate is not None
            if action.candidate.candidate_id in evaluated:
                raise OperatorPolicyError("operator journal evaluates a candidate more than once")
            validate_candidate_against_objective(action.candidate, run.objective)
            assert step.observation is not None
            validate_observation_against_objective(step.observation, run.objective)
            evaluated.add(action.candidate.candidate_id)
            evaluation_count += 1
        elif action.kind == OperatorActionKind.PROPOSE_COMMAND:
            assert action.command is not None
            if action.command.command_id in proposed_commands:
                raise OperatorPolicyError("operator journal proposes a command more than once")
            proposed_commands[action.command.command_id] = action.command
        elif action.kind == OperatorActionKind.EXECUTE_COMMAND:
            raise OperatorPolicyError(
                "operator command contract stages authority but does not support command commit"
            )
        additions = step.acquired_evidence
        if step.observation is not None:
            additions += step.observation.evidence
        if step.command_result is not None:
            additions += step.command_result.evidence
        known_evidence.extend(additions)
        prior_steps.append(step)

    if evaluation_count > authority.max_candidate_evaluations:
        raise OperatorPolicyError("operator journal exceeds its candidate evaluation budget")
    if run.status == OperatorRunStatus.BUDGET_EXHAUSTED and len(run.steps) != authority.max_steps:
        raise OperatorPolicyError("budget-exhausted journal does not consume its step budget")
