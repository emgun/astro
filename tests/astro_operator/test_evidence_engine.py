from __future__ import annotations

from hashlib import sha256

import pytest

from astro_operator.engine import run_operator
from astro_operator.errors import OperatorPolicyError
from astro_operator.evidence_tools import EvidenceToolRegistry
from astro_operator.models import (
    AcquisitionStatus,
    AllowedEvidenceTool,
    AuthorityGrant,
    AuthorityLevel,
    CandidateObservation,
    CandidateProposal,
    ClaimDisposition,
    ConclusionClaim,
    DesignVariable,
    EpistemicKind,
    EvidenceAcquisitionResult,
    EvidenceAssertion,
    EvidenceReference,
    EvidenceRequest,
    EvidenceToolSpec,
    MetricGoal,
    MissionObjective,
    OperatorAction,
    OperatorActionKind,
    WorldState,
)
from astro_operator.reasoner import ScriptedReasoner

SHA = "0" * 64


def _objective() -> MissionObjective:
    return MissionObjective(
        objective_id="evidence-native-test",
        summary="Acquire evidence and make a claim-backed conclusion.",
        design_variables=(
            DesignVariable(
                variable_id="mass",
                target="spacecraft_wet_mass_kg",
                lower_bound=400.0,
                upper_bound=600.0,
                unit="kg",
            ),
        ),
        metric_goals=(MetricGoal(metric_id="reserve", objective="maximize", unit="kg"),),
    )


def _authority(**updates: object) -> AuthorityGrant:
    payload: dict[str, object] = {
        "grant_id": "evidence-grant",
        "level": AuthorityLevel.RESEARCH,
        "mission_scope": "simulation-only evidence test",
        "allowed_actions": (
            OperatorActionKind.REQUEST_EVIDENCE,
            OperatorActionKind.FINISH,
        ),
        "allowed_evidence_tools": (
            AllowedEvidenceTool(
                tool_id="telemetry-snapshot",
                tool_version="1.0",
                request_kinds=("read_battery_soc",),
            ),
        ),
        "max_steps": 2,
        "max_candidate_evaluations": 0,
        "max_evidence_acquisitions": 1,
    }
    payload.update(updates)
    return AuthorityGrant.model_validate(payload)


class _UnusedEvaluator:
    def evaluate(self, candidate: CandidateProposal) -> CandidateObservation:
        raise AssertionError(f"unexpected candidate evaluation: {candidate.candidate_id}")


class _TelemetryTool:
    calls = 0
    spec = EvidenceToolSpec(
        tool_id="telemetry-snapshot",
        version="1.0",
        request_kind="read_battery_soc",
        parameter_schema_sha256=SHA,
        output_assertion_kinds=("battery_soc",),
    )

    def acquire(
        self, request: EvidenceRequest, world_state: WorldState
    ) -> EvidenceAcquisitionResult:
        del world_state
        self.calls += 1
        evidence = EvidenceReference(
            evidence_id="telemetry-frame-1",
            kind="simulated_telemetry_snapshot",
            epistemic_kind=EpistemicKind.OBSERVED,
            claim_scope="simulation spacecraft-a at epoch-1",
            path="telemetry/frame-1.json",
            sha256=SHA,
        )
        assertion = EvidenceAssertion(
            assertion_id="battery-soc-1",
            subject="spacecraft-a",
            predicate="battery_soc",
            value=0.82,
            epistemic_kind=EpistemicKind.OBSERVED,
            scope="simulation spacecraft-a at epoch-1",
            source_evidence_ids=(evidence.evidence_id,),
            producer_tool_id=self.spec.tool_id,
            producer_tool_version=self.spec.version,
        )
        return EvidenceAcquisitionResult(
            request=request,
            tool=self.spec,
            status=AcquisitionStatus.SUCCEEDED,
            evidence=(evidence,),
            assertions=(assertion,),
        )


def _request_action() -> OperatorAction:
    return OperatorAction(
        action_id="request-telemetry",
        kind=OperatorActionKind.REQUEST_EVIDENCE,
        rationale="Acquire the missing simulated battery state.",
        evidence_request=EvidenceRequest(
            request_id="battery-request-1",
            tool_id="telemetry-snapshot",
            tool_version="1.0",
            request_kind="read_battery_soc",
            parameters={"asset_id": "spacecraft-a"},
        ),
    )


def _finish_action() -> OperatorAction:
    conclusion = "The simulated battery state is above 80 percent."
    return OperatorAction(
        action_id="finish",
        kind=OperatorActionKind.FINISH,
        rationale="Conclude from the acquired typed assertion.",
        evidence_ids=("telemetry-frame-1",),
        conclusion=conclusion,
        conclusion_claims=(
            ConclusionClaim(
                claim_id="battery-above-threshold",
                statement="Battery state of charge is 0.82 in the simulated snapshot.",
                conclusion_sha256=sha256(conclusion.encode("utf-8")).hexdigest(),
                disposition=ClaimDisposition.SUPPORTED,
                assertion_ids=("battery-soc-1",),
            ),
        ),
    )


def test_operator_acquires_typed_evidence_reduces_state_and_finishes_with_claim() -> None:
    tool = _TelemetryTool()

    run = run_operator(
        objective=_objective(),
        authority=_authority(),
        reasoner=ScriptedReasoner((_request_action(), _finish_action())),
        evaluator=_UnusedEvaluator(),
        evidence_provider=EvidenceToolRegistry((tool,)),
    )

    assert run.schema_version == "1.2"
    assert tool.calls == 1
    assert run.world_state is not None
    assert tuple(item.assertion_id for item in run.world_state.assertions) == (
        "battery-soc-1",
    )
    assert run.world_state.assertions[0].assertion_sha256 is not None
    assert run.steps[0].acquisition_result is not None
    assert run.steps[-1].action.conclusion_claims[0].claim_id == "battery-above-threshold"


def test_disallowed_evidence_tool_is_rejected_before_dispatch() -> None:
    tool = _TelemetryTool()
    authority = _authority(
        allowed_evidence_tools=(
            AllowedEvidenceTool(
                tool_id="different-tool",
                tool_version="1.0",
                request_kinds=("read_battery_soc",),
            ),
        )
    )

    with pytest.raises(OperatorPolicyError, match="cannot serve"):
        run_operator(
            objective=_objective(),
            authority=authority,
            reasoner=ScriptedReasoner((_request_action(),)),
            evaluator=_UnusedEvaluator(),
            evidence_provider=EvidenceToolRegistry((tool,)),
        )

    assert tool.calls == 0


def test_typed_assertions_cannot_end_in_an_uncited_free_text_conclusion() -> None:
    finish = _finish_action().model_copy(update={"conclusion_claims": ()})

    with pytest.raises(OperatorPolicyError, match="claim-backed"):
        run_operator(
            objective=_objective(),
            authority=_authority(),
            reasoner=ScriptedReasoner((_request_action(), finish)),
            evaluator=_UnusedEvaluator(),
            evidence_provider=EvidenceToolRegistry((_TelemetryTool(),)),
        )
