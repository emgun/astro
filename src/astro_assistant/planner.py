import re

from astro_assistant.models import (
    ArtifactKind,
    AstroToolName,
    AstroWorkflowPlan,
    RiskLevel,
    WorkflowArtifact,
    WorkflowStep,
)
from astro_assistant.scenarios import resolve_local_od_scenario

UNSUPPORTED_PROMPT_MESSAGE = "deterministic planner currently supports the local OD demo only"


class DeterministicPlanner:
    def plan(self, prompt: str) -> AstroWorkflowPlan:
        normalized = prompt.lower()
        if not _matches_local_od_intent(normalized):
            raise ValueError(UNSUPPORTED_PROMPT_MESSAGE)
        return local_od_demo_plan(prompt)


def _matches_local_od_intent(normalized_prompt: str) -> bool:
    has_od_intent = (
        re.search(r"\bod\b", normalized_prompt) is not None
        or "orbit determination" in normalized_prompt
        or "orbit-determination" in normalized_prompt
    )
    has_local_demo_signal = re.search(r"\b(local|demo)\b", normalized_prompt) is not None

    return has_od_intent and has_local_demo_signal


def local_od_demo_plan(user_intent: str) -> AstroWorkflowPlan:
    scenario = resolve_local_od_scenario(user_intent)
    scenario_path = scenario.path
    measurements_json = f"{scenario.artifact_dir}/measurements.json"
    measurements_tdm = f"{scenario.artifact_dir}/measurements.tdm"
    estimate_json = f"{scenario.artifact_dir}/estimate.json"

    return AstroWorkflowPlan(
        plan_id="local-od-demo",
        title=f"Local Orbit Determination Demo: {scenario.scenario_id}",
        user_intent=user_intent,
        steps=[
            WorkflowStep(
                step_id="validate_scenario",
                tool=AstroToolName.VALIDATE_SCENARIO,
                description=f"Validate the checked-in scenario {scenario.scenario_id}.",
                inputs={"scenario_path": scenario_path},
                outputs=[WorkflowArtifact(path=scenario_path, kind=ArtifactKind.SCENARIO)],
                risk=RiskLevel.READ_ONLY,
            ),
            WorkflowStep(
                step_id="synth_measurements",
                tool=AstroToolName.SYNTH_MEASUREMENTS,
                description="Generate local synthetic range/range-rate measurements.",
                inputs={
                    "scenario_path": scenario_path,
                    "backend": "local",
                    "output": measurements_json,
                },
                outputs=[
                    WorkflowArtifact(path=measurements_json, kind=ArtifactKind.MEASUREMENTS_JSON)
                ],
                risk=RiskLevel.WRITES_ARTIFACTS,
            ),
            WorkflowStep(
                step_id="export_measurements_tdm",
                tool=AstroToolName.EXPORT_MEASUREMENTS,
                description="Export generated measurements to CCSDS TDM-style KVN.",
                inputs={
                    "measurements_path": measurements_json,
                    "format": "tdm",
                    "output": measurements_tdm,
                },
                outputs=[
                    WorkflowArtifact(path=measurements_tdm, kind=ArtifactKind.MEASUREMENTS_TDM)
                ],
                risk=RiskLevel.WRITES_ARTIFACTS,
            ),
            WorkflowStep(
                step_id="estimate_state",
                tool=AstroToolName.ESTIMATE_MEASUREMENTS,
                description="Estimate the initial state from the generated measurements.",
                inputs={
                    "scenario_path": scenario_path,
                    "measurements_path": measurements_json,
                    "backend": "local",
                    "output": estimate_json,
                },
                outputs=[WorkflowArtifact(path=estimate_json, kind=ArtifactKind.ESTIMATE_JSON)],
                risk=RiskLevel.WRITES_ARTIFACTS,
            ),
        ],
    )
