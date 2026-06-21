import re

from astro_assistant.models import (
    ArtifactKind,
    AstroToolName,
    AstroWorkflowPlan,
    RiskLevel,
    WorkflowArtifact,
    WorkflowStep,
)

UNSUPPORTED_PROMPT_MESSAGE = "deterministic planner currently supports the local OD demo only"


class DeterministicPlanner:
    def plan(self, prompt: str) -> AstroWorkflowPlan:
        normalized = prompt.lower()
        if not _matches_local_od_intent(normalized):
            raise ValueError(UNSUPPORTED_PROMPT_MESSAGE)
        return local_od_demo_plan(prompt)


def _matches_local_od_intent(normalized_prompt: str) -> bool:
    return (
        re.search(r"\bod\b", normalized_prompt) is not None
        or "orbit determination" in normalized_prompt
        or "orbit-determination" in normalized_prompt
    )


def local_od_demo_plan(user_intent: str) -> AstroWorkflowPlan:
    scenario_path = "examples/scenarios/leo_two_station_od.yaml"
    measurements_json = "/tmp/astro-assistant/measurements.json"
    measurements_tdm = "/tmp/astro-assistant/measurements.tdm"
    estimate_json = "/tmp/astro-assistant/estimate.json"

    return AstroWorkflowPlan(
        plan_id="local-od-demo",
        title="Local Orbit Determination Demo",
        user_intent=user_intent,
        steps=[
            WorkflowStep(
                step_id="validate_scenario",
                tool=AstroToolName.VALIDATE_SCENARIO,
                description="Validate the checked-in two-station OD scenario.",
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
