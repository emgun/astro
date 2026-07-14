from collections.abc import Callable

from astro_assistant.models import (
    AstroToolName,
    CommandSpec,
    ReviewAssuranceValidationInputs,
    RunCampaignInputs,
    SummarizeCampaignInputs,
    ValidateCampaignInputs,
    VerifyAssuranceValidationInputs,
    WorkflowStep,
)


def _required_str(step: WorkflowStep, key: str) -> str:
    value = step.inputs.get(key)
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{step.step_id} requires string input {key!r}")
    return value


def _optional_str(step: WorkflowStep, key: str, default: str) -> str:
    value = step.inputs.get(key, default)
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{step.step_id} requires string input {key!r}")
    return value


def _validate_scenario(step: WorkflowStep, cwd: str | None) -> CommandSpec:
    scenario_path = _required_str(step, "scenario_path")
    return CommandSpec(
        step_id=step.step_id,
        argv=["astro", "validate", scenario_path],
        cwd=cwd,
    )


def _synth_measurements(step: WorkflowStep, cwd: str | None) -> CommandSpec:
    scenario_path = _required_str(step, "scenario_path")
    output = _required_str(step, "output")
    backend = _optional_str(step, "backend", "local")
    return CommandSpec(
        step_id=step.step_id,
        argv=[
            "astro",
            "synth-measurements",
            scenario_path,
            "--backend",
            backend,
            "--output",
            output,
        ],
        cwd=cwd,
        writes=[output],
    )


def _export_measurements(step: WorkflowStep, cwd: str | None) -> CommandSpec:
    measurements_path = _required_str(step, "measurements_path")
    output = _required_str(step, "output")
    measurement_format = _optional_str(step, "format", "tdm")
    return CommandSpec(
        step_id=step.step_id,
        argv=[
            "astro",
            "export-measurements",
            measurements_path,
            "--format",
            measurement_format,
            "--output",
            output,
        ],
        cwd=cwd,
        writes=[output],
    )


def _estimate_measurements(step: WorkflowStep, cwd: str | None) -> CommandSpec:
    scenario_path = _required_str(step, "scenario_path")
    measurements_path = _required_str(step, "measurements_path")
    output = _required_str(step, "output")
    backend = _optional_str(step, "backend", "local")
    return CommandSpec(
        step_id=step.step_id,
        argv=[
            "astro",
            "estimate-measurements",
            scenario_path,
            measurements_path,
            "--backend",
            backend,
            "--output",
            output,
        ],
        cwd=cwd,
        writes=[output],
    )


def _validate_campaign(step: WorkflowStep, cwd: str | None) -> CommandSpec:
    inputs = ValidateCampaignInputs.model_validate(step.inputs)
    return CommandSpec(
        step_id=step.step_id,
        argv=["astro", "validate-campaign", inputs.definition_path],
        cwd=cwd,
    )


def _run_campaign(step: WorkflowStep, cwd: str | None) -> CommandSpec:
    inputs = RunCampaignInputs.model_validate(step.inputs)
    argv = [
        "astro",
        "run-campaign",
        inputs.definition_path,
        "--output-dir",
        inputs.output_dir,
    ]
    if inputs.resume:
        argv.append("--resume")
    return CommandSpec(
        step_id=step.step_id,
        argv=argv,
        cwd=cwd,
        writes=[inputs.output_dir],
    )


def _summarize_campaign(step: WorkflowStep, cwd: str | None) -> CommandSpec:
    inputs = SummarizeCampaignInputs.model_validate(step.inputs)
    return CommandSpec(
        step_id=step.step_id,
        argv=["astro", "summarize-campaign", inputs.output_dir],
        cwd=cwd,
    )


def _verify_assurance_validation(step: WorkflowStep, cwd: str | None) -> CommandSpec:
    inputs = VerifyAssuranceValidationInputs.model_validate(step.inputs)
    return CommandSpec(
        step_id=step.step_id,
        argv=["astro", "verify-assurance-validation", inputs.result_path],
        cwd=cwd,
    )


def _review_assurance_validation(step: WorkflowStep, cwd: str | None) -> CommandSpec:
    inputs = ReviewAssuranceValidationInputs.model_validate(step.inputs)
    argv = [
        "astro",
        "review-assurance-validation",
        inputs.result_path,
        "--output",
        inputs.output,
    ]
    writes = [inputs.output]
    if inputs.summary_output is not None:
        argv.extend(["--summary-output", inputs.summary_output])
        writes.append(inputs.summary_output)
    return CommandSpec(step_id=step.step_id, argv=argv, cwd=cwd, writes=writes)


_BUILDERS: dict[AstroToolName, Callable[[WorkflowStep, str | None], CommandSpec]] = {
    AstroToolName.VALIDATE_SCENARIO: _validate_scenario,
    AstroToolName.SYNTH_MEASUREMENTS: _synth_measurements,
    AstroToolName.EXPORT_MEASUREMENTS: _export_measurements,
    AstroToolName.ESTIMATE_MEASUREMENTS: _estimate_measurements,
    AstroToolName.VALIDATE_CAMPAIGN: _validate_campaign,
    AstroToolName.RUN_CAMPAIGN: _run_campaign,
    AstroToolName.SUMMARIZE_CAMPAIGN: _summarize_campaign,
    AstroToolName.VERIFY_ASSURANCE_VALIDATION: _verify_assurance_validation,
    AstroToolName.REVIEW_ASSURANCE_VALIDATION: _review_assurance_validation,
}


def build_command_spec(step: WorkflowStep, cwd: str | None = None) -> CommandSpec:
    try:
        builder = _BUILDERS[step.tool]
    except KeyError as exc:
        raise ValueError(f"Unsupported Astro tool {step.tool!r}") from exc
    return builder(step, cwd)
