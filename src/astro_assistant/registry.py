from collections.abc import Callable

from astro_assistant.models import AstroToolName, CommandSpec, WorkflowStep


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


_BUILDERS: dict[AstroToolName, Callable[[WorkflowStep, str | None], CommandSpec]] = {
    AstroToolName.VALIDATE_SCENARIO: _validate_scenario,
    AstroToolName.SYNTH_MEASUREMENTS: _synth_measurements,
    AstroToolName.EXPORT_MEASUREMENTS: _export_measurements,
    AstroToolName.ESTIMATE_MEASUREMENTS: _estimate_measurements,
}


def build_command_spec(step: WorkflowStep, cwd: str | None = None) -> CommandSpec:
    return _BUILDERS[step.tool](step, cwd)
