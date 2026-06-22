import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from astro_assistant.scenarios import (
    ARTIFACT_ROOT,
    SCENARIO_ROOT,
    SUPPORTED_LOCAL_OD_SCENARIOS,
    resolve_local_od_scenario,
)
from astro_core.errors import InvalidScenarioError, NumericalConvergenceError
from astro_core.io import load_scenario
from astro_core.models import ForceModelName, Scenario
from astro_dynamics.local import propagate_local
from astro_od.estimation import estimate_initial_state
from astro_od.measurements import generate_synthetic_measurements


@dataclass(frozen=True)
class LocalODSupportReport:
    supported: bool
    code: str
    message: str
    scenario_path: str | None = None
    scenario_id: str | None = None
    artifact_dir: str | None = None


def classify_local_od_support(prompt: str) -> LocalODSupportReport:
    normalized_prompt = prompt.lower()
    if not _matches_local_od_intent(normalized_prompt):
        return LocalODSupportReport(
            supported=False,
            code="unsupported_prompt",
            message="assistant verification currently supports local OD workflow prompts only",
        )

    try:
        resolved = resolve_local_od_scenario(prompt)
    except ValueError as exc:
        scenario_path = _extract_policy_safe_scenario_path(normalized_prompt)
        if scenario_path is None:
            return LocalODSupportReport(
                supported=False,
                code="path_policy",
                message=str(exc),
            )
        return _classify_path(scenario_path)

    return LocalODSupportReport(
        supported=True,
        code="supported",
        message="scenario is supported for the local OD assistant workflow",
        scenario_path=resolved.path,
        scenario_id=resolved.scenario_id,
        artifact_dir=resolved.artifact_dir,
    )


def _classify_path(scenario_path: str) -> LocalODSupportReport:
    path = PurePosixPath(scenario_path)
    if path.name in SUPPORTED_LOCAL_OD_SCENARIOS:
        resolved = resolve_local_od_scenario(scenario_path)
        return LocalODSupportReport(
            supported=True,
            code="supported",
            message="scenario is supported for the local OD assistant workflow",
            scenario_path=resolved.path,
            scenario_id=resolved.scenario_id,
            artifact_dir=resolved.artifact_dir,
        )

    concrete_path = Path(str(path))
    try:
        scenario = load_scenario(concrete_path)
    except InvalidScenarioError as exc:
        return LocalODSupportReport(
            supported=False,
            code="invalid_scenario",
            message=str(exc),
            scenario_path=str(path),
        )

    static_report = _static_classification(scenario, str(path))
    if static_report is not None:
        return static_report

    return _probe_local_od(scenario, str(path))


def _static_classification(
    scenario: Scenario, scenario_path: str
) -> LocalODSupportReport | None:
    artifact_dir = _artifact_dir_for_path(scenario_path)
    force_model = scenario.force_model
    if (
        force_model.gravity is ForceModelName.OREKIT_HIGH_FIDELITY
        or force_model.enabled_high_fidelity_flags()
    ):
        return LocalODSupportReport(
            supported=False,
            code="optional_backend",
            message=(
                "scenario requires an optional backend or high-fidelity force model "
                "outside the local OD assistant workflow"
            ),
            scenario_path=scenario_path,
            scenario_id=scenario.scenario_id,
            artifact_dir=artifact_dir,
        )
    if force_model.gravity not in {ForceModelName.TWO_BODY, ForceModelName.J2}:
        return LocalODSupportReport(
            supported=False,
            code="unsupported_force_model",
            message=f"local OD workflow does not support force model {force_model.gravity}",
            scenario_path=scenario_path,
            scenario_id=scenario.scenario_id,
            artifact_dir=artifact_dir,
        )
    if not scenario.ground_stations:
        return LocalODSupportReport(
            supported=False,
            code="missing_measurements",
            message="local OD workflow requires at least one measurement-producing ground station",
            scenario_path=scenario_path,
            scenario_id=scenario.scenario_id,
            artifact_dir=artifact_dir,
        )
    return None


def _probe_local_od(scenario: Scenario, scenario_path: str) -> LocalODSupportReport:
    artifact_dir = _artifact_dir_for_path(scenario_path)
    try:
        trajectory = propagate_local(scenario)
        measurements = generate_synthetic_measurements(scenario, trajectory)
        if not measurements:
            return LocalODSupportReport(
                supported=False,
                code="missing_measurements",
                message="local OD workflow generated no measurements for this scenario",
                scenario_path=scenario_path,
                scenario_id=scenario.scenario_id,
                artifact_dir=artifact_dir,
            )
        estimate_initial_state(scenario, measurements, backend="local")
    except NumericalConvergenceError as exc:
        message = str(exc)
        code = (
            "rank_deficient_geometry"
            if "rank deficient" in message
            else "estimation_not_converged"
        )
        return LocalODSupportReport(
            supported=False,
            code=code,
            message=message,
            scenario_path=scenario_path,
            scenario_id=scenario.scenario_id,
            artifact_dir=artifact_dir,
        )
    except ValueError as exc:
        return LocalODSupportReport(
            supported=False,
            code="unsupported_local_model",
            message=str(exc),
            scenario_path=scenario_path,
            scenario_id=scenario.scenario_id,
            artifact_dir=artifact_dir,
        )

    return LocalODSupportReport(
        supported=True,
        code="supported",
        message="scenario is supported for the local OD assistant workflow",
        scenario_path=scenario_path,
        scenario_id=scenario.scenario_id,
        artifact_dir=artifact_dir,
    )


def _matches_local_od_intent(normalized_prompt: str) -> bool:
    has_od_intent = (
        re.search(r"\bod\b", normalized_prompt) is not None
        or "orbit determination" in normalized_prompt
        or "orbit-determination" in normalized_prompt
    )
    has_local_signal = re.search(r"\blocal\b", normalized_prompt) is not None
    return has_od_intent and has_local_signal


def _extract_policy_safe_scenario_path(normalized_prompt: str) -> str | None:
    match = re.search(r"(?P<path>(?:\.?/|/)?[\w./-]+\.ya?ml)\b", normalized_prompt)
    if match is None:
        return None
    raw_path = match.group("path").removeprefix("./")
    path = PurePosixPath(raw_path)
    if path.is_absolute() or ".." in path.parts:
        return None
    if len(path.parts) == 1:
        path = SCENARIO_ROOT / path.name
    if len(path.parts) != 3 or PurePosixPath(*path.parts[:2]) != SCENARIO_ROOT:
        return None
    return str(path)


def _artifact_dir_for_path(scenario_path: str) -> str:
    return f"{ARTIFACT_ROOT}/{PurePosixPath(scenario_path).stem}"
