import re
from dataclasses import dataclass
from pathlib import PurePosixPath


@dataclass(frozen=True)
class ResolvedLocalODScenario:
    path: str
    scenario_id: str
    artifact_dir: str


SUPPORTED_LOCAL_OD_SCENARIOS: dict[str, str] = {
    "leo_doppler.yaml": "leo-doppler",
    "leo_geodetic_eop_table_topocentric.yaml": "leo-geodetic-eop-table-topocentric",
    "leo_geodetic_eop_topocentric.yaml": "leo-geodetic-eop-topocentric",
    "leo_geodetic_precession_nutation_topocentric.yaml": (
        "leo-geodetic-precession-nutation-topocentric"
    ),
    "leo_geodetic_topocentric.yaml": "leo-geodetic-topocentric",
    "leo_radiometric_links.yaml": "leo-radiometric-links",
    "leo_radiometric_media.yaml": "leo-radiometric-media",
    "leo_radiometric_weather_frequency.yaml": "leo-radiometric-weather-frequency",
    "leo_two_station_angles.yaml": "leo-two-station-angles",
    "leo_two_station_od.yaml": "leo-two-station-od",
    "leo_two_station_topocentric.yaml": "leo-two-station-topocentric",
}

DEFAULT_LOCAL_OD_SCENARIO = "leo_two_station_od.yaml"
SCENARIO_ROOT = PurePosixPath("examples/scenarios")
ARTIFACT_ROOT = "/tmp/astro-assistant"

_PATH_PATTERN = re.compile(r"(?P<path>(?:\.?/|/)?[\w./-]+\.ya?ml)\b", re.IGNORECASE)


def resolve_local_od_scenario(prompt: str) -> ResolvedLocalODScenario:
    normalized_prompt = prompt.lower()
    explicit_path = _extract_explicit_path(normalized_prompt)
    if explicit_path is not None:
        return _resolve_supported_path(explicit_path)

    alias_filename = _resolve_alias(normalized_prompt)
    if alias_filename is not None:
        return _resolved(alias_filename)

    if _looks_like_unknown_scenario_request(normalized_prompt):
        raise ValueError("could not resolve a supported local OD scenario from prompt")

    return _resolved(DEFAULT_LOCAL_OD_SCENARIO)


def _extract_explicit_path(normalized_prompt: str) -> str | None:
    match = _PATH_PATTERN.search(normalized_prompt)
    if match is None:
        return None
    return match.group("path").removeprefix("./")


def _resolve_supported_path(path_text: str) -> ResolvedLocalODScenario:
    path = PurePosixPath(path_text)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("scenario path must stay under examples/scenarios")
    filename = path.name
    if len(path.parts) == 1:
        if filename not in SUPPORTED_LOCAL_OD_SCENARIOS:
            raise ValueError(f"could not resolve a supported local OD scenario: {path_text}")
        return _resolved(filename)
    if not _is_under_scenario_root(path):
        raise ValueError("scenario path must stay under examples/scenarios")
    if filename not in SUPPORTED_LOCAL_OD_SCENARIOS:
        raise ValueError(f"could not resolve a supported local OD scenario: {path_text}")
    return _resolved(filename)


def _resolve_alias(normalized_prompt: str) -> str | None:
    for filename in SUPPORTED_LOCAL_OD_SCENARIOS:
        stem = filename.removesuffix(".yaml")
        aliases = {
            stem,
            stem.replace("_", " "),
            stem.removeprefix("leo_").replace("_", " "),
            SUPPORTED_LOCAL_OD_SCENARIOS[filename],
            SUPPORTED_LOCAL_OD_SCENARIOS[filename].replace("-", " "),
        }
        if any(_contains_phrase(normalized_prompt, alias) for alias in aliases):
            return filename
    return None


def _looks_like_unknown_scenario_request(normalized_prompt: str) -> bool:
    if "scenario" not in normalized_prompt:
        return False
    return "demo" not in normalized_prompt


def _contains_phrase(normalized_prompt: str, phrase: str) -> bool:
    return re.search(rf"\b{re.escape(phrase)}\b", normalized_prompt) is not None


def _is_under_scenario_root(path: PurePosixPath) -> bool:
    return len(path.parts) == 3 and PurePosixPath(*path.parts[:2]) == SCENARIO_ROOT


def _resolved(filename: str) -> ResolvedLocalODScenario:
    stem = filename.removesuffix(".yaml")
    return ResolvedLocalODScenario(
        path=str(SCENARIO_ROOT / filename),
        scenario_id=SUPPORTED_LOCAL_OD_SCENARIOS[filename],
        artifact_dir=f"{ARTIFACT_ROOT}/{stem}",
    )
