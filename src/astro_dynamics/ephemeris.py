from __future__ import annotations

import csv
from datetime import UTC, datetime
from io import StringIO

from astro_core.models import (
    CartesianState,
    ForceModelConfig,
    Trajectory,
    TrajectorySample,
)


def _format_oem_epoch(epoch: datetime) -> str:
    return epoch.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _format_oem_float(value: float) -> str:
    return repr(float(value))


def _parse_oem_epoch(raw_epoch: str) -> datetime:
    try:
        return datetime.fromisoformat(raw_epoch.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError as exc:
        raise ValueError(f"Invalid OEM epoch {raw_epoch!r}") from exc


def dump_trajectory_ephemeris_csv(trajectory: Trajectory) -> str:
    output = StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        [
            "scenario_id",
            "backend",
            "epoch",
            "x_km",
            "y_km",
            "z_km",
            "vx_km_s",
            "vy_km_s",
            "vz_km_s",
        ]
    )
    for sample in trajectory.samples:
        position = sample.state.position_km
        velocity = sample.state.velocity_km_s
        writer.writerow(
            [
                trajectory.scenario_id,
                trajectory.backend,
                sample.epoch.isoformat(),
                position[0],
                position[1],
                position[2],
                velocity[0],
                velocity[1],
                velocity[2],
            ]
        )
    return output.getvalue().rstrip("\n")


def dump_trajectory_oem(trajectory: Trajectory, *, originator: str = "ASTRO_SUITE") -> str:
    start_epoch = trajectory.samples[0].epoch
    stop_epoch = trajectory.samples[-1].epoch
    lines = [
        "CCSDS_OEM_VERS = 2.0",
        f"CREATION_DATE = {_format_oem_epoch(start_epoch)}",
        f"ORIGINATOR = {originator}",
        f"COMMENT scenario_id = {trajectory.scenario_id}",
        f"COMMENT backend = {trajectory.backend}",
        "META_START",
        f"OBJECT_NAME = {trajectory.scenario_id}",
        f"OBJECT_ID = {trajectory.scenario_id}",
        "CENTER_NAME = EARTH",
        "REF_FRAME = EME2000",
        "TIME_SYSTEM = UTC",
        f"START_TIME = {_format_oem_epoch(start_epoch)}",
        f"STOP_TIME = {_format_oem_epoch(stop_epoch)}",
        "META_STOP",
    ]
    for sample in trajectory.samples:
        position = sample.state.position_km
        velocity = sample.state.velocity_km_s
        values = [
            _format_oem_epoch(sample.epoch),
            *(_format_oem_float(component) for component in position),
            *(_format_oem_float(component) for component in velocity),
        ]
        lines.append(" ".join(values))
    return "\n".join(lines)


def load_trajectory_oem(payload: str, *, force_model: ForceModelConfig) -> Trajectory:
    metadata: dict[str, str] = {}
    comments: dict[str, str] = {}
    samples: list[TrajectorySample] = []
    in_metadata = False

    for raw_line in payload.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line == "META_START":
            in_metadata = True
            continue
        if line == "META_STOP":
            in_metadata = False
            continue
        if line.startswith("COMMENT "):
            raw_comment = line.removeprefix("COMMENT ")
            key, separator, value = raw_comment.partition("=")
            if separator:
                comments[key.strip()] = value.strip()
            continue
        if "=" in line:
            key, value = (component.strip() for component in line.split("=", maxsplit=1))
            if in_metadata or key in {"CCSDS_OEM_VERS", "CREATION_DATE", "ORIGINATOR"}:
                metadata[key] = value
            continue

        fields = line.split()
        if len(fields) != 7:
            raise ValueError("OEM state rows must contain epoch, 3 position, and 3 velocity fields")
        epoch = _parse_oem_epoch(fields[0])
        try:
            state_values = [float(value) for value in fields[1:]]
        except ValueError as exc:
            raise ValueError(f"Invalid numeric OEM state row: {line}") from exc
        samples.append(
            TrajectorySample(
                epoch=epoch,
                state=CartesianState(
                    position_km=tuple(state_values[:3]),
                    velocity_km_s=tuple(state_values[3:]),
                ),
            )
        )

    if metadata.get("CCSDS_OEM_VERS") != "2.0":
        raise ValueError("OEM ingest supports CCSDS_OEM_VERS = 2.0")
    if metadata.get("TIME_SYSTEM") != "UTC":
        raise ValueError("OEM ingest supports only TIME_SYSTEM = UTC")
    if metadata.get("REF_FRAME") != "EME2000":
        raise ValueError("OEM ingest supports only REF_FRAME = EME2000")
    if metadata.get("CENTER_NAME") != "EARTH":
        raise ValueError("OEM ingest supports only CENTER_NAME = EARTH")
    if not samples:
        raise ValueError("OEM ingest requires at least one state row")

    scenario_id = comments.get("scenario_id") or metadata.get("OBJECT_NAME")
    if scenario_id is None:
        raise ValueError("OEM ingest requires COMMENT scenario_id or OBJECT_NAME")

    backend = comments.get("backend", "oem")
    return Trajectory(
        scenario_id=scenario_id,
        samples=samples,
        force_model=force_model,
        backend=backend,
        metadata={
            "source_format": "ccsds_oem_kvn",
            "oem_version": metadata["CCSDS_OEM_VERS"],
            "oem_originator": metadata.get("ORIGINATOR", ""),
            "oem_object_name": metadata.get("OBJECT_NAME", ""),
            "oem_object_id": metadata.get("OBJECT_ID", ""),
            "oem_center_name": metadata["CENTER_NAME"],
            "oem_ref_frame": metadata["REF_FRAME"],
            "oem_time_system": metadata["TIME_SYSTEM"],
            "force_model_source": "scenario",
        },
    )
