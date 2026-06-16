from __future__ import annotations

import csv
from datetime import UTC, datetime
from io import StringIO

from astro_core.models import Trajectory


def _format_oem_epoch(epoch: datetime) -> str:
    return epoch.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _format_oem_float(value: float) -> str:
    return repr(float(value))


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
