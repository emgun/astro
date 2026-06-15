from __future__ import annotations

import csv
from io import StringIO

from astro_core.models import Trajectory


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
