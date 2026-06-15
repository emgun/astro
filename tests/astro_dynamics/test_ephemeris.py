from pathlib import Path

from astro_core.io import load_scenario
from astro_dynamics.ephemeris import dump_trajectory_ephemeris_csv
from astro_dynamics.local import propagate_local


def test_dump_trajectory_ephemeris_csv_writes_samples() -> None:
    scenario = load_scenario(Path("examples/scenarios/leo_two_body.yaml"))
    trajectory = propagate_local(scenario)

    payload = dump_trajectory_ephemeris_csv(trajectory)

    lines = payload.splitlines()
    assert lines[0] == (
        "scenario_id,backend,epoch,"
        "x_km,y_km,z_km,vx_km_s,vy_km_s,vz_km_s"
    )
    assert len(lines) == len(trajectory.samples) + 1
    assert lines[1].startswith("leo-two-body,local,2026-01-01T00:00:00+00:00,7000")
