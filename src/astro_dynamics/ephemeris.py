from __future__ import annotations

import csv
from datetime import UTC, datetime
from io import StringIO
from typing import cast

from astro_core.models import (
    AttitudeState,
    CartesianState,
    CovarianceSample,
    ForceModelConfig,
    Frame,
    Quaternion4,
    Trajectory,
    TrajectorySample,
)

OPM_COVARIANCE_ENTRIES: tuple[tuple[int, int, str], ...] = (
    (0, 0, "CX_X"),
    (1, 0, "CY_X"),
    (1, 1, "CY_Y"),
    (2, 0, "CZ_X"),
    (2, 1, "CZ_Y"),
    (2, 2, "CZ_Z"),
    (3, 0, "CX_DOT_X"),
    (3, 1, "CX_DOT_Y"),
    (3, 2, "CX_DOT_Z"),
    (3, 3, "CX_DOT_X_DOT"),
    (4, 0, "CY_DOT_X"),
    (4, 1, "CY_DOT_Y"),
    (4, 2, "CY_DOT_Z"),
    (4, 3, "CY_DOT_X_DOT"),
    (4, 4, "CY_DOT_Y_DOT"),
    (5, 0, "CZ_DOT_X"),
    (5, 1, "CZ_DOT_Y"),
    (5, 2, "CZ_DOT_Z"),
    (5, 3, "CZ_DOT_X_DOT"),
    (5, 4, "CZ_DOT_Y_DOT"),
    (5, 5, "CZ_DOT_Z_DOT"),
)


def _format_oem_epoch(epoch: datetime) -> str:
    return epoch.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _format_oem_float(value: float) -> str:
    return repr(float(value))


def _format_aem_float(value: float) -> str:
    return repr(float(value))


def _identity_matrix(size: int = 6) -> list[list[float]]:
    return [[1.0 if row == column else 0.0 for column in range(size)] for row in range(size)]


def _parse_oem_epoch(raw_epoch: str) -> datetime:
    try:
        return datetime.fromisoformat(raw_epoch.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError as exc:
        raise ValueError(f"Invalid OEM epoch {raw_epoch!r}") from exc


def _parse_kvn_assignment(line: str) -> tuple[str, str]:
    key, separator, value = line.partition("=")
    if not separator:
        raise ValueError(f"Invalid KVN assignment line: {line}")
    return key.strip(), value.strip()


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


def dump_trajectory_opm(trajectory: Trajectory, *, originator: str = "ASTRO_SUITE") -> str:
    """Export the first trajectory sample as a CCSDS OPM KVN orbit state message."""
    sample = trajectory.samples[0]
    position = sample.state.position_km
    velocity = sample.state.velocity_km_s
    lines = [
        "CCSDS_OPM_VERS = 2.0",
        f"CREATION_DATE = {_format_oem_epoch(sample.epoch)}",
        f"ORIGINATOR = {originator}",
        f"COMMENT scenario_id = {trajectory.scenario_id}",
        f"COMMENT backend = {trajectory.backend}",
        "META_START",
        f"OBJECT_NAME = {trajectory.scenario_id}",
        f"OBJECT_ID = {trajectory.scenario_id}",
        "CENTER_NAME = EARTH",
        "REF_FRAME = EME2000",
        "TIME_SYSTEM = UTC",
        "META_STOP",
        f"EPOCH = {_format_oem_epoch(sample.epoch)}",
        f"X = {_format_oem_float(position[0])}",
        f"Y = {_format_oem_float(position[1])}",
        f"Z = {_format_oem_float(position[2])}",
        f"X_DOT = {_format_oem_float(velocity[0])}",
        f"Y_DOT = {_format_oem_float(velocity[1])}",
        f"Z_DOT = {_format_oem_float(velocity[2])}",
    ]
    if sample.mass_kg is not None:
        lines.append(f"MASS = {_format_oem_float(sample.mass_kg)}")
    covariance_sample = next(
        (
            covariance
            for covariance in trajectory.covariance_history
            if covariance.epoch == sample.epoch
        ),
        None,
    )
    if covariance_sample is not None:
        lines.extend(
            [
                "COVARIANCE_START",
                "COV_REF_FRAME = EME2000",
            ]
        )
        covariance = covariance_sample.covariance
        for row_index, column_index, key in OPM_COVARIANCE_ENTRIES:
            lines.append(f"{key} = {_format_oem_float(covariance[row_index][column_index])}")
        lines.append("COVARIANCE_STOP")
    return "\n".join(lines)


def dump_trajectory_aem(trajectory: Trajectory, *, originator: str = "ASTRO_SUITE") -> str:
    attitude_samples = [sample for sample in trajectory.samples if sample.attitude is not None]
    if not attitude_samples:
        raise ValueError("AEM export requires at least one trajectory sample with attitude samples")

    start_epoch = attitude_samples[0].epoch
    stop_epoch = attitude_samples[-1].epoch
    lines = [
        "CCSDS_AEM_VERS = 2.0",
        f"CREATION_DATE = {_format_oem_epoch(start_epoch)}",
        f"ORIGINATOR = {originator}",
        f"COMMENT scenario_id = {trajectory.scenario_id}",
        f"COMMENT backend = {trajectory.backend}",
        "COMMENT quaternion_order = QC Q1 Q2 Q3",
        "META_START",
        f"OBJECT_NAME = {trajectory.scenario_id}",
        f"OBJECT_ID = {trajectory.scenario_id}",
        "CENTER_NAME = EARTH",
        "REF_FRAME_A = EME2000",
        "REF_FRAME_B = SC_BODY_1",
        "TIME_SYSTEM = UTC",
        f"START_TIME = {_format_oem_epoch(start_epoch)}",
        f"USEABLE_START_TIME = {_format_oem_epoch(start_epoch)}",
        f"USEABLE_STOP_TIME = {_format_oem_epoch(stop_epoch)}",
        f"STOP_TIME = {_format_oem_epoch(stop_epoch)}",
        "ATTITUDE_TYPE = QUATERNION",
        "META_STOP",
        "DATA_START",
    ]
    for sample in attitude_samples:
        assert sample.attitude is not None
        quaternion = sample.attitude.body_to_inertial_quaternion
        values = [
            _format_oem_epoch(sample.epoch),
            *(_format_aem_float(component) for component in quaternion),
        ]
        lines.append(" ".join(values))
    lines.append("DATA_STOP")
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
                    position_km=(state_values[0], state_values[1], state_values[2]),
                    velocity_km_s=(state_values[3], state_values[4], state_values[5]),
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


def load_trajectory_opm(payload: str, *, force_model: ForceModelConfig) -> Trajectory:
    metadata: dict[str, str] = {}
    comments: dict[str, str] = {}
    state_values: dict[str, str] = {}
    covariance_metadata: dict[str, str] = {}
    covariance_values: dict[str, str] = {}
    in_metadata = False
    in_covariance = False

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
        if line == "COVARIANCE_START":
            if in_covariance:
                raise ValueError("OPM covariance block has nested COVARIANCE_START")
            in_covariance = True
            continue
        if line == "COVARIANCE_STOP":
            if not in_covariance:
                raise ValueError("OPM covariance block has COVARIANCE_STOP without start")
            in_covariance = False
            continue
        if line.startswith("COMMENT "):
            raw_comment = line.removeprefix("COMMENT ")
            key, separator, value = raw_comment.partition("=")
            if separator:
                comments[key.strip()] = value.strip()
            continue
        if "=" in line:
            key, value = _parse_kvn_assignment(line)
            if in_covariance:
                if key == "COV_REF_FRAME":
                    covariance_metadata[key] = value
                else:
                    covariance_values[key] = value
            elif in_metadata or key in {
                "CCSDS_OPM_VERS",
                "CREATION_DATE",
                "ORIGINATOR",
            }:
                metadata[key] = value
            else:
                state_values[key] = value
            continue

        raise ValueError(f"Invalid OPM KVN line: {line}")

    if in_covariance:
        raise ValueError("OPM covariance block ended before COVARIANCE_STOP")
    if metadata.get("CCSDS_OPM_VERS") != "2.0":
        raise ValueError("OPM ingest supports CCSDS_OPM_VERS = 2.0")
    if metadata.get("TIME_SYSTEM") != "UTC":
        raise ValueError("OPM ingest supports only TIME_SYSTEM = UTC")
    if metadata.get("REF_FRAME") != "EME2000":
        raise ValueError("OPM ingest supports only REF_FRAME = EME2000")
    if metadata.get("CENTER_NAME") != "EARTH":
        raise ValueError("OPM ingest supports only CENTER_NAME = EARTH")

    required_state_keys = {"EPOCH", "X", "Y", "Z", "X_DOT", "Y_DOT", "Z_DOT"}
    missing_state_keys = sorted(required_state_keys - set(state_values))
    if missing_state_keys:
        raise ValueError("OPM ingest missing state keys: " + ", ".join(missing_state_keys))

    try:
        epoch = _parse_oem_epoch(state_values["EPOCH"])
        position = (
            float(state_values["X"]),
            float(state_values["Y"]),
            float(state_values["Z"]),
        )
        velocity = (
            float(state_values["X_DOT"]),
            float(state_values["Y_DOT"]),
            float(state_values["Z_DOT"]),
        )
        mass_kg = float(state_values["MASS"]) if "MASS" in state_values else None
    except ValueError as exc:
        raise ValueError("Invalid numeric OPM state value") from exc

    covariance_history: list[CovarianceSample] = []
    if covariance_values:
        covariance_frame = covariance_metadata.get("COV_REF_FRAME")
        if covariance_frame != "EME2000":
            raise ValueError("OPM covariance ingest supports only COV_REF_FRAME = EME2000")
        expected_covariance_keys = {key for _row, _column, key in OPM_COVARIANCE_ENTRIES}
        missing_covariance_keys = sorted(expected_covariance_keys - set(covariance_values))
        if missing_covariance_keys:
            raise ValueError(
                "OPM covariance block missing keys: " + ", ".join(missing_covariance_keys)
            )
        covariance_matrix = [[0.0 for _column in range(6)] for _row in range(6)]
        try:
            for row_index, column_index, key in OPM_COVARIANCE_ENTRIES:
                covariance_value = float(covariance_values[key])
                covariance_matrix[row_index][column_index] = covariance_value
                covariance_matrix[column_index][row_index] = covariance_value
        except ValueError as exc:
            raise ValueError("Invalid numeric OPM covariance value") from exc
        covariance_history.append(
            CovarianceSample(
                epoch=epoch,
                covariance=covariance_matrix,
                state_transition_matrix=_identity_matrix(),
                accumulated_state_transition_matrix=_identity_matrix(),
                metadata={
                    "source_format": "ccsds_opm_kvn",
                    "covariance_model": "imported_opm_single_epoch",
                    "covariance_reference_frame": covariance_frame,
                    "covariance_state_order": "X Y Z X_DOT Y_DOT Z_DOT",
                    "covariance_state_units": "km_and_km_per_s",
                    "state_transition_model": "identity",
                },
            )
        )

    scenario_id = comments.get("scenario_id") or metadata.get("OBJECT_NAME")
    if scenario_id is None:
        raise ValueError("OPM ingest requires COMMENT scenario_id or OBJECT_NAME")

    backend = comments.get("backend", "opm")
    return Trajectory(
        scenario_id=scenario_id,
        samples=[
            TrajectorySample(
                epoch=epoch,
                state=CartesianState(position_km=position, velocity_km_s=velocity),
                mass_kg=mass_kg,
            )
        ],
        force_model=force_model,
        backend=backend,
        covariance_history=covariance_history,
        metadata={
            "source_format": "ccsds_opm_kvn",
            "opm_version": metadata["CCSDS_OPM_VERS"],
            "opm_originator": metadata.get("ORIGINATOR", ""),
            "opm_object_name": metadata.get("OBJECT_NAME", ""),
            "opm_object_id": metadata.get("OBJECT_ID", ""),
            "opm_center_name": metadata["CENTER_NAME"],
            "opm_ref_frame": metadata["REF_FRAME"],
            "opm_time_system": metadata["TIME_SYSTEM"],
            "opm_state_units": "km_and_km_per_s",
            "opm_covariance_sample_count": len(covariance_history),
            "force_model_source": "scenario",
        },
    )


def load_trajectory_aem(payload: str, *, base_trajectory: Trajectory) -> Trajectory:
    metadata: dict[str, str] = {}
    comments: dict[str, str] = {}
    attitude_by_epoch: dict[datetime, AttitudeState] = {}
    in_metadata = False
    in_data = False

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
        if line == "DATA_START":
            in_data = True
            continue
        if line == "DATA_STOP":
            in_data = False
            continue
        if line.startswith("COMMENT "):
            raw_comment = line.removeprefix("COMMENT ")
            key, separator, value = raw_comment.partition("=")
            if separator:
                comments[key.strip()] = value.strip()
            continue
        if "=" in line:
            key, value = _parse_kvn_assignment(line)
            if in_metadata or key in {"CCSDS_AEM_VERS", "CREATION_DATE", "ORIGINATOR"}:
                metadata[key] = value
            continue

        if not in_data:
            raise ValueError("AEM attitude rows must appear between DATA_START and DATA_STOP")

        fields = line.split()
        if len(fields) != 5:
            raise ValueError("AEM quaternion rows must contain epoch and 4 quaternion fields")
        epoch = _parse_oem_epoch(fields[0])
        try:
            quaternion_values = tuple(float(value) for value in fields[1:])
        except ValueError as exc:
            raise ValueError(f"Invalid numeric AEM quaternion row: {line}") from exc
        if len(quaternion_values) != 4:
            raise ValueError("AEM quaternion rows must contain 4 quaternion fields")
        quaternion = cast(Quaternion4, quaternion_values)
        attitude_by_epoch[epoch] = AttitudeState(
            mode="inertial",
            frame=Frame.EME2000,
            body_to_inertial_quaternion=quaternion,
            metadata={
                "source_format": "ccsds_aem_kvn",
                "aem_attitude_type": metadata.get("ATTITUDE_TYPE", ""),
                "aem_ref_frame_a": metadata.get("REF_FRAME_A", ""),
                "aem_ref_frame_b": metadata.get("REF_FRAME_B", ""),
                "quaternion_order": comments.get("quaternion_order", "QC Q1 Q2 Q3"),
            },
        )

    if metadata.get("CCSDS_AEM_VERS") != "2.0":
        raise ValueError("AEM ingest supports CCSDS_AEM_VERS = 2.0")
    if metadata.get("TIME_SYSTEM") != "UTC":
        raise ValueError("AEM ingest supports only TIME_SYSTEM = UTC")
    if metadata.get("REF_FRAME_A") != "EME2000":
        raise ValueError("AEM ingest supports only REF_FRAME_A = EME2000")
    if metadata.get("ATTITUDE_TYPE") != "QUATERNION":
        raise ValueError("AEM ingest supports only ATTITUDE_TYPE = QUATERNION")
    if comments.get("quaternion_order", "QC Q1 Q2 Q3") != "QC Q1 Q2 Q3":
        raise ValueError("AEM ingest supports only quaternion_order = QC Q1 Q2 Q3")
    if not attitude_by_epoch:
        raise ValueError("AEM ingest requires at least one quaternion row")

    base_epochs = {sample.epoch for sample in base_trajectory.samples}
    unmatched_epochs = sorted(set(attitude_by_epoch) - base_epochs)
    if unmatched_epochs:
        raise ValueError(
            "AEM attitude rows must match base trajectory sample epochs; "
            f"unmatched epoch {unmatched_epochs[0].isoformat()}"
        )

    samples = [
        sample.model_copy(update={"attitude": attitude_by_epoch.get(sample.epoch)})
        if sample.epoch in attitude_by_epoch
        else sample
        for sample in base_trajectory.samples
    ]
    return base_trajectory.model_copy(
        update={
            "samples": samples,
            "metadata": {
                **base_trajectory.metadata,
                "attitude_source_format": "ccsds_aem_kvn",
                "aem_version": metadata["CCSDS_AEM_VERS"],
                "aem_originator": metadata.get("ORIGINATOR", ""),
                "aem_object_name": metadata.get("OBJECT_NAME", ""),
                "aem_object_id": metadata.get("OBJECT_ID", ""),
                "aem_ref_frame_a": metadata["REF_FRAME_A"],
                "aem_ref_frame_b": metadata.get("REF_FRAME_B", ""),
                "aem_time_system": metadata["TIME_SYSTEM"],
                "aem_attitude_type": metadata["ATTITUDE_TYPE"],
                "aem_attitude_sample_count": len(attitude_by_epoch),
            },
        }
    )
