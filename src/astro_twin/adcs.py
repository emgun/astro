from __future__ import annotations

from astro_twin.models import ADCSConfig, ADCSSample, TimelineGeometrySample


def compute_adcs_timeline(
    config: ADCSConfig,
    geometry: tuple[TimelineGeometrySample, ...],
) -> tuple[ADCSSample, ...]:
    return tuple(
        ADCSSample(
            elapsed_s=sample.elapsed_s,
            pointing_error_deg=config.max_pointing_error_deg,
            pointing_margin_deg=config.pointing_requirement_deg - config.max_pointing_error_deg,
            torque_margin_n_m=config.max_torque_n_m - config.required_slew_torque_n_m,
        )
        for sample in geometry
    )
