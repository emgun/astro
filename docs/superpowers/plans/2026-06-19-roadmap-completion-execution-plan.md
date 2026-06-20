# Roadmap Completion Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the remaining Astro Suite roadmap gaps through conservative, verified slices that improve suite-owned products first and keep standards-grade/live-backend claims explicit.

**Architecture:** Keep `astro_core` product models and public CLI outputs as the stable boundary. Treat local deterministic workflows as always-on release gates, optional engines as runtime-gated adapters, and official standards-grade integrations as separate validation campaigns. Do not let bridge formats, screening products, or research backends imply operational certification.

**Tech Stack:** Python 3.12, Pydantic v2, NumPy, SciPy, Typer, pytest, ruff, mypy, optional Orekit/RocketPy/Dymos/Tudat/JAX extras, Markdown docs.

---

## Execution Contract

Task: continue implementation toward the full flight-dynamics roadmap.

Done condition for this plan: each task lands as a small verified commit, the roadmap and validation docs distinguish implemented product surfaces from remaining standards/live-validation work, and the required local gates stay green.

Mutable files:

- `docs/validation-matrix.md`
- `docs/release-checklist.md`
- `docs/superpowers/plans/2026-06-15-roadmap-goals-implementation-plan.md`
- `README.md`
- `src/astro_od/calibration.py`
- `tests/astro_od/test_calibration.py`
- `tests/astro_cli/test_cli.py`
- Follow-on task files listed below.

Fixed files:

- Do not rewrite the core scenario schema outside task scope.
- Do not rename existing public commands unless a task explicitly says to.
- Do not remove optional backend gates because a local machine lacks the runtime.

Validation:

- Focused tests per task.
- `python -m ruff check ...` on changed Python files.
- `python -m mypy` for typed model/API changes.
- `git diff --check`.
- `python -m pytest -q` before a commit that claims roadmap progress.

Stop conditions:

- Official standards-grade DSN ODF/TNF decoding requires public format details and fixtures before implementation.
- Production-grade live validation for Orekit/Tudat/RocketPy/Dymos requires the matching runtime and data; if absent, add product-boundary tests and record the live gate as unavailable, not complete.
- If a task would broaden product claims beyond implemented behavior, update docs to narrow the claim before adding features.

## Roadmap Waves

1. **Truth-boundary cleanup:** split always-on release gates from optional/live validation so progress reporting is honest.
2. **OD/DSN product hardening:** improve existing suite-owned calibration and bridge products without claiming native standards processing.
3. **Screening evidence pack:** strengthen covariance, ACS, and conjunction screening assertions while preserving non-production labels.
4. **Backend live campaigns:** run or document unavailable live gates for Orekit, TudatPy, RocketPy, Dymos, and JAX.
5. **Release readiness:** package/build checks, final full-suite validation, and a roadmap status audit.

## Task 1: Validation Boundary Cleanup

**Files:**

- Modify: `docs/validation-matrix.md`
- Modify: `docs/release-checklist.md`
- Modify: `docs/superpowers/plans/2026-06-15-roadmap-goals-implementation-plan.md`

- [x] **Step 1: Split the validation matrix into local required gates and optional/backend gates**

Move rows that require optional runtimes out of the `Always-On Gates` table. Keep rows such as local propagation, local covariance, local OD, DSN bridge ingest tests, station calibration, local launch, and local research propagation in required gates. Move these rows under optional backend gates or reference tolerance sections:

```text
Orekit high-fidelity propagation
Tudat high-fidelity propagation
Orekit high-fidelity covariance
JAX OD sensitivity product
JAX research OD estimate
JAX research force flags
```

Expected wording:

```text
## Required Local Gates

These gates must run without optional backend runtimes.

## Optional Backend Gates

These gates run only when the matching runtime is installed. If a runtime is absent, capture the structured smoke-test or UnsupportedBackendError diagnostics instead of marking the capability complete.
```

- [x] **Step 2: Update release checklist language**

Keep `Required Local Gates` limited to commands that should run in a local install with default dependencies. Keep optional backend commands under `Optional Backend Gates`. Add this sentence below the optional heading:

```text
If a backend runtime is intentionally absent, capture the smoke command's structured unavailable JSON and treat the gate as not-run rather than failed or complete.
```

- [x] **Step 3: Update roadmap status language**

In `docs/superpowers/plans/2026-06-15-roadmap-goals-implementation-plan.md`, revise the `Still roadmap-level` bullets so they separate:

```text
suite-owned bridge/product hardening
standards-grade official decoder/estimator work
live backend validation campaigns
```

Keep these claims explicit:

```text
ASTRODSN1 is a suite-owned bridge, not a NASA binary ODF/TNF decoder.
Deterministic ACS products are screening primitives, not flight-qualified actuator/sensor models.
JAX OD workflows are research products, not operational differentiable OD services.
```

- [x] **Step 4: Verify docs**

Run:

```bash
git diff --check
```

Expected: no whitespace errors.

- [x] **Step 5: Commit**

Run:

```bash
git add docs/validation-matrix.md docs/release-checklist.md docs/superpowers/plans/2026-06-15-roadmap-goals-implementation-plan.md docs/superpowers/plans/2026-06-19-roadmap-completion-execution-plan.md
git commit -m "docs: clarify roadmap validation boundaries"
```

## Task 2: Station Calibration Residual Diagnostics

**Files:**

- Modify: `src/astro_od/calibration.py`
- Modify: `tests/astro_od/test_calibration.py`
- Modify: `tests/astro_cli/test_cli.py`
- Modify: `README.md`
- Modify: `docs/validation-matrix.md`
- Modify: `docs/superpowers/plans/2026-06-15-roadmap-goals-implementation-plan.md`

- [x] **Step 1: Add failing model/test assertions**

Extend `tests/astro_od/test_calibration.py::test_generate_station_calibration_product_summarizes_station_biases` with these assertions:

```python
assert product.measurement_count == 3
assert product.uncalibrated_measurement_count == 0
assert product.truth_metadata_key == "truth"
assert product.metadata["calibration_scope"] == "measurement_residual_summary"
assert product.metadata["grouping_keys"] == ["observer", "measurement_type", "units"]
assert product.metadata["residual_definition"] == "measurement_value_minus_truth_metadata"

assert dss14.bias_min == pytest.approx(0.15)
assert dss14.bias_max == pytest.approx(0.25)
assert dss14.bias_abs_mean == pytest.approx(0.2)
assert dss14.bias_std == pytest.approx(0.05)
assert dss14.sigma_mean == pytest.approx(0.5)
assert dss14.sigma_min == pytest.approx(0.5)
assert dss14.sigma_max == pytest.approx(0.5)
assert dss14.normalized_bias_rms == pytest.approx(0.41231056256176607)
```

Add a second test with one record lacking `truth` metadata:

```python
def test_generate_station_calibration_product_counts_uncalibrated_records() -> None:
    records = _station_calibration_records()
    records.append(
        MeasurementRecord(
            measurement_type=MeasurementType.RANGE,
            epoch="2026-01-01T00:02:00+00:00",
            observer="DSS-14",
            observed_object="demo-sat",
            value=12.0,
            sigma=0.5,
            units="km",
            metadata={},
        )
    )

    product = generate_station_calibration_product_from_measurements("dsn-demo", records)

    assert product.measurement_count == 3
    assert product.uncalibrated_measurement_count == 1
    assert product.metadata["source_measurement_count"] == 4
    assert product.metadata["calibrated_measurement_count"] == 3
```

- [x] **Step 2: Run the focused failing test**

Run:

```bash
python -m pytest tests/astro_od/test_calibration.py::test_generate_station_calibration_product_summarizes_station_biases tests/astro_od/test_calibration.py::test_generate_station_calibration_product_counts_uncalibrated_records -q
```

Expected before implementation: failure for missing `uncalibrated_measurement_count`, `truth_metadata_key`, and new entry fields.

- [x] **Step 3: Implement deterministic residual diagnostics**

Add these fields to `StationCalibrationEntry`:

```python
bias_std: FiniteFloat = Field(ge=0.0)
bias_abs_mean: FiniteFloat = Field(ge=0.0)
sigma_min: FiniteFloat = Field(gt=0.0)
sigma_max: FiniteFloat = Field(gt=0.0)
normalized_bias_rms: FiniteFloat = Field(ge=0.0)
```

Add these fields to `StationCalibrationProduct`:

```python
uncalibrated_measurement_count: int = Field(ge=0)
truth_metadata_key: str = "truth"
```

Use these formulas in `_station_calibration_entry`:

```python
bias_rms = sqrt(fmean([residual * residual for residual in residuals]))
return StationCalibrationEntry(
    ...
    bias_rms=bias_rms,
    bias_std=sqrt(fmean([(residual - bias_mean) ** 2 for residual in residuals])),
    bias_abs_mean=fmean([abs(residual) for residual in residuals]),
    sigma_min=min(sigmas),
    sigma_max=max(sigmas),
    normalized_bias_rms=bias_rms / sigma_mean,
)
```

Use this metadata in `generate_station_calibration_product_from_measurements`:

```python
product_metadata: dict[str, Any] = {
    "workflow": "station_calibration",
    "calibration_reference": "measurement_metadata_truth",
    "calibration_scope": "measurement_residual_summary",
    "calibration_limitations": [
        "requires suite measurement truth metadata",
        "does not solve station coordinates",
        "does not solve station clock or media calibration parameters",
        "does not decode native NASA ODF/TNF station calibration records",
    ],
    "grouping_keys": ["observer", "measurement_type", "units"],
    "residual_definition": "measurement_value_minus_truth_metadata",
    "source_measurement_count": len(measurement_records),
    "calibrated_measurement_count": calibrated_measurement_count,
}
```

Set:

```python
uncalibrated_measurement_count=len(measurement_records) - calibrated_measurement_count
```

- [x] **Step 4: Extend CLI assertions**

In `tests/astro_cli/test_cli.py::test_station_calibration_command_writes_json`, add:

```python
assert payload["truth_metadata_key"] == "truth"
assert payload["uncalibrated_measurement_count"] == 0
assert payload["metadata"]["calibration_scope"] == "measurement_residual_summary"
assert payload["metadata"]["grouping_keys"] == ["observer", "measurement_type", "units"]
assert payload["entries"][0]["normalized_bias_rms"] >= 0.0
```

- [x] **Step 5: Update docs**

Update `README.md` station calibration wording to:

```text
`astro station-calibration` summarizes per-station/per-measurement-type residual biases from suite measurement records carrying `truth` metadata. It reports grouped residual statistics and sigma-normalized diagnostics for auditability; it does not solve station coordinates, clock terms, media parameters, or native NASA ODF/TNF station calibration records.
```

Update `docs/validation-matrix.md` station calibration row to mention:

```text
mean/min/max/RMS, population standard deviation, mean absolute residual, sigma min/mean/max, normalized mean/RMS bias, calibrated/uncalibrated counts, and explicit non-native DSN limitations.
```

- [x] **Step 6: Verify**

Run:

```bash
python -m pytest tests/astro_od/test_calibration.py tests/astro_cli/test_cli.py::test_station_calibration_command_writes_json -q
python -m ruff check src/astro_od/calibration.py tests/astro_od/test_calibration.py tests/astro_cli/test_cli.py
python -m mypy
git diff --check
```

Expected: all pass.

- [x] **Step 7: Commit**

Run:

```bash
git add src/astro_od/calibration.py tests/astro_od/test_calibration.py tests/astro_cli/test_cli.py README.md docs/validation-matrix.md docs/superpowers/plans/2026-06-15-roadmap-goals-implementation-plan.md docs/superpowers/plans/2026-06-19-roadmap-completion-execution-plan.md
git commit -m "feat: add station calibration residual diagnostics"
```

## Task 3: DSN Bridge Rejection and Provenance Hardening

**Files:**

- Inspect: `src/astro_od/dsn.py`
- Inspect: `src/astro_od/io.py`
- Modify: `tests/astro_od/test_dsn_tracking.py`
- Modify as needed: `src/astro_od/dsn.py`
- Modify as needed: `docs/validation-matrix.md`

- [x] **Step 1: Identify current parser failure modes**

Run:

```bash
python -m pytest tests/astro_od/test_dsn_tracking.py -q
```

Expected: existing DSN bridge tests pass.

- [x] **Step 2: Add bridge-specific rejection tests**

Add tests that prove malformed bridge records fail with actionable errors:

```python
def test_load_dsn_binary_tracking_measurements_rejects_bad_magic(tmp_path: Path) -> None:
    path = tmp_path / "bad.bin"
    path.write_bytes(b"NOTODF01")

    with pytest.raises(InvalidMeasurementFileError, match="ASTRODSN1"):
        load_dsn_binary_tracking_measurements(path)
```

Use the actual loader names from `src/astro_od/dsn.py`; do not invent public APIs.

- [x] **Step 3: Ensure source-format metadata stays explicit**

Check that imported records include:

```python
record.metadata["source_format"]
record.metadata["tracking_format"]
```

Expected values must include the word `bridge` or otherwise clearly identify suite-owned interchange.

- [x] **Step 4: Verify**

Run:

```bash
python -m pytest tests/astro_od/test_dsn_tracking.py tests/astro_cli/test_cli.py::test_import_dsn_binary_tracking_command_writes_measurement_json -q
python -m ruff check src/astro_od/dsn.py tests/astro_od/test_dsn_tracking.py tests/astro_cli/test_cli.py
python -m mypy
git diff --check
```

Expected: all pass.

## Task 4: Screening Evidence Pack

**Files:**

- Inspect: `src/astro_dynamics/attitude.py`
- Inspect: `src/astro_dynamics/conjunction.py`
- Inspect: `src/astro_dynamics/local.py`
- Modify tests in `tests/astro_dynamics/`
- Modify docs in `docs/validation-matrix.md`

- [x] **Step 1: Add or tighten invariant tests**

Target deterministic product properties:

```text
covariance matrices stay symmetric within tolerance
ACS status reports tolerance miss/within_tolerance deterministically
ACS actuator screening reports commanded-vs-applied torque tracking error plus actuator saturation
and deadband fractions
conjunction assessment distinguishes screening-only from operational-candidate
```

- [x] **Step 2: Verify local screening only**

Run:

```bash
python -m pytest tests/astro_dynamics/test_attitude.py tests/astro_dynamics/test_conjunction.py tests/astro_dynamics/test_local.py -q
```

Expected: all pass without optional backends.

- [x] **Step 3: Update docs**

Docs must say these products are deterministic screening products, not flight-qualified ACS, production conjunction service, or drag/SRP/third-body production covariance validation.

## Task 5: Live Backend Campaign Ledger

**Files:**

- Create: `docs/validation/live-backend-campaigns.md`
- Modify: `docs/release-checklist.md`
- Modify: `docs/backend-installation.md` if command names drift.

- [x] **Step 1: Create live campaign ledger**

Record each backend with:

```text
Backend:
Required runtime:
Smoke command:
Live validation command:
Current local status:
Unavailable diagnostic:
Roadmap claim allowed:
Roadmap claim not allowed:
```

- [x] **Step 2: Run available smoke gates**

Run:

```bash
astro orekit-smoke
astro rocketpy-smoke
astro dymos-smoke
astro tudat-smoke
astro jax-smoke
```

If a command exits unavailable, capture the structured diagnostic in the ledger. Do not mark the live gate complete.

- [x] **Step 3: Verify docs**

Run:

```bash
git diff --check
```

Expected: no whitespace errors.

## Final Release Gate

- [x] Run `python -m pytest -q`.
- [x] Run `python -m ruff check .`.
- [x] Run `python -m mypy`.
- [x] Run `python -m build` if build tooling is installed.
- [x] Update this plan's checkboxes to reflect completed tasks.
- [x] Update the roadmap status in `docs/superpowers/plans/2026-06-15-roadmap-goals-implementation-plan.md`.
- [x] Commit final docs/status changes.

Final local gate evidence from 2026-06-19:

```text
python -m pytest -q  # 498 passed, 10 skipped
python -m ruff check .  # passed
python -m mypy  # passed
python -m build  # built sdist and wheel successfully
```

Available optional live gate evidence from 2026-06-19 and 2026-06-20:

```text
ASTRO_RUN_ROCKETPY_LIVE=1 python -m pytest tests/astro_backends/test_rocketpy_simulation.py::test_live_rocketpy_configured_launch_examples_return_suite_products -q
# 2026-06-20: 1 passed in 1.62s for the single-stage and two-stage adapter fixtures

python -m pytest tests/astro_backends/test_rocketpy_simulation.py::test_propagate_launch_rocketpy_rejects_additional_motors_until_backend_supports_them -q
# 2026-06-20: 1 passed in 0.28s, guarding RocketPy 1.11's one-motor limitation

ASTRO_RUN_DYMOS_LIVE=1 python -m pytest tests/astro_backends/test_dymos_optimization.py::test_live_dymos_optimization_returns_suite_product tests/astro_backends/test_dymos_optimization.py::test_live_dymos_pitch_program_optimization_executes_native_transcription -q
# 2026-06-20: 2 passed, 2 OpenMDAO warnings in 2.23s; verifies the default Dymos phase
# and native pitch-program transcription target-score metadata

astro optimize-launch examples/launch/pitch_program_two_stage.yaml --backend dymos --dymos-mode pitch-program --output /tmp/astro-dymos-target-seeking-launch.json
# 2026-06-20: wrote a suite launch optimization product with target_objective =
# minimize_final_normalized_target_insertion_error and target_score = 1.1572999135341908

ASTRO_RUN_DYMOS_LIVE=1 python -m pytest tests/astro_backends/test_dymos_optimization.py::test_live_dymos_optimization_returns_suite_product tests/astro_backends/test_dymos_optimization.py::test_live_dymos_pitch_program_optimization_executes_native_transcription tests/astro_backends/test_dymos_optimization.py::test_live_dymos_multistage_pitch_program_executes_native_multiphase -q
# 2026-06-20: 3 passed, 3 OpenMDAO warnings in 2.04s; verifies default, single-phase
# pitch-program, and linked multiphase stage-transcription Dymos products. The multiphase
# live test was extended with stage-local mass depletion metadata and rechecked at 1 passed,
# 1 OpenMDAO warning in 2.25s.

astro optimize-launch examples/launch/pitch_program_two_stage.yaml --backend dymos --dymos-mode multistage-pitch-program --output /tmp/astro-dymos-multistage-pitch-program-launch.json
# 2026-06-20: wrote a suite launch optimization product with source_backend =
# dymos_multistage_pitch_program, phase_count = 2, phase_topology = multiphase_stage_linked,
# linked_state_names = time/h/downrange/vr/vh, mass_model =
# stage_local_propellant_depletion_with_fixed_stage_initial_mass, and target_score =
# 0.7683617412541746

JAVA_HOME=/opt/homebrew/opt/openjdk/libexec/openjdk.jdk/Contents/Home PATH="/opt/homebrew/opt/openjdk/bin:$PATH" astro orekit-smoke
# available true, orekit_jpype 13.1.5.0

ASTRO_RUN_OREKIT_LIVE=1 python -m pytest tests/astro_backends/test_orekit_propagation.py::test_live_orekit_two_body_matches_local_reference tests/astro_backends/test_orekit_propagation.py::test_live_orekit_j2_matches_local_reference_scale tests/astro_backends/test_orekit_propagation.py::test_live_orekit_covariance_history_returns_suite_product -q
# 3 passed

JAVA_HOME=/opt/homebrew/opt/openjdk/libexec/openjdk.jdk/Contents/Home PATH="/opt/homebrew/opt/openjdk/bin:$PATH" ASTRO_RUN_OREKIT_LIVE=1 python -m pytest tests/astro_backends/test_orekit_propagation.py::test_live_orekit_high_fidelity_covariance_records_force_models -q
# 1 passed

ASTRO_RUN_OREKIT_LIVE=1 python -m pytest tests/astro_backends/test_orekit_estimation.py::test_live_orekit_native_od_executes_batch_estimator -q
# 1 passed

conda run -p /tmp/astro-tudat-live-env astro tudat-smoke
# available true, TudatPy 1.0.0

conda run -p /tmp/astro-tudat-live-env astro propagate examples/scenarios/leo_tudat_variational_covariance.yaml --backend tudat --output /tmp/astro-tudat-variational-covariance.json
# wrote native variational covariance trajectory

ASTRO_RUN_TUDAT_LIVE=1 conda run -p /tmp/astro-tudat-live-env python -m pytest tests/astro_backends/test_tudat_propagation.py::test_live_tudat_high_fidelity_covariance_records_force_models -q
# 1 passed

ASTRO_RUN_TUDAT_LIVE=1 conda run -p /tmp/astro-tudat-live-env python -m pytest tests/astro_backends/test_tudat_propagation.py::test_live_tudat_native_variational_covariance_records_force_models -q
# 1 passed

conda run -p /tmp/astro-tudat-live-env astro compare-tudat-campaign examples/scenarios/leo_two_body.yaml examples/scenarios/leo_j2.yaml --reference-backend local --position-tolerance-km 0.01 --velocity-tolerance-km-s 0.00003 --output /tmp/astro-tudat-reference-campaign-calibrated.json
# passed true, 2 scenarios passed

JAX release-checklist research propagation, OD sensitivity, and research-estimate commands
# all completed and wrote /tmp/astro-jax-*.json products
```
