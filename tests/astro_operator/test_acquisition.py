from __future__ import annotations

import stat
from pathlib import Path

import pytest

import astro_operator.recording as recording_module
from astro_operator.acquisition import (
    CallCappedReasoner,
    acquire_reasoner_behavior_recording,
)
from astro_operator.behavior import load_reasoner_behavior_corpus
from astro_operator.models import OperatorAction, OperatorActionKind
from astro_operator.reasoner import ScriptedReasoner
from astro_operator.recording import (
    RecordedFailureKind,
    load_reasoner_behavior_recording,
    reserve_reasoner_behavior_recording,
    write_reasoner_behavior_recording,
)

CORPUS = Path("examples/operator/reasoner_behavior_corpus.yaml")


def _finish_actions(count: int = 4) -> tuple[OperatorAction, ...]:
    return tuple(
        OperatorAction(
            action_id=f"finish-{index}",
            kind=OperatorActionKind.FINISH,
            rationale="Finish the bounded acquisition case.",
            conclusion="Recorded conclusion.",
        )
        for index in range(count)
    )


def test_acquisition_enforces_one_global_call_cap() -> None:
    recording = acquire_reasoner_behavior_recording(
        load_reasoner_behavior_corpus(CORPUS),
        ScriptedReasoner(_finish_actions()),
        recording_id="capped",
        max_calls=2,
    )

    assert recording.call_cap == 2
    assert recording.calls_attempted == 2
    assert [len(case.decisions) for case in recording.cases] == [1, 1, 0, 0]
    for case in recording.cases[2:]:
        assert case.terminal_failure is not None
        assert case.terminal_failure.kind == RecordedFailureKind.CANCELLED


def test_acquisition_checkpoints_before_and_after_each_completed_case() -> None:
    class InterruptingReasoner:
        def __init__(self) -> None:
            self.delegate = ScriptedReasoner(_finish_actions())
            self.calls = 0

        def decide(self, state):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("simulated process interruption")
            return self.delegate.decide(state)

    checkpoints = []
    with pytest.raises(RuntimeError, match="interruption"):
        acquire_reasoner_behavior_recording(
            load_reasoner_behavior_corpus(CORPUS),
            InterruptingReasoner(),
            recording_id="interrupted",
            max_calls=4,
            checkpoint=checkpoints.append,
        )

    assert [len(item.cases) for item in checkpoints] == [0, 1]
    assert all(not item.complete for item in checkpoints)
    assert checkpoints[-1].calls_attempted == 1


@pytest.mark.parametrize("max_calls", [0, -1, True, 1.5])
def test_call_cap_must_be_positive_strict_integer(max_calls: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        CallCappedReasoner(ScriptedReasoner(_finish_actions()), max_calls=max_calls)  # type: ignore[arg-type]


def test_cli_requires_explicit_provider_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from astro_cli.main import app
    from tests.astro_cli.helpers import make_cli_runner

    monkeypatch.setattr(
        "astro_cli.main.OpenRouterReasoner",
        lambda **kwargs: pytest.fail("adapter must not be constructed without confirmation"),
    )
    result = make_cli_runner().invoke(
        app,
        [
            "acquire-mission-reasoner-behavior",
            str(CORPUS),
            "--recording-id",
            "not-authorized",
            "--output",
            str(tmp_path / "recording.json"),
        ],
    )

    assert result.exit_code == 2
    assert not (tmp_path / "recording.json").exists()


def test_confirmed_cli_uses_injected_adapter_and_writes_private_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from astro_cli.main import app
    from tests.astro_cli.helpers import make_cli_runner

    monkeypatch.setattr(
        "astro_cli.main.OpenRouterReasoner",
        lambda **kwargs: ScriptedReasoner(_finish_actions()),
    )
    output = tmp_path / "recording.json"
    result = make_cli_runner().invoke(
        app,
        [
            "acquire-mission-reasoner-behavior",
            str(CORPUS),
            "--recording-id",
            "authorized-test",
            "--max-calls",
            "4",
            "--confirm-provider-calls",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    recording = load_reasoner_behavior_recording(output)
    assert recording.calls_attempted == 4
    assert recording.call_cap == 4
    assert recording.complete
    assert stat.S_IMODE(output.stat().st_mode) == 0o600


def test_reservation_refuses_existing_output_without_mutation(tmp_path: Path) -> None:
    output = tmp_path / "recording.json"
    output.write_text("user-owned", encoding="utf-8")
    recording = acquire_reasoner_behavior_recording(
        load_reasoner_behavior_corpus(CORPUS),
        ScriptedReasoner(_finish_actions()),
        recording_id="reserved",
        max_calls=4,
    )

    with pytest.raises(FileExistsError):
        reserve_reasoner_behavior_recording(output, recording)
    assert output.read_text(encoding="utf-8") == "user-owned"


def test_reservation_failure_never_exposes_partial_final_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "recording.json"
    recording = acquire_reasoner_behavior_recording(
        load_reasoner_behavior_corpus(CORPUS),
        ScriptedReasoner(_finish_actions()),
        recording_id="reservation-failure",
        max_calls=4,
    )

    def fail_link(source: Path, target: Path) -> None:
        raise OSError("simulated link interruption")

    monkeypatch.setattr(recording_module.os, "link", fail_link)
    with pytest.raises(OSError, match="link interruption"):
        reserve_reasoner_behavior_recording(output, recording)

    assert not output.exists()


def test_recording_commits_fsync_parent_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "recording.json"
    recording = acquire_reasoner_behavior_recording(
        load_reasoner_behavior_corpus(CORPUS),
        ScriptedReasoner(_finish_actions()),
        recording_id="directory-sync",
        max_calls=4,
    )
    synced: list[Path] = []
    monkeypatch.setattr(recording_module, "_fsync_directory", synced.append)

    reserve_reasoner_behavior_recording(output, recording)
    write_reasoner_behavior_recording(output, recording)

    assert synced == [tmp_path, tmp_path]


def test_atomic_write_preserves_prior_checkpoint_on_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "recording.json"
    recording = acquire_reasoner_behavior_recording(
        load_reasoner_behavior_corpus(CORPUS),
        ScriptedReasoner(_finish_actions()),
        recording_id="original",
        max_calls=4,
    )
    write_reasoner_behavior_recording(output, recording)

    def fail_replace(self: Path, target: Path) -> Path:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failure"):
        write_reasoner_behavior_recording(
            output, recording.model_copy(update={"recording_id": "new"})
        )

    assert load_reasoner_behavior_recording(output).recording_id == "original"
