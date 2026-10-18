"""Fake LiteRT and synthetic pixels only; never load an actual model or footage."""
import hashlib
import json
from pathlib import Path
import sys
import time
from types import ModuleType

import cv2
import numpy as np
import pytest

from src import video_movinet as engine
from scripts import benchmark_video_movinet as benchmark


class Runner:
    def __init__(self):
        self.calls = []
        self.invalid = None

    def get_input_details(self):
        return {"image": {"shape": np.array([1, 1, 172, 172, 3]), "dtype": np.float32},
                "state_buffer": {"shape": np.array([1, 1]), "dtype": np.float32},
                "state_frame_count": {"shape": np.array([1]), "dtype": np.int32}}

    def get_output_details(self):
        values = {key: value for key, value in self.get_input_details().items() if key != "image"}
        values["logits"] = {"shape": np.array([1, 600]), "dtype": np.float32}
        return values

    def __call__(self, **values):
        self.calls.append({name: value.copy() for name, value in values.items()})
        total = values["state_buffer"] + values["image"][..., 0].mean()
        count = values["state_frame_count"] + 1
        logits = np.zeros((1, 600), np.float32)
        logits[0, 0], logits[0, 1] = total.item(), count.item()
        result = {"logits": logits, "state_buffer": total, "state_frame_count": count}
        if self.invalid == "logits":
            result["logits"][0, 0] = np.nan
        elif self.invalid == "state":
            result["state_buffer"] = np.zeros((2, 2), np.float32)
        return result


@pytest.fixture
def fake_runtime(tmp_path, monkeypatch):
    path = tmp_path / "synthetic.tflite"
    path.write_bytes(b"not an actual model; fake interpreter only")
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(engine, "WEIGHTS_SHA256", checksum)
    monkeypatch.setitem(engine.FEATURE_CONFIG, "weights_sha256", checksum)
    runner, constructions = Runner(), []

    class Interpreter:
        def __init__(self, **kwargs):
            constructions.append(kwargs)

        def allocate_tensors(self):
            pass

        def get_signature_list(self):
            return {"serving_default": {"inputs": list(runner.get_input_details()), "outputs": list(runner.get_output_details())}}

        def get_signature_runner(self, key):
            assert key == "serving_default"
            return runner

    package, runtime = ModuleType("ai_edge_litert"), ModuleType("ai_edge_litert.interpreter")
    runtime.__file__ = str(tmp_path / "fake_interpreter.py")
    Path(runtime.__file__).write_text("fake runtime source for provenance")
    runtime.Interpreter = Interpreter
    package.interpreter = runtime
    monkeypatch.setitem(sys.modules, "ai_edge_litert", package)
    monkeypatch.setitem(sys.modules, "ai_edge_litert.interpreter", runtime)
    monkeypatch.setattr(engine.importlib.metadata, "version", lambda name: engine.RUNTIME_VERSION)
    return path, checksum, runner, constructions


def test_preprocessing_preserves_aspect_ratio_and_rgb_float_range():
    frame = np.full((80, 40, 3), (0, 0, 255), np.uint8)
    value = engine.preprocess_frame(frame)
    assert value.shape == (1, 1, 172, 172, 3) and value.dtype == np.float32 and value.flags.c_contiguous
    assert np.all(value[0, 0, :, :43] == 0) and np.all(value[0, 0, :, 129:] == 0)
    assert np.all(value[0, 0, :, 43:129, 0] == 1) and np.all(value[..., 1:] == 0)


def test_state_is_zeroed_per_window_and_final_logits_are_returned(fake_runtime):
    path, checksum, runner, constructions = fake_runtime
    encoder = engine.MoViNetEncoder(path, checksum)
    red = np.full((172, 172, 3), (0, 0, 255), np.uint8)
    one = encoder.encode_window([red, red])
    two = encoder.encode_window([red, red])
    assert one.shape == (600,) and one.dtype == np.float32 and np.array_equal(one, two)
    assert one[0] == one[1] == 2
    assert runner.calls[0]["state_frame_count"].item() == runner.calls[2]["state_frame_count"].item() == 0
    assert runner.calls[1]["state_frame_count"].item() == runner.calls[3]["state_frame_count"].item() == 1
    assert constructions == [{"model_content": path.read_bytes(), "num_threads": 2}]
    assert encoder.provenance["weights_sha256"] == checksum
    json.dumps(encoder.signature)  # NumPy shape scalars must not break reports.


def test_weight_tampering_is_rejected_before_interpreter_creation(fake_runtime):
    path, checksum, _, constructions = fake_runtime
    path.write_bytes(b"tampered")
    with pytest.raises(engine.VideoAnalysisError, match="SHA-256"):
        engine.MoViNetEncoder(path, checksum)
    assert constructions == []


def test_runtime_version_mismatch_fails_before_loading(fake_runtime, monkeypatch):
    path, checksum, _, constructions = fake_runtime
    monkeypatch.setattr(engine.importlib.metadata, "version", lambda name: "unapproved")
    with pytest.raises(engine.VideoAnalysisError, match="pinned version"):
        engine.MoViNetEncoder(path, checksum)
    assert constructions == []


def test_different_model_checksum_is_not_silently_accepted(fake_runtime):
    path, _, _, constructions = fake_runtime
    with pytest.raises(engine.VideoAnalysisError, match="approved model variant"):
        engine.MoViNetEncoder(path, "0" * 64)
    assert constructions == []


@pytest.mark.parametrize("kind", ["logits", "state"])
def test_invalid_model_outputs_fail(fake_runtime, kind):
    path, checksum, runner, _ = fake_runtime
    encoder = engine.MoViNetEncoder(path, checksum)
    runner.invalid = kind
    with pytest.raises(engine.VideoAnalysisError):
        encoder.encode_window([np.zeros((172, 172, 3), np.uint8)])


def test_deadline_prevents_any_model_invocation(fake_runtime):
    path, checksum, runner, _ = fake_runtime
    encoder = engine.MoViNetEncoder(path, checksum)
    with pytest.raises(TimeoutError):
        encoder.encode_window([np.zeros((172, 172, 3), np.uint8)], time.monotonic() - 1)
    assert runner.calls == []


def test_intervals_have_fixed_stride_and_one_short_tail():
    windows = engine.window_intervals(145)
    assert len(windows) == 70
    assert windows[0] == {"start_sec": 0., "end_sec": 8.}
    assert windows[-1] == {"start_sec": 138., "end_sec": 145}
    assert np.all(np.diff([row["start_sec"] for row in windows]) == 2)
    assert sum(row["end_sec"] - row["start_sec"] < 8 for row in windows) == 1
    assert engine.window_intervals(3) == [{"start_sec": 0., "end_sec": 3}]


def write_synthetic(path, duration=10):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10., (64, 96))
    assert writer.isOpened()
    for index in range(duration * 10):
        writer.write(np.full((96, 64, 3), (index, 0, 100), np.uint8))
    writer.release()


class StubEncoder:
    provenance = {"synthetic": True}

    def __init__(self):
        self.calls = []

    def encode_window(self, frames, deadline):
        self.calls.append(frames)
        assert 1 <= len(frames) <= 40
        return np.full(600, len(frames), np.float32)


def test_interval_keeps_original_times_indices_hashes_and_shapes(tmp_path):
    path = tmp_path / "synthetic.avi"
    write_synthetic(path)
    encoder = StubEncoder()
    result = engine.extract_interval(path, 1., 9., encoder=encoder)
    sample = result["sampling"]
    assert result["features"].shape == (600,) and np.all(result["features"] == 40)
    assert len(sample["actual_times_sec"]) == 40 and sample["actual_times_sec"][0] == 1.
    assert sample["source_frame_indices"][0] == 10 and sample["source_frame_indices"][-1] == 88
    assert np.allclose(sample["actual_times_sec"], sample["requested_times_sec"])
    assert all(len(value) == 64 for value in sample["source_frame_sha256"])
    assert sample["source_shapes"] == [[96, 64, 3]] * 40
    assert sample["feature_config"] == engine.FEATURE_CONFIG and len(encoder.calls) == 1


def test_full_extraction_reuses_decoded_frames_but_resets_each_window(tmp_path):
    path = tmp_path / "synthetic.avi"
    write_synthetic(path, duration=11)
    encoder = StubEncoder()
    result = engine.extract_video(path, encoder=encoder)
    assert len(result["sampling"]["actual_times_sec"]) == 55
    assert [(row["start_sec"], row["end_sec"]) for row in result["windows"]] == [(0., 8.), (2., 10.), (4., 11.)]
    assert [len(call) for call in encoder.calls] == [40, 40, 35]
    assert "clip_features" not in result
    assert result["windows"][1]["sampling"]["source_frame_indices"][0] == 20
    assert encoder.calls[0][10] is encoder.calls[1][0]


@pytest.mark.parametrize("start,end", [(0, 9), (-1, 7), (4, 3), (float("nan"), 4)])
def test_invalid_intervals_fail_without_model_calls(tmp_path, start, end):
    path = tmp_path / "synthetic.avi"
    write_synthetic(path)
    encoder = StubEncoder()
    with pytest.raises(engine.VideoAnalysisError):
        engine.extract_interval(path, start, end, encoder=encoder)
    assert encoder.calls == []


def test_benchmark_inspection_never_starts_runtime(monkeypatch, capsys):
    monkeypatch.setattr(benchmark, "run_benchmark", lambda *a: pytest.fail("Inspection must not execute"))
    benchmark.main([])
    assert json.loads(capsys.readouterr().out)["will_execute"] is False
