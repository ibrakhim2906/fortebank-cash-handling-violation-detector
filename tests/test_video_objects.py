"""Public geometry and synthetic pixels only; no supplied video is inspected."""
import hashlib
import time

import cv2
import numpy as np
import pytest

from src import video_objects as V


def raw_output(boxes=()):
    output = np.zeros((1, 8400, 85), dtype=np.float32)
    for index, (class_id, xyxy, object_probability, class_probability) in enumerate(boxes):
        x1, y1, x2, y2 = xyxy
        stride = V._STRIDES[index, 0]
        output[0, index, :2] = (np.asarray([(x1 + x2) / 2, (y1 + y2) / 2])
                                       / stride - V._GRIDS[index])
        output[0, index, 2:4] = np.log(np.asarray([x2 - x1, y2 - y1]) / stride)
        output[0, index, 4] = object_probability
        output[0, index, 5 + class_id] = class_probability
    return output


@pytest.fixture
def fake_assets(tmp_path, monkeypatch):
    payload = b"Synthetic YOLOX-S placeholder; never parsed as ONNX"
    path = tmp_path / "synthetic.onnx"
    path.write_bytes(payload)
    expected_hash = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(V, "WEIGHTS_SHA256", expected_hash)
    ready = {"model_path": str(path), "weights_sha256": expected_hash,
             "source_commit": V.FEATURE_CONFIG["source_commit"], "license": "Apache-2.0"}
    monkeypatch.setattr(V._assets, "verify_ready", lambda: dict(ready))
    nets = []

    class Net:
        backend, target, blob = None, None, None
        output = raw_output()

        def setPreferableBackend(self, value):
            self.backend = value

        def setPreferableTarget(self, value):
            self.target = value

        def getUnconnectedOutLayersNames(self):
            return ["synthetic-output"]

        def setInput(self, value):
            self.blob = value.copy()

        def forward(self, name):
            assert name == "synthetic-output"
            return self.output.copy()

    def load(buffer):
        assert buffer.tobytes() == payload
        net = Net()
        nets.append(net)
        return net

    monkeypatch.setattr(cv2.dnn, "readNetFromONNX", load)
    return path, ready, nets


def test_preprocessing_matches_rgb_top_left_letterbox_without_normalization():
    frame = np.full((100, 200, 3), (10, 20, 30), dtype=np.uint8)
    blob, ratio = V.preprocess(frame)
    assert ratio == 3.2
    assert blob.shape == (1, 3, 640, 640)
    assert blob.dtype == np.float32 and blob.flags.c_contiguous
    np.testing.assert_array_equal(blob[0, :, 0, 0], [30, 20, 10])
    np.testing.assert_array_equal(blob[0, :, 319, 639], [30, 20, 10])
    np.testing.assert_array_equal(blob[0, :, 320, 0], [114, 114, 114])
    np.testing.assert_array_equal(blob[0, :, 639, 639], [114, 114, 114])


def test_preprocessing_floors_resized_dimensions():
    blob, ratio = V.preprocess(np.zeros((43, 100, 3), dtype=np.uint8))
    assert ratio == 6.4
    assert int(43 * ratio) == 275
    assert blob[0, :, 274, 0].tolist() == [0, 0, 0]
    assert blob[0, :, 275, 0].tolist() == [114, 114, 114]


@pytest.mark.parametrize("frame", [None, np.zeros((10, 10), np.uint8),
                                    np.zeros((0, 10, 3), np.uint8), np.zeros((10, 10, 3), np.float32)])
def test_invalid_frames_fail_explicitly(frame):
    with pytest.raises(V.VideoAnalysisError, match="BGR frame"):
        V.preprocess(frame)


def test_anchors_have_correct_grid_order_and_strides():
    assert V._GRIDS.shape == (8400, 2)
    assert V._STRIDES.shape == (8400, 1)
    np.testing.assert_array_equal(V._GRIDS[[0, 1, 79, 80, 6399, 6400, 7999, 8000]],
                                  [[0, 0], [1, 0], [79, 0], [0, 1], [79, 79], [0, 0], [39, 39], [0, 0]])
    np.testing.assert_array_equal(V._STRIDES[[0, 6399, 6400, 7999, 8000, 8399], 0],
                                  [8, 8, 16, 16, 32, 32])


def test_retains_only_six_public_object_classes():
    boxes = [(class_id, (index * 70 + 10, 30, index * 70 + 40, 80), 0.9, 0.8)
             for index, class_id in enumerate(V.CLASS_IDS)]
    boxes.append((1, (500, 30, 550, 80), 0.99, 0.99))
    result = V.postprocess(raw_output(boxes), (640, 640, 3), 1)
    assert {row["class_id"] for row in result} == set(V.CLASS_IDS)
    assert set(V.CLASS_NAMES) == set(V.CLASS_IDS)
    assert len(result) == 6
    for row in result:
        assert set(row) == {"class_id", "score", "box_xyxy"}
        assert row["score"] == pytest.approx(0.72)
        assert all(0 <= value <= 1 for value in row["box_xyxy"])
    assert not any(word in name for name in V.CLASS_NAMES.values()
                   for word in ("cash", "hand ", "identity", "violation"))


def test_confidence_is_product_and_highest_class_is_selected():
    output = raw_output([
        (0, (10, 10, 30, 30), 0.5, 0.8),
        (0, (100, 10, 130, 30), 0.7, 0.4),
        (67, (200, 10, 230, 30), 0.9, 0.8),
    ])
    # An unretained class outranks the retained class at the third anchor.
    output[0, 2, 5 + 1] = 0.95
    result = V.postprocess(output, (640, 640, 3), 1)
    assert len(result) == 1
    assert result[0]["class_id"] == 0
    assert result[0]["score"] == pytest.approx(0.4)


def test_nms_suppresses_same_class_but_preserves_overlapping_different_classes():
    output = raw_output([
        (0, (100, 100, 200, 200), 0.99, 0.99),
        (0, (101, 101, 201, 201), 0.9, 0.9),
        (67, (100, 100, 200, 200), 0.95, 0.95),
    ])
    result = V.postprocess(output, (640, 640, 3), 1)
    assert [row["class_id"] for row in result] == [0, 67]


def test_boxes_scale_back_to_portrait_source_and_clip_padding():
    output = raw_output([
        (0, (80, 160, 240, 480), 0.99, 0.99),
        (67, (-10, -10, 330, 650), 0.9, 0.9),
        (63, (500, 100, 600, 200), 0.95, 0.95),
    ])
    result = V.postprocess(output, (800, 400, 3), 0.8)
    assert len(result) == 2
    by_class = {row["class_id"]: row for row in result}
    np.testing.assert_allclose(by_class[0]["box_xyxy"], [0.25, 0.25, 0.75, 0.75], atol=1e-6)
    np.testing.assert_allclose(by_class[67]["box_xyxy"], [0, 0, 1, 1], atol=1e-6)


def test_empty_valid_output_is_empty_list_and_does_not_mutate_raw():
    output = raw_output()
    before = output.copy()
    assert V.postprocess(output, (100, 100, 3), 6.4) == []
    np.testing.assert_array_equal(output, before)


@pytest.mark.parametrize("bad", [np.zeros((8400, 85)), np.zeros((1, 8400, 84)),
                                  np.zeros((1, 1, 85)), np.full((1, 8400, 85), np.nan)])
def test_wrong_raw_shape_or_nonfinite_values_raise_instead_of_empty_result(bad):
    with pytest.raises(V.VideoAnalysisError, match="invalid output"):
        V.postprocess(bad, (100, 100, 3), 6.4)


def test_invalid_probability_or_box_geometry_raises():
    output = raw_output()
    output[0, 0, 4] = 1.1
    with pytest.raises(V.VideoAnalysisError, match="probabilities"):
        V.postprocess(output, (100, 100, 3), 6.4)
    output = raw_output()
    output[0, 0, 2] = 1000
    with pytest.raises(V.VideoAnalysisError, match="box geometry"):
        V.postprocess(output, (100, 100, 3), 6.4)
    output[0, 0, 2] = -1000
    with pytest.raises(V.VideoAnalysisError, match="box geometry"):
        V.postprocess(output, (100, 100, 3), 6.4)


def test_global_detection_limit_is_deterministic():
    boxes = [(0, ((index % 20) * 30, (index // 20) * 30,
                  (index % 20) * 30 + 10, (index // 20) * 30 + 10), 0.9, 0.9)
             for index in range(120)]
    output = raw_output(boxes)
    first = V.postprocess(output, (640, 640, 3), 1)
    second = V.postprocess(output, (640, 640, 3), 1)
    assert len(first) == 100
    assert first == second


def test_loader_verifies_local_snapshot_and_uses_cpu(fake_assets):
    _, ready, nets = fake_assets
    detector = V.ObjectDetector()
    assert detector.provenance == ready
    assert nets[0].backend == cv2.dnn.DNN_BACKEND_OPENCV
    assert nets[0].target == cv2.dnn.DNN_TARGET_CPU
    assert detector.detect(np.zeros((100, 200, 3), np.uint8)) == []
    assert nets[0].blob.shape == (1, 3, 640, 640)


def test_tampered_weight_bytes_fail_before_model_loading(fake_assets):
    path, _, nets = fake_assets
    path.write_bytes(b"tampered local model")
    with pytest.raises(V.VideoAnalysisError, match="SHA-256"):
        V.ObjectDetector()
    assert not nets


def test_readiness_failure_prevents_model_loading(fake_assets, monkeypatch):
    def fail():
        raise ValueError("Synthetic provenance mismatch")

    monkeypatch.setattr(V._assets, "verify_ready", fail)
    with pytest.raises(V.VideoAnalysisError, match="provenance"):
        V.ObjectDetector()
    assert not fake_assets[2]


def test_wrong_source_commit_fails_before_model_loading(fake_assets):
    fake_assets[1]["source_commit"] = "unapproved"
    with pytest.raises(V.VideoAnalysisError, match="readiness provenance"):
        V.ObjectDetector()
    assert not fake_assets[2]


def test_invalid_runtime_output_is_failure_and_expired_deadline_stops(fake_assets):
    detector = V.ObjectDetector()
    fake_assets[2][0].output = np.zeros((1, 1, 1), np.float32)
    frame = np.zeros((100, 200, 3), np.uint8)
    with pytest.raises(V.VideoAnalysisError, match="invalid output"):
        detector.detect(frame)
    with pytest.raises(TimeoutError, match="deadline"):
        detector.detect(frame, deadline=time.monotonic() - 1)
