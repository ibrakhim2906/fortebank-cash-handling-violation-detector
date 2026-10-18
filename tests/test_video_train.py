"""Synthetic numeric-only checks for video development comparison."""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src import config, video_train
from src.video_eval import create_or_load_protocol, protocol_groups


@pytest.fixture
def numeric_development(tmp_path):
    rows = [{
        "video_id": f"synthetic-{index}", "path": f"not-real-{index}.mov",
        "sha256": hashlib.sha256(f"numeric-only-{index}".encode()).hexdigest(),
        "target": index % 2, "timestamp_sec": 5.0 if index % 2 else np.nan,
        "duration_sec": 10.0,
    } for index in range(40)]
    inventory = tmp_path / "inventory.csv"
    pd.DataFrame(rows).to_csv(inventory, index=False)
    protocol = create_or_load_protocol(inventory, tmp_path / "protocol.json")
    groups = protocol_groups(protocol)
    rng = np.random.default_rng(config.SEED)
    # This dict deliberately has no reserved-holdout records.
    records = {group["group_id"]: {
        "clip_features": np.asarray([index, rng.uniform(0.05, 0.95)]),
    } for index, group in enumerate(groups)}
    return protocol, groups, records


def test_fixed_lgbm_recipe_and_fold_local_preprocessing():
    pipeline = video_train.classifier("lgbm")
    estimator = pipeline.named_steps["lgbmclassifier"]
    expected = dict(n_estimators=100, learning_rate=0.05, num_leaves=7, max_depth=3,
                    min_child_samples=4, reg_lambda=5, n_jobs=2,
                    random_state=config.SEED, deterministic=True, verbosity=-1)
    for name, value in expected.items():
        assert estimator.get_params()[name] == value
    assert "simpleimputer" in pipeline.named_steps
    assert not hasattr(pipeline.named_steps["simpleimputer"], "statistics_")
    assert "standardscaler" in video_train.classifier("linear").named_steps


def test_oof_probabilities_align_by_id_and_training_never_sees_validation_labels(
    numeric_development, monkeypatch,
):
    protocol, groups, records = numeric_development
    fits, constructed = [], []

    class TrackingClassifier:
        def fit(self, x, y):
            fits.append((x.copy(), np.asarray(y).copy()))
            return self

        def predict_proba(self, x):
            return np.column_stack([1 - x[:, 1], x[:, 1]])

    def factory(name="linear"):
        constructed.append(name)
        return TrackingClassifier()

    monkeypatch.setattr(video_train, "classifier", factory)
    shuffled = np.asarray([group["target"] for group in groups])[::-1]
    result = video_train.evaluate_development(protocol, records, shuffled_labels=shuffled, model_name="lgbm")
    assert constructed == ["lgbm"] * 3
    assert result["model_name"] == "lgbm"
    assert result["group_ids"] == [group["group_id"] for group in groups]
    assert result["targets"] == shuffled.tolist()
    assert result["probabilities"] == [float(records[group["group_id"]]["clip_features"][1]) for group in groups]
    shuffled_by_id = dict(zip(result["group_ids"], shuffled))
    for fold, (fit_x, fit_y) in zip(protocol["folds"], fits):
        assert np.array_equal(fit_x, np.stack([records[key]["clip_features"] for key in fold["train_group_ids"]]))
        assert np.array_equal(fit_y, [shuffled_by_id[key] for key in fold["train_group_ids"]])
        assert not set(fold["valid_group_ids"]) & set(fold["train_group_ids"])
        assert not set(protocol["holdout_group_ids"]) & set(fold["train_group_ids"])
    assert result["model"] == result["linear"]  # Documented legacy evaluation alias.


def test_selection_prefers_higher_auc_and_linear_on_tie():
    evaluations = {"linear": {"model": {"roc_auc": 0.6}}, "lgbm": {"model": {"roc_auc": 0.7}}}
    assert video_train.select_model(evaluations) == "lgbm"
    evaluations["lgbm"]["model"]["roc_auc"] = 0.6
    assert video_train.select_model(dict(reversed(list(evaluations.items())))) == "linear"
    with pytest.raises(ValueError, match="finite"):
        video_train.select_model({"linear": {"model": {"roc_auc": float("nan")}}})


def test_weak_window_labels_exclude_ambiguous_positive_and_unlabeled_windows(monkeypatch):
    groups = [
        {"group_id": "negative", "target": 0, "localization_eligible": False, "timestamp_sec": None},
        {"group_id": "positive", "target": 1, "localization_eligible": True, "timestamp_sec": 5},
        {"group_id": "disputed", "target": 1, "localization_eligible": False, "timestamp_sec": None},
    ]
    records = {group["group_id"]: {"windows": [
        {"timestamp_sec": float(index), "features": np.asarray([group_number * 10 + index])}
        for index in (1, 5, 9)
    ]} for group_number, group in enumerate(groups)}
    captured = {}

    class WindowClassifier:
        @property
        def named_steps(self):
            return {"logisticregression": self}

        def set_params(self, **kwargs):
            return self

        def fit(self, x, y, **kwargs):
            captured.update(x=x, y=y, weights=kwargs["logisticregression__sample_weight"])
            return self

    monkeypatch.setattr(video_train, "classifier", WindowClassifier)
    video_train.fit_window_model(groups, records)
    assert captured["x"].ravel().tolist() == [1, 5, 9, 15]
    assert captured["y"].tolist() == [0, 0, 0, 1]
    assert captured["weights"][:3].sum() == pytest.approx(captured["weights"][3])


def test_invalid_shuffle_length_fails_before_training(numeric_development):
    protocol, _, records = numeric_development
    with pytest.raises(ValueError, match="one binary target"):
        video_train.evaluate_development(protocol, records, shuffled_labels=[0, 1])


def test_encoder_import_routing_uses_a_fixed_allowlist(monkeypatch):
    imported = []
    sentinel = object()

    def importer(name):
        imported.append(name)
        return sentinel

    monkeypatch.setattr(video_train.importlib, "import_module", importer)
    assert video_train.load_engine() is sentinel
    assert video_train.load_engine("mobilenet") is sentinel
    assert video_train.load_engine("qwen-actions") is sentinel
    assert imported == ["src.video_analysis", "src.video_pretrained", "src.video_qwen_actions"]
    with pytest.raises(ValueError, match="Unknown video encoder"):
        video_train.load_engine("arbitrary.external_module")
    assert len(imported) == 3


def test_engine_specific_caches_never_extract_holdout(numeric_development, tmp_path, monkeypatch):
    protocol, groups, _ = numeric_development
    real_digest = video_train.digest
    expected_digests = {str(config.ROOT / group["path"]): group["group_id"] for group in groups}
    def fake_digest(path):
        return expected_digests[str(path)] if str(path) in expected_digests else real_digest(path)

    monkeypatch.setattr(video_train, "digest", fake_digest)
    calls = []
    engines = {}
    for index, encoder in enumerate(("numeric", "mobilenet")):
        module_path = tmp_path / f"{encoder}.py"
        module_path.write_text(f"# fake {encoder} extractor\n")

        def extract(path, deadline, value=index):
            calls.append((value, str(path)))
            return {"clip_features": np.asarray([value]), "duration_sec": 10.,
                    "windows": [{"timestamp_sec": 1., "features": np.asarray([value])}]}

        engines[encoder] = SimpleNamespace(__file__=str(module_path),
                                           FEATURE_CONFIG={"encoder": encoder, "weights_sha256": str(index)},
                                           WEIGHTS_SHA256=str(index),
                                           extract_video=extract)
    monkeypatch.setattr(video_train, "load_engine", lambda encoder="numeric": engines[encoder])
    for encoder in engines:
        records = video_train.extract_development(protocol, tmp_path / encoder, encoder=encoder)
        assert set(records) == set(expected_digests.values())
        assert not set(records) & set(protocol["holdout_group_ids"])
        assert {record["engine"] for record in records.values()} == {encoder}
        for record in records.values():
            assert record["source_sha256"] == real_digest(engines[encoder].__file__)
            assert bool(record["inherited_sources"]) is (encoder == "mobilenet")
    assert len(calls) == 64
    assert all(path in expected_digests for _, path in calls)
    with pytest.raises(ValueError, match="different encoder"):
        video_train.extract_development(protocol, tmp_path / "wrong", source_cache=tmp_path / "numeric",
                                        encoder="mobilenet")
    assert len(calls) == 64  # Reject the foreign cache before any re-decoding.


def test_legacy_numeric_cache_compatibility_and_inherited_source_guard():
    provenance = {"engine": "numeric", "engine_module": "src.video_analysis",
                  "source_sha256": "numeric-source", "inherited_sources": {}}
    legacy = {"source_sha256": "numeric-source", "feature_config": {"resize": [96, 96]}}
    video_train.verify_feature_record(legacy, provenance, legacy["feature_config"])
    changed = dict(legacy, source_sha256="stale")
    with pytest.raises(ValueError, match="stale"):
        video_train.verify_feature_record(changed, provenance, legacy["feature_config"])
    pretrained = {"engine": "mobilenet", "engine_module": "src.video_pretrained",
                  "source_sha256": "pretrained-source", "inherited_sources": {"src/video_analysis.py": "helper-v2"}}
    stale_helper = {**pretrained, "feature_config": {}, "inherited_sources": {"src/video_analysis.py": "helper-v1"}}
    with pytest.raises(ValueError, match="stale"):
        video_train.verify_feature_record(stale_helper, pretrained, {})
    with pytest.raises(ValueError, match="different encoder"):
        video_train.verify_feature_record(legacy, pretrained, legacy["feature_config"])


def test_actual_comparison_requires_identical_frozen_protocol(tmp_path):
    fields = {"protocol_fingerprint": "frozen", "seed": config.SEED, "threshold": 0.5,
              "group_ids": ["one", "two"], "targets": [0, 1], "holdout_evaluated": False}
    baseline = {**fields, "linear": {"roc_auc": 0.51}, "comparisons": {"linear": {"roc_auc": 0.51}}}
    current = {**fields, "comparisons": {"linear": {"roc_auc": 0.60}}}
    (tmp_path / "development.json").write_text(json.dumps(baseline))
    comparison = video_train.representation_comparison(current, tmp_path)
    assert comparison["models"]["linear"]["before"]["roc_auc"] == 0.51
    assert comparison["models"]["linear"]["after"]["roc_auc"] == 0.60
    assert comparison["models"]["linear"]["roc_auc_delta"] == pytest.approx(0.09)
    with pytest.raises(ValueError, match="threshold"):
        video_train.representation_comparison(dict(current, threshold=0.3), tmp_path)
    with pytest.raises(ValueError, match="group_ids"):
        video_train.representation_comparison(dict(current, group_ids=["holdout"]), tmp_path)


def test_window_classifier_fits_only_outer_training_groups(numeric_development, monkeypatch):
    protocol, groups, records = numeric_development
    by_id = {g["group_id"]: g for g in groups}
    for index, group in enumerate(groups):
        records[group["group_id"]]["windows"] = [
            {"timestamp_sec": float(timestamp), "features": np.asarray([probability])}
            for timestamp, probability in ((1, .1), (5, .6), (9, .3))]
    fits = []

    class WindowModel:
        def predict_proba(self, x):
            return np.column_stack([1 - x[:, 0], x[:, 0]])

    def fit(training_groups, features):
        fits.append([group["group_id"] for group in training_groups])
        return WindowModel()

    monkeypatch.setattr(video_train, "fit_window_model", fit)
    monkeypatch.setattr(video_train, "classifier", lambda *args: pytest.fail("Clip classifier must not be fitted"))
    result = video_train.evaluate_development(protocol, records, model_name="window-linear")
    assert fits == [fold["train_group_ids"] for fold in protocol["folds"]]
    assert result["probabilities"] == [.6] * len(groups)
    assert set(result["candidate_times"].values()) == {5.}
    assert result["localization"]["n"] == sum(g["target"] and g["localization_eligible"] for g in by_id.values())
    assert not any(set(ids) & set(protocol["holdout_group_ids"]) for ids in fits)


def test_shuffle_moves_weak_supervision_as_an_atomic_bundle():
    groups = [
        {"group_id": "a", "target": 0, "localization_eligible": False, "timestamp_sec": None, "duration_sec": 10},
        {"group_id": "b", "target": 1, "localization_eligible": True, "timestamp_sec": 30, "duration_sec": 100},
        {"group_id": "c", "target": 1, "localization_eligible": False, "timestamp_sec": None, "duration_sec": 20},
    ]

    class Donors:
        def permutation(self, count):
            assert count == 3
            return np.asarray([1, 2, 0])

    shuffled = video_train.shuffle_supervision(groups, Donors())
    assert [(g["target"], g["localization_eligible"], g["timestamp_sec"]) for g in shuffled] == [
        (1, True, 3.), (1, False, None), (0, False, None)]
    assert [g["group_id"] for g in shuffled] == ["a", "b", "c"]
    assert [g["timestamp_sec"] for g in groups] == [None, 30, None]


def test_window_shuffle_uses_donor_labels_and_timestamps(numeric_development, monkeypatch):
    protocol, groups, records = numeric_development
    for group in groups:
        records[group["group_id"]]["windows"] = [{"timestamp_sec": 2., "features": np.asarray([.4])}]
    shuffled = video_train.shuffle_supervision(groups, np.random.default_rng(config.SEED))
    expected = {group["group_id"]: group for group in shuffled}
    fits = []

    class WindowModel:
        def predict_proba(self, x):
            return np.column_stack([1 - x[:, 0], x[:, 0]])

    def fit(training_groups, features):
        fits.append(training_groups)
        return WindowModel()

    monkeypatch.setattr(video_train, "fit_window_model", fit)
    result = video_train.evaluate_development(protocol, records, model_name="window-linear", shuffled_groups=shuffled)
    assert result["targets"] == [expected[g["group_id"]]["target"] for g in groups]
    assert "localization" not in result
    for fold, fitting_groups in zip(protocol["folds"], fits):
        assert fitting_groups == [expected[key] for key in fold["train_group_ids"]]
    with pytest.raises(ValueError, match="full shuffled supervision"):
        video_train.evaluate_development(protocol, records, model_name="window-linear", shuffled_labels=result["targets"])


def fake_qwen_engine(tmp_path, monkeypatch):
    """Pinned synthetic source files; no real runtime, weights or clips opened."""
    monkeypatch.setattr(config, "ROOT", tmp_path)
    module_path = tmp_path / "qwen_encoder.py"
    module_path.write_text("# synthetic action observation extractor\n")
    inherited = {}
    for relative in video_train.QWEN_SOURCE_PATHS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# synthetic helper {relative}\n")
        inherited[relative] = video_train.digest(path)
    return SimpleNamespace(
        __file__=str(module_path), WEIGHTS_SHA256="a" * 64,
        INHERITED_SOURCES=inherited,
        FEATURE_CONFIG={"projector_sha256": "b" * 64, "runtime_version": "synthetic-b11707"},
    )


def test_qwen_provenance_uses_only_pinned_inherited_source_allowlist(tmp_path, monkeypatch):
    engine = fake_qwen_engine(tmp_path, monkeypatch)
    provenance = video_train.engine_provenance("qwen-actions", engine)
    assert provenance["engine_module"] == "src.video_qwen_actions"
    assert provenance["weights_sha256"] == "a" * 64
    assert provenance["source_sha256"] == video_train.digest(engine.__file__)
    assert provenance["inherited_sources"] == engine.INHERITED_SOURCES
    assert provenance["inherited_sources"] is not engine.INHERITED_SOURCES
    engine.INHERITED_SOURCES["outside.py"] = "c" * 64
    with pytest.raises(ValueError, match="allowlist"):
        video_train.engine_provenance("qwen-actions", engine)
    del engine.INHERITED_SOURCES["outside.py"]
    (tmp_path / "src/video_actions.py").write_text("# changed synthetic helper\n")
    with pytest.raises(ValueError, match="pinned configuration"):
        video_train.engine_provenance("qwen-actions", engine)


def test_qwen_cache_rejects_weights_projector_runtime_or_helper_changes(tmp_path, monkeypatch):
    engine = fake_qwen_engine(tmp_path, monkeypatch)
    provenance = video_train.engine_provenance("qwen-actions", engine)
    item = {**provenance, "feature_config": engine.FEATURE_CONFIG}
    video_train.verify_feature_record(item, provenance, engine.FEATURE_CONFIG)
    variants = [
        {**item, "weights_sha256": "c" * 64},
        {**item, "inherited_sources": {}},
        {**item, "feature_config": {**engine.FEATURE_CONFIG, "projector_sha256": "c" * 64}},
        {**item, "feature_config": {**engine.FEATURE_CONFIG, "runtime_version": "synthetic-other"}},
    ]
    for stale in variants:
        with pytest.raises(ValueError, match="stale"):
            video_train.verify_feature_record(stale, provenance, engine.FEATURE_CONFIG)


def test_qwen_extraction_receives_development_video_paths_only(numeric_development, tmp_path, monkeypatch):
    protocol, groups, _ = numeric_development
    engine = fake_qwen_engine(tmp_path, monkeypatch)
    real_digest = video_train.digest
    expected = {str(config.ROOT / group["path"]): group["group_id"] for group in groups}

    def digest(path):
        return expected[str(path)] if str(path) in expected else real_digest(path)

    calls = []

    def extract(path, deadline):
        calls.append(str(path))
        # Only the local video path/deadline are passed, never targets, anchor
        # timestamps, group annotations or human/model action labels.
        assert str(path) in expected
        return {"clip_features": np.asarray([0., 1.]), "duration_sec": 10.,
                "windows": [{"timestamp_sec": 5., "features": np.asarray([0., 1.])}]}

    engine.extract_video = extract
    monkeypatch.setattr(video_train, "digest", digest)
    monkeypatch.setattr(video_train, "load_engine", lambda encoder="numeric": engine)
    records = video_train.extract_development(protocol, tmp_path / "cache", encoder="qwen-actions")
    assert set(records) == {group["group_id"] for group in groups}
    assert not set(records) & set(protocol["holdout_group_ids"])
    assert len(calls) == len(groups) == 32
    assert {record["engine"] for record in records.values()} == {"qwen-actions"}
    # A second extraction reuses matching cache without a local model call.
    video_train.extract_development(protocol, tmp_path / "cache", encoder="qwen-actions")
    assert len(calls) == 32
