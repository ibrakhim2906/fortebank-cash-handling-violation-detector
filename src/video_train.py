"""Development-only baseline on frozen, content-disjoint video groups."""
import argparse
import datetime as dt
import hashlib
import importlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config


MODEL_NAMES = ("linear", "lgbm", "window-linear")
ENGINE_MODULES = {"numeric": "src.video_analysis", "mobilenet": "src.video_pretrained",
                  "mobilenet-temporal": "src.video_temporal",
                  "qwen-actions": "src.video_qwen_actions",
                  "qwen-spatial-v3": "src.video_qwen_spatial_v3",
                  "object-tracks": "src.video_object_tracks"}
QWEN_SOURCE_PATHS = frozenset({
    "src/video_analysis.py", "src/video_action_proposals.py",
    "src/video_actions.py", "scripts/provision_action_model.py",
})
QWEN_SPATIAL_SOURCE_PATHS = QWEN_SOURCE_PATHS | {
    "src/video_qwen_actions.py", "src/video_local_observations.py",
}
OBJECT_SOURCE_PATHS = frozenset({
    "src/video_analysis.py", "src/video_objects.py", "scripts/fetch_video_objects.py",
})


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_engine(encoder="numeric"):
    """Load only one of the explicitly supported local feature extractors."""
    if encoder not in ENGINE_MODULES:
        raise ValueError(f"Unknown video encoder {encoder!r}; choose from {tuple(ENGINE_MODULES)}")
    return importlib.import_module(ENGINE_MODULES[encoder])


def engine_provenance(encoder="numeric", engine=None):
    engine = load_engine(encoder) if engine is None else engine
    provenance = {"engine": encoder, "engine_module": ENGINE_MODULES[encoder],
                  "source_sha256": digest(engine.__file__),
                  "inherited_sources": ({"src/video_analysis.py": digest(config.ROOT / "src/video_analysis.py")}
                                        if encoder != "numeric" else {})}
    if encoder == "mobilenet-temporal":
        provenance["inherited_sources"]["src/video_pretrained.py"] = digest(config.ROOT / "src/video_pretrained.py")
    if encoder in {"qwen-actions", "qwen-spatial-v3"}:
        # Only this new engine supplies its wider dependency map. Legacy
        # numeric/MobileNet provenance remains byte-for-byte compatible.
        inherited = getattr(engine, "INHERITED_SOURCES", None)
        expected = QWEN_SOURCE_PATHS if encoder == "qwen-actions" else QWEN_SPATIAL_SOURCE_PATHS
        if not isinstance(inherited, dict) or set(inherited) != expected:
            raise ValueError("Qwen action encoder requires the fixed inherited-source allowlist")
        if any(value != digest(config.ROOT / path) for path, value in inherited.items()):
            raise ValueError("Qwen action encoder inherited source differs from its pinned configuration")
        provenance["inherited_sources"] = dict(inherited)
    if encoder == "object-tracks":
        inherited = getattr(engine, "INHERITED_SOURCES", None)
        if not isinstance(inherited, dict) or set(inherited) != OBJECT_SOURCE_PATHS:
            raise ValueError("Object encoder requires its fixed inherited-source allowlist")
        if any(value != digest(config.ROOT / path) for path, value in inherited.items()):
            raise ValueError("Object encoder inherited source differs from its pinned configuration")
        provenance["inherited_sources"] = dict(inherited)
    if encoder != "numeric":
        provenance["weights_sha256"] = engine.WEIGHTS_SHA256
    return provenance


def verify_feature_record(item, provenance, feature_config):
    """Reject caches from another encoder/source, including inherited helpers.

    Numeric caches predating encoder routing remain valid only for the numeric
    engine and its unchanged source/configuration. No old cache is rewritten.
    """
    if (item.get("engine", "numeric") != provenance["engine"]
            or item.get("engine_module", ENGINE_MODULES["numeric"]) != provenance["engine_module"]
            or item.get("source_sha256") != provenance["source_sha256"]
            or item.get("inherited_sources", {}) != provenance["inherited_sources"]
            or item.get("weights_sha256") != provenance.get("weights_sha256")
            or item.get("feature_config") != feature_config):
        raise ValueError("Feature cache is stale or uses a different encoder; use a fresh run directory")


def classifier(model_name="linear"):
    if model_name == "linear":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=0.1, solver="liblinear", max_iter=1000,
                                                class_weight="balanced", random_state=config.SEED))
    if model_name == "lgbm":
        from lightgbm import LGBMClassifier
        return make_pipeline(SimpleImputer(strategy="median"), LGBMClassifier(
            n_estimators=100, learning_rate=0.05, num_leaves=7, max_depth=3,
            min_child_samples=4, reg_lambda=5, n_jobs=2, random_state=config.SEED,
            deterministic=True, verbosity=-1,
        ))
    raise ValueError(f"Unknown video model {model_name!r}; choose from {MODEL_NAMES}")


def extract_development(protocol, cache, source_cache=None, encoder="numeric"):
    from src.video_eval import protocol_groups
    engine = load_engine(encoder)
    cache = Path(cache)
    source_cache = Path(source_cache) if source_cache is not None else cache
    cache.mkdir(parents=True, exist_ok=True)
    provenance = engine_provenance(encoder, engine)
    records = {}
    groups = protocol_groups(protocol, "development")
    for index, group in enumerate(groups):
        path = config.ROOT / group["path"]
        if digest(path) != group["group_id"]:
            raise ValueError("Video content no longer matches frozen protocol")
        cached = source_cache / f"{group['group_id']}.joblib"
        if cached.exists():
            item = joblib.load(cached)
            verify_feature_record(item, provenance, engine.FEATURE_CONFIG)
        else:
            started = time.monotonic()
            result = engine.extract_video(path, deadline=started + 240)
            item = {**provenance, "feature_config": engine.FEATURE_CONFIG,
                    "clip_features": np.asarray(result["clip_features"], dtype=float),
                    "windows": result["windows"], "duration_sec": result["duration_sec"],
                    "extraction_sec": time.monotonic() - started}
            if encoder == "qwen-spatial-v3":
                item["observation_audit"] = result["observation_audit"]
                item["sampling"] = result["sampling"]
            if encoder == "object-tracks":
                item["object_audit"] = result["object_audit"]
                item["sampling"] = result["sampling"]
            # An explicitly supplied older cache is read-only. Missing entries
            # are derived in the new run, never added to a previous run.
            cached = cache / f"{group['group_id']}.joblib"
            joblib.dump(item, cached)
        records[group["group_id"]] = item
        print(f"Extracted development group {index + 1}/{len(groups)} in {item['extraction_sec']:.3f}s", flush=True)
    if set(records) & set(protocol["holdout_group_ids"]):
        raise ValueError("Reserved holdout cannot enter development extraction")
    return records


def fit_window_model(groups, records):
    vectors, labels, weights = [], [], []
    for group in groups:
        windows = records[group["group_id"]]["windows"]
        if group["target"] == 0:
            selected = list(range(len(windows)))
        elif group["localization_eligible"]:
            selected = [int(np.argmin([abs(w["timestamp_sec"] - group["timestamp_sec"]) for w in windows]))]
        else:
            continue
        for index in selected:
            vectors.append(windows[index]["features"])
            labels.append(group["target"])
            weights.append(1 / len(selected))
    y, weights = np.asarray(labels), np.asarray(weights)
    if set(y) != {0, 1}:
        raise ValueError("Window model requires both positive and negative training clips")
    # Give each source clip equal total weight, then balance the two classes.
    for target in (0, 1):
        mask = y == target
        weights[mask] *= len(y) / (2 * weights[mask].sum())
    model = classifier()
    model.named_steps["logisticregression"].set_params(class_weight=None)
    model.fit(np.asarray(vectors), y, logisticregression__sample_weight=weights)
    return model


def shuffle_supervision(groups, rng):
    """Permute full weak-label bundles, including event location and ambiguity.

    A donor's relative event position is mapped to the recipient clip duration.
    Labels and timestamp eligibility travel together; the recipient's original
    class, event timestamp and eligibility are never used for supervision.
    """
    bundles = []
    for group in groups:
        eligible = bool(group["target"] and group["localization_eligible"])
        fraction = (float(group["timestamp_sec"]) / float(group["duration_sec"])) if eligible else None
        bundles.append((int(group["target"]), eligible, fraction))
    shuffled = []
    for group, donor_index in zip(groups, rng.permutation(len(groups))):
        target, eligible, fraction = bundles[int(donor_index)]
        shuffled.append({**group, "target": target, "localization_eligible": eligible,
                         "timestamp_sec": fraction * float(group["duration_sec"]) if eligible else None})
    return shuffled


def evaluate_development(protocol, records, shuffled_labels=None, model_name="linear", shuffled_groups=None):
    from src.video_eval import localization_metrics, protocol_groups, video_metrics
    groups = protocol_groups(protocol, "development")
    if shuffled_labels is not None and shuffled_groups is not None:
        raise ValueError("Supply shuffled labels or full supervision bundles, not both")
    if model_name == "window-linear" and shuffled_labels is not None:
        raise ValueError("Window-model sanity requires full shuffled supervision bundles")
    is_shuffled = shuffled_labels is not None or shuffled_groups is not None
    if shuffled_groups is not None:
        if (len(shuffled_groups) != len(groups)
                or {g["group_id"] for g in shuffled_groups} != {g["group_id"] for g in groups}):
            raise ValueError("Shuffled supervision must provide every development group exactly once")
        supplied = {g["group_id"]: g for g in shuffled_groups}
        groups = [supplied[g["group_id"]] for g in groups]
    by_id = {g["group_id"]: g for g in groups}
    ordered = [g["group_id"] for g in groups]
    targets = {g["group_id"]: g["target"] for g in groups}
    if shuffled_labels is not None:
        if len(shuffled_labels) != len(ordered) or not np.isin(shuffled_labels, [0, 1]).all():
            raise ValueError("Shuffled labels must provide one binary target per development group")
        targets = dict(zip(ordered, map(int, shuffled_labels)))
    predictions, event_times, fold_reports = {}, {}, []
    for index, fold in enumerate(protocol["folds"]):
        train_ids, valid_ids = fold["train_group_ids"], fold["valid_group_ids"]
        x_train = np.stack([records[key]["clip_features"] for key in train_ids])
        x_valid = np.stack([records[key]["clip_features"] for key in valid_ids])
        y_train, y_valid = [targets[key] for key in train_ids], [targets[key] for key in valid_ids]
        if model_name == "window-linear":
            window_model = fit_window_model([by_id[key] for key in train_ids], records)
            probabilities = []
            for key in valid_ids:
                windows = records[key]["windows"]
                scores = window_model.predict_proba(np.stack([w["features"] for w in windows]))[:, 1]
                candidate = int(np.argmax(scores))
                probabilities.append(float(scores[candidate]))
                event_times[key] = float(windows[candidate]["timestamp_sec"])
        else:
            model = classifier(model_name).fit(x_train, y_train)
            probabilities = model.predict_proba(x_valid)[:, 1]
        predictions.update(zip(valid_ids, map(float, probabilities)))
        if not is_shuffled and model_name != "window-linear":
            window_model = fit_window_model([by_id[key] for key in train_ids], records)
            for key in valid_ids:
                windows = records[key]["windows"]
                window_probabilities = window_model.predict_proba(np.stack([w["features"] for w in windows]))[:, 1]
                event_times[key] = float(windows[int(np.argmax(window_probabilities))]["timestamp_sec"])
        fold_reports.append({"fold": index, "train_groups": len(train_ids), "valid_groups": len(valid_ids),
                             "shared_groups": 0, **video_metrics(y_valid, probabilities)})
    y = [targets[key] for key in ordered]
    p = [predictions[key] for key in ordered]
    model_metrics = video_metrics(y, p)
    result = {"dummy": video_metrics(y, np.full(len(y), 0.5)),
              # Keep this internal legacy alias for existing callers. New
              # callers and reports use model/model_name or comparisons.
              "linear": model_metrics, "model": model_metrics, "model_name": model_name,
              "folds": fold_reports,
              "group_ids": ordered, "targets": y, "probabilities": p}
    if not is_shuffled:
        eligible = [g for g in groups if g["target"] and g["localization_eligible"]]
        result["localization"] = localization_metrics(
            [g["timestamp_sec"] for g in eligible], [event_times[g["group_id"]] for g in eligible],
            duration_sec=[g["duration_sec"] for g in eligible])
        result["localization"]["scope"] = "Candidate time on eligible positive clips, independent of verdict; sparse labels"
        result["candidate_times"] = event_times
    return result


def select_model(evaluations):
    """Choose once by pooled development AUC; prefer linear on an exact tie."""
    if not evaluations or any(name not in MODEL_NAMES for name in evaluations):
        raise ValueError("Provide nonempty evaluations of supported video models")
    for evaluation in evaluations.values():
        auc = evaluation["model"]["roc_auc"]
        if auc is None or not np.isfinite(auc):
            raise ValueError("Model selection requires a finite development ROC-AUC")
    return max(evaluations, key=lambda name: (
        evaluations[name]["model"]["roc_auc"], name == "linear",
    ))


def log_experiments(result, note, run):
    path = config.ROOT / "experiments.md"
    with path.open("a") as stream:
        for model, values in {"dummy": result["dummy"], **result["comparisons"]}.items():
            if model == "dummy":
                decision = "baseline floor"
            elif result["model"]["roc_auc"] > 0.95:
                decision = "drop pending suspicious-score audit"
            elif model != result["selected_model"]:
                decision = "drop: lower development ROC-AUC"
            elif values["roc_auc"] <= result["dummy"]["roc_auc"]:
                decision = "drop for detection quality; mechanics artifact only"
            else:
                decision = "pending sanity and runtime checks: development candidate"
            stream.write(f"| {dt.datetime.now():%m-%d %H:%M} | VIDEO {note} [{result.get('engine', 'numeric')}/{model}] | "
                         f"roc_auc={values['roc_auc']:.6f} | {decision} | "
                         f"Frozen group CV; F1={values['f1']:.6f}; {run.relative_to(config.ROOT)}; no holdout evaluated |\n")


def representation_comparison(result, baseline_run):
    """Compare actual saved scores only when development groups/policy match."""
    baseline_run = Path(baseline_run).resolve()
    baseline = json.loads((baseline_run / "development.json").read_text())
    for key in ("protocol_fingerprint", "seed", "threshold", "group_ids", "targets"):
        if baseline.get(key) != result.get(key):
            raise ValueError(f"Baseline comparison differs on frozen development field {key}")
    if baseline.get("holdout_evaluated") is not False:
        raise ValueError("Baseline comparison requires a development-only run")
    previous = baseline["comparisons"] if "comparisons" in baseline else {"linear": baseline["linear"]}
    return {"baseline_run": str(baseline_run), "baseline_engine": baseline.get("engine", "numeric"),
            "models": {name: {"before": previous[name], "after": metrics,
                              "roc_auc_delta": metrics["roc_auc"] - previous[name]["roc_auc"]}
                       for name, metrics in result["comparisons"].items() if name in previous}}


def main(argv=None):
    from src.video_eval import create_or_load_protocol, protocol_groups
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", choices=tuple(ENGINE_MODULES), default="numeric")
    parser.add_argument("--run-dir", type=Path, help="Fresh output directory; defaults to an encoder-specific run")
    parser.add_argument("--inventory", type=Path, default=config.DATA_PROC / "fortebank-20261017/inventory.csv")
    parser.add_argument("--protocol", type=Path, default=config.DATA_PROC / "fortebank-20261017/video_protocol.json")
    parser.add_argument("--models", help="Fixed models; default linear,lgbm, or window-linear for temporal encoder")
    parser.add_argument("--cache-dir", type=Path, help="Read and copy verified features from an earlier run")
    parser.add_argument("--baseline-run-dir", type=Path, help="Compare actual saved development scores under the same protocol")
    parser.add_argument("--note", default="P2 numeric visual baseline")
    args = parser.parse_args(argv)
    models = (args.models or ("window-linear" if args.encoder == "mobilenet-temporal" else "linear,lgbm")).split(",")
    if not models or len(models) != len(set(models)) or any(name not in MODEL_NAMES for name in models):
        parser.error(f"--models must contain unique names from {MODEL_NAMES}")
    if ("window-linear" in models) != (args.encoder == "mobilenet-temporal") or ("window-linear" in models and len(models) != 1):
        parser.error("window-linear is the sole model supported by mobilenet-temporal; use clip classifiers with other encoders")
    engine = load_engine(args.encoder)
    provenance = engine_provenance(args.encoder, engine)
    run = (args.run_dir or config.DATA_PROC / ("video-baseline" if args.encoder == "numeric" else f"video-{args.encoder}")).resolve()
    run.mkdir(parents=True, exist_ok=True)
    if (run / "development.json").exists() or (run / "models/baseline.joblib").exists():
        raise ValueError("Completed video run already exists; choose a fresh --run-dir")
    protocol = create_or_load_protocol(args.inventory, args.protocol)
    print("Frozen protocol:", protocol["fingerprint"], flush=True)
    print("Holdout groups reserved (not decoded/evaluated):", len(protocol["holdout_group_ids"]), flush=True)
    for index, fold in enumerate(protocol["folds"]):
        print(f"Fold {index}: {len(fold['train_group_ids'])} train / {len(fold['valid_group_ids'])} valid; shared groups=0", flush=True)
    # Copy trusted cached descriptors into the new run. This preserves
    # a self-contained immutable run while avoiding any repeat video decoding.
    cache = (run / "cache").resolve()
    if args.encoder == "qwen-spatial-v3":
        if args.cache_dir is not None:
            raise ValueError("Spatial v3 pilot owns its fresh run cache; do not provide an alternate cache")
        from scripts.check_spatial_pilot import ensure_pilot
        pilot = ensure_pilot(protocol, run)
        print("SPATIAL PILOT", json.dumps(pilot), flush=True)
        if not pilot["passed"]:
            raise RuntimeError("Spatial observation pilot failed its fixed operational gate; stop this recipe")
    records = extract_development(protocol, cache, source_cache=args.cache_dir, encoder=args.encoder)
    if args.cache_dir is not None:
        for key, item in records.items():
            destination = cache / f"{key}.joblib"
            if not destination.exists():
                joblib.dump(item, destination)
    evaluations = {name: evaluate_development(protocol, records, model_name=name) for name in models}
    selected = select_model(evaluations)
    result = evaluations[selected]
    result.update(comparisons={name: evaluation["model"] for name, evaluation in evaluations.items()},
                  selected_model=selected, selection_policy="Highest pooled development ROC-AUC; exact tie prefers linear",
                  cache_dir=str(cache), source_cache_dir=str(args.cache_dir.resolve()) if args.cache_dir else None)
    result.update(protocol_fingerprint=protocol["fingerprint"], seed=config.SEED,
                  **provenance, feature_config=engine.FEATURE_CONFIG, threshold=0.5, holdout_evaluated=False,
                  limitations=["Visual similarity model; no verified event taxonomy or legal mapping",
                               "Camera/session independence beyond exact duplicates not established",
                               "Only 32 development recordings; high score uncertainty",
                               "Localization uses sparse timestamps and weak window labels"])
    if args.encoder in {"qwen-actions", "qwen-spatial-v3"}:
        result["limitations"].append(
            "Pretrained action observations are numeric inputs evaluated against original binary targets; "
            "they are not independently verified action labels or action-type accuracy evidence"
        )
    if args.encoder == "object-tracks":
        result["limitations"].append(
            "Object detections and anonymous tracks are pretrained predictions; no cash, hand, identity, "
            "verified action taxonomy or automatic rule association is supplied by this encoder"
        )
    if args.baseline_run_dir is not None:
        result["representation_comparison"] = representation_comparison(result, args.baseline_run_dir)
    (run / "development.json").write_text(json.dumps(result, indent=2) + "\n")
    log_experiments(result, args.note, run)
    print("DUMMY", json.dumps(result["dummy"]), flush=True)
    for name, metrics in result["comparisons"].items():
        print(name.upper(), json.dumps(metrics), flush=True)
    print("SELECTED", selected, flush=True)
    print("LOCALIZATION", json.dumps(result["localization"]), flush=True)
    if "representation_comparison" in result:
        for name, comparison in result["representation_comparison"]["models"].items():
            print(f"BEFORE -> AFTER ({name}, identical frozen development groups): "
                  f"ROC-AUC {comparison['before']['roc_auc']:.6f} -> {comparison['after']['roc_auc']:.6f}", flush=True)
    if result["model"]["roc_auc"] > 0.95:
        raise RuntimeError("Suspicious development AUC above 0.95; report and audit before serving")
    groups = protocol_groups(protocol, "development")
    x = np.stack([records[g["group_id"]]["clip_features"] for g in groups])
    y = [g["target"] for g in groups]
    model = None if selected == "window-linear" else classifier(selected).fit(x, y)
    window_model = fit_window_model(groups, records)
    bundle = {"schema_version": 1, "clip_model": model, "window_model": window_model,
              "threshold": 0.5, "feature_names": {"clip": engine.CLIP_FEATURE_NAMES, "window": engine.WINDOW_FEATURE_NAMES},
              "feature_config": engine.FEATURE_CONFIG, **provenance,
              "protocol_fingerprint": protocol["fingerprint"], "training_groups": [g["group_id"] for g in groups],
              "selected_model": selected,
              "stage": "development-only", "seed": config.SEED,
              "limitations": result["limitations"]}
    models = run / "models"
    models.mkdir(exist_ok=True)
    joblib.dump(bundle, models / "baseline.joblib")
    (run / "model_manifest.json").write_text(json.dumps({
        "model_sha256": digest(models / "baseline.joblib"), "protocol_fingerprint": protocol["fingerprint"],
        **provenance, "feature_config": engine.FEATURE_CONFIG,
        "selected_model": selected, "cache_dir": str(cache),
        "stage": "development-only", "training_groups": bundle["training_groups"]}, indent=2) + "\n")
    print("Development artifact saved:", models / "baseline.joblib", flush=True)


if __name__ == "__main__":
    main()
