"""Development-only local video leakage checks; never decode reserved holdout."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score

from src import config
from src.video_train import (digest, engine_provenance, evaluate_development,
                             load_engine, shuffle_supervision, verify_feature_record)


def main(argv=None):
    from src.video_eval import load_protocol, protocol_groups
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=config.DATA_PROC / "video-baseline")
    parser.add_argument("--protocol", type=Path, default=config.DATA_PROC / "fortebank-20261017/video_protocol.json")
    parser.add_argument("--shuffle-trials", type=int, default=20)
    args = parser.parse_args(argv)
    if args.shuffle_trials < 1:
        parser.error("--shuffle-trials must be positive")
    protocol = load_protocol(args.protocol)
    development = json.loads((args.run_dir / "development.json").read_text())
    encoder = development.get("engine", "numeric")
    engine = load_engine(encoder)
    provenance = engine_provenance(encoder, engine)
    if development["protocol_fingerprint"] != protocol["fingerprint"] or development["feature_config"] != engine.FEATURE_CONFIG:
        raise ValueError("Training run differs from the loaded protocol or feature configuration")
    # Legacy numeric reports omitted provenance. Every newly routed run includes
    # it and must agree with the current encoder and inherited decoder helpers.
    if "engine" in development:
        verify_feature_record(development, provenance, engine.FEATURE_CONFIG)
    selected_model = development.get("selected_model", "linear")
    cache_dir = Path(development.get("cache_dir", args.run_dir / "cache"))
    groups = protocol_groups(protocol, "development")
    records = {}
    for group in groups:
        item = joblib.load(cache_dir / f"{group['group_id']}.joblib")
        verify_feature_record(item, provenance, engine.FEATURE_CONFIG)
        if digest(config.ROOT / group["path"]) != group["group_id"]:
            raise ValueError("Input video differs from frozen content group")
        records[group["group_id"]] = item
    holdout_ids = set(protocol["holdout_group_ids"])
    assert not set(records) & holdout_ids
    assert len(records) == len(groups)
    x = np.stack([records[g["group_id"]]["clip_features"] for g in groups])
    if x.shape[1] != len(engine.CLIP_FEATURE_NAMES) or not np.isfinite(x).all():
        raise ValueError("Cached feature dimensions or values differ from the encoder schema")
    y = np.asarray([g["target"] for g in groups])
    suspicious = []
    for index, name in enumerate(engine.CLIP_FEATURE_NAMES):
        if len(np.unique(x[:, index])) < 2:
            continue
        auc = float(roc_auc_score(y, x[:, index]))
        oriented = max(auc, 1 - auc)
        if oriented > 0.90:
            suspicious.append({"feature": name, "auc": oriented})
    rng = np.random.default_rng(config.SEED)
    shuffled_scores = []
    for index in range(args.shuffle_trials):
        if selected_model == "window-linear":
            shuffled = shuffle_supervision(groups, rng)
            result = evaluate_development(protocol, records, shuffled_groups=shuffled, model_name=selected_model)
        else:
            shuffled = rng.permutation(y)
            result = evaluate_development(protocol, records, shuffled_labels=shuffled, model_name=selected_model)
        shuffled_scores.append(result["model"]["roc_auc"])
    mean = float(np.mean(shuffled_scores))
    problems = []
    if suspicious:
        problems.append("Single-feature AUC above 0.90 requires review before claiming leakage checks passed")
    if abs(mean - 0.5) > 0.10:
        problems.append("Mean shuffled-target AUC differs materially from chance; investigate")
    result = {"seed": config.SEED, **provenance, "selected_model": selected_model,
              "feature_count": len(engine.CLIP_FEATURE_NAMES), "development_groups": len(groups),
              "holdout_groups_decoded": 0, "shared_groups": 0,
              "protocol_fingerprint": protocol["fingerprint"],
              "shuffled_auc_mean": mean, "shuffled_auc_std": float(np.std(shuffled_scores)),
              "shuffled_auc_trials": shuffled_scores, "dummy_auc": 0.5,
              "shuffle_policy": ("Atomic target/eligibility/relative-timestamp bundle permutation"
                                 if selected_model == "window-linear" else "Video target permutation"),
              "suspicious_single_features": suspicious, "problems": problems,
              "limitations": [f"{args.shuffle_trials} permutations on {len(groups)} clips; diagnostic not proof of no leakage",
                              "Single-feature scan is unadjusted for multiple comparisons; pretrained channels are numerous and correlated",
                              "Perceptual/session grouping and event semantics remain unverified",
                              "No adversarial comparison with reserved or jury videos was performed"]}
    if encoder in {"qwen-actions", "qwen-spatial-v3"}:
        result["limitations"].append(
            "This audit tests original binary detection targets; pretrained action proposals are not verified action truth"
        )
    (args.run_dir / "sanity.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Video sanity — development only", "",
             f"Encoder: {encoder}; classifier: {selected_model}; clip features tested: {len(engine.CLIP_FEATURE_NAMES)}.",
             f"Development groups: {len(groups)}; shared groups: 0; holdout groups decoded: 0.",
             f"Shuffled-target ROC-AUC across {args.shuffle_trials} permutations: mean {mean:.6f}; std {np.std(shuffled_scores):.6f}; dummy 0.500000.",
             f"Single-feature AUC > 0.90: {len(suspicious)}.", "", "## Problems",
             *([f"- {problem}" for problem in problems] or ["- None detected by these checks."]),
             "", "The single-feature scan is unadjusted for multiple comparisons. Thousands of correlated pretrained channels can yield high in-sample AUC by chance; flags still require review.",
             "", "These checks do not establish event meaning, legal grounding, session independence or deployment accuracy."]
    (args.run_dir / "sanity.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "shuffled_auc_trials"}), flush=True)
    if problems:
        raise SystemExit("Video sanity checks require review; do not claim a passed audit")


if __name__ == "__main__":
    main()
