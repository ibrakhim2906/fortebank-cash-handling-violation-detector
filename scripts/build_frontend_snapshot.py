"""Publish aggregate measured results only; never copy clips, labels, or frames."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECIPES = [
    ("Original 8B · strict", "Original 8B", "Strict confirmation", "mission-v1-dev-honest.json"),
    ("Source-resolution 8B", "Source 8B", "Source-resolution", "mission-source-dev-30-honest.json"),
    ("32B · proposal policy", "32B proposal", "Proposal policy", "mission-32b-proposal-dev30-report.json"),
]


def read_source(relative: str):
    path = ROOT / "data/processed" / relative
    content = path.read_bytes()
    return json.loads(content), hashlib.sha256(content).hexdigest()


def build_snapshot():
    recipes = []
    for name, short, variant, filename in RECIPES:
        value, digest = read_source(filename)
        row = {key: value[key] for key in (
            "n_videos", "accuracy", "recall", "tp", "tn", "fp", "fn", "labelled_events", "hit_1s", "hit_2s", "failed"
        )}
        assert all(isinstance(row[key], (int, float)) for key in row)
        assert row["n_videos"] == 30 and row["labelled_events"] == 16
        assert abs(row["accuracy"] - (row["tp"] + row["tn"]) / row["n_videos"]) < 1e-9
        row.update(
            name=name, short_name=short, variant=variant,
            source=f"data/processed/{filename}", source_sha256=digest,
            correct=row["tp"] + row["tn"],
            localized_2s=round(row["hit_2s"] * row["labelled_events"]),
        )
        recipes.append(row)
    checkpoint, digest = read_source("mission-checkpoint-1/runtime-and-primary-audit.json")
    summary, summary_digest = read_source("mission-checkpoint-1/summary.json")
    # The summary API subset contains only clean uploads; the separate runtime
    # audit supplies the true maximum over ALL 44 observed jobs, including transforms.
    return {
        "schema_version": 1,
        "snapshot_label": "18 Oct 2026 · checkpoint 1",
        "target_accuracy": 0.90,
        "target_status": "aspirational_not_achieved",
        "independent_test": False,
        "recipes": recipes,
        "checkpoint": {
            "source": "data/processed/mission-checkpoint-1/runtime-and-primary-audit.json",
            "source_sha256": digest,
            "summary_sha256": summary_digest,
            "api_jobs": checkpoint["api_jobs"],
            "failed_jobs": summary["api"]["failed"],
            "worst_sec": checkpoint["worst_all_observed_api_sec"],
            "limit_sec": 300,
            "full_set_qualified": checkpoint["full_set_qualified"],
            "transform_maxima_sec": [t["worst_sec"] for t in checkpoint["transforms"]],
        },
    }


if __name__ == "__main__":
    snapshot = build_snapshot()
    destination = ROOT / "service/static/evaluation.json"
    destination.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"Saved {destination.relative_to(ROOT)}: aggregate-only, {len(snapshot['recipes'])} measured recipes")
    for recipe in snapshot["recipes"]:
        print(f"{recipe['name']}: accuracy={recipe['accuracy']:.6f}, recall={recipe['recall']:.6f}, timed={recipe['localized_2s']}/{recipe['labelled_events']}")
