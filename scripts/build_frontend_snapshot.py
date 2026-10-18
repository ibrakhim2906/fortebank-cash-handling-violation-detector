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
    ("32B · storage veto (selected)", "32B selected", "Storage veto", "mission-claude-selected-dev30-honest.json"),
    ("32B · storage veto + rescue", "32B + rescue", "Rescue recall pass", "mission-claude-recall-rescue-dev30-honest.json"),
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
    runtime, digest = read_source("mission-checkpoint-3-storage-qualified-v1/actual-api-runtime.json")
    clean = runtime["clean_primary_qualified"]
    return {
        "schema_version": 1,
        "snapshot_label": "18 Oct 2026 · checkpoint 3 + rescue",
        "target_accuracy": 0.90,
        "target_status": "aspirational_not_achieved",
        "independent_test": False,
        "recipes": recipes,
        "checkpoint": {
            "source": "data/processed/mission-checkpoint-3-storage-qualified-v1/actual-api-runtime.json",
            "source_sha256": digest,
            "api_jobs": runtime["unique_total_actual_job_ids"],
            "clean_uploads": clean["n_videos"],
            "failed_jobs": clean["failed"],
            "worst_sec": clean["worst_sec"],
            "repeat_worst_sec": runtime["qualified_repeat_worst_sec"],
            "limit_sec": 300,
            "full_set_qualified": runtime["all40_primary_completed"] is True and runtime["actual_api_latency_qualified"] is True,
            "transform_maxima_sec": [],
        },
    }


if __name__ == "__main__":
    snapshot = build_snapshot()
    destination = ROOT / "service/static/evaluation.json"
    destination.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"Saved {destination.relative_to(ROOT)}: aggregate-only, {len(snapshot['recipes'])} measured recipes")
    for recipe in snapshot["recipes"]:
        print(f"{recipe['name']}: accuracy={recipe['accuracy']:.6f}, recall={recipe['recall']:.6f}, timed={recipe['localized_2s']}/{recipe['labelled_events']}")
