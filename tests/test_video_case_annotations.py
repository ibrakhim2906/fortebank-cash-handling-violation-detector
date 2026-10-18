"""Synthetic documents and metadata only; no bank footage, models or proposals."""
from copy import deepcopy
import hashlib
import json
import stat
from zipfile import ZipFile

import pandas as pd
import pytest

from src import video_case_annotations as annotations
from src.video_eval import _fingerprint, create_or_load_protocol


@pytest.fixture
def source(tmp_path, monkeypatch):
    processed = tmp_path / "processed"
    monkeypatch.setattr(annotations.config, "DATA_PROC", processed)
    rows = [{"video_id": f"Х-{i}", "path": f"videos/Х-{i}.mov",
             "sha256": hashlib.sha256(f"synthetic-{i}".encode()).hexdigest(),
             "target": int(i <= 21), "timestamp_sec": 10.0 if i <= 21 else None,
             "duration_sec": 100.0} for i in range(1, 41)]
    rows[3]["timestamp_sec"] = 57.0
    rows[12]["timestamp_sec"] = 47.0
    rows.append({**rows[3], "video_id": "Х-47", "path": "videos/Х-47.mov", "timestamp_sec": 68.0})
    rows.append({**rows[39], "video_id": "Х-48", "path": "videos/Х-48.mov"})
    inventory = tmp_path / "inventory.csv"
    pd.DataFrame(rows).to_csv(inventory, index=False)
    protocol_path = tmp_path / "protocol.json"
    protocol = create_or_load_protocol(inventory, protocol_path)
    csv_rows, note_rows = ["video,second,type"], []
    for row in rows:
        if not row["target"]:
            continue
        alias, second = row["video_id"], int(row["timestamp_sec"])
        types = ["type_a", "type_b"] if alias == "Х-4" else ["type_b"] if alias == "Х-47" else ["type_a"]
        csv_rows.extend(f"{alias},{second},{kind}" for kind in types)
        note_rows.append(f"| {alias} | {second} | Human observation 50000, not inferred by a model | "
                         + ", ".join(f"`{kind}`" for kind in types) + " |")
    members = {
        "docs/annotations.md": ("# Human notes\n" + "\n".join(note_rows) + "\n").encode(),
        "docs/case/types.csv": ("\n".join(csv_rows) + "\n").encode(),
        "docs/case/rules.seed.json": json.dumps({"types": {
            "type_a": {"name": "Synthetic A", "enabled": True},
            "type_b": {"name": "Synthetic B", "enabled": False}}}).encode(),
        "docs/case/legal.md": b"# Candidate source; inert, not verified.\n",
    }
    archive = tmp_path / "candidate.zip"
    def rewrite():
        with ZipFile(archive, "w") as z:
            for name, payload in members.items():
                z.writestr(name, payload)
        monkeypatch.setattr(annotations, "ARCHIVE_SHA256", hashlib.sha256(archive.read_bytes()).hexdigest())
    rewrite()
    return {"archive": archive, "protocol_path": protocol_path, "protocol": protocol,
            "members": members, "rewrite": rewrite, "out": processed / "annotations"}


def run_import(source, version="v1"):
    return annotations.import_annotations(source["archive"], source["protocol_path"], source["out"], version)


def test_roundtrip_preserves_human_origin_and_inert_source_bytes(source):
    summary = run_import(source)
    assert summary == {"status": "imported", "schema_version": 1, "annotation_version": "v1",
                       "recording_groups": 40, "source_aliases": 42, "type_assignments": 23,
                       "grouped_events": 22, "event_types": 2,
                       "human_annotated_recording_groups": 21, "flagged_events": 1,
                       "active_rules": 0, "machine_proposals_imported": 0,
                       "independent_holdout_remaining": 0}
    directory = source["out"] / "v1"
    loaded = annotations.load_annotations(directory, source["protocol_path"])
    assert loaded["origin"] == annotations.ORIGIN
    assert not loaded["origin"]["organizer_original_type_ground_truth"]
    assert not loaded["origin"]["machine_proposal"]
    assert not loaded["candidate_rules_executable"]
    assert loaded["active_rules"] == []
    for name, data in source["members"].items():
        assert (directory / "source" / name).read_bytes() == data
    assert json.loads((directory / "source/docs/case/rules.seed.json").read_bytes())["types"]["type_a"]["enabled"]
    assert all(stat.S_IMODE(p.stat().st_mode) == 0o600 for p in directory.rglob("*") if p.is_file())


def test_duplicate_content_keeps_two_anchors_and_multilabel_event(source):
    run_import(source)
    d = annotations.load_annotations(source["out"] / "v1", source["protocol_path"])
    recording = next(g for g in d["recordings"] if "Х-47" in g["aliases"])
    events = [e for e in d["events"] if e["group_id"] == recording["group_id"]]
    assert [e["timecode_sec"] for e in events] == [57.0, 68.0]
    assert events[0]["type_ids"] == ["type_a", "type_b"]
    assert len(events[0]["source_assignments"]) == 2
    assert len(recording["human_event_ids"]) == 2
    assert not recording["organizer_weak_labels"]["localization_eligible_in_original_protocol"]


def test_all_reviewed_groups_are_development_without_changing_protocol(source):
    before = source["protocol_path"].read_bytes()
    run_import(source)
    d = annotations.load_annotations(source["out"] / "v1", source["protocol_path"])
    assert source["protocol_path"].read_bytes() == before
    assert len([g for g in d["recordings"] if g["historical_partition"] == "former_reserved_consumed"]) == 8
    assert {g["evaluation_role"] for g in d["recordings"]} == {"development"}
    assert d["evaluation_disclosure"]["independent_holdout_remaining"] == 0
    assert all(g["no_human_events_does_not_establish_action_absence"] for g in d["recordings"])
    flagged = [e for e in d["events"] if e["flags"]]
    assert len(flagged) == 1 and flagged[0]["timecode_sec"] == 47
    assert "Human observation" in flagged[0]["human_notes"][0]["description_original"]


def test_archive_digest_checked_before_any_output(source):
    source["archive"].write_bytes(source["archive"].read_bytes() + b"changed")
    with pytest.raises(ValueError, match="archive SHA-256"):
        run_import(source)
    assert not source["out"].exists()


def test_versions_never_overwrite_and_new_version_preserves_prior(source):
    run_import(source)
    files = {str(p.relative_to(source["out"])): p.read_bytes()
             for p in source["out"].rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError):
        run_import(source)
    run_import(source, "v2")
    for name, data in files.items():
        assert (source["out"] / name).read_bytes() == data


def test_normalized_latin_cyrillic_collision_rejected(source):
    d = deepcopy(source["protocol"])
    first = next(g for g in d["groups"] if "Х-1" in g["video_ids"])
    second = next(g for g in d["groups"] if "Х-2" in g["video_ids"])
    first["video_ids"][0] = "X-1"
    first["timestamp_annotations"][0]["video_id"] = "X-1"
    second["video_ids"][0] = "Х-1"
    second["timestamp_annotations"][0]["video_id"] = "Х-1"
    d["fingerprint"] = _fingerprint(d)
    source["protocol_path"].write_text(json.dumps(d))
    with pytest.raises(ValueError, match="alias collision"):
        run_import(source)


@pytest.mark.parametrize("text,match", [
    ("video,second,type\nХ-999,10,type_a\n", "absent from frozen protocol"),
    ("video,second,type\nХ-1,10,unrecognized\n", "unknown type"),
    ("video,second,type\nХ-1,nan,type_a\n", "outside recording"),
    ("video,second,type\nХ-1,101,type_a\n", "outside recording"),
    ("video,second,type\nХ-1,12,type_a\n", "original binary/anchor"),
    ("video,second,type\nХ-40,10,type_a\n", "original binary/anchor"),
    ("video,second,type\nХ-1,10,type_a\nХ-1,10,type_a\n", "Duplicate human"),
])
def test_invalid_assignments_rejected_without_writing(source, text, match):
    source["members"]["docs/case/types.csv"] = text.encode()
    source["rewrite"]()
    with pytest.raises(ValueError, match=match):
        run_import(source)
    assert not source["out"].exists()


def test_missing_note_and_unassigned_note_cannot_silently_drop_targets(source):
    original = source["members"]["docs/annotations.md"]
    source["members"]["docs/annotations.md"] = original.replace(b"`type_a`", b"`wrong_type`", 1)
    source["rewrite"]()
    with pytest.raises(ValueError, match="not supported"):
        run_import(source)


@pytest.mark.parametrize("mutation", ["origin", "active_rules", "event", "source"])
def test_loader_rejects_source_or_normalized_provenance_drift(source, mutation):
    run_import(source)
    directory = source["out"] / "v1"
    if mutation == "source":
        path = directory / "source/docs/annotations.md"
        path.write_bytes(path.read_bytes() + b"\nChanged note")
    else:
        path = directory / "normalized.json"
        d = json.loads(path.read_bytes())
        if mutation == "origin":
            d["origin"]["organizer_original_type_ground_truth"] = True
        elif mutation == "active_rules":
            d["active_rules"] = ["type_a"]
        else:
            d["events"][0]["type_ids"] = ["type_b"]
        path.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="checksum mismatch|differs from preserved"):
        annotations.load_annotations(directory, source["protocol_path"])


def test_raw_or_escaping_outputs_are_denied(source):
    with pytest.raises(ValueError, match="processed data"):
        annotations.import_annotations(source["archive"], source["protocol_path"], source["out"].parent.parent / "raw", "v1")
    with pytest.raises(ValueError, match="Invalid annotation version"):
        run_import(source, "../overwrite")


def test_cli_emits_aggregate_only(source, capsys):
    from scripts.import_case_annotations import main
    main(["--archive", str(source["archive"]), "--protocol", str(source["protocol_path"]),
          "--out-dir", str(source["out"])])
    printed = capsys.readouterr().out
    assert json.loads(printed)["type_assignments"] == 23
    assert "Human observation" not in printed and "Х-" not in printed
    assert '"type_a"' not in printed and '"type_b"' not in printed
