"""Immutable team-human development annotations; never inference rules or footage.

The original organizer binary/anchor metadata, supplied human type notes and
earlier machine proposals are separate sources. All supplied clips were released
for review; the historical protocol is preserved without a new holdout claim.
"""
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import unicodedata
from zipfile import ZipFile

from src import config
from src.video_eval import load_protocol, validate_protocol


ARCHIVE_SHA256 = "133867918f704d22d87dc4770b6ae4854e5206d993fe7bd542630c3d1e9d797c"
DEFAULT_ARCHIVE = Path("/mnt/c/Users/ibrak/Downloads/kit-video-v5.zip")
DEFAULT_PROTOCOL = config.DATA_PROC / "fortebank-20261017/video_protocol.json"
DEFAULT_OUTPUT = config.DATA_PROC / "team-annotations-v5"
SOURCE_MEMBERS = (
    "docs/annotations.md", "docs/case/types.csv",
    "docs/case/rules.seed.json", "docs/case/legal.md",
)
ORIGIN = {
    "kind": "team_human_reviewed_development_annotation",
    "confirmation": "Human-reviewed notes after watching the videos",
    "confirmation_source": "human_chat_reply",
    "organizer_original_type_ground_truth": False,
    "machine_proposal": False,
    "legal_applicability_verified": False,
}


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def canonical_alias(value):
    """Normalize Latin/Cyrillic X and optional .mov; never guess other names."""
    match = re.fullmatch(r"[XxХх]-([1-9][0-9]*)(?:\.mov)?",
                         unicodedata.normalize("NFC", value.strip()), re.IGNORECASE)
    if not match:
        raise ValueError("Unsupported video alias")
    return "Х-" + str(int(match.group(1)))


def _alias_index(protocol):
    validate_protocol(protocol)
    result = {}
    for group in protocol["groups"]:
        for alias in group["video_ids"]:
            key = canonical_alias(alias)
            if key in result:
                raise ValueError("Normalized video alias collision")
            result[key] = (group, alias)
    return result


def _notes(payload):
    notes = {}
    for line_number, line in enumerate(payload.decode("utf-8-sig").splitlines(), 1):
        cells = [part.strip() for part in line.split("|")]
        if len(cells) != 6 or not cells[1] or not cells[2].isdigit():
            continue
        # Parenthetical duplicate annotation is retained verbatim below.
        alias = canonical_alias(cells[1].split()[0])
        key = (alias, float(cells[2]))
        if key in notes:
            raise ValueError("Repeated human annotation note")
        notes[key] = {"source_line": line_number, "verbatim_row": line,
                      "description_original": cells[3],
                      "type_ids": re.findall(r"`([a-z_]+)`", cells[4])}
    return notes


def _normalize(protocol, members, source_manifest_sha256, version):
    aliases = _alias_index(protocol)
    seed = json.loads(members["docs/case/rules.seed.json"])
    taxonomy = seed.get("types")
    if not isinstance(taxonomy, dict) or not taxonomy:
        raise ValueError("Candidate taxonomy is missing")
    notes = _notes(members["docs/annotations.md"])
    reader = csv.DictReader(io.StringIO(members["docs/case/types.csv"].decode("utf-8-sig")))
    if reader.fieldnames != ["video", "second", "type"]:
        raise ValueError("Unexpected type annotation columns")
    events, assignments, seen = {}, [], set()
    for row_number, row in enumerate(reader, 2):
        key = canonical_alias(row["video"])
        if key not in aliases:
            raise ValueError("Human annotation alias is absent from frozen protocol")
        group, exact_alias = aliases[key]
        try:
            second = float(row["second"])
        except (TypeError, ValueError) as exc:
            raise ValueError("Human annotation timestamp is invalid") from exc
        if not math.isfinite(second) or not 0 <= second <= group["duration_sec"]:
            raise ValueError("Human annotation timestamp is outside recording")
        official_anchor = next(x["timestamp_sec"] for x in group["timestamp_annotations"]
                               if x["video_id"] == exact_alias)
        if group["target"] != 1 or official_anchor is None or abs(second - official_anchor) > 1e-6:
            raise ValueError("Human annotation conflicts with original binary/anchor metadata")
        type_id = row["type"]
        if type_id not in taxonomy:
            raise ValueError("Human annotation has unknown type")
        identity = (key, second, type_id)
        if identity in seen:
            raise ValueError("Duplicate human type assignment")
        seen.add(identity)
        note = notes.get((key, second))
        if note is None or type_id not in note["type_ids"]:
            raise ValueError("Type assignment is not supported by supplied human note")
        event_key = (group["group_id"], second)
        if event_key not in events:
            events[event_key] = {
                "event_id": _sha(_canonical({"group_id": group["group_id"], "timecode_sec": second})),
                "group_id": group["group_id"], "timecode_sec": second,
                "type_ids": [], "source_assignments": [], "human_notes": [],
                "origin": dict(ORIGIN), "status": "human_reviewed_development_target",
                "visibility_and_rule_applicability": "requires_qualified_evidence",
                "flags": [],
            }
        event = events[event_key]
        assignment = {"source_row": row_number, "source_video_alias": row["video"],
                      "protocol_video_alias": exact_alias, "group_id": group["group_id"],
                      "timecode_sec": second, "type_id": type_id, "event_id": event["event_id"]}
        assignments.append(assignment)
        event["source_assignments"].append(dict(assignment))
        if type_id not in event["type_ids"]:
            event["type_ids"].append(type_id)
        attributed_note = {"source_video_alias": row["video"], **note}
        if attributed_note not in event["human_notes"]:
            event["human_notes"].append(attributed_note)
        if key == "Х-13":
            event["flags"] = [{"code": "prior_assistant_semantic_disagreement",
                               "resolution": "Human note unchanged; retain source timing and reconcile evidence separately. Prior assistant review observed storage around49–50s and closure around51s; this does not erase the human target at47s.",
                               "reference": "reports/case_video_review.md"}]
    expected = {(key, second, type_id) for (key, second), note in notes.items()
                for type_id in note["type_ids"]}
    if not assignments or seen != expected:
        raise ValueError("Human notes and type assignments do not cover the same targets")
    positive_ids = {g["group_id"] for g in protocol["groups"] if g["target"]}
    if {event["group_id"] for event in events.values()} != positive_ids:
        raise ValueError("Human targets must cover each original positive recording")
    if {row["type_id"] for row in assignments} != set(taxonomy):
        raise ValueError("Human taxonomy and assignments do not cover the same types")
    historical = set(protocol["development_group_ids"])
    recordings = [{
        "group_id": g["group_id"], "source_sha256": g["sha256"],
        "aliases": list(g["video_ids"]), "paths": list(g["paths"]),
        "duration_sec": g["duration_sec"], "evaluation_role": "development",
        "historical_partition": "original_development" if g["group_id"] in historical else "former_reserved_consumed",
        "organizer_weak_labels": {"binary_target": g["target"],
                                  "timestamp_annotations": g["timestamp_annotations"],
                                  "localization_eligible_in_original_protocol": g["localization_eligible"]},
        "human_event_ids": sorted(e["event_id"] for e in events.values() if e["group_id"] == g["group_id"]),
        "no_human_events_does_not_establish_action_absence": True,
    } for g in protocol["groups"]]
    result = {
        "schema_version": 1, "annotation_version": version,
        "origin": dict(ORIGIN), "protocol_fingerprint": protocol["fingerprint"],
        "source_manifest_sha256": source_manifest_sha256,
        "archive_sha256": ARCHIVE_SHA256, "recordings": recordings,
        "assignments": assignments,
        "events": sorted(events.values(), key=lambda e: (e["group_id"], e["timecode_sec"])),
        "taxonomy": {key: {"name": value["name"], "name_en": value.get("name_en")}
                     for key, value in taxonomy.items()},
        "candidate_rules_executable": False, "active_rules": [],
        "evaluation_disclosure": {"all_supplied_recordings_reviewed": True,
                                  "independent_holdout_remaining": 0,
                                  "frozen_protocol_changed": False,
                                  "new_independent_performance_claim": False},
        "machine_proposals_imported": 0,
    }
    result["fingerprint"] = _sha(_canonical(result))
    return result


def _write_exclusive(path, payload):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(payload)


def import_annotations(archive_path=DEFAULT_ARCHIVE, protocol_path=DEFAULT_PROTOCOL,
                       out_dir=DEFAULT_OUTPUT, version="v1"):
    """Read pinned documentation only; create a new exclusive immutable version."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", version):
        raise ValueError("Invalid annotation version")
    output = Path(out_dir).resolve()
    if not output.is_relative_to(config.DATA_PROC.resolve()):
        raise ValueError("Annotation output must be derived processed data")
    payload = Path(archive_path).read_bytes()
    if _sha(payload) != ARCHIVE_SHA256:
        raise ValueError("Supplied archive SHA-256 does not match approved source")
    with ZipFile(io.BytesIO(payload)) as archive:
        if any(archive.namelist().count(name) != 1 for name in SOURCE_MEMBERS):
            raise ValueError("Archive source members must be unique and complete")
        if any(archive.getinfo(name).file_size > 2_000_000 for name in SOURCE_MEMBERS):
            raise ValueError("Source document exceeds bounded import size")
        members = {name: archive.read(name) for name in SOURCE_MEMBERS}
    protocol = load_protocol(protocol_path)
    manifest = {"schema_version": 1, "archive_sha256": ARCHIVE_SHA256,
                "origin": dict(ORIGIN), "candidate_rules_inert": True,
                "sources": [{"archive_member": name, "preserved_file": "source/" + name,
                             "size_bytes": len(data), "sha256": _sha(data)}
                            for name, data in members.items()]}
    manifest_bytes = _json_bytes(manifest)
    normalized = _normalize(protocol, members, _sha(manifest_bytes), version)
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    destination = output / version
    destination.mkdir(mode=0o700)  # Existing versions are never replaced, even after failure.
    for name, data in members.items():
        _write_exclusive(destination / "source" / name, data)
    _write_exclusive(destination / "source_manifest.json", manifest_bytes)
    _write_exclusive(destination / "normalized.json", _json_bytes(normalized))
    summary = annotation_summary(normalized)
    _write_exclusive(destination / "summary.json", _json_bytes(summary))
    return summary


def load_annotations(version_dir, protocol_path=DEFAULT_PROTOCOL):
    """Verify preserved member hashes and rebuild normalized targets before use."""
    directory = Path(version_dir)
    manifest_bytes = (directory / "source_manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if (manifest.get("schema_version") != 1 or manifest.get("archive_sha256") != ARCHIVE_SHA256
            or manifest.get("origin") != ORIGIN or manifest.get("candidate_rules_inert") is not True):
        raise ValueError("Annotation source provenance differs from approved human origin")
    members = {}
    for entry in manifest["sources"]:
        name = entry["archive_member"]
        if name not in SOURCE_MEMBERS or name in members or entry["preserved_file"] != "source/" + name:
            raise ValueError("Unexpected or repeated source member")
        data = (directory / entry["preserved_file"]).read_bytes()
        if _sha(data) != entry["sha256"] or len(data) != entry["size_bytes"]:
            raise ValueError("Preserved source document checksum mismatch")
        members[name] = data
    if set(members) != set(SOURCE_MEMBERS):
        raise ValueError("Source manifest does not preserve every required document")
    actual = json.loads((directory / "normalized.json").read_bytes())
    expected = _normalize(load_protocol(protocol_path), members, _sha(manifest_bytes), actual["annotation_version"])
    if actual != expected:
        raise ValueError("Normalized annotation content or provenance differs from preserved sources")
    return actual


def annotation_summary(annotations):
    """Only nonsensitive aggregate output; no labels, aliases or note text."""
    recordings = annotations["recordings"]
    return {"status": "imported", "schema_version": 1,
            "annotation_version": annotations["annotation_version"],
            "recording_groups": len(recordings),
            "source_aliases": sum(len(g["aliases"]) for g in recordings),
            "type_assignments": len(annotations["assignments"]),
            "grouped_events": len(annotations["events"]),
            "event_types": len(annotations["taxonomy"]),
            "human_annotated_recording_groups": sum(bool(g["human_event_ids"]) for g in recordings),
            "flagged_events": sum(bool(e["flags"]) for e in annotations["events"]),
            "active_rules": len(annotations["active_rules"]),
            "machine_proposals_imported": annotations["machine_proposals_imported"],
            "independent_holdout_remaining": 0}
