"""Preserve a release and inspect QuickTime headers without decoding footage."""
import argparse
import hashlib
import json
import re
import shutil
import stat
import struct
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import pandas as pd

from src import config


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def preserve(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        assert digest(source) == digest(destination), f"Existing file differs: {destination}"
    else:
        with source.open("rb") as incoming, destination.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing)
    assert digest(source) == digest(destination)


def atoms(data, start=0, end=None):
    end = len(data) if end is None else end
    while start + 8 <= end:
        size, kind = struct.unpack_from(">I4s", data, start)
        header = 8
        if size == 1:
            assert start + 16 <= end, "Truncated extended atom"
            size = struct.unpack_from(">Q", data, start + 8)[0]
            header = 16
        elif size == 0:
            size = end - start
        assert size >= header and start + size <= end, "Invalid QuickTime atom bounds"
        yield kind, start + header, start + size
        start += size


def child(data, start, end, kind):
    return next(((a, b) for k, a, b in atoms(data, start, end) if k == kind), None)


def media_clock(data, bounds):
    a, _ = bounds
    version = data[a]
    assert version in (0, 1), f"Unknown movie/media header version {version}"
    if version == 1:
        scale = struct.unpack_from(">I", data, a + 20)[0]
        duration = struct.unpack_from(">Q", data, a + 24)[0]
    else:
        scale, duration = struct.unpack_from(">II", data, a + 12)
    assert scale > 0, "Invalid media timescale"
    return scale, duration


def movie_metadata(path):
    # Parse container bytes only. No video decoding, pixel output, audio analysis or AI.
    data = path.read_bytes()
    moov = child(data, 0, len(data), b"moov")
    assert moov, "Missing movie header"
    clock = child(data, *moov, b"mvhd")
    assert clock, "Missing movie clock"
    scale, duration = media_clock(data, clock)
    result = {"duration_sec": duration / scale, "video_tracks": 0, "audio_tracks": 0}
    for kind, start, end in atoms(data, *moov):
        if kind != b"trak":
            continue
        mdia = child(data, start, end, b"mdia")
        if not mdia:
            continue
        hdlr = child(data, *mdia, b"hdlr")
        assert hdlr, "Missing handler"
        handler = data[hdlr[0] + 8:hdlr[0] + 12]
        if handler == b"soun":
            result["audio_tracks"] += 1
        if handler != b"vide":
            continue
        result["video_tracks"] += 1
        tkhd = child(data, start, end, b"tkhd")
        assert tkhd, "Missing video track header"
        width, height = struct.unpack_from(">II", data, tkhd[1] - 8)
        result.update(width=width / 65536, height=height / 65536)
        matrix = struct.unpack_from(">9i", data, tkhd[1] - 44)
        result["display_matrix"] = list(matrix)
        mdhd = child(data, *mdia, b"mdhd")
        assert mdhd, "Missing video clock"
        timescale, media_duration = media_clock(data, mdhd)
        result["video_media_duration_sec"] = media_duration / timescale
        minf = child(data, *mdia, b"minf")
        assert minf, "Missing media information"
        stbl = child(data, *minf, b"stbl")
        assert stbl, "Missing sample table"
        stsd = child(data, *stbl, b"stsd")
        assert stsd, "Missing sample description"
        result["codec"] = data[stsd[0] + 12:stsd[0] + 16].decode("ascii")
        stts = child(data, *stbl, b"stts")
        assert stts, "Missing sample timing"
        count = struct.unpack_from(">I", data, stts[0] + 4)[0]
        assert stts[0] + 8 + count * 8 <= stts[1], "Truncated timing table"
        entries = [struct.unpack_from(">II", data, stts[0] + 8 + i * 8) for i in range(count)]
        samples = sum(n for n, _ in entries)
        ticks = sum(n * delta for n, delta in entries)
        result["frame_count"] = samples
        result["average_fps"] = samples * timescale / ticks if ticks else None
    assert result["video_tracks"] == 1, "Expected one video track"
    return result


def timestamp_seconds(value):
    if pd.isna(value):
        return None
    value = str(value).lower().strip()
    pieces = re.findall(r"(\d+)\s*(минут[аы]?|секунд[аы]?)", value)
    assert pieces, f"Unknown timestamp format: {value!r}"
    return sum(int(n) * (60 if unit.startswith("минут") else 1) for n, unit in pieces)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--brief", type=Path, required=True)
    parser.add_argument("--name", default="fortebank-20261017")
    args = parser.parse_args()
    assert re.fullmatch(r"[a-z0-9-]+", args.name), "Invalid release name"
    raw = config.DATA_RAW / args.name
    output = config.DATA_PROC / args.name
    raw.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    sources = []
    for source in (args.archive, args.brief):
        destination = raw / source.name
        preserve(source, destination)
        sources.append({"source": str(source), "preserved": str(destination),
                        "bytes": destination.stat().st_size, "sha256": digest(destination)})
    members = []
    with ZipFile(raw / args.archive.name) as archive:
        for entry in archive.infolist():
            relative = PurePosixPath(entry.filename)
            assert not relative.is_absolute() and ".." not in relative.parts
            assert "\\" not in entry.filename and not stat.S_ISLNK(entry.external_attr >> 16)
            target = output.joinpath(*relative.parts)
            assert target.resolve().is_relative_to(output.resolve()), "Unsafe archive path"
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                with archive.open(entry) as incoming, target.open("xb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
            with archive.open(entry) as incoming:
                expected = hashlib.file_digest(incoming, "sha256").hexdigest()
            assert digest(target) == expected, f"Extracted bytes differ: {target}"
            members.append({"member": entry.filename, "bytes": entry.file_size,
                            "sha256": expected})
    labels = pd.read_excel(output / "open/open_labels.xlsx")
    assert labels.iloc[:, 0].is_unique and labels.shape[1] == 3
    lookup = {str(row.iloc[0]): row for _, row in labels.iterrows()}
    videos = sorted((output / "open/videos").glob("*.mov"))
    assert len(videos) == len(labels) and {p.stem for p in videos} == set(lookup)
    rows = []
    for path in videos:
        label = lookup[path.stem]
        seconds = timestamp_seconds(label.iloc[1])
        assert label.iloc[2] in ("нарушение", "без нарушения")
        target = int(label.iloc[2] == "нарушение")
        assert (seconds is not None) == bool(target)
        metadata = movie_metadata(path)
        rows.append({"video_id": path.stem, "path": str(path.relative_to(config.ROOT)),
                     "bytes": path.stat().st_size, "sha256": digest(path), "target": target,
                     "timestamp_raw": None if pd.isna(label.iloc[1]) else str(label.iloc[1]),
                     "timestamp_sec": seconds, **metadata})
    frame = pd.DataFrame(rows)
    conflicting = set()
    for key, group in frame.groupby("sha256", sort=True):
        assert group.target.nunique() == 1, f"Conflicting video verdicts for hash {key}"
        if group.timestamp_sec.nunique() > 1:
            conflicting.add(key)
    frame["group_id"] = frame.sha256
    frame["timestamp_in_bounds"] = frame.timestamp_sec.isna() | (
        (frame.timestamp_sec >= 0) & (frame.timestamp_sec < frame.duration_sec))
    frame["localization_eligible"] = (frame.target.eq(1) & frame.timestamp_in_bounds
                                       & ~frame.sha256.isin(conflicting))
    manifest = {"seed": config.SEED, "source_kind": "human-supplied local files",
                "source_url": None, "sources": sources, "archive_members": members,
                "video_count": len(frame), "unique_hashes": frame.sha256.nunique(),
                "label_counts": frame.target.value_counts().to_dict(),
                "unique_label_counts": frame.drop_duplicates("sha256").target.value_counts().to_dict(),
                "duplicate_groups": [group.video_id.tolist() for _, group in frame.groupby("sha256") if len(group)>1],
                "timestamp_review_groups": sorted(conflicting),
                "out_of_bounds_ids": frame.loc[~frame.timestamp_in_bounds, "video_id"].tolist(),
                "footage_decoded": False, "footage_uploaded": False,
                "metadata_method": "QuickTime container headers, no pixels decoded"}
    frame.to_csv(output / "inventory.csv", index=False)
    (output / "provenance.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({k: manifest[k] for k in ("video_count", "unique_hashes", "label_counts",
                     "unique_label_counts", "duplicate_groups", "out_of_bounds_ids")}, ensure_ascii=False))
    print(frame[["duration_sec", "average_fps", "width", "height", "audio_tracks", "codec"]].describe(include="all").to_string())
    print("Localization eligible:", int(frame.localization_eligible.sum()), "positive annotation rows")
    print("Inventory:", output / "inventory.csv")
    print("Provenance:", output / "provenance.json")


if __name__ == "__main__":
    main()
