"""Bounded private static-observer qualification; default prints inspection only."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import cv2
import numpy as np

from scripts import check_temporal_protocol as transport
from scripts.provision_action_model import verify_ready
from scripts.serve_action_model import command
from src import config, video_case_observations as observer


NAMES = ()  # Export-only: private case defaults removed.
PROTOCOL = config.DATA_PROC / "fortebank-20261017/video_protocol.json"
OUTPUT = config.DATA_PROC / "case-observation-checkpoint-v1"
SOURCES = ("src/video_case_observations.py", "scripts/check_case_observations.py",
           "scripts/check_temporal_protocol.py", "scripts/provision_action_model.py",
           "scripts/serve_action_model.py")
# These narrow state priors are attributed to prior assistant visual review.
# They are not human frame gold, official action labels or an accuracy oracle.
PRIORS = {}  # Export-only: private case defaults removed.


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def private_write(path, value):
    payload = value if isinstance(value, bytes) else observer.canonical(value) + b"\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)


def select(protocol):
    selected = []
    for name in NAMES:
        found = [g for g in protocol["groups"] if name in g["video_ids"]]
        if len(found) != 1:
            raise ValueError("Fixed diagnostic selection is missing or ambiguous")
        g = found[0]
        selected.append({k: g[k] for k in ("group_id", "path", "sha256", "duration_sec")} | {"name": name})
    if len({g["group_id"] for g in selected}) != 6:
        raise ValueError("Fixed diagnostic requires six unique supplied groups")
    return selected


def _source(selection):
    path = (config.ROOT / selection["path"]).resolve()
    allowed = (config.DATA_PROC / "fortebank-20261017/open/videos").resolve()
    if not path.is_relative_to(allowed) or digest(path) != selection["sha256"]:
        raise ValueError("Diagnostic source does not match its supplied-group provenance")
    return path


def prepare(output=OUTPUT):
    output = Path(output).resolve()
    if not output.is_relative_to(config.DATA_PROC.resolve()) or output.exists():
        raise ValueError("Diagnostic preparation requires a fresh processed-data directory")
    model = verify_ready()
    protocol = json.loads(PROTOCOL.read_text())
    selected = select(protocol)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    (output / "samples").mkdir(mode=0o700)
    snapshots = output / "sources"; snapshots.mkdir(mode=0o700)
    source_hashes = {}
    for source in SOURCES:
        data = (config.ROOT / source).read_bytes()
        source_hashes[source] = hashlib.sha256(data).hexdigest()
        private_write(snapshots / source.replace("/", "__"), data)
    samples = []
    for selection in selected:
        cap = cv2.VideoCapture(str(_source(selection)))
        try:
            fps = cap.get(cv2.CAP_PROP_FPS); count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if not cap.isOpened() or not math.isfinite(fps) or not 0 < fps <= 120 or count <= 0:
                raise ValueError("Diagnostic video header is invalid")
            if abs(count / fps - selection["duration_sec"]) > .25:
                raise ValueError("Diagnostic duration differs from supplied protocol")
            wanted = {min(count - 1, round(selection["duration_sec"] * f * fps)): i
                      for i, f in enumerate(observer.RECIPE["fractions"])}
            if len(wanted) != 3:
                raise ValueError("Diagnostic fractional samples collide")
            for index in range(max(wanted) + 1):
                if not cap.grab():
                    raise ValueError("Diagnostic sequential frame decoding failed")
                if index not in wanted:
                    continue
                ok, frame = cap.retrieve()
                if not ok:
                    raise ValueError("Diagnostic selected-frame retrieval failed")
                position = wanted[index]; ordinal = len(samples)
                files = []
                for view_index, payload in enumerate(observer.image_views(frame)):
                    file = f"samples/{ordinal:03d}-{view_index}.jpg"
                    private_write(output / file, payload)
                    files.append({"file": file, "sha256": hashlib.sha256(payload).hexdigest()})
                samples.append({"ordinal": ordinal, "group_id": selection["group_id"],
                                "fraction": observer.RECIPE["fractions"][position],
                                "source_frame_index": index, "source_fps": fps,
                                "timestamp_sec": index / fps,
                                "decoder_timestamp_sec": cap.get(cv2.CAP_PROP_POS_MSEC) / 1000,
                                "source_frame_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
                                "source_shape": list(frame.shape), "views": files,
                                "assistant_prior_states": PRIORS[selection["name"]].get(position, {})})
        finally:
            cap.release()
    synthetic = np.full((848, 464, 3), 128, np.uint8)
    synthetic_views = []
    for i, payload in enumerate(observer.image_views(synthetic)):
        filename = f"samples/synthetic-{i}.jpg"; private_write(output / filename, payload)
        synthetic_views.append({"file": filename, "sha256": hashlib.sha256(payload).hexdigest()})
    manifest = {"schema_version": 1, "purpose": "bounded_static_observer_diagnostic",
                "recipe": observer.RECIPE, "recipe_sha256": observer.fingerprint(observer.RECIPE),
                "source_hashes": source_hashes, "protocol_sha256": digest(PROTOCOL),
                "model": {k: model[k] for k in ("model_id", "model_revision", "weights_sha256", "projector_sha256", "runtime_release", "alias")},
                "expected_server_command": command(config.ROOT, model),
                "selections": selected, "samples": samples, "synthetic_views": synthetic_views,
                "repeat_ordinal": 0, "total_requests_max": 20,
                "prior_origin": "Attributed assistant visual review, not human frame gold",
                "review_report_sha256": digest(config.ROOT / "reports/case_video_review.md"),
                "selection_uses_labels_or_anchors": False,
                "all_supplied_groups_are_development": True, "independent_holdout_claim": False,
                "semantic_accuracy": None, "promotion_authorized": False}
    if len(samples) != 18:
        raise ValueError("Diagnostic sample count differs from frozen recipe")
    private_write(output / "manifest.json", manifest)
    return manifest


def validate_manifest(output):
    output = Path(output).resolve()
    if not output.is_relative_to(config.DATA_PROC.resolve()):
        raise ValueError("Diagnostic output must remain inside processed data")
    manifest = json.loads((output / "manifest.json").read_text())
    if (manifest["recipe_sha256"] != observer.fingerprint(observer.RECIPE)
            or observer.canonical(manifest["recipe"]) != observer.canonical(observer.RECIPE)
            or manifest["selections"] != select(json.loads(PROTOCOL.read_text()))
            or digest(PROTOCOL) != manifest["protocol_sha256"]
            or len(manifest["samples"]) != 18 or manifest["repeat_ordinal"] != 0):
        raise ValueError("Diagnostic manifest differs from its frozen selection/recipe")
    expected_samples = [(s, position, fraction) for s in manifest['selections']
                        for position, fraction in enumerate(observer.RECIPE['fractions'])]
    for ordinal, (sample, (selection, position, fraction)) in enumerate(zip(manifest['samples'], expected_samples, strict=True)):
        if (sample['ordinal'] != ordinal or sample['group_id'] != selection['group_id']
                or sample['fraction'] != fraction
                or sample['assistant_prior_states'] != PRIORS[selection['name']].get(position, {})
                or not 0 <= sample['timestamp_sec'] <= selection['duration_sec']
                or len(sample['views']) != 2):
            raise ValueError("Diagnostic samples differ from the fixed selection or prior states")
    for source, expected in manifest["source_hashes"].items():
        if digest(config.ROOT / source) != expected:
            raise ValueError("Diagnostic source changed after preparation")
    for selection in manifest["selections"]:
        _source(selection)
    return manifest


def _images(output, views):
    values = []
    for view in views:
        path = (output / view["file"]).resolve()
        if not path.is_relative_to((output / "samples").resolve()) or digest(path) != view["sha256"]:
            raise ValueError("Diagnostic JPEG provenance differs")
        values.append(path.read_bytes())
    return values


def attest_server(expected):
    matches = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = (proc / "cmdline").read_bytes().rstrip(b"\0").decode().split("\0")
        except (OSError, UnicodeError):
            continue
        if args and Path(args[0]).name == "llama-server" and "8003" in args:
            if args != expected:
                raise ValueError("Local vision listener uses a different launch profile")
            matches.append(int(proc.name))
    if len(matches) != 1:
        raise ValueError("Local vision process identity cannot be attested")
    advertised = transport.request_json("/v1/models", timeout=5).get("data")
    if not isinstance(advertised, list) or len(advertised) != 1 or advertised[0].get("id") != observer.MODEL_ALIAS:
        raise ValueError("Local vision advertised model differs")
    return {"pid": matches[0], "command_sha256": observer.fingerprint(expected)}


@contextmanager
def ready_server(expected):
    """Reuse an attested service or own a fresh existing verified launcher."""
    process = None
    with socket.socket() as sock:
        occupied = sock.connect_ex(("127.0.0.1", 8003)) == 0
    if not occupied:
        with open(os.devnull, 'wb') as silent:
            process = subprocess.Popen([sys.executable, '-m', 'scripts.serve_action_model'],
                                       cwd=config.ROOT, stdout=silent, stderr=silent,
                                       start_new_session=True)
        deadline = time.monotonic() + 60
        try:
            while True:
                if process.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError("Verified local vision launcher did not become ready")
                try:
                    transport.request_json('/health', timeout=2)
                    break
                except (OSError, RuntimeError, TimeoutError):
                    time.sleep(.2)
            yield attest_server(expected)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL); process.wait(timeout=5)
    else:
        yield attest_server(expected)


def comparison(rows):
    checks = []
    for row in rows:
        for key, expected in row['assistant_prior_states'].items():
            checks.append({'field': key, 'agrees': row['states'][key] == expected,
                           'unknown': row['states'][key] == 'unknown'})
    return {'attributed_prior_checks': len(checks),
            'prior_state_agreements': sum(c['agrees'] for c in checks),
            'prior_state_disagreements': sum(not c['agrees'] and not c['unknown'] for c in checks),
            'prior_state_unknowns': sum(c['unknown'] for c in checks),
            'semantic_accuracy': None, 'prior_checks_are_not_accuracy': True}


def run(output=OUTPUT):
    output = Path(output).resolve(); manifest = validate_manifest(output)
    manifest_sha = digest(output / 'manifest.json')
    private_write(output / 'started.json', {'manifest_sha256': manifest_sha, 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    records = []; started = time.monotonic(); stage = 'service'; failure = None
    summary = {'schema_version':1, 'status':'failed', 'manifest_sha256':manifest_sha,
               'semantic_accuracy':None, 'promotion_authorized':False,
               'model_training_performed':False, 'independent_holdout_claim':False}
    try:
        model = verify_ready()
        if any(model[k] != v for k,v in manifest['model'].items()):
            raise ValueError('Verified model differs from frozen manifest')
        with ready_server(manifest['expected_server_command']) as attestation:
            private_write(output / 'runtime_attestation.json', attestation)
            requests = [('synthetic', None, manifest['synthetic_views'])]
            requests += [('target', sample, sample['views']) for sample in manifest['samples']]
            requests += [('repeat', manifest['samples'][0], manifest['samples'][0]['views'])]
            for kind, sample, views in requests:
                stage = kind
                if digest(output/'manifest.json') != manifest_sha:
                    raise ValueError('Frozen manifest changed during diagnostic')
                for source, expected in manifest['source_hashes'].items():
                    if digest(config.ROOT/source) != expected:
                        raise ValueError('Diagnostic source changed during execution')
                if attest_server(manifest['expected_server_command']) != attestation:
                    raise ValueError('Local vision process changed during diagnostic')
                remaining = observer.RECIPE['total_timeout_sec'] - (time.monotonic()-started)
                if remaining <= 0 or len(records) >= manifest['total_requests_max']:
                    raise TimeoutError('Static diagnostic exceeded its frozen budget')
                call_start = time.monotonic()
                value = observer.observe(_images(output, views), timeout=min(45, remaining))
                row = {'kind':kind, 'ordinal':sample['ordinal'] if sample else None,
                       'states':value['states'], 'body_sha256':value['body_sha256'],
                       'response':value['response'], 'elapsed_sec':time.monotonic()-call_start,
                       'assistant_prior_states':sample['assistant_prior_states'] if sample else {}}
                records.append(row); private_write(output/f'response-{len(records):02d}.json', row)
                print(json.dumps({'completed_requests':len(records),'maximum_requests':20}), flush=True)
                if kind == 'synthetic' and (row['states']['person_state'] != 'absent' or row['states']['banknotes_visible'] != 'no'):
                    raise ValueError('Synthetic blank control hallucinated person or banknotes')
            targets = [r for r in records if r['kind']=='target']
            repeat = records[-1]
            repeated = repeat['body_sha256']==targets[0]['body_sha256'] and repeat['states']==targets[0]['states']
            metrics = comparison(targets)
            summary.update(metrics, status='completed', repeatability_passed=repeated,
                           supports_expansion=(repeated and metrics['prior_state_agreements']==metrics['attributed_prior_checks']),
                           target_requests_completed=len(targets),
                           unique_state_vectors=len({observer.fingerprint(r['states']) for r in targets}))
        validate_manifest(output)
        verified_after = verify_ready()
        if any(verified_after[k] != v for k,v in manifest['model'].items()):
            raise ValueError('Model assets changed during execution')
        # Video labels are attached only after all model calls; never prompts or priors.
        by_group = {g['group_id']:g for g in json.loads(PROTOCOL.read_text())['groups']}
        private_write(output/'post_inference_label_context.json', {
            'origin':'Original organizer binary labels; contextual only, no deployed verdict evaluation',
            'labels':[{ 'group_id':s['group_id'], 'target':by_group[s['group_id']]['target'] } for s in manifest['selections']]})
    except Exception as error:
        failure = {'stage':stage, 'exception_type':type(error).__name__, 'reason':str(error) if isinstance(error,(ValueError,TimeoutError)) else 'Local observer operational failure'}
        private_write(output/'failure.json', failure)
        summary.update(status='failed', supports_expansion=False)
    summary.update(completed_requests=len(records), elapsed_sec=time.monotonic()-started,
                   failure_stage=failure['stage'] if failure else None,
                   request_elapsed_sec_total=sum(r['elapsed_sec'] for r in records),
                   request_elapsed_sec_max=max((r['elapsed_sec'] for r in records), default=None),
                   zero_cache_n_requests=sum(r['response'].get('timings',{}).get('cache_n')==0 for r in records),
                   positive_cache_n_requests=sum((r['response'].get('timings',{}).get('cache_n') or 0)>0 for r in records),
                   missing_cache_n_requests=sum(r['response'].get('timings',{}).get('cache_n') is None for r in records))
    private_write(output/'summary.json', summary)
    print(json.dumps(summary), flush=True)
    return 0 if summary['status']=='completed' else 2


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    actions=parser.add_mutually_exclusive_group()
    actions.add_argument('--prepare',action='store_true');actions.add_argument('--run',action='store_true')
    parser.add_argument('--output',type=Path,default=OUTPUT);args=parser.parse_args(argv)
    if args.run:
        return run(args.output)
    if args.prepare:
        prepare(args.output)
        print(json.dumps({'status':'prepared','selected_recordings':6,'target_requests':18,'maximum_requests':20,'will_execute':False}))
    else:
        print(json.dumps({'will_execute':False,'selected_recordings':6,'target_requests':18,'maximum_requests':20,'total_timeout_sec':900,'promotion_authorized':False}))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
