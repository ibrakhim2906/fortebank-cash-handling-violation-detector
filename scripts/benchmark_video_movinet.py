"""Synthetic-only MoViNet CPU benchmark; inspection is default, no downloads."""
import argparse
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import resource
import tempfile
import time

import cv2
import numpy as np

from src import config
from src import video_movinet as engine


def _private_json(path, value):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")


def synthetic_frame(index):
    frame = np.zeros((848, 464, 3), np.uint8)
    left, top = 24 + index % 320, 80 + index % 640
    frame[top:top + 64, left:left + 64] = (40, 80, 240)
    return frame


def _synthetic_video(path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30., (464, 848))
    try:
        if not writer.isOpened():
            raise ValueError("Synthetic stress video could not be encoded")
        for index in range(4350):
            writer.write(synthetic_frame(index))
    finally:
        writer.release()


def _benchmark(output):
    # Provisioning is separate; this call verifies approved model/license/runtime.
    from scripts.provision_movinet import verify_ready
    ready = verify_ready()
    started = time.monotonic()
    encoder = engine.MoViNetEncoder(ready["model_path"], ready["weights_sha256"])
    load_sec = time.monotonic() - started
    frames = [synthetic_frame(index) for index in range(40)]
    times, vectors = [], []
    for _ in range(2):
        call_started = time.monotonic()
        vectors.append(encoder.encode_window(frames, deadline=started + 240))
        times.append(time.monotonic() - call_started)
    _private_json(output / "runtime.json", {"provenance": encoder.provenance, "signature": encoder.signature,
                                           "feature_config": engine.FEATURE_CONFIG})
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        path = Path(temporary) / "synthetic-145s.avi"
        _synthetic_video(path)
        cap = cv2.VideoCapture(str(path))
        try:
            source_fps = cap.get(cv2.CAP_PROP_FPS)
            source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if not cap.isOpened() or source_fps != 30 or source_frames != 4350:
                raise ValueError("Synthetic stress source does not have the declared30FPS/4350frames")
        finally:
            cap.release()
        stress = engine.extract_video(path, encoder=encoder, deadline=started + 240)
    matrix = np.stack([row["features"] for row in stress["windows"]])
    result = {"purpose": "synthetic_cpu_feasibility_only", "load_sec": load_sec,
              "eight_second_window_sec": times, "repeat_exact": bool(np.array_equal(*vectors)),
              "repeat_max_abs_difference": float(np.max(np.abs(vectors[0] - vectors[1]))),
              "stress_duration_sec": stress["duration_sec"], "stress_windows": len(stress["windows"]),
              "stress_source_fps": source_fps, "stress_source_frames": source_frames,
              "stress_source_codec": "MJPEG/AVI; not a MOV/API compliance test",
              "stress_sampled_source_frames": len(stress["sampling"]["actual_times_sec"]),
              "stress_model_frame_invocations": sum(len(row["sampling"]["actual_times_sec"]) for row in stress["windows"]),
              "stress_extraction_sec": stress["extraction_sec"],
              "stress_feature_dimension": int(matrix.shape[1]), "stress_all_finite": bool(np.isfinite(matrix).all()),
              "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
              "seed": config.SEED, "bank_videos_used": 0, "holdout_videos_used": 0,
              "semantic_accuracy_verified": False, "promotion_authorized": False,
              "provenance": encoder.provenance, "total_elapsed_sec": time.monotonic() - started}
    result["passed"] = (result["repeat_exact"] and result["stress_all_finite"]
                        and abs(result["stress_duration_sec"] - 145) < 1e-6
                        and result["stress_windows"] == 70 and result["stress_sampled_source_frames"] == 725
                        and result["stress_extraction_sec"] < 180)
    return result


def _worker(output):
    descriptor = os.open(os.devnull, os.O_WRONLY)
    os.dup2(descriptor, 1)
    os.dup2(descriptor, 2)
    os.close(descriptor)
    try:
        result = _benchmark(Path(output))
    except Exception as error:
        result = {"passed": False, "operational_failure": type(error).__name__,
                  "bank_videos_used": 0, "holdout_videos_used": 0, "synthetic_failure_reason": str(error)[:1000]}
    result["benchmark_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    _private_json(Path(output) / "benchmark.json", result)


def run_benchmark(output, *, timeout=270):
    if type(timeout) not in (int, float) or not 0 < timeout <= 270:
        raise ValueError("Synthetic benchmark timeout must be bounded by 270 seconds")
    output = Path(output).resolve()
    if not output.is_relative_to(config.DATA_PROC.resolve()):
        raise ValueError("Synthetic benchmark output must be inside data/processed")
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    for name, path in (("benchmark_source.py", Path(__file__)), ("encoder_source.py", Path(engine.__file__))):
        descriptor = os.open(output / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(path.read_bytes())
    context = multiprocessing.get_context("spawn")
    process = context.Process(target=_worker, args=(str(output),))
    try:
        process.start()
        process.join(timeout=timeout)
        if process.is_alive():
            process.terminate()
            process.join(timeout=1)
            if process.is_alive():
                process.kill()
                process.join(timeout=1)
            raise TimeoutError("Synthetic benchmark exceeded its owned-process deadline")
        if process.exitcode != 0 or not (output / "benchmark.json").is_file():
            raise RuntimeError("Synthetic benchmark process did not produce a completed report")
    finally:
        if process.is_alive():
            process.kill()
            process.join(timeout=1)
        process.close()
    result = json.loads((output / "benchmark.json").read_text())
    if result["benchmark_source_sha256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise RuntimeError("Benchmark source changed during execution")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=config.DATA_PROC / "movinet-synthetic-benchmark")
    args = parser.parse_args(argv)
    if not args.run:
        print(json.dumps({"will_execute": False, "runtime": engine.RUNTIME_DISTRIBUTION,
                          "runtime_version": engine.RUNTIME_VERSION, "weights_sha256": engine.WEIGHTS_SHA256,
                          "stress_duration_sec": 145, "bank_videos_used": 0}))
        return
    try:
        result = run_benchmark(args.output_dir)
    except Exception:
        raise SystemExit("Synthetic MoViNet benchmark failed; details suppressed") from None
    print(json.dumps(result), flush=True)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
