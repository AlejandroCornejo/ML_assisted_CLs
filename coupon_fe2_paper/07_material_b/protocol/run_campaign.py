"""Orchestrate frozen material-B chunks as independent restartable processes."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    from .prepare_design import digest
except ImportError:
    from prepare_design import digest


HERE = Path(__file__).resolve().parent
BASE = HERE.parent
DRIVER = HERE / "run_data_stage.py"
SPEC = HERE / "data_protocol_v1.json"
DESIGN = BASE / "results/data_protocol_design_v1.npz"
DESIGN_REPORT = BASE / "results/data_protocol_design_v1.json"


def jobs() -> list[dict]:
    groups = [
        ("reference", "reference", "working", 1),
        ("fit", "fit", "working", 64),
        ("validation", "validation", "working", 8),
        ("test", "test", "working", 8),
        ("paths", "paths", "working", 10),
        ("audit_working", "audit", "working", 4),
        ("audit_check", "audit", "audit", 4),
        ("cold", "cold", "working", 4),
    ]
    return [dict(name=f"{prefix}_{index:03d}", split=split, mesh=mesh,
                 chunk_index=index, chunk_count=count)
            for prefix, split, mesh, count in groups for index in range(count)]


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def run_job(job: dict, output: Path) -> dict:
    folder = output / "chunks" / job["name"]
    log = output / "logs" / (job["name"] + ".log")
    command = [sys.executable, str(DRIVER), "--spec", str(SPEC), "--design", str(DESIGN),
               "--design-report", str(DESIGN_REPORT), "--split", job["split"],
               "--mesh", job["mesh"], "--chunk-index", str(job["chunk_index"]),
               "--chunk-count", str(job["chunk_count"]), "--out", str(folder),
               "--confirm-full"]
    resumed = folder.exists()
    if resumed:
        command.append("--resume")
    started = time.perf_counter()
    with log.open("a" if resumed else "x") as stream:
        stream.write("COMMAND " + " ".join(command) + "\n")
        stream.flush()
        completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
    return dict(**job, returncode=completed.returncode, resumed=resumed,
                elapsed_seconds=time.perf_counter()-started, log=str(log), output=str(folder))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not (1 <= args.workers <= 16):
        raise ValueError("workers must lie between 1 and 16")
    output = args.out.resolve()
    manifest_path = output / "manifest.json"
    planned = jobs()
    if args.resume:
        manifest = json.loads(manifest_path.read_text())
        if (manifest["driver_sha256"] != digest(DRIVER)
                or manifest["orchestrator_sha256"] != digest(Path(__file__))
                or manifest["protocol_sha256"] != digest(SPEC)
                or manifest["design_sha256"] != digest(DESIGN)
                or manifest["plan"] != planned):
            raise ValueError("Cannot resume after changing driver, protocol, design or plan")
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / "chunks").mkdir()
        (output / "logs").mkdir()
        manifest = dict(status="running", protocol_sha256=digest(SPEC), design_sha256=digest(DESIGN),
                        driver_sha256=digest(DRIVER), orchestrator_sha256=digest(Path(__file__)),
                        plan=planned, jobs=[], workers=args.workers,
                        started_utc=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()))
        atomic_json(manifest_path, manifest)
    completed_names = {row["name"] for row in manifest["jobs"] if row["returncode"] == 0}
    pending = [job for job in planned if job["name"] not in completed_names]
    failures = []
    running = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        iterator = iter(pending)
        for _ in range(min(args.workers, len(pending))):
            job = next(iterator, None)
            if job is not None:
                running[pool.submit(run_job, job, output)] = job
        while running:
            done, _ = concurrent.futures.wait(running, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                job = running.pop(future)
                row = future.result()
                manifest["jobs"] = [old for old in manifest["jobs"] if old["name"] != row["name"]]
                manifest["jobs"].append(row)
                manifest["jobs"].sort(key=lambda value: value["name"])
                atomic_json(manifest_path, manifest)
                print(row["name"], "return", row["returncode"],
                      f"{row['elapsed_seconds']:.1f}s", flush=True)
                if row["returncode"] != 0:
                    failures.append(row)
            while not failures and len(running) < args.workers:
                job = next(iterator, None)
                if job is None:
                    break
                running[pool.submit(run_job, job, output)] = job
    manifest["status"] = "failed" if failures else "complete"
    manifest["failed_jobs"] = [row["name"] for row in failures]
    manifest["completed_utc"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    atomic_json(manifest_path, manifest)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
