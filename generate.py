"""Generate MessyKitchens-style synthetic scenes.

    python generate.py --level easy --n_scenes 600 --seed 0 --out out/easy

Scenes are simulated and rendered by Blender workers (blender/scene_worker.py, one process per
GPU slot, each handling a chunk of scenes), then annotated on the CPU while the GPUs keep
rendering. The run is resumable: finished scenes are skipped, failed scenes are retried with
fresh layouts up to --max_retries times and then recorded in failures.jsonl. --redo regenerates
finished scenes (for example ones validate.py rejected) the same way, with fresh layouts.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import yaml

from mksynth.config import REPO_ROOT, load_config, load_yaml
from mksynth.layout import LEVELS, scene_seed

BLENDER_INSTALL = REPO_ROOT / ".blender"
WORKER = REPO_ROOT / "blender" / "scene_worker.py"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--level", required=True, choices=LEVELS)
    parser.add_argument("--n_scenes", type=int, required=True, help="training scenes (train objects only)")
    parser.add_argument("--n_val", type=int, default=None, help="validation scenes (val objects only); default 5%% of n_scenes")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides, e.g. render.samples=64")
    parser.add_argument("--gpus", default=None, help="comma-separated GPU ids (CUDA_VISIBLE_DEVICES per worker); default: one local worker")
    parser.add_argument("--workers_per_gpu", type=int, default=1, help=">1 overlaps physics on the CPU with rendering")
    parser.add_argument("--device_type", default=None, help="Cycles device type: OPTIX, CUDA, METAL (default: BlenderProc's choice)")
    parser.add_argument("--chunk", type=int, default=5, help="scenes per Blender process")
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--annotate_workers", type=int, default=None, help="default: a quarter of the CPU quota")
    parser.add_argument("--keep_raw", action="store_true", help="keep raw/views.npz after annotation")
    parser.add_argument("--start", type=int, default=0, help="first scene index (for splitting a run across machines)")
    parser.add_argument("--redo", nargs="*", default=[], metavar="SCENE_ID",
                        help="regenerate these scenes with fresh layouts (needs --max_retries >= 1)")
    return parser.parse_args(argv)


def check_assets(cfg: dict) -> None:
    """Every asset used must have an allowed licence and must not come from the held-out benchmark."""
    assets_cfg = load_yaml(REPO_ROOT / cfg["assets"]["config"])
    allowed = set(assets_cfg["allowed_licences"])
    holdout_sources = set(assets_cfg["holdout"]["sources"])
    table = json.loads((REPO_ROOT / cfg["assets"]["table"]).read_text())
    bad = [a["id"] for a in table.values() if a["licence"] not in allowed or a["source"] in holdout_sources]
    if bad:
        raise SystemExit(f"assets with disallowed licence or held-out source: {bad}")
    shared = {a["id"] for a in table.values() if a["split"] == "train"} & {a["id"] for a in table.values() if a["split"] == "val"}
    if shared:
        raise SystemExit(f"assets in both splits: {sorted(shared)}")


def scene_specs(args, cfg: dict, out_dir: Path) -> list[dict]:
    n_val = args.n_val if args.n_val is not None else round(0.05 * args.n_scenes)
    specs = []
    for split, count in (("train", args.n_scenes), ("val", n_val)):
        for index in range(args.start, args.start + count):
            scene_id = f"{split}_{args.level}_{index:05d}"
            specs.append({
                "scene_id": scene_id, "level": args.level, "split": split, "index": index,
                "seed": scene_seed(args.seed, split, args.level, index, cfg["splits"]["val_seed_offset"]),
                "global_seed": args.seed, "out_dir": str(out_dir / "scenes" / scene_id),
            })
    return specs


def scene_state(spec: dict) -> tuple[str, int]:
    directory = Path(spec["out_dir"])
    if (directory / "scene.json").is_file():
        return "annotated", 0
    if (directory / "raw" / "DONE").is_file():
        return "rendered", 0
    failed = directory / "raw" / "FAILED.json"
    if failed.is_file():
        return "failed", json.loads(failed.read_text()).get("retries", 0)
    return "todo", 0


def annotate_one(scene_dir: str, cfg: dict, keep_raw: bool) -> tuple[str, str]:
    from mksynth.annotate import annotate_scene  # imported in the worker process (open3d)

    table = json.loads((REPO_ROOT / cfg["assets"]["table"]).read_text())
    try:
        annotate_scene(Path(scene_dir), cfg, table, keep_raw_views=keep_raw)
        return scene_dir, "ok"
    except Exception as error:  # noqa: BLE001 - recorded, scene is re-annotated on resume
        return scene_dir, f"annotate failed: {error!r}"


def run_worker(job_path: Path, log_path: Path, gpu: str | None) -> int:
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    command = [str(Path(sys.executable).parent / "blenderproc"), "run", "--blender-install-path", str(BLENDER_INSTALL),
               str(WORKER), str(job_path)]
    with open(log_path, "a") as log:
        log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(command)}\n")
        log.flush()
        return subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=REPO_ROOT).returncode


def cpu_budget() -> float:
    """CPUs this process may use: the cgroup quota when there is one (containers), else the cores."""
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            return int(quota) / int(period)
    except (OSError, ValueError):
        pass
    return float(len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1)


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.config, args.set)
    if args.device_type:
        cfg["render"]["devices"] = args.device_type
    n_slots = len(args.gpus.split(",")) * args.workers_per_gpu if args.gpus else args.workers_per_gpu
    if not cfg["render"].get("threads"):
        # Each Blender process would otherwise start one thread per visible core.
        cfg["render"]["threads"] = max(2, int(0.75 * cpu_budget()) // n_slots)
    check_assets(cfg)
    out_dir = Path(args.out).resolve()
    (out_dir / "scenes").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)
    (out_dir / "jobs").mkdir(exist_ok=True)
    snapshot = out_dir / "config.yaml"
    if snapshot.is_file() and load_yaml(snapshot) != cfg:
        print(f"warning: config differs from the snapshot in {snapshot}; finished scenes keep their old settings")
    snapshot.write_text(yaml.safe_dump(cfg, sort_keys=False))

    specs = scene_specs(args, cfg, out_dir)
    unknown = set(args.redo) - {spec["scene_id"] for spec in specs}
    if unknown:
        raise SystemExit(f"--redo: not scenes of this run: {sorted(unknown)}")
    for spec in specs:
        if spec["scene_id"] in args.redo:  # marked as failed once, so the retry round makes a fresh layout
            directory = Path(spec["out_dir"])
            shutil.rmtree(directory, ignore_errors=True)
            (directory / "raw").mkdir(parents=True)
            (directory / "raw" / "FAILED.json").write_text(json.dumps({**spec, "retries": 0, "reason": "redo",
                                                                       "detail": "requested with --redo"}))
    started = time.perf_counter()
    annotate_pool = ProcessPoolExecutor(args.annotate_workers or max(1, int(cpu_budget()) // 4))
    pending_annotations = []

    def annotate_ready(batch):
        for spec in batch:
            if scene_state(spec)[0] == "rendered":
                pending_annotations.append(annotate_pool.submit(annotate_one, spec["out_dir"], cfg, args.keep_raw))

    annotate_ready(specs)  # rendered in an earlier, interrupted run
    slots = [g.strip() for g in args.gpus.split(",")] if args.gpus else [None]
    slots = [g for g in slots for _ in range(args.workers_per_gpu)]
    for retry in range(args.max_retries + 1):
        todo = []
        for spec in specs:
            state, retries = scene_state(spec)
            if state == "todo" or (state == "failed" and retry > 0 and retries == retry - 1):
                todo.append({**spec, "retries": retry, "attempt_offset": retry * cfg["physics"]["max_attempts"]})
        if not todo:
            continue
        print(f"[generate] round {retry}: {len(todo)} scenes on {len(slots)} worker slot(s)", flush=True)
        chunks = queue.Queue()
        for i in range(0, len(todo), args.chunk):
            chunks.put(todo[i:i + args.chunk])

        def slot_loop(slot_index: int, gpu: str | None):
            while True:
                try:
                    batch = chunks.get_nowait()
                except queue.Empty:
                    return
                job_path = out_dir / "jobs" / f"{batch[0]['scene_id']}_r{retry}.json"
                job_path.write_text(json.dumps({"config": cfg, "scenes": batch}))
                code = run_worker(job_path, out_dir / "logs" / f"worker_{slot_index}.log", gpu)
                for spec in batch:
                    if scene_state(spec)[0] == "todo":  # the process died before reaching this scene
                        failed = Path(spec["out_dir"]) / "raw"
                        failed.mkdir(parents=True, exist_ok=True)
                        (failed / "FAILED.json").write_text(json.dumps({**spec, "reason": "worker_crash", "detail": f"exit code {code}"}))
                annotate_ready(batch)
                done = sum(scene_state(s)[0] in ("rendered", "annotated") for s in specs)
                print(f"[generate] slot {slot_index}: chunk {batch[0]['scene_id']} exit {code}; "
                      f"{done}/{len(specs)} rendered, {time.perf_counter() - started:.0f}s", flush=True)

        threads = [threading.Thread(target=slot_loop, args=(i, gpu)) for i, gpu in enumerate(slots)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    for future in pending_annotations:
        scene_dir, status = future.result()
        if status != "ok":
            print(f"[generate] {Path(scene_dir).name}: {status}", flush=True)
    annotate_pool.shutdown()

    from mksynth.mod_export import collect

    stats = collect(out_dir)
    print(json.dumps(stats, indent=2))
    print(f"[generate] finished in {time.perf_counter() - started:.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()
