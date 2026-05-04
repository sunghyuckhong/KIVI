#!/usr/bin/env python3
"""Parallel scheduler for run-* sweeps.

Detects idle GPUs via ``nvidia-smi`` (or uses an explicit list), groups them
into TP-sized streams, and runs ``(variant, task)`` cells in parallel — one
cell per stream at a time. As each cell finishes, the freed stream picks the
next pending cell from the queue.

Usage (from Makefile -- not normally called by hand):

    python scripts/parallel_sweep.py \\
        --runner scripts/run_eval_qwen3.sh \\
        --runner_args --size 32b \\
        --variants bf16 fp8 pertoken smkv \\
        --tasks gsm8k_cot minerva_math500 gpqa_main_cot_n_shot \\
        --tp 2 \\
        [--gpus 0,1,2,3]   # default: auto-detect idle (< 2GB used)

Behavior:
    - If --gpus given, uses exactly that pool. Errors if not divisible by --tp.
    - If --gpus not given, scans nvidia-smi for GPUs with memory.used < 2GB.
    - Pool gets chunked into consecutive --tp-sized groups: pool[0:tp],
      pool[tp:2tp], etc. Trailing remainder is dropped.

Each cell's stdout/stderr goes to ``logs/run_out/parallel_<variant>_<task>_g<gpus>.log``
in addition to being captured for failure reporting.

Exit code: 0 if all cells succeeded, 1 otherwise (also prints which cells failed).
"""
import argparse
import os
import shlex
import subprocess
import sys
import time
from collections import deque


def query_idle_gpus(threshold_mb: int = 2000) -> list[int]:
    """Return GPU indices with memory.used < threshold_mb."""
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used",
         "--format=csv,noheader,nounits"],
        text=True,
    )
    idle: list[int] = []
    for line in out.strip().splitlines():
        idx, used = (s.strip() for s in line.split(","))
        if int(used) < threshold_mb:
            idle.append(int(idx))
    return idle


def chunk(lst: list[int], size: int) -> list[list[int]]:
    """Group lst into size-sized chunks (drop incomplete trailing chunk)."""
    return [lst[i:i + size] for i in range(0, len(lst) - size + 1, size)]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runner", required=True,
                   help="path to scripts/run_eval_<family>.sh")
    p.add_argument("--runner_args", default="",
                   help="extra args passed to the runner before --variant/--task/--gpus, "
                        "as a single space-separated string (shlex-split internally). "
                        "E.g. --runner_args=\"--size 8b\"")
    p.add_argument("--variants", nargs="+", required=True)
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument("--tp", type=int, default=1,
                   help="GPUs per cell (1 for 8B/7B, 2 for 32B/33B/70B with TP=2)")
    p.add_argument("--gpus", default=None,
                   help="Comma-separated explicit pool. Default: auto-detect idle.")
    p.add_argument("--ns", type=int, default=512)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--idle_threshold_mb", type=int, default=2000,
                   help="GPU is idle if memory.used < this (default 2000 = 2GB)")
    return p.parse_args()


def main():
    args = parse_args()

    # ---- resolve GPU pool ----
    if args.gpus:
        pool = [int(g.strip()) for g in args.gpus.split(",") if g.strip()]
        print(f"[parallel] explicit GPU pool: {pool}")
    else:
        pool = query_idle_gpus(args.idle_threshold_mb)
        print(f"[parallel] auto-detected idle GPUs (< {args.idle_threshold_mb}MB): {pool}")

    groups = chunk(pool, args.tp)
    if not groups:
        print(f"[parallel] ERROR: pool={pool} has no complete TP={args.tp} groups")
        sys.exit(1)

    n_streams = len(groups)
    streams_str = ", ".join("[" + ",".join(str(g) for g in grp) + "]" for grp in groups)
    print(f"[parallel] {n_streams} stream(s) × {args.tp} GPU(s): {streams_str}")

    # ---- build job queue ----
    jobs = [(v, t) for v in args.variants for t in args.tasks]
    print(f"[parallel] {len(jobs)} cells: variants={args.variants} × tasks={args.tasks}")

    log_dir = os.path.join("logs", "run_out")
    os.makedirs(log_dir, exist_ok=True)

    # ---- schedule loop: at most n_streams concurrent subprocesses ----
    queue: deque[tuple[str, str]] = deque(jobs)
    free_groups: deque[list[int]] = deque(groups)
    running: list[tuple[subprocess.Popen, list[int], str, str, float, str]] = []
    failed: list[tuple[str, str, list[int], int]] = []
    n_done = 0
    t_start = time.time()

    def launch(grp: list[int], variant: str, task: str):
        gpu_str = ",".join(str(g) for g in grp)
        log_path = os.path.join(
            log_dir,
            f"parallel_{variant}_{task}_g{gpu_str.replace(',', '_')}.log",
        )
        cmd = ["bash", args.runner, "--variant", variant, "--task", task,
               "--gpus", gpu_str, "--ns", str(args.ns),
               "--alpha", str(args.alpha), "--beta", str(args.beta)]
        cmd.extend(shlex.split(args.runner_args))
        logf = open(log_path, "w")
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
        # keep logf open for the process; we'll close it after wait()
        proc._logf = logf  # type: ignore[attr-defined]
        return proc, log_path

    while queue or running:
        # fill any free streams
        while queue and free_groups:
            grp = free_groups.popleft()
            v, t = queue.popleft()
            proc, log_path = launch(grp, v, t)
            t0 = time.time()
            running.append((proc, grp, v, t, t0, log_path))
            print(f"[parallel] START  {v:<10} {t:<35} gpu=[{','.join(str(g) for g in grp)}]  "
                  f"→ {log_path}")

        # poll
        time.sleep(2)
        for entry in list(running):
            proc, grp, v, t, t0, log_path = entry
            rc = proc.poll()
            if rc is None:
                continue
            running.remove(entry)
            free_groups.append(grp)
            proc._logf.close()  # type: ignore[attr-defined]
            dt = time.time() - t0
            n_done += 1
            status = "✅ PASS" if rc == 0 else f"❌ FAIL(rc={rc})"
            print(f"[parallel] DONE   {v:<10} {t:<35} gpu=[{','.join(str(g) for g in grp)}]  "
                  f"{status}  {dt:.0f}s  ({n_done}/{len(jobs)} cells; "
                  f"elapsed {time.time()-t_start:.0f}s)")
            if rc != 0:
                failed.append((v, t, grp, rc))

    elapsed = time.time() - t_start
    if failed:
        print(f"\n[parallel] {len(failed)} cell(s) FAILED in {elapsed:.0f}s:")
        for v, t, grp, rc in failed:
            gpu_str = ",".join(str(g) for g in grp)
            print(f"  {v}/{t} on [{gpu_str}] (rc={rc})")
        sys.exit(1)
    print(f"\n[parallel] ✅ all {len(jobs)} cells passed in {elapsed:.0f}s "
          f"({n_streams}-way parallel)")


if __name__ == "__main__":
    main()
