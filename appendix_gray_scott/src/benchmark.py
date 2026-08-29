from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gray_scott import SimConfig, run_timed, simulate_numba


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Gray-Scott solver backends.")
    parser.add_argument("--out-dir", type=Path, default=Path("../results"))
    parser.add_argument("--nx", type=int, default=96)
    parser.add_argument("--ny", type=int, default=96)
    parser.add_argument("--steps", type=int, default=360)
    parser.add_argument("--save-every", type=int, default=60)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workers", nargs="+", type=int, default=[1, 2, 4])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = SimConfig(nx=args.nx, ny=args.ny, steps=args.steps, save_every=args.save_every)

    # Compile once so the numba timing measures solver throughput, not compilation.
    warmup = SimConfig(nx=16, ny=16, steps=4, save_every=2)
    simulate_numba(warmup, args.seed)

    if args.repeats < 1:
        raise ValueError("repeats must be at least 1")
    if 1 not in args.workers:
        raise ValueError("workers must include 1 so speedup has a baseline")
    rows: list[dict[str, str | float | int]] = []
    for backend in ["python", "numpy", "numba"]:
        bench_cfg = cfg if backend != "python" else SimConfig(nx=48, ny=48, steps=120, save_every=40)
        timings = []
        for repeat in range(args.repeats):
            data, seconds = run_timed(backend, bench_cfg, args.seed + repeat)
            timings.append(seconds)
        seconds = float(np.median(timings))
        throughput = bench_cfg.steps * bench_cfg.nx * bench_cfg.ny / seconds / 1e6
        rows.append(
            {
                "category": "solver_core",
                "mode": backend,
                "grid": f"{bench_cfg.ny}x{bench_cfg.nx}",
                "steps": bench_cfg.steps,
                "samples": 1,
                "workers": 1,
                "repeats": args.repeats,
                "seconds": seconds,
                "mcell_updates_per_s": throughput,
                "speedup": 0.0,
                "parallel_efficiency": None,
                "note": f"shape={tuple(data.shape)}",
            }
        )
    python_rate = float(rows[0]["mcell_updates_per_s"])
    for row in rows:
        row["speedup"] = float(row["mcell_updates_per_s"]) / python_rate

    parallel_rows = []
    for workers in sorted(set(min(value, args.samples) for value in args.workers)):
        if workers < 1:
            continue
        executor = "thread"
        command = [
            sys.executable,
            str(Path(__file__).with_name("run_case.py")),
            "--out",
            str(args.out_dir / f"bench_workers_{workers}.h5"),
            "--backend",
            "numba",
            "--executor",
            executor,
            "--samples",
            str(args.samples),
            "--workers",
            str(workers),
            "--seed",
            str(args.seed),
            "--nx",
            str(args.nx),
            "--ny",
            str(args.ny),
            "--steps",
            str(args.steps),
            "--save-every",
            str(args.save_every),
        ]
        timings = []
        for _ in range(args.repeats):
            tic = __import__("time").perf_counter()
            subprocess.run(command, check=True, capture_output=True, text=True)
            timings.append(__import__("time").perf_counter() - tic)
        seconds = float(np.median(timings))
        throughput = args.samples * cfg.steps * cfg.nx * cfg.ny / seconds / 1e6
        parallel_rows.append(
            {
                "category": "end_to_end",
                "mode": f"numba_thread_{workers}",
                "grid": f"{cfg.ny}x{cfg.nx}",
                "steps": cfg.steps,
                "samples": args.samples,
                "workers": workers,
                "repeats": args.repeats,
                "seconds": seconds,
                "mcell_updates_per_s": throughput,
                "speedup": 0.0,
                "parallel_efficiency": 0.0,
                "note": f"{args.samples} samples, end-to-end HDF5 write",
            }
        )
    one_worker = next(float(row["seconds"]) for row in parallel_rows if int(row["workers"]) == 1)
    for row in parallel_rows:
        speedup = one_worker / float(row["seconds"])
        row["speedup"] = speedup
        row["parallel_efficiency"] = speedup / int(row["workers"])
    rows.extend(parallel_rows)

    csv_path = args.out_dir / "benchmark.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    core = rows[:3]
    modes = [str(row["mode"]) for row in core]
    speeds = [float(row["mcell_updates_per_s"]) for row in core]
    bars = axes[0].bar(modes, speeds, color=["#6c757d", "#2a9d8f", "#e76f51"])
    axes[0].set(ylabel="Mcell-updates/s", title="Single-sample solver core")
    for bar, value in zip(bars, speeds):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}", ha="center", va="bottom", fontsize=9)
    worker_counts = [int(row["workers"]) for row in parallel_rows]
    speedups = [float(row["speedup"]) for row in parallel_rows]
    efficiencies = [float(row["parallel_efficiency"]) for row in parallel_rows]
    axes[1].plot(worker_counts, speedups, "o-", label="measured")
    axes[1].plot(worker_counts, worker_counts, "--", color="gray", label="ideal")
    axes[1].set(xlabel="Workers", ylabel="Speedup vs 1 worker", title="End-to-end sample parallelism")
    axes[1].legend()
    axes[2].plot(worker_counts, efficiencies, "o-", color="#7a4eab")
    axes[2].axhline(1.0, ls="--", color="gray")
    axes[2].set(xlabel="Workers", ylabel="Parallel efficiency", title="Efficiency including HDF5 I/O", ylim=(0, max(1.05, max(efficiencies) * 1.1)))
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.savefig(args.out_dir / "benchmark_speedup.png", dpi=180)
    plt.close(fig)

    summary = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "method": f"median of {args.repeats} runs; Numba warmed before measurement",
        "core_speedup_basis": "ratio of normalized per-cell throughput; Python uses a reduced 48x48x120 case to keep runtime practical",
        "rows": rows,
        "interpretation": {
            "best_core_mode": max(core, key=lambda row: float(row["mcell_updates_per_s"]))["mode"],
            "best_end_to_end_workers": max(parallel_rows, key=lambda row: float(row["mcell_updates_per_s"]))["workers"],
            "parallel_scaling_is_sublinear": all(float(row["parallel_efficiency"]) <= 1.05 for row in parallel_rows),
        },
    }
    (args.out_dir / "benchmark_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"wrote {csv_path}")
    print(f"wrote {args.out_dir / 'benchmark_speedup.png'}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
