from __future__ import annotations

import argparse
import json
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np

from gray_scott import BACKENDS, SimConfig, run_timed, simulate_numba


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a mini PDEBench Gray-Scott dataset.")
    parser.add_argument("--out", type=Path, default=Path("../results/grayscott_dataset.h5"))
    parser.add_argument("--backend", choices=sorted(BACKENDS), default="numba")
    parser.add_argument("--executor", choices=["process", "thread"], default="process")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--ny", type=int, default=128)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--save-every", type=int, default=30)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--du", type=float, default=0.16)
    parser.add_argument("--dv", type=float, default=0.08)
    parser.add_argument("--feed", type=float, default=0.060)
    parser.add_argument("--kill", type=float, default=0.062)
    parser.add_argument("--noise", type=float, default=0.02)
    return parser.parse_args()


def _run_one(task: tuple[str, SimConfig, int, int]) -> tuple[int, int, float, np.ndarray]:
    backend, cfg, sample_id, seed = task
    data, seconds = run_timed(backend, cfg, seed)
    return sample_id, seed, seconds, data


def write_hdf5(
    path: Path,
    cfg: SimConfig,
    backend: str,
    results: list[tuple[int, int, float, np.ndarray]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.linspace(0.0, 1.0, cfg.nx, endpoint=False, dtype=np.float32)
    y = np.linspace(0.0, 1.0, cfg.ny, endpoint=False, dtype=np.float32)
    t = np.arange(cfg.frames, dtype=np.float32) * cfg.save_every * cfg.dt

    with h5py.File(path, "w") as h5:
        h5.attrs["case"] = "PDEBench Gray-Scott reaction diffusion mini case"
        h5.attrs["backend"] = backend
        h5.attrs["config"] = json.dumps(cfg.to_dict())
        h5.create_dataset("grid/x", data=x)
        h5.create_dataset("grid/y", data=y)
        h5.create_dataset("grid/t", data=t)
        timings = h5.create_group("timings")
        samples = h5.create_group("samples")
        for sample_id, seed, seconds, data in sorted(results):
            group = samples.create_group(f"{sample_id:04d}")
            group.attrs["seed"] = seed
            group.attrs["seconds"] = seconds
            group.create_dataset(
                "data",
                data=data,
                dtype="float32",
                chunks=(1, cfg.ny, cfg.nx, 2),
                compression="gzip",
                compression_opts=4,
            )
            timings.attrs[f"{sample_id:04d}"] = seconds


def main() -> None:
    args = parse_args()
    cfg = SimConfig(nx=args.nx, ny=args.ny, steps=args.steps, save_every=args.save_every, dt=args.dt, du=args.du, dv=args.dv, feed=args.feed, kill=args.kill, noise=args.noise)
    cfg.validate()
    workers = max(1, min(args.workers, args.samples))
    tasks = [
        (args.backend, cfg, sample_id, args.seed + sample_id)
        for sample_id in range(args.samples)
    ]

    if args.backend == "numba":
        warmup = SimConfig(nx=16, ny=16, steps=4, save_every=2)
        simulate_numba(warmup, args.seed)

    wall_start = time.perf_counter()
    if workers == 1:
        results = [_run_one(task) for task in tasks]
    else:
        results = []
        executor_cls = ThreadPoolExecutor if args.executor == "thread" else ProcessPoolExecutor
        with executor_cls(max_workers=workers) as pool:
            futures = [pool.submit(_run_one, task) for task in tasks]
            for future in as_completed(futures):
                sample_id, seed, seconds, data = future.result()
                print(f"sample {sample_id:04d} seed={seed} finished in {seconds:.3f}s")
                results.append((sample_id, seed, seconds, data))

    write_hdf5(args.out, cfg, args.backend, results)
    wall_seconds = time.perf_counter() - wall_start
    total_cell_updates = args.samples * cfg.steps * cfg.nx * cfg.ny
    solve_seconds = sum(item[2] for item in results)
    print(f"wrote {args.out}")
    print(f"mean sample time: {np.mean([item[2] for item in results]):.3f}s")
    print(f"aggregate solver throughput: {total_cell_updates / solve_seconds / 1e6:.2f} Mcell-updates/s")
    report = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "backend": args.backend,
        "executor": args.executor,
        "workers": workers,
        "samples": args.samples,
        "configuration": cfg.to_dict(),
        "wall_seconds_including_hdf5": wall_seconds,
        "mean_sample_seconds": float(np.mean([item[2] for item in results])),
        "aggregate_solver_mcell_updates_per_s": total_cell_updates / solve_seconds / 1e6,
        "end_to_end_mcell_updates_per_s": total_cell_updates / wall_seconds / 1e6,
        "output_bytes": args.out.stat().st_size,
    }
    (args.out.parent / "generation_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
