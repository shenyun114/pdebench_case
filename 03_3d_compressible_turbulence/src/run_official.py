"""Invoke PDEBench's unmodified multi-GPU 3D compressible-flow generator."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import time
from pathlib import Path

import jax
import numpy as np

from common import git_commit, load_config, locate_field_files, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("dataset", "benchmark"), required=True)
    return parser.parse_args()


def visible_devices(count: int) -> str:
    return ",".join(str(index) for index in range(count))


def run_solver(
    *, solver_dir: Path, output_dir: Path, sim: dict, gpu_count: int,
    log_path: Path, cache_dir: Path
) -> dict:
    if sim["samples"] % gpu_count:
        raise ValueError("sample count must be divisible by GPU count")
    output_dir.mkdir(parents=True, exist_ok=False)
    save_path = str(output_dir.resolve()) + "/"
    compat_launcher = Path(__file__).with_name("jax_loc_compat.py")
    command = [
        os.environ.get("PYTHON", os.sys.executable),
        str(compat_launcher),
        str(solver_dir / "CFD_multi_Hydra.py"),
        "+args=3D_Multi_TurbM1.yaml",
        f"++args.save={save_path}",
        f"++args.nx={sim['resolution']}",
        f"++args.ny={sim['resolution']}",
        f"++args.nz={sim['resolution']}",
        f"++args.numbers={sim['samples']}",
        f"++args.init_key={sim['seed']}",
        f"++args.M0={sim['mach']}",
        f"++args.k_tot={sim['k_total']}",
        f"++args.gamma={sim['gamma']}",
        f"++args.eta={sim['eta']}",
        f"++args.zeta={sim['zeta']}",
        f"++args.CFL={sim['cfl']}",
        f"++args.fin_time={sim['final_time']}",
        f"++args.dt_save={sim['save_interval']}",
        f"++args.show_steps={sim['show_steps']}",
        "++args.if_show=0",
        "++args.if_rand_param=false",
        "++args.init_mode_Multi=3D_Turbs",
    ]
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": visible_devices(gpu_count),
            "JAX_PLATFORMS": "cuda",
            "JAX_COMPILATION_CACHE_DIR": str(cache_dir.resolve()),
            "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "PYTHONUNBUFFERED": "1",
        }
    )
    started = time.perf_counter()
    process = subprocess.run(
        command, cwd=solver_dir, env=env, text=True, capture_output=True
    )
    elapsed = time.perf_counter() - started
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "$ " + " ".join(command) + "\n\nSTDOUT\n" + process.stdout
        + "\nSTDERR\n" + process.stderr,
        encoding="utf-8",
    )
    if process.returncode:
        raise RuntimeError(f"PDEBench solver failed; see {log_path}")
    files = locate_field_files(output_dir)
    shape = list(np.load(files["D"], mmap_mode="r").shape)
    cells_advanced = int(np.prod(shape[:2]) * np.prod(shape[2:]))
    return {
        "gpu_count": gpu_count,
        "visible_devices": visible_devices(gpu_count),
        "elapsed_seconds": elapsed,
        "shape": shape,
        "field_files": {name: path.name for name, path in files.items()},
        "cell_snapshots": cells_advanced,
        "throughput_mcell_snapshots_per_second": cells_advanced / elapsed / 1.0e6,
        "command": command,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    work_dir = args.work_dir.resolve()
    repo_override = os.environ.get("PDEBENCH_ROOT")
    repo = (
        Path(repo_override).expanduser().resolve()
        if repo_override
        else (work_dir.parent / "PDEBench").resolve()
    )
    commit = git_commit(repo)
    if commit != cfg["case"]["expected_commit"]:
        raise SystemExit(f"unexpected PDEBench commit: {commit}")
    solver_dir = repo / "pdebench/data_gen/data_gen_NLE/CompressibleFluid"
    cache_dir = work_dir / "jax-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if jax.default_backend() != "gpu":
        raise SystemExit(f"GPU JAX required, got {jax.default_backend()}")
    available = len(jax.devices("gpu"))

    if args.mode == "dataset":
        sim = dict(cfg["simulation"])
        gpu_count = min(8, available, sim["samples"])
        raw_dir = work_dir / "raw_dataset"
        result = run_solver(
            solver_dir=solver_dir,
            output_dir=raw_dir,
            sim=sim,
            gpu_count=gpu_count,
            log_path=work_dir / "logs/dataset.log",
            cache_dir=cache_dir,
        )
        result.update(
            {
                "pdebench_commit": commit,
                "jax_version": jax.__version__,
                "jax_backend": jax.default_backend(),
                "available_gpu_count": available,
                "devices": [str(device) for device in jax.devices("gpu")],
                "simulation": sim,
            }
        )
        write_json(work_dir / "dataset_run.json", result)
        print(f"dataset: {result['shape']} in {result['elapsed_seconds']:.3f} s")
        return

    bench = cfg["benchmark"]
    if not bench["enabled"]:
        print("benchmark disabled")
        return
    sim = dict(cfg["simulation"])
    sim.update(
        {
            "resolution": bench["resolution"],
            "samples": bench["samples"],
            "final_time": bench["final_time"],
            "save_interval": bench["save_interval"],
        }
    )
    requested = [int(value) for value in bench["gpu_counts"]]
    if max(requested) > available:
        raise SystemExit(f"requested {max(requested)} GPUs, only {available} available")
    rows = []
    benchmark_root = work_dir / "benchmark"
    benchmark_root.mkdir(parents=True, exist_ok=True)
    for gpu_count in requested:
        # One unreported warm-up populates the persistent XLA cache for this pmap shape.
        warm_dir = benchmark_root / f"warmup_{gpu_count}gpu"
        warm = run_solver(
            solver_dir=solver_dir, output_dir=warm_dir, sim=sim,
            gpu_count=gpu_count,
            log_path=work_dir / f"logs/benchmark_warmup_{gpu_count}gpu.log",
            cache_dir=cache_dir,
        )
        shutil.rmtree(warm_dir)
        for repeat in range(int(bench["repeats"])):
            output_dir = benchmark_root / f"run_{gpu_count}gpu_r{repeat + 1}"
            result = run_solver(
                solver_dir=solver_dir, output_dir=output_dir, sim=sim,
                gpu_count=gpu_count,
                log_path=work_dir / f"logs/benchmark_{gpu_count}gpu_r{repeat + 1}.log",
                cache_dir=cache_dir,
            )
            result["repeat"] = repeat + 1
            result["warmup_elapsed_seconds"] = warm["elapsed_seconds"]
            rows.append(result)
            shutil.rmtree(output_dir)

    medians = {}
    baseline = None
    for gpu_count in requested:
        values = [row["elapsed_seconds"] for row in rows if row["gpu_count"] == gpu_count]
        elapsed = float(np.median(values))
        if baseline is None:
            baseline = elapsed
        medians[str(gpu_count)] = {
            "elapsed_seconds_median": elapsed,
            "speedup": baseline / elapsed,
            "parallel_efficiency": baseline / elapsed / gpu_count,
            "repeats": values,
        }
    metrics = {
        "definition": "fixed-total-sample strong scaling; warm-up excluded; includes process startup, initialization, solve, device-to-host transfer and NPY output",
        "parallelism": "sample-level jax.pmap outside device-level jax.vmap; no spatial domain decomposition",
        "pdebench_commit": commit,
        "jax_version": jax.__version__,
        "available_gpu_count": available,
        "simulation": sim,
        "results": medians,
    }
    write_json(benchmark_root / "benchmark_metrics.json", metrics)
    with (benchmark_root / "benchmark.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("gpu_count", "repeat", "elapsed_seconds", "warmup_elapsed_seconds", "throughput_mcell_snapshots_per_second"),
        )
        writer.writeheader()
        writer.writerows({key: row[key] for key in writer.fieldnames} for row in rows)
    print("benchmark:", {key: round(value["elapsed_seconds_median"], 3) for key, value in medians.items()})


if __name__ == "__main__":
    main()
