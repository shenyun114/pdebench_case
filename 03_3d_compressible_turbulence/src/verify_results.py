"""Machine-readable acceptance checks for the complete 3D CFD workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from common import load_config, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    work = args.work_dir.resolve()
    checks = {}
    dataset = work / "cfd3d_dataset.h5"
    checks["hdf5_exists"] = dataset.exists() and dataset.stat().st_size > 0
    if checks["hdf5_exists"]:
        with h5py.File(dataset, "r") as handle:
            expected = int(cfg["simulation"]["resolution"])
            density = handle["solution/density"]
            checks["shape"] = density.shape[0] == int(cfg["simulation"]["samples"]) and density.shape[2:] == (expected,) * 3
            checks["five_fields"] = set(handle["solution"].keys()) == {"density", "pressure", "velocity_x", "velocity_y", "velocity_z"}
            checks["finite_positive_thermodynamics"] = bool(
                np.isfinite(density[:]).all()
                and np.isfinite(handle["solution/pressure"][:]).all()
                and density[:].min() > 0
                and handle["solution/pressure"][:].min() > 0
            )
            checks["time_coordinate"] = len(handle["grid/t"]) == density.shape[1]
    physical_path = work / "results/physical_metrics.json"
    checks["physical_metrics"] = physical_path.exists()
    if physical_path.exists():
        physical = json.loads(physical_path.read_text(encoding="utf-8"))
        checks["mass_drift_below_2pct"] = physical["mass_relative_drift"] < 0.02
        checks["energy_drift_below_5pct"] = physical["total_energy_relative_drift"] < 0.05
        checks["nontrivial_vorticity"] = physical["final"]["rms_vorticity"] > 0
    required = [
        "orthogonal_slices.png",
        "density_vorticity_isosurfaces.png",
        "conservation_and_flow_diagnostics.png",
        "kinetic_energy_spectrum.png",
        "turbulence_evolution.gif",
    ]
    if cfg["benchmark"]["enabled"]:
        required.append("multi_gpu_scaling.png")
    checks["visualizations"] = all((work / "results" / name).stat().st_size > 1000 for name in required)
    if cfg["benchmark"]["enabled"]:
        benchmark_path = work / "benchmark/benchmark_metrics.json"
        checks["benchmark_metrics"] = benchmark_path.exists()
        if benchmark_path.exists():
            benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
            expected_gpu = {str(value) for value in cfg["benchmark"]["gpu_counts"]}
            checks["gpu_groups"] = set(benchmark["results"]) == expected_gpu
            checks["sample_parallel_definition"] = "no spatial domain decomposition" in benchmark["parallelism"]
    else:
        checks["benchmark_metrics"] = True
        checks["gpu_groups"] = True
        checks["sample_parallel_definition"] = True
    passed = all(checks.values())
    write_json(work / "verification.json", {"passed": passed, "checks": checks})
    if not passed:
        failed = [key for key, value in checks.items() if not value]
        raise SystemExit("verification failed: " + ", ".join(failed))
    if cfg["benchmark"]["enabled"]:
        print("PASS: official 3D CFD generation, multi-GPU benchmark and physical postprocessing are valid")
    else:
        print("PASS: official 3D CFD generation and physical postprocessing are valid")


if __name__ == "__main__":
    main()
