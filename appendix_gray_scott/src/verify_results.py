#!/usr/bin/env python3
"""Machine-readable acceptance checks for the complete Gray-Scott workflow."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml


def main() -> None:
    root = Path(sys.argv[1])
    cfg = yaml.safe_load((root / "resolved_config.yaml").read_text(encoding="utf-8"))
    sim, dataset = cfg["simulation"], cfg["dataset"]
    expected_frames = int(sim["steps"]) // int(sim["save_every"]) + 1
    with h5py.File(root / "grayscott_dataset.h5") as handle:
        sample_ids = sorted(handle["samples"].keys())
        data = handle[f"samples/{sample_ids[0]}/data"][:]
        shapes_match = all(handle[f"samples/{item}/data"].shape == data.shape for item in sample_ids)
        seeds = [int(handle[f"samples/{item}"].attrs["seed"]) for item in sample_ids]
    physics = json.loads((root / "results/physical_metrics.json").read_text(encoding="utf-8"))
    benchmark = json.loads((root / "benchmark/benchmark_metrics.json").read_text(encoding="utf-8"))
    consistency = json.loads((root / "backend_consistency.json").read_text(encoding="utf-8"))
    required = [
        "sample_0000_fields.png", "sample_0000_physics_diagnostics.png",
        "sample_0000_derived_physics.png", "sample_0000_kymograph.png",
        "sample_0000_radial_spectrum.png", "dataset_final_v_montage.png",
        "sample_0000_v.gif", "physical_metrics.json",
    ]
    sweep_required = ["parameter_sweep.h5", "parameter_sweep_metrics.csv", "parameter_sweep_patterns.png", "parameter_sweep_metrics.png"]
    checks = {
        "sample_count_matches_config": len(sample_ids) == int(dataset["samples"]),
        "sample_shape_matches_config": data.shape == (expected_frames, int(sim["ny"]), int(sim["nx"]), 2),
        "all_sample_shapes_match": shapes_match,
        "seeds_are_reproducible": seeds == list(range(int(dataset["seed"]), int(dataset["seed"]) + int(dataset["samples"]))),
        "all_values_finite": bool(np.isfinite(data).all()),
        "backend_consistency_test_passes": consistency["passed"] and consistency["numpy_numba_max_abs"] < 2e-5,
        "physical_pattern_survives": physics["interpretation"]["v_pattern_survives"],
        "numba_core_faster_than_python": next(row["speedup"] for row in benchmark["rows"] if row["mode"] == "numba") > 1.0,
        "benchmark_contains_requested_workers": {row["workers"] for row in benchmark["rows"] if row["category"] == "end_to_end"} == set(cfg["benchmark"]["workers"]),
        "all_visuals_exist": all((root / "results" / name).is_file() for name in required),
        "gif_has_material_size": (root / "results/sample_0000_v.gif").stat().st_size > 10_000,
        "parameter_sweep_complete": all((root / "parameter_sweep" / name).is_file() for name in sweep_required),
    }
    report = {"checks": checks, "passed": all(checks.values())}
    (root / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("Gray-Scott verification failed")
    print("PASS: Gray-Scott preprocessing, optimized generation, benchmark and postprocessing are valid")


if __name__ == "__main__":
    main()
