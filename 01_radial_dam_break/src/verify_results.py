#!/usr/bin/env python3
"""Verify fields, conservation diagnostics and visual artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np


def main() -> None:
    data_path, result_dir = Path(sys.argv[1]), Path(sys.argv[2])
    with h5py.File(data_path) as handle:
        keys = set(handle.keys())
        h = handle["h"][:]
        u = handle["u"][:]
        v = handle["v"][:]
        nx, ny, nt = len(handle["x"]), len(handle["y"]), len(handle["t"])
    metrics = json.loads((result_dir / "physical_metrics.json").read_text())
    resolution = json.loads((result_dir / "resolution_metrics.json").read_text())
    required = [
        "water_depth_snapshots.png",
        "surface_evolution_3d.png",
        "velocity_and_froude.png",
        "radial_profiles.png",
        "conservation_diagnostics.png",
        "shallow_water_evolution.gif",
        "resolution_study.png",
    ]
    checks = {
        "all_fields_present": {"h", "u", "v", "hu", "hv", "x", "y", "t"} <= keys,
        "field_shape_matches_coordinates": h.shape == (nt, nx, ny) and u.shape == h.shape and v.shape == h.shape,
        "all_values_finite": bool(np.isfinite(h).all() and np.isfinite(u).all() and np.isfinite(v).all()),
        "positive_water_depth": bool(h.min() > 0.0),
        "relative_mass_drift_below_1e-5": metrics["conservation"]["max_relative_mass_drift"] < 1e-5,
        "symmetry_error_below_1e-3": metrics["symmetry"]["max_rotation_relative_l1"] < 1e-3,
        "all_visuals_exist": all((result_dir / name).is_file() for name in required),
        "gif_has_material_size": (result_dir / "shallow_water_evolution.gif").stat().st_size > 100_000,
        "self_convergence_error_decreases": resolution["acceptance"]["l2_error_decreases_with_refinement"],
        "study_mass_conservation_passes": resolution["acceptance"]["all_mass_drifts_below_1e-5"],
    }
    print(json.dumps(checks, indent=2))
    if not all(checks.values()):
        raise SystemExit("shallow-water verification failed")
    print("PASS: shallow-water fields, physics diagnostics and visualizations are valid")


if __name__ == "__main__":
    main()
