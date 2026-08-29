#!/usr/bin/env python3
"""Verify the reaction-diffusion data, derived diagnostics, figures and GIF."""

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
        u, v = handle["u"][:], handle["v"][:]
    metrics = json.loads((result_dir / "physical_metrics.json").read_text())
    resolution = json.loads((result_dir / "resolution_metrics.json").read_text())
    required = [
        "field_snapshots.png",
        "coupled_state.png",
        "phase_portrait.png",
        "mechanism_balance.png",
        "pattern_diagnostics.png",
        "spatial_spectrum.png",
        "reaction_diffusion_evolution.gif",
        "resolution_study.png",
    ]
    checks = {
        "all_fields_present": {"u", "v", "x", "y", "t"} <= keys,
        "fields_have_matching_3d_shape": u.ndim == 3 and v.shape == u.shape,
        "all_values_finite": bool(np.isfinite(u).all() and np.isfinite(v).all()),
        "diffusion_smooths_initial_noise": metrics["pattern"]["u_gradient_energy_final"] < metrics["pattern"]["u_gradient_energy_initial"],
        "fields_become_correlated": metrics["coupling"]["final_correlation"] > 0.8,
        "all_visuals_exist": all((result_dir / name).is_file() for name in required),
        "gif_has_material_size": (result_dir / "reaction_diffusion_evolution.gif").stat().st_size > 100_000,
        "exact_pdebench_laplacian_used": metrics["mechanisms"]["laplacian_definition"].startswith("exact PDEBench"),
        "grid_consistency_errors_decrease": resolution["acceptance"]["u_error_decreases"] and resolution["acceptance"]["v_error_decreases"],
    }
    print(json.dumps(checks, indent=2))
    if not all(checks.values()):
        raise SystemExit("reaction-diffusion verification failed")
    print("PASS: reaction-diffusion fields, mechanisms and visualizations are valid")


if __name__ == "__main__":
    main()
