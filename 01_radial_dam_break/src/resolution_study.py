#!/usr/bin/env python3
"""Three-grid self-convergence study for the PDEBench radial dam break."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pdebench.data_gen.src.sim_radial_dam_break import RadialDamBreak2D  # noqa: E402


def block_average(field: np.ndarray, factor: int) -> np.ndarray:
    ny, nx = field.shape
    return field.reshape(ny // factor, factor, nx // factor, factor).mean(axis=(1, 3))


def final_state(n: int, cfg: dict) -> tuple[np.ndarray, float, float]:
    scenario = RadialDamBreak2D(
        xdim=n,
        ydim=n,
        grav=float(cfg["gravity"]),
        dam_radius=float(cfg["dam_radius"]),
        inner_height=float(cfg["inner_height"]),
    )
    started = time.perf_counter()
    scenario.run(T=float(cfg["final_time"]), tsteps=int(cfg["frames"]) - 1)
    runtime = time.perf_counter() - started
    h = np.asarray(scenario.save_state["h"], dtype=np.float64)
    dx = 5.0 / n
    drift = float(np.max(np.abs((h.sum((1, 2)) - h[0].sum()) / h[0].sum())))
    return h[-1], runtime, drift


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-data", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    full = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cfg, study = full["simulation"], full["resolution_study"]
    grids = [int(value) for value in study["coarse_grids"]]
    reference_grid = int(study["reference_grid"])
    if cfg["nx"] != cfg["ny"] or int(cfg["nx"]) != reference_grid:
        raise ValueError("resolution study requires a square main run matching reference_grid")
    with h5py.File(args.reference_data) as handle:
        reference = handle["h"][-1].astype(np.float64)
        reference_runtime = float(handle.attrs["runtime_seconds"])
        all_h = handle["h"][:].astype(np.float64)
    reference_drift = float(
        np.max(np.abs((all_h.sum((1, 2)) - all_h[0].sum()) / all_h[0].sum()))
    )

    rows = []
    for grid in grids:
        field, runtime, drift = final_state(grid, cfg)
        factor = reference_grid // grid
        target = block_average(reference, factor)
        difference = field - target
        rows.append(
            {
                "grid": grid,
                "runtime_seconds": runtime,
                "relative_l1_error_vs_block_averaged_reference": float(
                    np.mean(np.abs(difference)) / np.mean(np.abs(target))
                ),
                "relative_l2_error_vs_block_averaged_reference": float(
                    np.sqrt(np.mean(difference**2)) / np.sqrt(np.mean(target**2))
                ),
                "max_relative_mass_drift": drift,
            }
        )
    rows.append(
        {
            "grid": reference_grid,
            "runtime_seconds": reference_runtime,
            "relative_l1_error_vs_block_averaged_reference": 0.0,
            "relative_l2_error_vs_block_averaged_reference": 0.0,
            "max_relative_mass_drift": reference_drift,
        }
    )
    observed_order = float(
        np.log(rows[0]["relative_l2_error_vs_block_averaged_reference"] / rows[1]["relative_l2_error_vs_block_averaged_reference"])
        / np.log(2.0)
    )
    report = {
        "study_type": "three-grid self-convergence; finest numerical solution is the reference, not an exact solution",
        "reference_grid": reference_grid,
        "rows": rows,
        "observed_l2_order_between_coarse_grids": observed_order,
        "acceptance": {
            "l2_error_decreases_with_refinement": rows[1]["relative_l2_error_vs_block_averaged_reference"] < rows[0]["relative_l2_error_vs_block_averaged_reference"],
            "all_mass_drifts_below_1e-5": all(row["max_relative_mass_drift"] < 1e-5 for row in rows),
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "resolution_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    coarse = rows[:-1]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), constrained_layout=True)
    axes[0].loglog([r["grid"] for r in coarse], [r["relative_l2_error_vs_block_averaged_reference"] for r in coarse], "o-", label="relative L2")
    axes[0].loglog([r["grid"] for r in coarse], [r["relative_l1_error_vs_block_averaged_reference"] for r in coarse], "s--", label="relative L1")
    axes[0].set(title=f"Self-convergence (observed p={observed_order:.2f})", xlabel="Grid N (N x N)", ylabel="Error vs 128-grid reference")
    axes[1].plot([r["grid"] for r in rows], [r["runtime_seconds"] for r in rows], "o-", color="#d95f02")
    axes[1].set(title="Runtime cost", xlabel="Grid N", ylabel="Seconds")
    axes[2].semilogy([r["grid"] for r in rows], [max(r["max_relative_mass_drift"], 1e-16) for r in rows], "o-", color="#1b9e77")
    axes[2].axhline(1e-5, ls="--", color="gray", label="acceptance threshold")
    axes[2].set(title="Mass-conservation stability", xlabel="Grid N", ylabel="Max relative drift")
    for axis in axes:
        axis.grid(alpha=0.25, which="both")
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(handles, labels)
    fig.savefig(args.output / "resolution_study.png", dpi=180)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
