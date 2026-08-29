#!/usr/bin/env python3
"""Grid-consistency study using PDEBench's exact ODE and Laplacian definitions."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import yaml
from scipy.integrate import solve_ivp

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pdebench.data_gen.src.sim_diff_react import Simulator  # noqa: E402
from pdebench_operators import neumann_laplacian  # noqa: E402


def block_average(field: np.ndarray, factor: int) -> np.ndarray:
    ny, nx = field.shape
    return field.reshape(ny // factor, factor, nx // factor, factor).mean(axis=(1, 3))


def solve_projected(n: int, high_u0: np.ndarray, high_v0: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray, float]:
    factor = high_u0.shape[0] // n
    u0, v0 = block_average(high_u0, factor), block_average(high_v0, factor)
    sim = Simulator(Du=float(cfg["du"]), Dv=float(cfg["dv"]), k=float(cfg["reaction_k"]), t=float(cfg["final_time"]), tdim=int(cfg["frames"]), xdim=n, ydim=n, seed=int(cfg["seed"]))
    sim.lap = neumann_laplacian(n, n, sim.dx, sim.dy)
    initial = np.concatenate([u0.ravel(), v0.ravel()])
    started = time.perf_counter()
    solution = solve_ivp(sim.rc_ode, (0, sim.T), initial, t_eval=sim.t)
    runtime = time.perf_counter() - started
    if not solution.success:
        raise RuntimeError(solution.message)
    cells = n * n
    u = solution.y[:cells, -1].reshape(n, n)
    v = solution.y[cells:, -1].reshape(n, n)
    return u, v, runtime


def relative_l2(value: np.ndarray, reference: np.ndarray) -> float:
    return float(np.sqrt(np.mean((value - reference) ** 2)) / np.sqrt(np.mean(reference**2)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-data", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    full = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cfg, study = full["simulation"], full["resolution_study"]
    grids = [int(value) for value in study["coarse_grids"]]
    ref_grid = int(study["reference_grid"])
    with h5py.File(args.reference_data) as handle:
        high_u0, high_v0 = handle["u"][0].astype(float), handle["v"][0].astype(float)
        high_uf, high_vf = handle["u"][-1].astype(float), handle["v"][-1].astype(float)
        reference_runtime = float(handle.attrs["runtime_seconds"])
    rows = []
    for grid in grids:
        u, v, runtime = solve_projected(grid, high_u0, high_v0, cfg)
        factor = ref_grid // grid
        target_u, target_v = block_average(high_uf, factor), block_average(high_vf, factor)
        rows.append({"grid": grid, "runtime_seconds": runtime, "relative_l2_u": relative_l2(u, target_u), "relative_l2_v": relative_l2(v, target_v), "final_correlation_uv": float(np.corrcoef(u.ravel(), v.ravel())[0, 1])})
    rows.append({"grid": ref_grid, "runtime_seconds": reference_runtime, "relative_l2_u": 0.0, "relative_l2_v": 0.0, "final_correlation_uv": float(np.corrcoef(high_uf.ravel(), high_vf.ravel())[0, 1])})
    order_u = float(np.log(rows[0]["relative_l2_u"] / rows[1]["relative_l2_u"]) / np.log(2))
    order_v = float(np.log(rows[0]["relative_l2_v"] / rows[1]["relative_l2_v"]) / np.log(2))
    report = {
        "study_type": "projected-initial-condition grid consistency; finest numerical solution is not an exact solution",
        "operator": "PDEBench Simulator.rc_ode plus its exact five-point Neumann Laplacian",
        "initial_condition_note": "The 128-grid seeded random field is block-averaged onto coarse grids so all runs represent the same resolved initial condition.",
        "rows": rows,
        "observed_l2_order_u": order_u,
        "observed_l2_order_v": order_v,
        "acceptance": {"u_error_decreases": rows[1]["relative_l2_u"] < rows[0]["relative_l2_u"], "v_error_decreases": rows[1]["relative_l2_v"] < rows[0]["relative_l2_v"]},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "resolution_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), constrained_layout=True)
    axes[0].loglog(grids, [r["relative_l2_u"] for r in rows[:-1]], "o-", label=f"u, p={order_u:.2f}")
    axes[0].loglog(grids, [r["relative_l2_v"] for r in rows[:-1]], "s--", label=f"v, p={order_v:.2f}")
    axes[0].set(title="Projected-IC self-convergence", xlabel="Grid N", ylabel="Relative L2 error")
    axes[1].plot([r["grid"] for r in rows], [r["runtime_seconds"] for r in rows], "o-", color="#d95f02")
    axes[1].set(title="Runtime cost", xlabel="Grid N", ylabel="Seconds")
    axes[2].plot([r["grid"] for r in rows], [r["final_correlation_uv"] for r in rows], "o-", color="#1b9e77")
    axes[2].set(title="Physical-statistic consistency", xlabel="Grid N", ylabel="Final corr(u,v)")
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
