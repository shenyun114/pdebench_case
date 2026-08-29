#!/usr/bin/env python3
"""Generate one reproducible sample with PDEBench's reaction-diffusion solver."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import h5py
import numpy as np
import scipy
import yaml

from pdebench.data_gen.src.sim_diff_react import Simulator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))["simulation"]

    simulator = Simulator(
        Du=float(config["du"]),
        Dv=float(config["dv"]),
        k=float(config["reaction_k"]),
        t=float(config["final_time"]),
        tdim=int(config["frames"]),
        xdim=int(config["nx"]),
        ydim=int(config["ny"]),
        seed=int(config["seed"]),
    )
    started = time.perf_counter()
    data = simulator.generate_sample().astype(np.float32)
    elapsed = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as handle:
        handle.create_dataset("u", data=data[..., 0], compression="gzip", shuffle=True)
        handle.create_dataset("v", data=data[..., 1], compression="gzip", shuffle=True)
        handle.create_dataset("x", data=simulator.x.astype(np.float32))
        handle.create_dataset("y", data=simulator.y.astype(np.float32))
        handle.create_dataset("t", data=simulator.t.astype(np.float32))
        handle.attrs["Du"] = float(config["du"])
        handle.attrs["Dv"] = float(config["dv"])
        handle.attrs["k"] = float(config["reaction_k"])
        handle.attrs["seed"] = int(config["seed"])
        handle.attrs["boundary"] = "homogeneous Neumann"
        handle.attrs["solver"] = "PDEBench Simulator / SciPy solve_ivp RK45"
        handle.attrs["runtime_seconds"] = elapsed
    report = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "shape": list(data.shape),
        "runtime_seconds": elapsed,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "configuration": config,
        "config_file": str(args.config.resolve()),
        "pdebench_commit": __import__("subprocess").check_output(
            ["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True
        ).strip(),
        "solver": "pdebench.data_gen.src.sim_diff_react.Simulator",
    }
    (args.output.parent / "simulation_info.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
