#!/usr/bin/env python3
"""Run PDEBench's official radial-dam-break solver and retain all state fields."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from pdebench.data_gen.src.sim_radial_dam_break import RadialDamBreak2D


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))["simulation"]
    nx, ny = int(config["nx"]), int(config["ny"])
    frames = int(config["frames"])
    final_time = float(config["final_time"])
    gravity = float(config["gravity"])
    dam_radius = float(config["dam_radius"])
    inner_height = float(config["inner_height"])
    if inner_height != 2.0:
        raise ValueError(
            "Pinned PDEBench commit hard-codes h_in=2.0 in set_initial_conditions; "
            "use inner_height: 2.0 to avoid recording a parameter the solver ignores."
        )

    scenario = RadialDamBreak2D(
        xdim=nx,
        ydim=ny,
        grav=gravity,
        dam_radius=dam_radius,
        inner_height=inner_height,
    )
    started = time.perf_counter()
    scenario.run(T=final_time, tsteps=frames - 1)
    elapsed = time.perf_counter() - started

    state = scenario.save_state
    fields = {
        name: np.asarray(state[name], dtype=np.float32)
        for name in ("h", "u", "v", "hu", "hv")
    }
    x = np.asarray(state["x"], dtype=np.float32)
    y = np.asarray(state["y"], dtype=np.float32)
    t = np.asarray(state["t"], dtype=np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as handle:
        for name, values in fields.items():
            handle.create_dataset(name, data=values, compression="gzip", shuffle=True)
        handle.create_dataset("x", data=x)
        handle.create_dataset("y", data=y)
        handle.create_dataset("t", data=t)
        handle.attrs["gravity"] = gravity
        handle.attrs["dam_radius"] = dam_radius
        handle.attrs["inner_height"] = inner_height
        handle.attrs["outer_height"] = 1.0
        handle.attrs["solver"] = "PDEBench RadialDamBreak2D / PyClaw Roe with entropy fix"
        handle.attrs["boundary"] = "extrapolation"
        handle.attrs["runtime_seconds"] = elapsed

    report = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "shape": list(fields["h"].shape),
        "runtime_seconds": elapsed,
        "numpy": np.__version__,
        "configuration": config,
        "config_file": str(args.config.resolve()),
        "pdebench_commit": __import__("subprocess").check_output(
            ["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True
        ).strip(),
        "solver": "pdebench.data_gen.src.sim_radial_dam_break.RadialDamBreak2D",
    }
    (args.output.parent / "simulation_info.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
