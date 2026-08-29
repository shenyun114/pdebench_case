#!/usr/bin/env python3
"""Validate PDEBench's Burgers HDF5 schema and basic physical invariants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("hdf5", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with h5py.File(args.hdf5, "r") as handle:
        keys = sorted(handle.keys())
        u = handle["tensor"][:]
        x = handle["x-coordinate"][:]
        t = handle["t-coordinate"][:]
        viscosity = float(handle.attrs["Nu"])

    mass = u.mean(axis=-1)
    energy = np.mean(u * u, axis=-1)
    mass_drift = np.abs(mass - mass[:, :1])
    positive_energy_step = np.maximum(np.diff(energy, axis=1), 0.0)
    checks = {
        "required_keys": keys == ["t-coordinate", "tensor", "x-coordinate"],
        "rank_is_batch_time_x": u.ndim == 3,
        "coordinate_lengths_match": u.shape[1:] == (len(t), len(x)),
        "all_values_finite": bool(np.isfinite(u).all()),
        "x_strictly_increasing": bool(np.all(np.diff(x) > 0)),
        "t_strictly_increasing": bool(np.all(np.diff(t) > 0)),
        "mass_conserved": bool(mass_drift.max() < 2e-5),
        "viscous_energy_nonincreasing": bool(positive_energy_step.max() < 2e-5),
    }
    report = {
        "file": str(args.hdf5.resolve()),
        "shape": list(u.shape),
        "dtype": str(u.dtype),
        "viscosity": viscosity,
        "x_range": [float(x[0]), float(x[-1])],
        "t_range": [float(t[0]), float(t[-1])],
        "max_abs_mass_drift": float(mass_drift.max()),
        "max_positive_energy_step": float(positive_energy_step.max()),
        "checks": checks,
        "passed": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("HDF5 validation failed")


if __name__ == "__main__":
    main()

