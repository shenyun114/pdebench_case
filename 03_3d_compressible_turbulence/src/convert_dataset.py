"""Convert PDEBench's five NPY arrays into one documented HDF5 artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from common import load_config, locate_field_files, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    work = args.work_dir.resolve()
    raw = work / "raw_dataset"
    fields = locate_field_files(raw)
    arrays = {name: np.load(path, mmap_mode="r") for name, path in fields.items()}
    shape = arrays["D"].shape
    if any(array.shape != shape for array in arrays.values()):
        raise SystemExit("the five PDEBench fields have inconsistent shapes")
    output = work / "cfd3d_dataset.h5"
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    nt, n = shape[1], shape[2]
    coords = {}
    for axis in "xyz":
        coords[axis] = np.load(raw / f"{axis}_coordinate.npy")
    raw_t = np.load(raw / "t_coordinate.npy")
    # Upstream allocates one extra coordinate. The field-array length is authoritative.
    coords["t"] = raw_t[:nt]
    names = {
        "D": "density",
        "Vx": "velocity_x",
        "Vy": "velocity_y",
        "Vz": "velocity_z",
        "P": "pressure",
    }
    with h5py.File(output, "w") as handle:
        handle.attrs.update(
            {
                "case": cfg["case"]["name"],
                "source": "PDEBench CFD_multi_Hydra.py, unmodified upstream checkout",
                "pdebench_commit": cfg["case"]["expected_commit"],
                "parallelism": "sample-level pmap(vmap(evolve)); no spatial decomposition",
                "gamma": cfg["simulation"]["gamma"],
                "mach_initial": cfg["simulation"]["mach"],
                "boundary_condition": "periodic",
            }
        )
        grid = handle.create_group("grid")
        for name, values in coords.items():
            grid.create_dataset(name, data=values)
        solution = handle.create_group("solution")
        chunks = (1, 1, n, n, n)
        for source_name, target_name in names.items():
            solution.create_dataset(
                target_name,
                data=arrays[source_name],
                chunks=chunks,
                compression="lzf",
                shuffle=True,
            )
    info = {
        "path": str(output),
        "shape_per_field": list(shape),
        "layout": "[sample, time, x, y, z]",
        "fields": list(names.values()),
        "coordinate_lengths": {name: int(len(value)) for name, value in coords.items()},
        "upstream_t_coordinate_length": int(len(raw_t)),
        "upstream_coordinate_trimmed": bool(len(raw_t) != nt),
        "size_bytes": output.stat().st_size,
    }
    write_json(work / "dataset_info.json", info)
    print(f"HDF5: {shape}, {output.stat().st_size / 2**20:.2f} MiB")


if __name__ == "__main__":
    main()
