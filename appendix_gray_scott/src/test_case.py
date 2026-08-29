from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gray_scott import SimConfig, simulate_numba, simulate_numpy, simulate_python


def test_backends_are_consistent() -> dict[str, object]:
    cfg = SimConfig(nx=24, ny=20, steps=12, save_every=4, noise=0.01)
    py_data = simulate_python(cfg, seed=3)
    np_data = simulate_numpy(cfg, seed=3)
    nb_data = simulate_numba(cfg, seed=3)
    assert py_data.shape == (4, 20, 24, 2)
    assert np.all(np.isfinite(np_data))
    assert np.allclose(py_data, np_data, rtol=2e-5, atol=2e-5)
    assert np.allclose(np_data, nb_data, rtol=2e-5, atol=2e-5)
    return {
        "shape": list(py_data.shape),
        "all_values_finite": bool(np.isfinite(py_data).all() and np.isfinite(np_data).all() and np.isfinite(nb_data).all()),
        "python_numpy_max_abs": float(np.max(np.abs(py_data - np_data))),
        "numpy_numba_max_abs": float(np.max(np.abs(np_data - nb_data))),
        "rtol": 2e-5,
        "atol": 2e-5,
        "passed": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = test_backends_are_consistent()
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("smoke test passed")
