#!/usr/bin/env python3
"""Record the software and accelerator used by a run."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path

import h5py
import jax
import matplotlib
import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except subprocess.CalledProcessError:
        commit = None
    report = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu,
        "numpy": np.__version__,
        "h5py": h5py.__version__,
        "matplotlib": matplotlib.__version__,
        "jax": jax.__version__,
        "pdebench_commit": commit,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

