#!/usr/bin/env python3
"""Fail fast when a pipeline result is incomplete or numerically invalid."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    result_dir = Path(sys.argv[1])
    validation = json.loads((result_dir / "data_validation.json").read_text())
    metrics = json.loads((result_dir / "metrics.json").read_text())
    required = [
        "training_curve.png",
        "rollout_comparison.png",
        "rollout_rmse.png",
        "solution_snapshots.png",
        "physics_diagnostics.png",
        "burgers_truth_vs_fno.gif",
        "physics_metrics.json",
        "predictions.npz",
        "best_fno1d.pt",
        "history.json",
        "system_info.json",
    ]
    checks = {
        "data_validation_passed": validation["passed"],
        "all_predictions_finite": metrics["rollout"]["all_predictions_finite"],
        "training_loss_decreased": metrics["convergence_ratio"] < 0.25,
        "validation_mse_below_smoke_threshold": metrics["best_validation_mse"] < 0.1,
        "all_artifacts_exist": all((result_dir / name).is_file() for name in required),
        "gif_has_material_size": (result_dir / "burgers_truth_vs_fno.gif").stat().st_size > 100_000,
    }
    print(json.dumps(checks, indent=2))
    if not all(checks.values()):
        raise SystemExit("result verification failed")
    print("PASS: data, convergence, rollout and artifacts are valid")


if __name__ == "__main__":
    main()
