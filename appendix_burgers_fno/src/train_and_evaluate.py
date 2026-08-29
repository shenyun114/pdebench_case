#!/usr/bin/env python3
"""Train PDEBench's official FNO1d and evaluate an autoregressive rollout."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pdebench.models.fno.fno import FNO1d  # noqa: E402
from pdebench.models.fno.utils import FNODatasetSingle  # noqa: E402
from pdebench.models.metrics import metric_func  # noqa: E402


def rollout(
    model: nn.Module,
    xx: torch.Tensor,
    yy: torch.Tensor,
    grid: torch.Tensor,
    initial_step: int,
) -> torch.Tensor:
    """Predict every future frame by feeding each prediction back into the window."""
    prediction = yy[..., :initial_step, :]
    window = xx
    flattened_shape = list(window.shape[:-2]) + [-1]
    for _ in range(initial_step, yy.shape[-2]):
        next_frame = model(window.reshape(flattened_shape), grid)
        prediction = torch.cat((prediction, next_frame), dim=-2)
        window = torch.cat((window[..., 1:, :], next_frame), dim=-2)
    return prediction


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, initial_step: int
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    predictions, targets = [], []
    for xx, yy, grid in loader:
        xx, yy, grid = xx.to(device), yy.to(device), grid.to(device)
        predictions.append(rollout(model, xx, yy, grid, initial_step).cpu())
        targets.append(yy.cpu())
    return torch.cat(predictions), torch.cat(targets)


def save_plots(
    output: Path,
    history: list[dict],
    pred: np.ndarray,
    truth: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    initial_step: int,
) -> None:
    """Save publication-friendly static figures without requiring a desktop."""
    epochs = [row["epoch"] for row in history]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.semilogy(epochs, [row["train_mse"] for row in history], label="training")
    ax.semilogy(epochs, [row["val_mse"] for row in history], label="validation")
    ax.set(xlabel="Epoch", ylabel="Rollout MSE", title="FNO1d convergence")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "training_curve.png", dpi=180)
    plt.close(fig)

    absolute_error = np.abs(pred - truth)
    extent = [float(x[0]), float(x[-1]), float(t[-1]), float(t[0])]
    value_limit = max(abs(truth.min()), abs(truth.max()))
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0), constrained_layout=True)
    image_truth = axes[0].imshow(
        truth, aspect="auto", extent=extent, cmap="RdBu_r",
        vmin=-value_limit, vmax=value_limit,
    )
    axes[1].imshow(
        pred, aspect="auto", extent=extent, cmap="RdBu_r",
        vmin=-value_limit, vmax=value_limit,
    )
    image_error = axes[2].imshow(
        absolute_error, aspect="auto", extent=extent, cmap="magma"
    )
    titles = ["PDE solver (truth)", "FNO rollout", "Absolute error"]
    for axis, title in zip(axes, titles):
        axis.set(title=title, xlabel="x", ylabel="t")
        axis.axhline(t[initial_step - 1], color="white", ls="--", lw=0.9)
    fig.colorbar(image_truth, ax=axes[:2], shrink=0.85, label="u")
    fig.colorbar(image_error, ax=axes[2], shrink=0.85, label="|error|")
    fig.savefig(output / "rollout_comparison.png", dpi=180)
    plt.close(fig)

    rmse_by_time = np.sqrt(np.mean((pred - truth) ** 2, axis=1))
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(t[initial_step:], rmse_by_time[initial_step:], lw=2)
    ax.set(xlabel="t", ylabel="RMSE", title="Autoregressive error accumulation")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "rollout_rmse.png", dpi=180)
    plt.close(fig)

    indices = [initial_step - 1, len(t) // 2, len(t) - 1]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.6), sharey=True)
    for axis, index in zip(axes, indices):
        axis.plot(x, truth[index], color="black", lw=2, label="truth")
        axis.plot(x, pred[index], color="#d95f02", lw=1.7, ls="--", label="FNO")
        axis.set(title=f"t = {t[index]:.3f}", xlabel="x")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("u(x,t)")
    axes[-1].legend()
    fig.suptitle("Burgers solution snapshots")
    fig.tight_layout()
    fig.savefig(output / "solution_snapshots.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--initial-step", type=int, default=5)
    parser.add_argument("--modes", type=int, default=12)
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)

    train_set = FNODatasetSingle(
        args.data.name, saved_folder=args.data.parent, initial_step=args.initial_step
    )
    validation_set = FNODatasetSingle(
        args.data.name,
        saved_folder=args.data.parent,
        initial_step=args.initial_step,
        if_test=True,
    )
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    validation_loader = DataLoader(
        validation_set, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    model = FNO1d(
        num_channels=1,
        modes=args.modes,
        width=args.width,
        initial_step=args.initial_step,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=max(args.epochs // 2, 1), gamma=0.5
    )
    loss_function = nn.MSELoss()
    history: list[dict] = []
    best_loss = float("inf")
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_total, train_items = 0.0, 0
        for xx, yy, grid in train_loader:
            xx, yy, grid = xx.to(device), yy.to(device), grid.to(device)
            prediction = rollout(model, xx, yy, grid, args.initial_step)
            loss = loss_function(
                prediction[..., args.initial_step:, :],
                yy[..., args.initial_step:, :],
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_total += loss.item() * yy.shape[0]
            train_items += yy.shape[0]

        prediction_val, truth_val = evaluate(
            model, validation_loader, device, args.initial_step
        )
        validation_loss = loss_function(
            prediction_val[..., args.initial_step:, :],
            truth_val[..., args.initial_step:, :],
        ).item()
        row = {
            "epoch": epoch,
            "train_mse": train_total / train_items,
            "val_mse": validation_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        (args.output / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        print(
            f"epoch={epoch:03d} train_mse={row['train_mse']:.6e} "
            f"val_mse={validation_loss:.6e}",
            flush=True,
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                },
                args.output / "best_fno1d.pt",
            )
        scheduler.step()

    checkpoint = torch.load(args.output / "best_fno1d.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    prediction_tensor, truth_tensor = evaluate(
        model, validation_loader, device, args.initial_step
    )
    official = metric_func(
        prediction_tensor, truth_tensor, if_mean=True, initial_step=args.initial_step
    )
    official_names = [
        "rmse",
        "normalized_rmse",
        "conservation_error",
        "max_error",
        "boundary_rmse",
        "fourier_rmse",
    ]
    official_metrics = {}
    for name, value in zip(official_names, official):
        array = value.detach().cpu().numpy()
        official_metrics[name] = float(array) if array.size == 1 else array.tolist()

    prediction = prediction_tensor.numpy()[..., 0]
    truth = truth_tensor.numpy()[..., 0]
    forecast_error = (
        prediction[..., args.initial_step:] - truth[..., args.initial_step:]
    )
    rmse_by_step = np.sqrt(np.mean(forecast_error * forecast_error, axis=(0, 1)))
    reference = np.sqrt(
        np.mean(truth[..., args.initial_step:] ** 2, axis=(0, 1))
    )
    with h5py.File(args.data, "r") as handle:
        x = handle["x-coordinate"][:]
        t = handle["t-coordinate"][:]
    truth_mass = truth.mean(axis=1)
    prediction_mass = prediction.mean(axis=1)
    elapsed = time.perf_counter() - started
    report = {
        "implementation": {
            "model": "pdebench.models.fno.fno.FNO1d",
            "dataset": "pdebench.models.fno.utils.FNODatasetSingle",
            "training": "autoregressive rolling-window rollout",
        },
        "runtime": {
            "device": str(device),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "seconds": elapsed,
        },
        "configuration": {
            **vars(args),
            "data": str(args.data),
            "output": str(args.output),
        },
        "samples": {"train": len(train_set), "validation": len(validation_set)},
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": int(checkpoint["epoch"]),
        "best_validation_mse": float(checkpoint["loss"]),
        "convergence_ratio": history[-1]["train_mse"] / history[0]["train_mse"],
        "official_pdebench_metrics": official_metrics,
        "rollout": {
            "forecast_steps": int(truth.shape[-1] - args.initial_step),
            "final_step_rmse": float(rmse_by_step[-1]),
            "final_step_nrmse": float(rmse_by_step[-1] / reference[-1]),
            "max_abs_prediction": float(np.abs(prediction).max()),
            "all_predictions_finite": bool(np.isfinite(prediction).all()),
            "mean_abs_mass_error": float(
                np.mean(
                    np.abs(
                        prediction_mass[:, args.initial_step:]
                        - truth_mass[:, args.initial_step:]
                    )
                )
            ),
        },
    }
    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    np.savez_compressed(
        args.output / "predictions.npz",
        prediction=prediction,
        truth=truth,
        x=x,
        t=t,
    )
    save_plots(
        args.output,
        history,
        prediction[0].T,
        truth[0].T,
        x,
        t,
        args.initial_step,
    )
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    main()
