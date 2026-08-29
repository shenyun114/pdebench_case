#!/usr/bin/env python3
"""Add physics diagnostics and a truth-versus-FNO animation to saved rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import animation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-step", type=int, default=5)
    args = parser.parse_args()
    arrays = np.load(args.predictions)
    prediction = arrays["prediction"]
    truth = arrays["truth"]
    x, t = arrays["x"], arrays["t"]

    mass_truth = truth.mean(axis=1)
    mass_prediction = prediction.mean(axis=1)
    energy_truth = 0.5 * np.mean(truth * truth, axis=1)
    energy_prediction = 0.5 * np.mean(prediction * prediction, axis=1)
    variation_truth = np.sum(np.abs(np.diff(truth, axis=1)), axis=1)
    variation_prediction = np.sum(np.abs(np.diff(prediction, axis=1)), axis=1)
    final_truth_spectrum = np.abs(np.fft.rfft(truth[:, :, -1], axis=1)).mean(axis=0)
    final_prediction_spectrum = np.abs(np.fft.rfft(prediction[:, :, -1], axis=1)).mean(axis=0)
    modes = np.arange(len(final_truth_spectrum))

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.0), constrained_layout=True)
    axes[0, 0].plot(t, mass_truth.mean(axis=0), color="black", label="truth")
    axes[0, 0].plot(t, mass_prediction.mean(axis=0), color="#d95f02", ls="--", label="FNO")
    axes[0, 0].set(title="Spatial mean (mass)", xlabel="t", ylabel="mean(u)")
    axes[0, 1].plot(t, energy_truth.mean(axis=0), color="black", label="truth")
    axes[0, 1].plot(t, energy_prediction.mean(axis=0), color="#d95f02", ls="--", label="FNO")
    axes[0, 1].set(title="Kinetic energy proxy", xlabel="t", ylabel="0.5 mean(u²)")
    axes[1, 0].plot(t, variation_truth.mean(axis=0), color="black", label="truth")
    axes[1, 0].plot(t, variation_prediction.mean(axis=0), color="#d95f02", ls="--", label="FNO")
    axes[1, 0].set(title="Total variation", xlabel="t", ylabel="TV(u)")
    axes[1, 1].semilogy(modes[1:], final_truth_spectrum[1:], color="black", label="truth")
    axes[1, 1].semilogy(modes[1:], final_prediction_spectrum[1:], color="#d95f02", ls="--", label="FNO")
    axes[1, 1].axvline(12, color="gray", ls=":", label="12 retained modes")
    axes[1, 1].set(title="Mean Fourier spectrum at t=1", xlabel="Fourier mode", ylabel="Amplitude")
    for axis in axes.flat:
        axis.grid(alpha=0.22)
        axis.legend(fontsize=8)
        axis.axvline(t[args.initial_step - 1], color="#7570b3", lw=0.8, ls="--")
    fig.suptitle("Burgers rollout: physical and spectral diagnostics", fontsize=15)
    fig.savefig(args.output / "physics_diagnostics.png", dpi=180)
    plt.close(fig)

    sample = 0
    y_min = float(min(truth[sample].min(), prediction[sample].min())) - 0.1
    y_max = float(max(truth[sample].max(), prediction[sample].max())) + 0.1
    error_max = float(np.abs(prediction[sample] - truth[sample]).max())
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.0), constrained_layout=True)
    truth_line, = axes[0].plot(x, truth[sample, :, 0], color="black", lw=2, label="PDE solver")
    prediction_line, = axes[0].plot(x, prediction[sample, :, 0], color="#d95f02", lw=1.8, ls="--", label="FNO")
    error_fill = axes[1].fill_between(x, 0, np.abs(prediction[sample, :, 0] - truth[sample, :, 0]), color="#7570b3")
    axes[0].set(xlabel="x", ylabel="u(x,t)", ylim=(y_min, y_max), title="Solution profile")
    axes[1].set(xlabel="x", ylabel="|error|", ylim=(0, error_max * 1.05), title="Pointwise absolute error")
    axes[0].legend()
    for axis in axes:
        axis.grid(alpha=0.22)
    title = fig.suptitle("t = 0.000 | observed input", fontsize=14)

    def update(index: int):
        nonlocal error_fill
        truth_line.set_ydata(truth[sample, :, index])
        prediction_line.set_ydata(prediction[sample, :, index])
        error_fill.remove()
        error_fill = axes[1].fill_between(
            x,
            0,
            np.abs(prediction[sample, :, index] - truth[sample, :, index]),
            color="#7570b3",
        )
        phase = "observed input" if index < args.initial_step else "autoregressive forecast"
        title.set_text(f"t = {t[index]:.3f} | {phase}")
        return truth_line, prediction_line, error_fill, title

    movie = animation.FuncAnimation(fig, update, frames=len(t), interval=100, blit=False)
    movie.save(
        args.output / "burgers_truth_vs_fno.gif",
        writer=animation.PillowWriter(fps=10),
        dpi=110,
    )
    plt.close(fig)

    future = slice(args.initial_step, None)
    report = {
        "truth": {
            "max_mass_drift": float(np.max(np.abs(mass_truth - mass_truth[:, :1]))),
            "energy_change_fraction": float(
                (energy_truth[:, -1].mean() - energy_truth[:, 0].mean())
                / energy_truth[:, 0].mean()
            ),
        },
        "prediction": {
            "mean_abs_mass_error_future": float(
                np.mean(np.abs(mass_prediction[:, future] - mass_truth[:, future]))
            ),
            "final_energy_relative_error": float(
                abs(energy_prediction[:, -1].mean() - energy_truth[:, -1].mean())
                / energy_truth[:, -1].mean()
            ),
            "final_total_variation_relative_error": float(
                abs(variation_prediction[:, -1].mean() - variation_truth[:, -1].mean())
                / variation_truth[:, -1].mean()
            ),
        },
    }
    (args.output / "physics_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
