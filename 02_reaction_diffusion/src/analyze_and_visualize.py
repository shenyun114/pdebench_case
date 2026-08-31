#!/usr/bin/env python3
"""Explain reaction, diffusion and emergent patterns with quantitative visuals."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from pdebench_operators import apply_laplacian, neumann_laplacian  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))
from case_utils.animation import (  # noqa: E402
    axes_pixel_boxes,
    capture_rgb,
    freeze_figure_layout,
    save_fixed_palette_gif,
)


def gradients(field: np.ndarray, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    # Stored field layout is (time, y, x).
    gx = np.gradient(field, dx, axis=2)
    gy = np.gradient(field, dy, axis=1)
    return gx, gy


def isotropic_spectrum(field: np.ndarray, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    centered = field - field.mean()
    power = np.abs(np.fft.fft2(centered)) ** 2
    kx = np.fft.fftfreq(field.shape[0], d=dx)
    ky = np.fft.fftfreq(field.shape[1], d=dy)
    kxx, kyy = np.meshgrid(kx, ky, indexing="ij")
    radius = np.sqrt(kxx * kxx + kyy * kyy)
    bins = np.linspace(0, radius.max(), min(field.shape) // 2 + 1)
    indices = np.digitize(radius.ravel(), bins) - 1
    sums = np.bincount(indices, weights=power.ravel(), minlength=len(bins))[: len(bins) - 1]
    counts = np.bincount(indices, minlength=len(bins))[: len(bins) - 1]
    spectrum = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    centers = 0.5 * (bins[:-1] + bins[1:])
    return centers, spectrum


def characteristic_length(field: np.ndarray, dx: float, dy: float) -> float:
    frequency, spectrum = isotropic_spectrum(field, dx, dy)
    valid = frequency > 0
    mean_frequency = np.sum(frequency[valid] * spectrum[valid]) / np.sum(spectrum[valid])
    return float(1.0 / mean_frequency)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.data) as handle:
        u, v = handle["u"][:], handle["v"][:]
        x, y, t = handle["x"][:], handle["y"][:], handle["t"][:]
        du, dv, reaction_k = float(handle.attrs["Du"]), float(handle.attrs["Dv"]), float(handle.attrs["k"])

    dx, dy = float(np.mean(np.diff(x))), float(np.mean(np.diff(y)))
    gx_u, gy_u = gradients(u, dx, dy)
    gx_v, gy_v = gradients(v, dx, dy)
    lap = neumann_laplacian(len(x), len(y), dx, dy)
    lap_u = apply_laplacian(u.astype(np.float64), lap)
    lap_v = apply_laplacian(v.astype(np.float64), lap)
    reaction_u = u - u**3 - reaction_k - v
    reaction_v = u - v
    diffusion_u = du * lap_u
    diffusion_v = dv * lap_v
    gradient_energy_u = np.mean(gx_u**2 + gy_u**2, axis=(1, 2))
    gradient_energy_v = np.mean(gx_v**2 + gy_v**2, axis=(1, 2))
    correlation = np.asarray([
        np.corrcoef(frame_u.ravel(), frame_v.ravel())[0, 1]
        for frame_u, frame_v in zip(u, v)
    ])
    mean_u, mean_v = u.mean(axis=(1, 2)), v.mean(axis=(1, 2))
    std_u, std_v = u.std(axis=(1, 2)), v.std(axis=(1, 2))
    rms_reaction_u = np.sqrt(np.mean(reaction_u**2, axis=(1, 2)))
    rms_reaction_v = np.sqrt(np.mean(reaction_v**2, axis=(1, 2)))
    rms_diffusion_u = np.sqrt(np.mean(diffusion_u**2, axis=(1, 2)))
    rms_diffusion_v = np.sqrt(np.mean(diffusion_v**2, axis=(1, 2)))
    length_u = np.asarray([characteristic_length(frame, dx, dy) for frame in u])
    length_v = np.asarray([characteristic_length(frame, dx, dy) for frame in v])

    metrics = {
        "data": {
            "shape": list(u.shape),
            "dx": dx,
            "dy": dy,
            "time_range": [float(t[0]), float(t[-1])],
            "u_range": [float(u.min()), float(u.max())],
            "v_range": [float(v.min()), float(v.max())],
        },
        "coupling": {
            "initial_correlation": float(correlation[0]),
            "final_correlation": float(correlation[-1]),
            "final_mean_u": float(mean_u[-1]),
            "final_mean_v": float(mean_v[-1]),
            "final_std_u": float(std_u[-1]),
            "final_std_v": float(std_v[-1]),
        },
        "pattern": {
            "u_gradient_energy_initial": float(gradient_energy_u[0]),
            "u_gradient_energy_final": float(gradient_energy_u[-1]),
            "v_gradient_energy_initial": float(gradient_energy_v[0]),
            "v_gradient_energy_final": float(gradient_energy_v[-1]),
            "u_characteristic_length_initial": float(length_u[0]),
            "u_characteristic_length_final": float(length_u[-1]),
            "v_characteristic_length_initial": float(length_v[0]),
            "v_characteristic_length_final": float(length_v[-1]),
        },
        "mechanisms": {
            "laplacian_definition": "exact PDEBench sparse five-point operator with homogeneous Neumann boundary",
            "initial_rms_reaction_u": float(rms_reaction_u[0]),
            "final_rms_reaction_u": float(rms_reaction_u[-1]),
            "initial_rms_diffusion_u": float(rms_diffusion_u[0]),
            "final_rms_diffusion_u": float(rms_diffusion_u[-1]),
            "initial_rms_reaction_v": float(rms_reaction_v[0]),
            "final_rms_reaction_v": float(rms_reaction_v[-1]),
            "initial_rms_diffusion_v": float(rms_diffusion_v[0]),
            "final_rms_diffusion_v": float(rms_diffusion_v[-1]),
        },
    }
    (args.output / "physical_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    extent = [float(x[0]), float(x[-1]), float(y[0]), float(y[-1])]
    indices = [0, 5, 20, 50, 100]
    u_limit = float(np.percentile(np.abs(u[5:]), 99.7))
    v_limit = float(np.percentile(np.abs(v[5:]), 99.7))
    fig, axes = plt.subplots(2, 5, figsize=(16.8, 6.5), constrained_layout=True)
    for column, index in enumerate(indices):
        image_u = axes[0, column].imshow(u[index], origin="lower", extent=extent, cmap="RdBu_r", vmin=-u_limit, vmax=u_limit)
        image_v = axes[1, column].imshow(v[index], origin="lower", extent=extent, cmap="PiYG", vmin=-v_limit, vmax=v_limit)
        axes[0, column].set_title(f"t={t[index]:.2f}")
        axes[1, column].set_xlabel("x")
        for row in range(2):
            axes[row, column].set_aspect("equal")
            if column == 0:
                axes[row, column].set_ylabel("y")
    fig.colorbar(image_u, ax=axes[0], shrink=0.8, label="Activator u")
    fig.colorbar(image_v, ax=axes[1], shrink=0.8, label="Inhibitor v")
    fig.suptitle("From random fluctuations to coupled reaction-diffusion domains", fontsize=15)
    fig.savefig(args.output / "field_snapshots.png", dpi=180)
    plt.close(fig)

    final = -1
    reaction_limit = float(np.percentile(np.abs(reaction_u[final]), 99.5))
    mismatch_limit = float(np.percentile(np.abs(u[final] - v[final]), 99.5))
    fig, axes = plt.subplots(1, 4, figsize=(15.8, 3.8), constrained_layout=True)
    panels = [
        (u[final], "RdBu_r", -u_limit, u_limit, "Activator u"),
        (v[final], "PiYG", -v_limit, v_limit, "Inhibitor v"),
        (u[final] - v[final], "PuOr", -mismatch_limit, mismatch_limit, "Coupling mismatch u-v"),
        (reaction_u[final], "coolwarm", -reaction_limit, reaction_limit, "Local reaction Ru"),
    ]
    for axis, (field, cmap, low, high, title) in zip(axes, panels):
        image = axis.imshow(field, origin="lower", extent=extent, cmap=cmap, vmin=low, vmax=high)
        axis.set(title=title, xlabel="x", ylabel="y")
        axis.set_aspect("equal")
        fig.colorbar(image, ax=axis, shrink=0.76)
    fig.suptitle("Coupled state and local reaction at t=5", fontsize=15)
    fig.savefig(args.output / "coupled_state.png", dpi=180)
    plt.close(fig)

    phase_indices = [0, 5, 20, 50, 100]
    phase_colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(phase_indices)))
    rng = np.random.default_rng(2026)
    sample_points = rng.choice(u.shape[1] * u.shape[2], size=500, replace=False)
    u_curve = np.linspace(-1.1, 1.1, 500)
    fig, axis = plt.subplots(figsize=(7.4, 6.2), constrained_layout=True)
    for color, index in zip(phase_colors, phase_indices):
        axis.scatter(u[index].ravel()[sample_points], v[index].ravel()[sample_points], s=7, alpha=0.24, color=color, label=f"t={t[index]:.2f}")
    axis.plot(u_curve, u_curve - u_curve**3 - reaction_k, color="black", lw=2, label="du/dt nullcline")
    axis.plot(u_curve, u_curve, color="#d95f02", lw=2, ls="--", label="dv/dt nullcline")
    axis.plot(mean_u, mean_v, color="#7570b3", lw=2.2, label="spatial-mean trajectory")
    axis.set(xlabel="u", ylabel="v", title="Local phase portrait and reaction nullclines")
    axis.set_xlim(-1.1, 1.1)
    axis.set_ylim(-1.1, 1.1)
    axis.grid(alpha=0.22)
    axis.legend(ncol=2, fontsize=8)
    fig.savefig(args.output / "phase_portrait.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.0), constrained_layout=True)
    axes[0, 0].semilogy(t, rms_reaction_u, label="reaction |Ru|")
    axes[0, 0].semilogy(t, rms_diffusion_u, label="diffusion |Du lap u|")
    axes[0, 0].set(title="Activator mechanism balance", xlabel="t", ylabel="RMS rate")
    axes[0, 1].semilogy(t, rms_reaction_v, label="reaction |Rv|")
    axes[0, 1].semilogy(t, rms_diffusion_v, label="diffusion |Dv lap v|")
    axes[0, 1].set(title="Inhibitor mechanism balance", xlabel="t", ylabel="RMS rate")
    axes[1, 0].plot(t, mean_u, label="mean u")
    axes[1, 0].plot(t, mean_v, label="mean v")
    axes[1, 0].set(title="Spatial means (not conserved)", xlabel="t", ylabel="Mean value")
    axes[1, 1].plot(t, std_u, label="std u")
    axes[1, 1].plot(t, std_v, label="std v")
    axes[1, 1].set(title="Pattern contrast", xlabel="t", ylabel="Spatial standard deviation")
    for axis in axes.flat:
        axis.grid(alpha=0.22)
        axis.legend()
    fig.suptitle("Reaction, diffusion and amplitude evolution", fontsize=15)
    fig.savefig(args.output / "mechanism_balance.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.0), constrained_layout=True)
    axes[0, 0].semilogy(t, gradient_energy_u, label="u")
    axes[0, 0].semilogy(t, gradient_energy_v, label="v")
    axes[0, 0].set(title="Gradient energy: small-scale roughness", xlabel="t", ylabel="mean |grad field|²")
    axes[0, 1].plot(t, correlation, color="#1b9e77", label="corr(u,v)")
    axes[0, 1].set(title="u-v spatial correlation", xlabel="t", ylabel="Pearson correlation", ylim=(-0.05, 1.0))
    axes[1, 0].plot(t, length_u, label="u")
    axes[1, 0].plot(t, length_v, label="v")
    axes[1, 0].set(title="Spectral characteristic length", xlabel="t", ylabel="Length scale")
    axes[1, 1].hist(u[-1].ravel(), bins=60, density=True, alpha=0.6, label="u")
    axes[1, 1].hist(v[-1].ravel(), bins=60, density=True, alpha=0.6, label="v")
    axes[1, 1].set(title="Final field-value distribution", xlabel="Value", ylabel="Density")
    for axis in axes.flat:
        axis.grid(alpha=0.22)
        axis.legend()
    fig.suptitle("Pattern formation diagnostics", fontsize=15)
    fig.savefig(args.output / "pattern_diagnostics.png", dpi=180)
    plt.close(fig)

    spectrum_indices = [0, 5, 20, 50, 100]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5), constrained_layout=True)
    for index, color in zip(spectrum_indices, phase_colors):
        frequency_u, spectrum_u = isotropic_spectrum(u[index], dx, dy)
        frequency_v, spectrum_v = isotropic_spectrum(v[index], dx, dy)
        axes[0].loglog(frequency_u[1:], spectrum_u[1:], color=color, label=f"t={t[index]:.2f}")
        axes[1].loglog(frequency_v[1:], spectrum_v[1:], color=color, label=f"t={t[index]:.2f}")
    axes[0].set(title="Activator isotropic spectrum", xlabel="Spatial frequency", ylabel="Power")
    axes[1].set(title="Inhibitor isotropic spectrum", xlabel="Spatial frequency", ylabel="Power")
    for axis in axes:
        axis.grid(alpha=0.22, which="both")
        axis.legend(fontsize=8)
    fig.savefig(args.output / "spatial_spectrum.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.4), constrained_layout=True)
    image_u = axes[0].imshow(u[0], origin="lower", extent=extent, cmap="RdBu_r", vmin=-u_limit, vmax=u_limit)
    image_v = axes[1].imshow(v[0], origin="lower", extent=extent, cmap="PiYG", vmin=-v_limit, vmax=v_limit)
    image_r = axes[2].imshow(reaction_u[0], origin="lower", extent=extent, cmap="coolwarm", vmin=-reaction_limit, vmax=reaction_limit)
    for axis, title in zip(axes, ["Activator u", "Inhibitor v", "Reaction source Ru"]):
        axis.set(title=title, xlabel="x", ylabel="y")
        axis.set_aspect("equal")
    bar_u = fig.colorbar(image_u, ax=axes[0], shrink=0.78)
    bar_v = fig.colorbar(image_v, ax=axes[1], shrink=0.78)
    bar_r = fig.colorbar(image_r, ax=axes[2], shrink=0.78)
    title = fig.suptitle("t=0.00 | corr(u,v)=0.000", fontsize=14)

    def update(index: int):
        image_u.set_data(u[index])
        image_v.set_data(v[index])
        image_r.set_data(reaction_u[index])
        title.set_text(f"t={t[index]:.2f} | corr(u,v)={correlation[index]:.3f}")
        return image_u, image_v, image_r, title

    animation_indices = list(range(0, len(t), 4))
    freeze_figure_layout(fig, dpi=90)
    colorbar_boxes = axes_pixel_boxes(fig, [bar_u.ax, bar_v.ax, bar_r.ax])
    frames = []
    for index in animation_indices:
        update(index)
        frames.append(capture_rgb(fig))
    save_fixed_palette_gif(
        frames,
        args.output / "reaction_diffusion_evolution.gif",
        duration_ms=100,
        static_boxes=colorbar_boxes,
    )
    plt.close(fig)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
