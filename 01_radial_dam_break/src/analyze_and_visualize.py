#!/usr/bin/env python3
"""Compute shallow-water diagnostics and create static and animated visuals."""

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
from matplotlib.colors import Normalize  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))
from case_utils.animation import (  # noqa: E402
    axes_pixel_boxes,
    capture_rgb,
    freeze_figure_layout,
    save_fixed_palette_gif,
)


def radial_average(
    field: np.ndarray, radius: np.ndarray, edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.digitize(radius.ravel(), edges) - 1
    valid = (indices >= 0) & (indices < len(edges) - 1)
    sums = np.bincount(
        indices[valid], weights=field.ravel()[valid], minlength=len(edges) - 1
    )
    counts = np.bincount(indices[valid], minlength=len(edges) - 1)
    mean = np.divide(
        sums,
        counts,
        out=np.full_like(sums, np.nan, dtype=float),
        where=counts > 0,
    )
    centers = 0.5 * (edges[:-1] + edges[1:])
    finite = np.isfinite(mean)
    mean = np.interp(centers, centers[finite], mean[finite])
    return centers, mean


def style_axis(axis: plt.Axes) -> None:
    axis.grid(alpha=0.22, linewidth=0.7)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.data) as handle:
        h = handle["h"][:]
        u = handle["u"][:]
        v = handle["v"][:]
        hu = handle["hu"][:]
        hv = handle["hv"][:]
        x = handle["x"][:]
        y = handle["y"][:]
        t = handle["t"][:]
        gravity = float(handle.attrs["gravity"])
        dam_radius = float(handle.attrs["dam_radius"])

    dx, dy = float(np.mean(np.diff(x))), float(np.mean(np.diff(y)))
    cell_area = dx * dy
    speed = np.sqrt(u * u + v * v)
    froude = speed / np.sqrt(gravity * h)
    kinetic_density = 0.5 * h * speed * speed
    potential_density = 0.5 * gravity * h * h
    total_energy_density = kinetic_density + potential_density
    # Accumulate conserved quantities in float64 even though fields are stored compactly.
    mass = h.sum(axis=(1, 2), dtype=np.float64) * cell_area
    momentum_x = hu.sum(axis=(1, 2), dtype=np.float64) * cell_area
    momentum_y = hv.sum(axis=(1, 2), dtype=np.float64) * cell_area
    energy = total_energy_density.sum(axis=(1, 2), dtype=np.float64) * cell_area
    relative_mass = (mass - mass[0]) / mass[0]
    relative_energy = (energy - energy[0]) / energy[0]
    rotation_error = np.sum(np.abs(h - np.rot90(h, axes=(1, 2))), axis=(1, 2)) / np.sum(
        np.abs(h), axis=(1, 2)
    )

    xx, yy = np.meshgrid(x, y, indexing="ij")
    radius = np.sqrt(xx * xx + yy * yy)
    radial_velocity = np.divide(
        u * xx[None] + v * yy[None],
        radius[None],
        out=np.zeros_like(u),
        where=radius[None] > 0,
    )
    radial_edges = np.linspace(0.0, min(abs(x).max(), abs(y).max()), 100)
    radial_centers = 0.5 * (radial_edges[:-1] + radial_edges[1:])
    radial_h = []
    radial_ur = []
    wavefront_radius = []
    for frame_h, frame_ur in zip(h, radial_velocity):
        _, h_mean = radial_average(frame_h, radius, radial_edges)
        _, ur_mean = radial_average(frame_ur, radius, radial_edges)
        radial_h.append(h_mean)
        radial_ur.append(ur_mean)
        gradient = np.abs(np.gradient(h_mean, radial_centers))
        gradient[radial_centers < 0.15] = -np.inf
        wavefront_radius.append(float(radial_centers[np.argmax(gradient)]))
    radial_h = np.asarray(radial_h)
    radial_ur = np.asarray(radial_ur)
    wavefront_radius = np.asarray(wavefront_radius)

    metrics = {
        "data": {
            "shape": list(h.shape),
            "dx": dx,
            "dy": dy,
            "time_range": [float(t[0]), float(t[-1])],
            "water_depth_range": [float(h.min()), float(h.max())],
        },
        "conservation": {
            "initial_mass": float(mass[0]),
            "final_mass": float(mass[-1]),
            "max_relative_mass_drift": float(np.abs(relative_mass).max()),
            "initial_energy": float(energy[0]),
            "final_energy": float(energy[-1]),
            "relative_energy_change": float(relative_energy[-1]),
            "max_abs_x_momentum": float(np.abs(momentum_x).max()),
            "max_abs_y_momentum": float(np.abs(momentum_y).max()),
        },
        "flow": {
            "max_speed": float(speed.max()),
            "max_froude": float(froude.max()),
            "max_kinetic_energy_density": float(kinetic_density.max()),
            "initial_wavefront_radius": float(wavefront_radius[0]),
            "final_wavefront_radius": float(wavefront_radius[-1]),
        },
        "symmetry": {
            "max_rotation_relative_l1": float(rotation_error.max()),
            "final_rotation_relative_l1": float(rotation_error[-1]),
        },
        "interpretation": {
            "mass_is_conserved": bool(np.abs(relative_mass).max() < 1e-5),
            "flow_remains_subcritical": bool(froude.max() < 1.0),
            "numerical_energy_is_nonincreasing": bool(np.max(np.diff(energy)) < 1e-5),
        },
    }
    (args.output / "physical_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    extent = [float(x[0]), float(x[-1]), float(y[0]), float(y[-1])]
    snapshot_indices = [0, 10, 25, 50, 100]
    depth_norm = Normalize(vmin=float(h.min()), vmax=float(h.max()))

    fig, axes = plt.subplots(1, 5, figsize=(17.2, 3.6), constrained_layout=True)
    for axis, index in zip(axes, snapshot_indices):
        image = axis.imshow(
            h[index].T,
            origin="lower",
            extent=extent,
            cmap="Blues",
            norm=depth_norm,
        )
        axis.contour(x, y, h[index].T, levels=[1.05, 1.2, 1.4], colors="white", linewidths=0.6)
        axis.set(title=f"t = {t[index]:.2f}", xlabel="x")
        axis.set_aspect("equal")
    axes[0].set_ylabel("y")
    fig.colorbar(image, ax=axes, shrink=0.82, label="Water depth h")
    fig.suptitle("Radial dam-break: outward bore and inward rarefaction", fontsize=15)
    fig.savefig(args.output / "water_depth_snapshots.png", dpi=180)
    plt.close(fig)

    surface_indices = [0, 35, 100]
    x_surface, y_surface = np.meshgrid(x[::4], y[::4], indexing="ij")
    fig = plt.figure(figsize=(15.2, 4.4), constrained_layout=True)
    for plot_index, frame_index in enumerate(surface_indices, 1):
        axis = fig.add_subplot(1, 3, plot_index, projection="3d")
        surface = axis.plot_surface(
            x_surface,
            y_surface,
            h[frame_index, ::4, ::4],
            cmap="Blues",
            norm=depth_norm,
            linewidth=0,
            antialiased=True,
        )
        axis.set(xlabel="x", ylabel="y", zlabel="h", title=f"t = {t[frame_index]:.2f}")
        axis.set_zlim(float(h.min()) - 0.03, float(h.max()) + 0.03)
        axis.view_init(elev=30, azim=-55)
    fig.colorbar(surface, ax=fig.axes, shrink=0.65, pad=0.06, label="Water depth h")
    fig.suptitle("Free-surface evolution", fontsize=15)
    fig.savefig(args.output / "surface_evolution_3d.png", dpi=180)
    plt.close(fig)

    flow_indices = [20, 50, 100]
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 8.2), constrained_layout=True)
    speed_limit = float(speed.max())
    froude_limit = float(froude.max())
    skip = 8
    for column, frame_index in enumerate(flow_indices):
        top = axes[0, column]
        speed_image = top.imshow(
            speed[frame_index].T,
            origin="lower",
            extent=extent,
            cmap="magma",
            vmin=0,
            vmax=speed_limit,
        )
        top.quiver(
            xx[::skip, ::skip].T,
            yy[::skip, ::skip].T,
            u[frame_index, ::skip, ::skip].T,
            v[frame_index, ::skip, ::skip].T,
            color="white",
            scale=8,
            width=0.004,
        )
        top.set(title=f"Speed and direction, t={t[frame_index]:.2f}")
        bottom = axes[1, column]
        froude_image = bottom.imshow(
            froude[frame_index].T,
            origin="lower",
            extent=extent,
            cmap="viridis",
            vmin=0,
            vmax=froude_limit,
        )
        bottom.set(title=f"Froude number, t={t[frame_index]:.2f}", xlabel="x")
        for axis in (top, bottom):
            axis.set_aspect("equal")
            axis.set_ylabel("y" if column == 0 else "")
    fig.colorbar(speed_image, ax=axes[0], shrink=0.78, label="Speed |u|")
    fig.colorbar(froude_image, ax=axes[1], shrink=0.78, label="Fr = |u|/sqrt(gh)")
    fig.savefig(args.output / "velocity_and_froude.png", dpi=180)
    plt.close(fig)

    profile_indices = [0, 10, 25, 50, 75, 100]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.5), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(profile_indices)))
    for color, frame_index in zip(colors, profile_indices):
        label = f"t={t[frame_index]:.2f}"
        axes[0].plot(radial_centers, radial_h[frame_index], color=color, label=label)
        axes[1].plot(radial_centers, radial_ur[frame_index], color=color, label=label)
    axes[0].axvline(dam_radius, color="gray", ls="--", lw=1, label="initial dam")
    axes[0].set(xlabel="Radius r", ylabel="Azimuthal mean depth", title="Radial water-depth profiles")
    axes[1].set(xlabel="Radius r", ylabel="Azimuthal mean radial velocity", title="Radial velocity profiles")
    for axis in axes:
        style_axis(axis)
    axes[1].legend(ncol=2, fontsize=8)
    fig.savefig(args.output / "radial_profiles.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.8), constrained_layout=True)
    axes[0, 0].plot(t, relative_mass, lw=2)
    axes[0, 0].set(title="Mass conservation", xlabel="t", ylabel="(M-M0)/M0")
    axes[0, 1].plot(t, relative_energy, color="#d95f02", lw=2)
    axes[0, 1].set(title="Numerical total-energy change", xlabel="t", ylabel="(E-E0)/E0")
    axes[1, 0].plot(t, momentum_x, label="Px")
    axes[1, 0].plot(t, momentum_y, ls="--", label="Py")
    axes[1, 0].set(title="Domain-integrated momentum", xlabel="t", ylabel="Momentum")
    axes[1, 0].legend()
    speed_line = axes[1, 1].plot(t, speed.max(axis=(1, 2)), label="max speed", color="#1b9e77")
    froude_axis = axes[1, 1].twinx()
    froude_line = froude_axis.plot(t, froude.max(axis=(1, 2)), label="max Froude", color="#7570b3")
    axes[1, 1].set(title="Peak flow intensity", xlabel="t", ylabel="Max speed")
    froude_axis.set_ylabel("Max Froude")
    lines = speed_line + froude_line
    axes[1, 1].legend(lines, [line.get_label() for line in lines], loc="best")
    for axis in axes.flat:
        style_axis(axis)
    fig.suptitle("Physical diagnostics", fontsize=15)
    fig.savefig(args.output / "conservation_diagnostics.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), constrained_layout=True)
    depth_image = axes[0].imshow(
        h[0].T, origin="lower", extent=extent, cmap="Blues", norm=depth_norm
    )
    speed_image = axes[1].imshow(
        speed[0].T,
        origin="lower",
        extent=extent,
        cmap="magma",
        vmin=0,
        vmax=speed_limit,
    )
    quiver = axes[1].quiver(
        xx[::skip, ::skip].T,
        yy[::skip, ::skip].T,
        u[0, ::skip, ::skip].T,
        v[0, ::skip, ::skip].T,
        color="white",
        scale=8,
        width=0.004,
    )
    axes[0].set_title("Water depth h")
    axes[1].set_title("Speed and velocity direction")
    for axis in axes:
        axis.set(xlabel="x", ylabel="y")
        axis.set_aspect("equal")
    depth_bar = fig.colorbar(depth_image, ax=axes[0], shrink=0.82, label="h")
    speed_bar = fig.colorbar(speed_image, ax=axes[1], shrink=0.82, label="|u|")
    time_text = fig.suptitle("t = 0.00 | relative mass drift = 0.00e+00", fontsize=14)

    animation_indices = list(range(0, len(t), 2))

    def update(frame_number: int):
        index = animation_indices[frame_number]
        depth_image.set_data(h[index].T)
        speed_image.set_data(speed[index].T)
        quiver.set_UVC(
            u[index, ::skip, ::skip].T,
            v[index, ::skip, ::skip].T,
        )
        time_text.set_text(
            f"t = {t[index]:.2f} | relative mass drift = {relative_mass[index]:+.2e}"
        )
        return depth_image, speed_image, quiver, time_text

    freeze_figure_layout(fig, dpi=105)
    colorbar_boxes = axes_pixel_boxes(fig, [depth_bar.ax, speed_bar.ax])
    frames = []
    for frame_number in range(len(animation_indices)):
        update(frame_number)
        frames.append(capture_rgb(fig))
    save_fixed_palette_gif(
        frames,
        args.output / "shallow_water_evolution.gif",
        duration_ms=83,
        static_boxes=colorbar_boxes,
    )
    plt.close(fig)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
