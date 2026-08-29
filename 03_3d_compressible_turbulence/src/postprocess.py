"""Physics diagnostics and publication-style 3D visualizations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes

from common import load_config, write_json


def derivatives(vx: np.ndarray, vy: np.ndarray, vz: np.ndarray, dx: float):
    ddx = lambda value, axis: (np.roll(value, -1, axis=axis) - np.roll(value, 1, axis=axis)) / (2 * dx)
    divergence = ddx(vx, 0) + ddx(vy, 1) + ddx(vz, 2)
    wx = ddx(vz, 1) - ddx(vy, 2)
    wy = ddx(vx, 2) - ddx(vz, 0)
    wz = ddx(vy, 0) - ddx(vx, 1)
    return divergence, np.sqrt(wx * wx + wy * wy + wz * wz)


def spectrum(vx: np.ndarray, vy: np.ndarray, vz: np.ndarray):
    n = vx.shape[0]
    energy = sum(np.abs(np.fft.fftn(value)) ** 2 for value in (vx, vy, vz)) / (2 * n**6)
    freq = np.fft.fftfreq(n) * n
    radius = np.sqrt(
        freq[:, None, None] ** 2 + freq[None, :, None] ** 2 + freq[None, None, :] ** 2
    )
    shell = np.rint(radius).astype(int)
    max_shell = n // 2
    values = np.bincount(shell.ravel(), weights=energy.ravel(), minlength=max_shell + 1)[: max_shell + 1]
    return np.arange(max_shell + 1), values


def save_slices(result_dir: Path, rho, pressure, speed, extent):
    fields = [(rho, r"Density $\rho$"), (pressure, r"Pressure $p$"), (speed, r"Speed $|v|$")]
    planes = ((0, "x = L/2"), (1, "y = L/2"), (2, "z = L/2"))
    fig, axes = plt.subplots(3, 3, figsize=(13, 11), constrained_layout=True)
    mid = rho.shape[0] // 2
    for row, (field, label) in enumerate(fields):
        vmin, vmax = np.percentile(field, [1, 99])
        for col, (axis, plane) in enumerate(planes):
            image = np.take(field, mid, axis=axis).T
            artist = axes[row, col].imshow(image, origin="lower", extent=extent, cmap="viridis", vmin=vmin, vmax=vmax)
            axes[row, col].set_title(f"{label}, {plane}")
            axes[row, col].set_xlabel("coordinate 1")
            axes[row, col].set_ylabel("coordinate 2")
            fig.colorbar(artist, ax=axes[row, col], shrink=0.78)
    fig.suptitle("Final-time orthogonal slices (shared scale within each row)", fontsize=14)
    fig.savefig(result_dir / "orthogonal_slices.png", dpi=170)
    plt.close(fig)


def add_isosurface(ax, field, level, color, face_limit, title):
    vertices, faces, _, _ = marching_cubes(field, level=level)
    if len(faces) > face_limit:
        stride = int(np.ceil(len(faces) / face_limit))
        faces = faces[::stride]
    mesh = Poly3DCollection(vertices[faces], alpha=0.50, linewidths=0.0)
    mesh.set_facecolor(color)
    ax.add_collection3d(mesh)
    n = field.shape[0] - 1
    ax.set(xlim=(0, n), ylim=(0, n), zlim=(0, n), title=title, xlabel="x", ylabel="y", zlabel="z")
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=24, azim=38)
    return len(vertices), len(faces)


def render_frame(rho, vort, div, time_value, limits):
    mid = rho.shape[0] // 2
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    items = [
        (rho[mid].T, r"Density $\rho$", "viridis", limits["density"]),
        (vort[mid].T, r"Vorticity $|\omega|$", "magma", limits["vorticity"]),
        (div[mid].T, r"Dilatation $\nabla\cdot v$", "coolwarm", limits["divergence"]),
    ]
    for ax, (field, title, cmap, (vmin, vmax)) in zip(axes, items):
        image = ax.imshow(field, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("y index")
        ax.set_ylabel("z index")
        fig.colorbar(image, ax=ax, shrink=0.75)
    fig.suptitle(f"3D compressible turbulence: central x-slice, t={time_value:.3f}")
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    work = args.work_dir.resolve()
    result_dir = work / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    sample = int(cfg["postprocess"]["sample_index"])
    gamma = float(cfg["simulation"]["gamma"])
    with h5py.File(work / "cfd3d_dataset.h5", "r") as handle:
        t = handle["grid/t"][:]
        x = handle["grid/x"][:]
        rho = handle["solution/density"][sample]
        vx = handle["solution/velocity_x"][sample]
        vy = handle["solution/velocity_y"][sample]
        vz = handle["solution/velocity_z"][sample]
        pressure = handle["solution/pressure"][sample]
    dx = float(x[1] - x[0])
    speed = np.sqrt(vx * vx + vy * vy + vz * vz)
    nt = len(t)
    divergence = np.empty_like(rho)
    vorticity = np.empty_like(rho)
    diagnostics = []
    spectra = {}
    chosen = sorted(set((0, nt // 2, nt - 1)))
    for index in range(nt):
        divergence[index], vorticity[index] = derivatives(vx[index], vy[index], vz[index], dx)
        kinetic = 0.5 * rho[index] * speed[index] ** 2
        total_energy = pressure[index] / (gamma - 1.0) + kinetic
        diagnostics.append(
            {
                "time": float(t[index]),
                "mean_density": float(rho[index].mean()),
                "mass": float(rho[index].sum() * dx**3),
                "kinetic_energy": float(kinetic.sum() * dx**3),
                "total_energy": float(total_energy.sum() * dx**3),
                "mean_pressure": float(pressure[index].mean()),
                "rms_divergence": float(np.sqrt(np.mean(divergence[index] ** 2))),
                "rms_vorticity": float(np.sqrt(np.mean(vorticity[index] ** 2))),
            }
        )
        if index in chosen:
            k, ek = spectrum(vx[index], vy[index], vz[index])
            spectra[str(index)] = {"k": k.tolist(), "energy": ek.tolist()}

    save_slices(result_dir, rho[-1], pressure[-1], speed[-1], [x[0], x[-1], x[0], x[-1]])

    fig = plt.figure(figsize=(12, 5), constrained_layout=True)
    axes = [fig.add_subplot(1, 2, index + 1, projection="3d") for index in range(2)]
    rho_level = float(np.percentile(rho[-1], cfg["postprocess"]["density_isosurface_percentile"]))
    vort_level = float(np.percentile(vorticity[-1], cfg["postprocess"]["vorticity_isosurface_percentile"]))
    rho_counts = add_isosurface(axes[0], rho[-1], rho_level, "#2a9d8f", cfg["postprocess"]["isosurface_face_limit"], rf"Density isosurface $\rho={rho_level:.3f}$")
    vort_counts = add_isosurface(axes[1], vorticity[-1], vort_level, "#e76f51", cfg["postprocess"]["isosurface_face_limit"], rf"Vorticity isosurface $|\omega|={vort_level:.2f}$")
    fig.suptitle("Three-dimensional compressible structures and vortex tubes")
    fig.savefig(result_dir / "density_vorticity_isosurfaces.png", dpi=180)
    plt.close(fig)

    values = {key: np.array([row[key] for row in diagnostics]) for key in diagnostics[0] if key != "time"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    axes[0, 0].plot(t, values["mass"] / values["mass"][0] - 1, marker="o")
    axes[0, 0].set(title="Relative mass drift", ylabel=r"$M/M_0-1$")
    axes[0, 1].plot(t, values["kinetic_energy"], marker="o", label="kinetic")
    axes[0, 1].plot(t, values["total_energy"], marker="s", label="total")
    axes[0, 1].set(title="Volume-integrated energy", ylabel="energy")
    axes[0, 1].legend()
    axes[1, 0].plot(t, values["mean_pressure"], marker="o")
    axes[1, 0].set(title="Mean pressure", xlabel="time", ylabel=r"$\langle p\rangle$")
    axes[1, 1].plot(t, values["rms_vorticity"], marker="o", label=r"RMS $|\omega|$")
    axes[1, 1].plot(t, values["rms_divergence"], marker="s", label=r"RMS $\nabla\cdot v$")
    axes[1, 1].set(title="Rotational and compressive activity", xlabel="time", ylabel="RMS")
    axes[1, 1].legend()
    for ax in axes.flat:
        ax.grid(alpha=0.25)
    fig.savefig(result_dir / "conservation_and_flow_diagnostics.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)
    for index in chosen:
        values_k = spectra[str(index)]
        k_values = np.asarray(values_k["k"])[1:]
        e_values = np.asarray(values_k["energy"])[1:]
        ax.loglog(k_values, np.maximum(e_values, 1e-20), marker="o", ms=3, label=f"t={t[index]:.3f}")
    ref_k = np.arange(2, max(4, rho.shape[-1] // 3))
    scale = max(np.asarray(spectra[str(chosen[0])]["energy"])[2], 1e-20)
    ax.loglog(ref_k, scale * (ref_k / 2) ** (-5 / 3), "k--", label=r"$k^{-5/3}$ guide")
    ax.set(xlabel="integer wavenumber k", ylabel=r"shell kinetic energy $E(k)$", title="Isotropic kinetic-energy spectrum")
    ax.grid(which="both", alpha=0.25)
    ax.legend()
    fig.savefig(result_dir / "kinetic_energy_spectrum.png", dpi=180)
    plt.close(fig)

    perf_path = work / "benchmark/benchmark_metrics.json"
    if perf_path.exists():
        perf = json.loads(perf_path.read_text(encoding="utf-8"))["results"]
        gpu = np.array(sorted(int(key) for key in perf))
        elapsed = np.array([perf[str(value)]["elapsed_seconds_median"] for value in gpu])
        speedup_values = elapsed[0] / elapsed
        efficiency = speedup_values / gpu
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
        axes[0].plot(gpu, elapsed, "o-")
        axes[0].set(title="Wall time", xlabel="GPUs", ylabel="seconds")
        axes[1].plot(gpu, speedup_values, "o-", label="measured")
        axes[1].plot(gpu, gpu, "k--", label="ideal")
        axes[1].set(title="Strong-scaling speedup", xlabel="GPUs", ylabel="speedup")
        axes[1].legend()
        axes[2].plot(gpu, efficiency, "o-")
        axes[2].axhline(1, color="k", ls="--")
        axes[2].set(title="Parallel efficiency", xlabel="GPUs", ylabel=r"$S_p/p$")
        for ax in axes:
            ax.set_xticks(gpu)
            ax.grid(alpha=0.25)
        fig.savefig(result_dir / "multi_gpu_scaling.png", dpi=180)
        plt.close(fig)

    frame_indices = np.linspace(0, nt - 1, min(nt, int(cfg["postprocess"]["gif_frames"])), dtype=int)
    div_limit = float(np.percentile(np.abs(divergence), 99))
    animation_limits = {
        "density": tuple(float(value) for value in np.percentile(rho, [1, 99])),
        "vorticity": (0.0, float(np.percentile(vorticity, 99))),
        "divergence": (-div_limit, div_limit),
    }
    frames = [render_frame(rho[index], vorticity[index], divergence[index], t[index], animation_limits) for index in frame_indices]
    imageio.mimsave(result_dir / "turbulence_evolution.gif", frames, duration=0.65, loop=0)

    mass_drift = abs(diagnostics[-1]["mass"] / diagnostics[0]["mass"] - 1)
    energy_drift = abs(diagnostics[-1]["total_energy"] / diagnostics[0]["total_energy"] - 1)
    metrics = {
        "sample_index": sample,
        "time_steps": nt,
        "grid_spacing": dx,
        "mass_relative_drift": mass_drift,
        "total_energy_relative_drift": energy_drift,
        "initial": diagnostics[0],
        "final": diagnostics[-1],
        "density_range_final": [float(rho[-1].min()), float(rho[-1].max())],
        "pressure_range_final": [float(pressure[-1].min()), float(pressure[-1].max())],
        "isosurfaces": {
            "density_level": rho_level,
            "density_vertices_faces": rho_counts,
            "vorticity_level": vort_level,
            "vorticity_vertices_faces": vort_counts,
        },
        "animation_color_limits": animation_limits,
        "diagnostics": diagnostics,
        "spectra": spectra,
    }
    write_json(result_dir / "physical_metrics.json", metrics)
    print(f"mass drift={mass_drift:.3e}, energy drift={energy_drift:.3e}")


if __name__ == "__main__":
    main()
