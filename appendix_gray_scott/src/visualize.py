from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a Gray-Scott HDF5 dataset.")
    parser.add_argument("--input", type=Path, default=Path("../results/grayscott_dataset.h5"))
    parser.add_argument("--out-dir", type=Path, default=Path("../results"))
    parser.add_argument("--sample", default="0000")
    return parser.parse_args()


def laplacian(a: np.ndarray) -> np.ndarray:
    return (
        np.roll(a, 1, axis=0)
        + np.roll(a, -1, axis=0)
        + np.roll(a, 1, axis=1)
        + np.roll(a, -1, axis=1)
        - 4.0 * a
    )


def gradient_energy(a: np.ndarray) -> np.ndarray:
    grad_y = 0.5 * (np.roll(a, -1, axis=-2) - np.roll(a, 1, axis=-2))
    grad_x = 0.5 * (np.roll(a, -1, axis=-1) - np.roll(a, 1, axis=-1))
    return grad_x * grad_x + grad_y * grad_y


def interface_length(field: np.ndarray) -> float:
    threshold = float(np.quantile(field, 0.75))
    mask = field > threshold
    vertical = np.logical_xor(mask, np.roll(mask, 1, axis=0)).sum()
    horizontal = np.logical_xor(mask, np.roll(mask, 1, axis=1)).sum()
    return float(vertical + horizontal) / field.size


def radial_spectrum(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = field - field.mean()
    power = np.abs(np.fft.fftshift(np.fft.fft2(centered))) ** 2
    yy, xx = np.indices(field.shape)
    cy = (field.shape[0] - 1) / 2.0
    cx = (field.shape[1] - 1) / 2.0
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.int32)
    radial_sum = np.bincount(radius.ravel(), weights=power.ravel())
    radial_count = np.bincount(radius.ravel())
    spectrum = radial_sum / np.maximum(radial_count, 1)
    k = np.arange(spectrum.size)
    return k[1:], spectrum[1:]


def plot_fields(data: np.ndarray, out_dir: Path, sample: str) -> None:
    frame_ids = [0, data.shape[0] // 2, data.shape[0] - 1]
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.2), constrained_layout=True)
    for col, frame in enumerate(frame_ids):
        for row, field_id in enumerate([0, 1]):
            ax = axes[row, col]
            im = ax.imshow(data[frame, :, :, field_id], cmap="viridis", origin="lower")
            ax.set_title(f"{'u' if field_id == 0 else 'v'} frame {frame}")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(out_dir / f"sample_{sample}_fields.png", dpi=180)
    plt.close(fig)


def compute_metrics(data: np.ndarray, feed: float, kill: float) -> dict[str, np.ndarray]:
    u = data[:, :, :, 0]
    v = data[:, :, :, 1]
    reaction = u * v * v
    grad = gradient_energy(u) + gradient_energy(v)
    dvdt_reaction = reaction - (feed + kill) * v
    metrics = {
        "mean_u": u.mean(axis=(1, 2)),
        "mean_v": v.mean(axis=(1, 2)),
        "total_concentration": (u + v).mean(axis=(1, 2)),
        "reaction_rate": reaction.mean(axis=(1, 2)),
        "feed_supply": (feed * (1.0 - u)).mean(axis=(1, 2)),
        "v_removal": ((feed + kill) * v).mean(axis=(1, 2)),
        "gradient_energy": grad.mean(axis=(1, 2)),
        "dvdt_reaction_rms": np.sqrt((dvdt_reaction * dvdt_reaction).mean(axis=(1, 2))),
        "interface_length": np.array([interface_length(frame) for frame in v]),
    }
    return metrics


def plot_statistics(data: np.ndarray, time: np.ndarray, out_dir: Path, sample: str, feed: float, kill: float) -> dict[str, np.ndarray]:
    metrics = compute_metrics(data, feed, kill)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    axes[0, 0].plot(time, metrics["mean_u"], label="mean u")
    axes[0, 0].plot(time, metrics["mean_v"], label="mean v")
    axes[0, 0].plot(time, metrics["total_concentration"], label="mean(u+v)")
    axes[0, 0].set_title("Domain inventory")
    axes[0, 0].set_xlabel("time")
    axes[0, 0].legend()

    axes[0, 1].plot(time, metrics["reaction_rate"], label="reaction u*v^2")
    axes[0, 1].plot(time, metrics["feed_supply"], label="feed F*(1-u)")
    axes[0, 1].plot(time, metrics["v_removal"], label="removal (F+k)*v")
    axes[0, 1].set_title("Reaction-source balance")
    axes[0, 1].set_xlabel("time")
    axes[0, 1].legend()

    axes[1, 0].plot(time, metrics["gradient_energy"], color="#2a9d8f", label="gradient energy")
    axes[1, 0].plot(time, metrics["interface_length"], color="#e76f51", label="interface length")
    axes[1, 0].set_title("Pattern sharpness")
    axes[1, 0].set_xlabel("time")
    axes[1, 0].legend()

    axes[1, 1].plot(time, metrics["dvdt_reaction_rms"], color="#457b9d", label="reaction RMS")
    axes[1, 1].set_title("Local reaction activity")
    axes[1, 1].set_xlabel("time")
    axes[1, 1].legend()
    fig.savefig(out_dir / f"sample_{sample}_physics_diagnostics.png", dpi=180)
    fig.savefig(out_dir / f"sample_{sample}_statistics.png", dpi=180)
    plt.close(fig)

    with (out_dir / f"sample_{sample}_statistics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "time", *metrics.keys()])
        for i in range(data.shape[0]):
            writer.writerow([i, time[i], *[values[i] for values in metrics.values()]])
    return metrics


def plot_derived_fields(data: np.ndarray, out_dir: Path, sample: str, feed: float, kill: float) -> None:
    final = data[-1]
    u = final[:, :, 0]
    v = final[:, :, 1]
    reaction = u * v * v
    diffusion_v = laplacian(v)
    grad_v = np.sqrt(gradient_energy(v))
    source_balance = reaction - (feed + kill) * v

    fields = [
        (v, "final v concentration", "magma"),
        (reaction, "reaction intensity u*v^2", "inferno"),
        (grad_v, "|grad v| diffusion front", "viridis"),
        (source_balance, "local v source balance", "coolwarm"),
        (diffusion_v, "Laplacian(v)", "coolwarm"),
        (v - data[0, :, :, 1], "v final - initial", "cividis"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.4), constrained_layout=True)
    for ax, (field, title, cmap) in zip(axes.ravel(), fields):
        im = ax.imshow(field, cmap=cmap, origin="lower")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(out_dir / f"sample_{sample}_derived_physics.png", dpi=180)
    plt.close(fig)


def plot_kymograph(data: np.ndarray, time: np.ndarray, out_dir: Path, sample: str) -> None:
    v = data[:, :, :, 1]
    center_y = v.shape[1] // 2
    center_x = v.shape[2] // 2
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), constrained_layout=True)
    extent_x = [0, v.shape[2], float(time[0]), float(time[-1])]
    extent_y = [0, v.shape[1], float(time[0]), float(time[-1])]
    im0 = axes[0].imshow(v[:, center_y, :], cmap="magma", aspect="auto", origin="lower", extent=extent_x)
    axes[0].set_title("v kymograph along x")
    axes[0].set_xlabel("x index")
    axes[0].set_ylabel("time")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(v[:, :, center_x], cmap="magma", aspect="auto", origin="lower", extent=extent_y)
    axes[1].set_title("v kymograph along y")
    axes[1].set_xlabel("y index")
    axes[1].set_ylabel("time")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    fig.savefig(out_dir / f"sample_{sample}_kymograph.png", dpi=180)
    plt.close(fig)


def plot_spectrum(data: np.ndarray, out_dir: Path, sample: str) -> None:
    frame_ids = [0, data.shape[0] // 2, data.shape[0] - 1]
    fig, ax = plt.subplots(figsize=(7.6, 4.8), constrained_layout=True)
    for frame in frame_ids:
        k, spectrum = radial_spectrum(data[frame, :, :, 1])
        ax.semilogy(k, spectrum + 1e-20, label=f"frame {frame}")
    ax.set_title("Radially averaged power spectrum of v")
    ax.set_xlabel("radial wave number")
    ax.set_ylabel("power")
    ax.legend()
    fig.savefig(out_dir / f"sample_{sample}_radial_spectrum.png", dpi=180)
    plt.close(fig)


def plot_sample_montage(h5: h5py.File, out_dir: Path) -> None:
    sample_ids = list(h5["samples"].keys())[:8]
    cols = min(4, len(sample_ids))
    rows = int(np.ceil(len(sample_ids) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, sample_id in zip(axes, sample_ids):
        v = h5[f"samples/{sample_id}/data"][-1, :, :, 1]
        im = ax.imshow(v, cmap="magma", origin="lower")
        ax.set_title(f"sample {sample_id}")
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[len(sample_ids) :]:
        ax.axis("off")
    fig.colorbar(im, ax=axes[: len(sample_ids)], fraction=0.025)
    fig.savefig(out_dir / "dataset_final_v_montage.png", dpi=180)
    plt.close(fig)


def make_gif(data: np.ndarray, time: np.ndarray, out_dir: Path, sample: str) -> None:
    images = []
    v_field = data[:, :, :, 1]
    vmin = float(v_field.min())
    vmax = float(v_field.max())
    for frame in range(data.shape[0]):
        fig, ax = plt.subplots(figsize=(4.2, 4.2), constrained_layout=True)
        ax.imshow(v_field[frame], cmap="magma", origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(f"v concentration, t={time[frame]:.0f}")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        images.append(image)
        plt.close(fig)
    imageio.mimsave(out_dir / f"sample_{sample}_v.gif", images, duration=0.12)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.input, "r") as h5:
        cfg = json.loads(h5.attrs["config"])
        feed = float(cfg["feed"])
        kill = float(cfg["kill"])
        data = h5[f"samples/{args.sample}/data"][:]
        time = h5["grid/t"][:]
        plot_sample_montage(h5, args.out_dir)
    plot_fields(data, args.out_dir, args.sample)
    metrics = plot_statistics(data, time, args.out_dir, args.sample, feed, kill)
    plot_derived_fields(data, args.out_dir, args.sample, feed, kill)
    plot_kymograph(data, time, args.out_dir, args.sample)
    plot_spectrum(data, args.out_dir, args.sample)
    make_gif(data, time, args.out_dir, args.sample)
    report = {
        "sample": args.sample,
        "shape": list(data.shape),
        "time_range": [float(time[0]), float(time[-1])],
        "u_range": [float(data[..., 0].min()), float(data[..., 0].max())],
        "v_range": [float(data[..., 1].min()), float(data[..., 1].max())],
        "initial": {key: float(value[0]) for key, value in metrics.items()},
        "final": {key: float(value[-1]) for key, value in metrics.items()},
        "interpretation": {
            "all_values_finite": bool(np.isfinite(data).all()),
            "v_pattern_survives": bool(metrics["mean_v"][-1] > 1e-4),
            "reaction_remains_active": bool(metrics["reaction_rate"][-1] > 1e-6),
        },
    }
    (args.out_dir / "physical_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.out_dir / f'sample_{args.sample}_fields.png'}")
    print(f"wrote {args.out_dir / f'sample_{args.sample}_physics_diagnostics.png'}")
    print(f"wrote {args.out_dir / f'sample_{args.sample}_derived_physics.png'}")
    print(f"wrote {args.out_dir / f'sample_{args.sample}_kymograph.png'}")
    print(f"wrote {args.out_dir / f'sample_{args.sample}_radial_spectrum.png'}")
    print(f"wrote {args.out_dir / 'dataset_final_v_montage.png'}")
    print(f"wrote {args.out_dir / f'sample_{args.sample}_v.gif'}")


if __name__ == "__main__":
    main()
