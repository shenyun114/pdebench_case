from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from gray_scott import SimConfig, simulate_numba
from visualize import gradient_energy, radial_spectrum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a feed/kill parameter sweep for Gray-Scott patterns.")
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--nx", type=int, default=96)
    parser.add_argument("--ny", type=int, default=96)
    parser.add_argument("--steps", type=int, default=720)
    parser.add_argument("--save-every", type=int, default=120)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--feeds", nargs="+", type=float, default=[0.035, 0.045, 0.055, 0.065])
    parser.add_argument("--kills", nargs="+", type=float, default=[0.055, 0.060, 0.065, 0.070])
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--du", type=float, default=0.16)
    parser.add_argument("--dv", type=float, default=0.08)
    parser.add_argument("--noise", type=float, default=0.02)
    return parser.parse_args()


def spectral_centroid(field: np.ndarray) -> tuple[float, float]:
    k, spectrum = radial_spectrum(field)
    total_power = float(np.sum(spectrum))
    if len(k) == 0 or total_power < 1e-12 or float(field.max() - field.min()) < 1e-6:
        return 0.0, 0.0
    weights = spectrum + 1e-20
    centroid = float(np.sum(k * weights) / np.sum(weights))
    high_mask = k >= max(2, int(0.18 * min(field.shape)))
    high_fraction = float(np.sum(weights[high_mask]) / np.sum(weights))
    return centroid, high_fraction


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    feeds = np.array(args.feeds, dtype=np.float32)
    kills = np.array(args.kills, dtype=np.float32)
    final_v = np.empty((len(kills), len(feeds), args.ny, args.nx), dtype=np.float32)
    mean_v = np.empty((len(kills), len(feeds)), dtype=np.float32)
    pattern_energy = np.empty_like(mean_v)
    reaction_rate = np.empty_like(mean_v)
    spectral_center = np.empty_like(mean_v)
    high_freq_fraction = np.empty_like(mean_v)

    # Warm up JIT before timing or sweeping.
    simulate_numba(SimConfig(nx=16, ny=16, steps=4, save_every=2), args.seed)

    rows = []
    for iy, kill in enumerate(kills):
        for ix, feed in enumerate(feeds):
            cfg = SimConfig(
                nx=args.nx,
                ny=args.ny,
                steps=args.steps,
                save_every=args.save_every,
                feed=float(feed),
                kill=float(kill),
                dt=args.dt,
                du=args.du,
                dv=args.dv,
                noise=args.noise,
            )
            data = simulate_numba(cfg, args.seed + iy * len(feeds) + ix)
            u = data[-1, :, :, 0]
            v = data[-1, :, :, 1]
            final_v[iy, ix] = v
            mean_v[iy, ix] = v.mean()
            pattern_energy[iy, ix] = gradient_energy(v).mean()
            reaction_rate[iy, ix] = (u * v * v).mean()
            spectral_center[iy, ix], high_freq_fraction[iy, ix] = spectral_centroid(v)
            rows.append(
                [
                    feed,
                    kill,
                    mean_v[iy, ix],
                    pattern_energy[iy, ix],
                    reaction_rate[iy, ix],
                    spectral_center[iy, ix],
                    high_freq_fraction[iy, ix],
                ]
            )

    with h5py.File(args.out_dir / "parameter_sweep.h5", "w") as h5:
        h5.create_dataset("feed", data=feeds)
        h5.create_dataset("kill", data=kills)
        h5.create_dataset("final_v", data=final_v, compression="gzip", compression_opts=4)
        h5.create_dataset("mean_v", data=mean_v)
        h5.create_dataset("pattern_energy", data=pattern_energy)
        h5.create_dataset("reaction_rate", data=reaction_rate)
        h5.create_dataset("spectral_centroid_k", data=spectral_center)
        h5.create_dataset("high_frequency_fraction", data=high_freq_fraction)

    with (args.out_dir / "parameter_sweep_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "feed",
                "kill",
                "mean_v",
                "pattern_energy",
                "reaction_rate",
                "spectral_centroid_k",
                "high_frequency_fraction",
            ]
        )
        writer.writerows(rows)

    fig, axes = plt.subplots(len(kills), len(feeds), figsize=(2.3 * len(feeds), 2.3 * len(kills)), constrained_layout=True)
    axes = np.atleast_2d(axes)
    vmin = float(final_v.min())
    vmax = float(final_v.max())
    for iy, kill in enumerate(kills):
        for ix, feed in enumerate(feeds):
            ax = axes[iy, ix]
            ax.imshow(final_v[iy, ix], cmap="magma", origin="lower", vmin=vmin, vmax=vmax)
            ax.set_title(f"F={feed:.3f}, k={kill:.3f}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.savefig(args.out_dir / "parameter_sweep_patterns.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.8), constrained_layout=True)
    heatmaps = [
        (mean_v, "mean final v"),
        (pattern_energy, "pattern gradient energy"),
        (reaction_rate, "mean reaction u*v^2"),
        (spectral_center, "spectral centroid k"),
        (high_freq_fraction, "high-frequency power fraction"),
    ]
    for ax, (metric, title) in zip(axes.ravel(), heatmaps):
        im = ax.imshow(metric, origin="lower", cmap="viridis", aspect="auto")
        ax.set_title(title)
        ax.set_xticks(range(len(feeds)), [f"{v:.3f}" for v in feeds])
        ax.set_yticks(range(len(kills)), [f"{v:.3f}" for v in kills])
        ax.set_xlabel("feed F")
        ax.set_ylabel("kill k")
        fig.colorbar(im, ax=ax, fraction=0.046)
    axes.ravel()[-1].axis("off")
    fig.savefig(args.out_dir / "parameter_sweep_metrics.png", dpi=180)
    plt.close(fig)

    print(f"wrote {args.out_dir / 'parameter_sweep.h5'}")
    print(f"wrote {args.out_dir / 'parameter_sweep_patterns.png'}")
    print(f"wrote {args.out_dir / 'parameter_sweep_metrics.png'}")


if __name__ == "__main__":
    main()
