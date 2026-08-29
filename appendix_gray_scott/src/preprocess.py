from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gray_scott import SimConfig, make_initial_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a Gray-Scott PDEBench case.")
    parser.add_argument("--out-dir", type=Path, default=Path("../results"))
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--ny", type=int, default=128)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--save-every", type=int, default=30)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--du", type=float, default=0.16)
    parser.add_argument("--dv", type=float, default=0.08)
    parser.add_argument("--feed", type=float, default=0.060)
    parser.add_argument("--kill", type=float, default=0.062)
    parser.add_argument("--noise", type=float, default=0.02)
    return parser.parse_args()


def save_preview(cfg: SimConfig, seed: int, out_dir: Path) -> None:
    u, v = make_initial_state(cfg, seed)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), constrained_layout=True)
    for ax, field, title in zip(axes, [u, v], ["u: activator", "v: inhibitor"]):
        im = ax.imshow(field, cmap="viridis", origin="lower")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(out_dir / "initial_condition.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = SimConfig(nx=args.nx, ny=args.ny, steps=args.steps, save_every=args.save_every, dt=args.dt, du=args.du, dv=args.dv, feed=args.feed, kill=args.kill, noise=args.noise)
    cfg.validate()

    x = np.linspace(0.0, 1.0, cfg.nx, endpoint=False, dtype=np.float32)
    y = np.linspace(0.0, 1.0, cfg.ny, endpoint=False, dtype=np.float32)
    t = np.arange(cfg.frames, dtype=np.float32) * cfg.save_every * cfg.dt
    np.savez(args.out_dir / "grid.npz", x=x, y=y, t=t)

    metadata = {
        "case": "PDEBench Gray-Scott reaction diffusion",
        "samples": args.samples,
        "seed": args.seed,
        "config": cfg.to_dict(),
        "fields": ["u", "v"],
        "boundary": "periodic",
        "layout": "HDF5 samples/<id>/data with shape [time, y, x, field]",
    }
    (args.out_dir / "case_config.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    save_preview(cfg, args.seed, args.out_dir)
    print(f"wrote {args.out_dir / 'case_config.json'}")
    print(f"wrote {args.out_dir / 'grid.npz'}")
    print(f"wrote {args.out_dir / 'initial_condition.png'}")


if __name__ == "__main__":
    main()
