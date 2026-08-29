from __future__ import annotations

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


def add_box(ax, xy, text, color):
    x, y = xy
    rect = Rectangle((x, y), 2.45, 0.78, facecolor=color, edgecolor="#263238", linewidth=1.2)
    ax.add_patch(rect)
    ax.text(x + 1.225, y + 0.39, text, ha="center", va="center", fontsize=10)


def add_arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=15, linewidth=1.2, color="#263238"))


def workflow(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.axis("off")
    labels = [
        ("Preprocess\nconfig + grid", "#d8f3dc"),
        ("Parallel solve\nper seed", "#b7e4c7"),
        ("HDF5 dataset\nPDEBench layout", "#95d5b2"),
        ("Postprocess\nfigures + metrics", "#74c69d"),
    ]
    xs = [0.2, 3.0, 5.8, 8.6]
    for x, (label, color) in zip(xs, labels):
        add_box(ax, (x, 1.15), label, color)
    for x in [2.65, 5.45, 8.25]:
        add_arrow(ax, (x, 1.54), (x + 0.3, 1.54))
    ax.set_xlim(0, 11.3)
    ax.set_ylim(0, 3)
    fig.savefig(out_dir / "workflow.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def stencil(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    ax.set_aspect("equal")
    ax.axis("off")
    for y in range(5):
        for x in range(5):
            face = "#ffd166" if (x, y) == (2, 2) else "#e9ecef"
            if (x, y) in [(1, 2), (3, 2), (2, 1), (2, 3)]:
                face = "#8ecae6"
            ax.add_patch(Rectangle((x, y), 0.9, 0.9, facecolor=face, edgecolor="#495057"))
    ax.text(2.45, 2.45, "center", ha="center", va="center", fontsize=9)
    ax.text(2.45, 4.55, "5-point periodic Laplacian", ha="center", va="bottom", fontsize=12)
    ax.text(2.45, -0.25, "lap(a)=left+right+up+down-4*center", ha="center", va="top", fontsize=10)
    ax.set_xlim(-0.2, 5.1)
    ax.set_ylim(-0.6, 5.2)
    fig.savefig(out_dir / "stencil.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def parallel_model(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    ax.axis("off")
    add_box(ax, (0.25, 2.2), "Seed list\n0..N-1", "#caf0f8")
    for i, x in enumerate([3.1, 3.1, 3.1]):
        add_box(ax, (x, 3.0 - i * 1.05), f"Worker {i}\nNumba solver", "#90e0ef")
        add_arrow(ax, (2.7, 2.55), (3.05, 3.38 - i * 1.05))
        add_arrow(ax, (5.6, 3.38 - i * 1.05), (6.15, 2.55))
    add_box(ax, (6.2, 2.2), "Parent process\nHDF5 writer", "#48cae4")
    add_arrow(ax, (8.7, 2.55), (9.25, 2.55))
    add_box(ax, (9.3, 2.2), "Dataset\n+ timings", "#00b4d8")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.4)
    fig.savefig(out_dir / "parallel_model.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    workflow(out_dir)
    stencil(out_dir)
    parallel_model(out_dir)
    print(f"wrote diagrams to {out_dir}")


if __name__ == "__main__":
    main()
