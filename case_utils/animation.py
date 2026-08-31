"""Create GIFs with fixed layouts and one global color palette."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import GifImagePlugin, Image, ImageSequence


PixelBox = tuple[int, int, int, int]


def freeze_figure_layout(fig, *, dpi: int) -> None:
    """Resolve constrained layout once, then keep every axes position fixed."""
    fig.set_dpi(dpi)
    fig.canvas.draw()
    positions = [(axis, axis.get_position().frozen()) for axis in fig.axes]
    fig.set_layout_engine("none")
    for axis, position in positions:
        axis.set_position(position)
    fig.canvas.draw()


def capture_rgb(fig) -> np.ndarray:
    """Return an owned RGB image of the current Matplotlib canvas."""
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()


def axes_pixel_boxes(fig, axes: Iterable) -> list[PixelBox]:
    """Convert Matplotlib axes bounds to top-left-origin image pixel boxes."""
    fig.canvas.draw()
    height = int(fig.canvas.get_width_height()[1])
    boxes: list[PixelBox] = []
    for axis in axes:
        x0, y0, x1, y1 = axis.get_window_extent().extents
        boxes.append(
            (
                max(0, int(np.floor(x0))),
                max(0, height - int(np.ceil(y1))),
                int(np.ceil(x1)),
                height - int(np.floor(y0)),
            )
        )
    return boxes


def _assert_static_regions(frames: list[np.ndarray], boxes: Iterable[PixelBox]) -> None:
    reference = frames[0]
    for box_index, (x0, y0, x1, y1) in enumerate(boxes):
        expected = reference[y0:y1, x0:x1]
        for frame_index, frame in enumerate(frames[1:], start=1):
            if not np.array_equal(expected, frame[y0:y1, x0:x1]):
                raise RuntimeError(
                    f"static animation region {box_index} changed in frame {frame_index}"
                )


def _global_palette_frames(frames: list[np.ndarray]) -> list[Image.Image]:
    palette_seed = Image.fromarray(frames[0]).quantize(
        colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE
    )
    palette_source = Image.new("P", (1, 1))
    palette_source.putpalette(palette_seed.getpalette())
    converted: list[Image.Image] = []
    for frame in frames:
        converted.append(
            Image.fromarray(frame).quantize(
                palette=palette_source, dither=Image.Dither.NONE
            )
        )
    return converted


def _write_global_palette_gif(
    frames: list[Image.Image], output: Path, *, duration_ms: int
) -> None:
    """Write full frames without Pillow's per-frame palette normalization."""
    with output.open("wb") as stream:
        header, _ = GifImagePlugin.getheader(
            frames[0],
            palette=frames[0].getpalette(),
            info={"loop": 0, "optimize": False},
        )
        for block in header:
            stream.write(block)
        for frame in frames:
            for block in GifImagePlugin.getdata(
                frame,
                duration=duration_ms,
                disposal=1,
                include_color_table=False,
            ):
                stream.write(block)
        stream.write(b";")


def save_fixed_palette_gif(
    frames: list[np.ndarray],
    output: Path,
    *,
    duration_ms: int,
    static_boxes: Iterable[PixelBox] = (),
) -> None:
    """Save full GIF frames with a global palette and verify static regions."""
    if not frames:
        raise ValueError("at least one animation frame is required")
    shape = frames[0].shape
    if len(shape) != 3 or shape[2] != 3:
        raise ValueError(f"expected RGB frames, got {shape}")
    if any(frame.shape != shape for frame in frames):
        raise ValueError("all animation frames must have the same shape")

    boxes = list(static_boxes)
    _assert_static_regions(frames, boxes)
    paletted = _global_palette_frames(frames)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_global_palette_gif(paletted, output, duration_ms=duration_ms)

    with Image.open(output) as animation:
        decoded = [
            np.asarray(frame.convert("RGB"))
            for frame in ImageSequence.Iterator(animation)
        ]
    if len(decoded) != len(frames):
        raise RuntimeError(f"GIF frame count changed: {len(frames)} -> {len(decoded)}")
    _assert_static_regions(decoded, boxes)
