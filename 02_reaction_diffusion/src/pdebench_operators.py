"""Discrete operators matching PDEBench's pinned reaction-diffusion source."""

from __future__ import annotations

import numpy as np
from scipy.sparse import diags


def neumann_laplacian(nx: int, ny: int, dx: float, dy: float):
    """Return PDEBench's five-point cell-centred Laplacian with zero flux."""
    main = -2 * np.ones(nx) / dx**2 - 2 * np.ones(nx) / dy**2
    main[0] = -1 / dx**2 - 2 / dy**2
    main[-1] = -1 / dx**2 - 2 / dy**2
    main = np.tile(main, ny)
    main[:nx] = -2 / dx**2 - 1 / dy**2
    main[nx * (ny - 1) :] = -2 / dx**2 - 1 / dy**2
    main[0] = main[nx - 1] = -1 / dx**2 - 1 / dy**2
    main[nx * (ny - 1)] = main[-1] = -1 / dx**2 - 1 / dy**2

    left = np.tile(np.r_[0.0, np.ones(nx - 1)], ny)[1:] / dx**2
    right = np.tile(np.r_[np.ones(nx - 1), 0.0], ny)[:-1] / dx**2
    vertical = np.ones(nx * (ny - 1)) / dy**2
    return diags(
        [main, left, right, vertical, vertical],
        [0, -1, 1, -nx, nx],
        format="csr",
    )


def apply_laplacian(field: np.ndarray, lap) -> np.ndarray:
    """Apply the sparse operator to fields shaped (time, y, x)."""
    flat = field.reshape(field.shape[0], -1)
    return (lap @ flat.T).T.reshape(field.shape)
