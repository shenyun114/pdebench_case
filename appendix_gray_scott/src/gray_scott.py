from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Callable

import numpy as np

try:
    from numba import njit
except Exception:  # pragma: no cover - exercised when numba is absent
    njit = None


@dataclass(frozen=True)
class SimConfig:
    nx: int = 128
    ny: int = 128
    steps: int = 900
    save_every: int = 30
    dt: float = 1.0
    du: float = 0.16
    dv: float = 0.08
    feed: float = 0.060
    kill: float = 0.062
    noise: float = 0.02

    @property
    def frames(self) -> int:
        return self.steps // self.save_every + 1

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)

    def validate(self) -> None:
        if self.nx < 8 or self.ny < 8:
            raise ValueError("nx and ny must both be at least 8")
        if self.steps <= 0 or self.save_every <= 0 or self.steps % self.save_every:
            raise ValueError("steps must be positive and exactly divisible by save_every")
        if self.dt <= 0 or self.du < 0 or self.dv < 0:
            raise ValueError("dt must be positive and diffusion coefficients non-negative")


def make_initial_state(cfg: SimConfig, seed: int) -> tuple[np.ndarray, np.ndarray]:
    cfg.validate()
    rng = np.random.default_rng(seed)
    u = np.ones((cfg.ny, cfg.nx), dtype=np.float32)
    v = np.zeros((cfg.ny, cfg.nx), dtype=np.float32)

    radius_y = max(4, cfg.ny // 10)
    radius_x = max(4, cfg.nx // 10)
    cy = cfg.ny // 2 + rng.integers(-cfg.ny // 12, cfg.ny // 12 + 1)
    cx = cfg.nx // 2 + rng.integers(-cfg.nx // 12, cfg.nx // 12 + 1)
    y0, y1 = max(0, cy - radius_y), min(cfg.ny, cy + radius_y)
    x0, x1 = max(0, cx - radius_x), min(cfg.nx, cx + radius_x)

    u[y0:y1, x0:x1] = 0.50
    v[y0:y1, x0:x1] = 0.25
    u += cfg.noise * rng.standard_normal(u.shape, dtype=np.float32)
    v += cfg.noise * rng.standard_normal(v.shape, dtype=np.float32)
    return np.clip(u, 0.0, 1.0), np.clip(v, 0.0, 1.0)


def _lap_numpy(a: np.ndarray) -> np.ndarray:
    return (
        np.roll(a, 1, axis=0)
        + np.roll(a, -1, axis=0)
        + np.roll(a, 1, axis=1)
        + np.roll(a, -1, axis=1)
        - 4.0 * a
    )


def simulate_numpy(cfg: SimConfig, seed: int) -> np.ndarray:
    u, v = make_initial_state(cfg, seed)
    out = np.empty((cfg.frames, cfg.ny, cfg.nx, 2), dtype=np.float32)
    out[0, :, :, 0] = u
    out[0, :, :, 1] = v

    frame = 1
    for step in range(1, cfg.steps + 1):
        uvv = u * v * v
        u += cfg.dt * (cfg.du * _lap_numpy(u) - uvv + cfg.feed * (1.0 - u))
        v += cfg.dt * (cfg.dv * _lap_numpy(v) + uvv - (cfg.feed + cfg.kill) * v)
        if step % cfg.save_every == 0:
            out[frame, :, :, 0] = u
            out[frame, :, :, 1] = v
            frame += 1
    return out


def simulate_python(cfg: SimConfig, seed: int) -> np.ndarray:
    u, v = make_initial_state(cfg, seed)
    out = np.empty((cfg.frames, cfg.ny, cfg.nx, 2), dtype=np.float32)
    out[0, :, :, 0] = u
    out[0, :, :, 1] = v

    frame = 1
    for step in range(1, cfg.steps + 1):
        new_u = np.empty_like(u)
        new_v = np.empty_like(v)
        for y in range(cfg.ny):
            ym = (y - 1) % cfg.ny
            yp = (y + 1) % cfg.ny
            for x in range(cfg.nx):
                xm = (x - 1) % cfg.nx
                xp = (x + 1) % cfg.nx
                lap_u = u[ym, x] + u[yp, x] + u[y, xm] + u[y, xp] - 4.0 * u[y, x]
                lap_v = v[ym, x] + v[yp, x] + v[y, xm] + v[y, xp] - 4.0 * v[y, x]
                uvv = u[y, x] * v[y, x] * v[y, x]
                new_u[y, x] = u[y, x] + cfg.dt * (
                    cfg.du * lap_u - uvv + cfg.feed * (1.0 - u[y, x])
                )
                new_v[y, x] = v[y, x] + cfg.dt * (
                    cfg.dv * lap_v + uvv - (cfg.feed + cfg.kill) * v[y, x]
                )
        u, v = new_u, new_v
        if step % cfg.save_every == 0:
            out[frame, :, :, 0] = u
            out[frame, :, :, 1] = v
            frame += 1
    return out


if njit is not None:

    @njit(cache=True, nogil=True)
    def _simulate_numba_kernel(
        u: np.ndarray,
        v: np.ndarray,
        steps: int,
        save_every: int,
        dt: float,
        du: float,
        dv: float,
        feed: float,
        kill: float,
    ) -> np.ndarray:
        ny, nx = u.shape
        frames = steps // save_every + 1
        out = np.empty((frames, ny, nx, 2), dtype=np.float32)
        next_u = np.empty_like(u)
        next_v = np.empty_like(v)

        for y in range(ny):
            for x in range(nx):
                out[0, y, x, 0] = u[y, x]
                out[0, y, x, 1] = v[y, x]

        frame = 1
        for step in range(1, steps + 1):
            for y in range(ny):
                ym = y - 1 if y > 0 else ny - 1
                yp = y + 1 if y < ny - 1 else 0
                for x in range(nx):
                    xm = x - 1 if x > 0 else nx - 1
                    xp = x + 1 if x < nx - 1 else 0
                    lap_u = u[ym, x] + u[yp, x] + u[y, xm] + u[y, xp] - 4.0 * u[y, x]
                    lap_v = v[ym, x] + v[yp, x] + v[y, xm] + v[y, xp] - 4.0 * v[y, x]
                    uvv = u[y, x] * v[y, x] * v[y, x]
                    next_u[y, x] = u[y, x] + dt * (du * lap_u - uvv + feed * (1.0 - u[y, x]))
                    next_v[y, x] = v[y, x] + dt * (dv * lap_v + uvv - (feed + kill) * v[y, x])

            tmp = u
            u = next_u
            next_u = tmp
            tmp = v
            v = next_v
            next_v = tmp

            if step % save_every == 0:
                for y in range(ny):
                    for x in range(nx):
                        out[frame, y, x, 0] = u[y, x]
                        out[frame, y, x, 1] = v[y, x]
                frame += 1
        return out


def simulate_numba(cfg: SimConfig, seed: int) -> np.ndarray:
    if njit is None:
        raise RuntimeError("numba is not installed")
    u, v = make_initial_state(cfg, seed)
    return _simulate_numba_kernel(
        u,
        v,
        cfg.steps,
        cfg.save_every,
        cfg.dt,
        cfg.du,
        cfg.dv,
        cfg.feed,
        cfg.kill,
    )


BACKENDS: dict[str, Callable[[SimConfig, int], np.ndarray]] = {
    "python": simulate_python,
    "numpy": simulate_numpy,
    "numba": simulate_numba,
}


def run_timed(backend: str, cfg: SimConfig, seed: int) -> tuple[np.ndarray, float]:
    start = perf_counter()
    data = BACKENDS[backend](cfg, seed)
    return data, perf_counter() - start
