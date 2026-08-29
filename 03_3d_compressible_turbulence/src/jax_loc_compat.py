"""Run an upstream PDEBench script with its historical ``Tracer.loc`` alias.

PDEBench commit 4ff3e3a uses ``u.loc[idx].set(...)`` in the nonlinear-equation
utilities. JAX exposes the identical indexed-update operation as ``u.at``.
This launcher supplies the former name at runtime and leaves the upstream
checkout byte-for-byte unchanged.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

from jax._src.core import Tracer


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: jax_loc_compat.py UPSTREAM_SCRIPT [ARGS ...]")
    upstream = Path(sys.argv[1]).resolve()
    Tracer.loc = property(lambda value: value.at)  # type: ignore[attr-defined]
    sys.argv = [str(upstream), *sys.argv[2:]]
    runpy.run_path(str(upstream), run_name="__main__")


if __name__ == "__main__":
    main()
