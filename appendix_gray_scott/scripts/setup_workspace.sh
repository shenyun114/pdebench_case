#!/usr/bin/env bash
set -euo pipefail

CASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${PDEBENCH_CASE_DATA:-/home/ubuntu/data}"
WORK_ROOT="${1:-${DATA_ROOT}/pdebench-gray-scott-demo}"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "错误：请先激活 pdebench-gray-scott（或兼容环境）。" >&2
  exit 2
fi

mkdir -p "${WORK_ROOT}/artifacts"
python - <<'PY'
import h5py, imageio, matplotlib, numba, numpy, yaml
print("Gray-Scott dependencies: OK")
print("numpy", numpy.__version__, "numba", numba.__version__, "h5py", h5py.__version__)
PY
PYTHONPYCACHEPREFIX="${WORK_ROOT}/python-cache" python -m py_compile "${CASE_DIR}"/src/*.py
echo "工作区就绪：${WORK_ROOT}"
