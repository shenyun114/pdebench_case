#!/usr/bin/env bash
set -euo pipefail

CASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${PDEBENCH_CASE_DATA:-/home/ubuntu/data}"
WORK_ROOT="${1:-${DATA_ROOT}/pdebench-cfd3d-demo}"
CONFIG="${2:-${CASE_DIR}/configs/default.yaml}"
PDEBENCH_ROOT="${WORK_ROOT}/PDEBench"
PDEBENCH_COMMIT="4ff3e3a4aa1561721b5571fa3a048a0a463e0568"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "错误：请先激活 cfd3d 环境。" >&2
  exit 2
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "错误：配置文件不存在：${CONFIG}" >&2
  exit 3
fi
mkdir -p "${WORK_ROOT}/artifacts/logs"
export PYTHONPYCACHEPREFIX="${WORK_ROOT}/python-cache"

if [[ ! -d "${PDEBENCH_ROOT}/.git" ]]; then
  git clone https://github.com/pdebench/PDEBench.git "${PDEBENCH_ROOT}"
fi
if ! git -C "${PDEBENCH_ROOT}" diff --quiet || ! git -C "${PDEBENCH_ROOT}" diff --cached --quiet; then
  echo "错误：${PDEBENCH_ROOT} 存在未提交改动，请使用新的工作目录。" >&2
  exit 4
fi
git -C "${PDEBENCH_ROOT}" fetch --quiet origin "${PDEBENCH_COMMIT}"
git -C "${PDEBENCH_ROOT}" checkout --quiet --detach "${PDEBENCH_COMMIT}"

python - <<'PY'
import h5py, hydra, imageio, jax, matplotlib, numpy, pandas, scipy, skimage, yaml
print("JAX", jax.__version__, "backend", jax.default_backend())
print("GPU devices", len(jax.devices("gpu")), jax.devices("gpu"))
assert jax.default_backend() == "gpu", "GPU-enabled JAX is required"
assert len(jax.devices("gpu")) >= 1
print("numpy", numpy.__version__, "h5py", h5py.__version__, "matplotlib", matplotlib.__version__)
PY
python -m py_compile "${CASE_DIR}"/src/*.py

COMMIT="$(git -C "${PDEBENCH_ROOT}" rev-parse HEAD)"
if [[ "${COMMIT}" != "${PDEBENCH_COMMIT}" ]]; then
  echo "错误：PDEBench 提交不匹配：${COMMIT}" >&2
  exit 5
fi
if [[ -n "$(git -C "${PDEBENCH_ROOT}" status --short)" ]]; then
  echo "错误：PDEBench 上游仓库存在改动；本案例要求干净检出。" >&2
  git -C "${PDEBENCH_ROOT}" status --short >&2
  exit 6
fi
echo "工作区就绪：${WORK_ROOT}；PDEBench ${COMMIT}（干净）"
