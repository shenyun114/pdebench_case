#!/usr/bin/env bash
set -euo pipefail

CASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${PDEBENCH_CASE_DATA:-/home/ubuntu/data}"
WORK_ROOT="${1:-${DATA_ROOT}/pdebench-cfd3d-demo}"
CONFIG="${2:-${CASE_DIR}/configs/default.yaml}"
PDEBENCH_ROOT="${PDEBENCH_ROOT:-${DATA_ROOT}/pdebench-upstream/PDEBench}"
PDEBENCH_COMMIT="4ff3e3a4aa1561721b5571fa3a048a0a463e0568"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "错误：请先激活 cfd3d 环境。" >&2
  exit 2
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "错误：配置文件不存在：${CONFIG}" >&2
  exit 3
fi
mkdir -p "${WORK_ROOT}/artifacts/logs" "$(dirname "${PDEBENCH_ROOT}")"
export PYTHONPYCACHEPREFIX="${WORK_ROOT}/python-cache"

if [[ ! -d "${PDEBENCH_ROOT}/.git" ]]; then
  git clone https://github.com/pdebench/PDEBench.git "${PDEBENCH_ROOT}"
fi
if ! git -C "${PDEBENCH_ROOT}" diff --quiet || ! git -C "${PDEBENCH_ROOT}" diff --cached --quiet; then
  echo "错误：共享源码 ${PDEBENCH_ROOT} 存在未提交改动，请清理改动或指定新的 PDEBENCH_ROOT。" >&2
  exit 4
fi
git -C "${PDEBENCH_ROOT}" fetch --quiet origin "${PDEBENCH_COMMIT}"
git -C "${PDEBENCH_ROOT}" checkout --quiet --detach "${PDEBENCH_COMMIT}"

REQUIRED_BACKEND="$(python -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1], encoding="utf-8"))["case"].get("backend", "gpu"))' "${CONFIG}")"
if [[ "${REQUIRED_BACKEND}" != "cpu" && "${REQUIRED_BACKEND}" != "gpu" ]]; then
  echo "错误：case.backend 必须是 cpu 或 gpu，当前为 ${REQUIRED_BACKEND}。" >&2
  exit 5
fi
export PDEBENCH_REQUIRED_BACKEND="${REQUIRED_BACKEND}"
if [[ "${REQUIRED_BACKEND}" == "cpu" ]]; then
  export JAX_PLATFORMS=cpu
else
  export JAX_PLATFORMS=cuda
fi

python - <<'PY'
import os
import h5py, hydra, imageio, jax, matplotlib, numpy, pandas, scipy, skimage, yaml
required = os.environ["PDEBENCH_REQUIRED_BACKEND"]
print("JAX", jax.__version__, "backend", jax.default_backend())
print("devices", len(jax.devices()), jax.devices())
assert jax.default_backend() == required, f"{required} backend is required"
assert len(jax.devices()) >= 1
print("numpy", numpy.__version__, "h5py", h5py.__version__, "matplotlib", matplotlib.__version__)
PY
python -m py_compile "${CASE_DIR}"/src/*.py

COMMIT="$(git -C "${PDEBENCH_ROOT}" rev-parse HEAD)"
if [[ "${COMMIT}" != "${PDEBENCH_COMMIT}" ]]; then
  echo "错误：PDEBench 提交不匹配：${COMMIT}" >&2
  exit 6
fi
if [[ -n "$(git -C "${PDEBENCH_ROOT}" status --short)" ]]; then
  echo "错误：PDEBench 上游仓库存在改动；本案例要求干净检出。" >&2
  git -C "${PDEBENCH_ROOT}" status --short >&2
  exit 7
fi
echo "工作区就绪：${WORK_ROOT}；共享源码 ${PDEBENCH_ROOT}；后端 ${REQUIRED_BACKEND}；PDEBench ${COMMIT}（干净）"
