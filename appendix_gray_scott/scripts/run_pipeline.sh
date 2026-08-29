#!/usr/bin/env bash
set -euo pipefail

CASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${PDEBENCH_CASE_DATA:-/home/ubuntu/data}"
WORK_ROOT="${1:-${DATA_ROOT}/pdebench-gray-scott-demo}"
CONFIG="${2:-${CASE_DIR}/configs/default.yaml}"
ARTIFACTS="${WORK_ROOT}/artifacts"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "错误：请先激活 pdebench-gray-scott（或兼容环境）。" >&2
  exit 2
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "错误：配置文件不存在：${CONFIG}" >&2
  exit 3
fi
if [[ -e "${ARTIFACTS}/grayscott_dataset.h5" ]]; then
  echo "错误：结果已经存在。为避免覆盖，请更换 WORK_ROOT：${WORK_ROOT}" >&2
  exit 4
fi

mkdir -p "${ARTIFACTS}"
cp "${CONFIG}" "${ARTIFACTS}/resolved_config.yaml"
exec > >(tee "${ARTIFACTS}/pipeline.log") 2>&1
export PYTHONPYCACHEPREFIX="${WORK_ROOT}/python-cache"
export NUMBA_CACHE_DIR="${WORK_ROOT}/numba-cache"
python "${CASE_DIR}/src/pipeline.py" \
  --case-dir "${CASE_DIR}" \
  --work-dir "${ARTIFACTS}" \
  --config "${ARTIFACTS}/resolved_config.yaml"

echo "Gray-Scott 案例完成：${ARTIFACTS}"
echo "完整控制台日志：${ARTIFACTS}/pipeline.log"
