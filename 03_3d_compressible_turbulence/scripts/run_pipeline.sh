#!/usr/bin/env bash
set -euo pipefail

CASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${PDEBENCH_CASE_DATA:-/home/ubuntu/data}"
WORK_ROOT="${1:-${DATA_ROOT}/pdebench-cfd3d-demo}"
CONFIG_INPUT="${2:-${CASE_DIR}/configs/default.yaml}"
ARTIFACTS="${WORK_ROOT}/artifacts"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "错误：请先激活 cfd3d 环境。" >&2
  exit 2
fi
if [[ -e "${ARTIFACTS}/cfd3d_dataset.h5" ]]; then
  echo "错误：为避免覆盖已有数据，请更换 WORK_ROOT：${WORK_ROOT}" >&2
  exit 3
fi

bash "${CASE_DIR}/scripts/setup_workspace.sh" "${WORK_ROOT}" "${CONFIG_INPUT}"
mkdir -p "${ARTIFACTS}"
cp "${CONFIG_INPUT}" "${ARTIFACTS}/resolved_config.yaml"
CONFIG="${ARTIFACTS}/resolved_config.yaml"
exec > >(tee "${ARTIFACTS}/pipeline.log") 2>&1
export PYTHONPYCACHEPREFIX="${WORK_ROOT}/python-cache"
export MPLCONFIGDIR="${WORK_ROOT}/matplotlib-cache"

echo "[1/5] 调用未修改的 PDEBench 多 GPU 求解器生成 3D 数据"
python "${CASE_DIR}/src/run_official.py" --config "${CONFIG}" --work-dir "${ARTIFACTS}" --mode dataset
echo "[2/5] 合并五个官方 NPY 场为带元数据的 HDF5"
python "${CASE_DIR}/src/convert_dataset.py" --config "${CONFIG}" --work-dir "${ARTIFACTS}"
echo "[3/5] 进行固定总样本数的 1/2/4/8 GPU 强扩展测试"
python "${CASE_DIR}/src/run_official.py" --config "${CONFIG}" --work-dir "${ARTIFACTS}" --mode benchmark
echo "[4/5] 计算守恒量、涡量、散度、能谱并生成 PNG/GIF"
python "${CASE_DIR}/src/postprocess.py" --config "${CONFIG}" --work-dir "${ARTIFACTS}"
echo "[5/5] 自动验收"
python "${CASE_DIR}/src/verify_results.py" --config "${CONFIG}" --work-dir "${ARTIFACTS}"

echo "3D 可压缩湍流案例完成：${ARTIFACTS}"
echo "完整日志：${ARTIFACTS}/pipeline.log"
