#!/usr/bin/env bash
set -euo pipefail

CASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${PDEBENCH_CASE_DATA:-/home/ubuntu/data}"
WORK_ROOT="${1:-${DATA_ROOT}/pdebench-swe-demo}"
CONFIG="${2:-${CASE_DIR}/configs/default.yaml}"
REPO="${PDEBENCH_ROOT:-${DATA_ROOT}/pdebench-upstream/PDEBench}"
ARTIFACTS="${WORK_ROOT}/artifacts"
DATA="${ARTIFACTS}/radial_dam_break.h5"
RESULTS="${ARTIFACTS}/results"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "错误：请先激活 pdebench-swe（或已安装 Clawpack 的兼容环境）。" >&2
  exit 2
fi
if [[ ! -d "${REPO}/pdebench" ]]; then
  echo "错误：请先运行 scripts/setup_workspace.sh ${WORK_ROOT}" >&2
  exit 3
fi
if [[ -e "${DATA}" ]]; then
  echo "错误：${DATA} 已存在。为避免覆盖，请换一个 WORK_ROOT。" >&2
  exit 4
fi

mkdir -p "${RESULTS}"
cp "${CONFIG}" "${ARTIFACTS}/resolved_config.yaml"
exec > >(tee "${ARTIFACTS}/pipeline.log") 2>&1
cd "${ARTIFACTS}"
echo "[1/4] 运行 PDEBench/PyClaw 二维浅水波求解器"
PYTHONPATH="${REPO}" python "${CASE_DIR}/src/simulate_shallow_water.py" \
  --output "${DATA}" --config "${ARTIFACTS}/resolved_config.yaml" --repo "${REPO}"
echo "[2/4] 计算物理诊断并生成静态图和 GIF"
python "${CASE_DIR}/src/analyze_and_visualize.py" \
  --data "${DATA}" --output "${RESULTS}"
echo "[3/4] 执行 32/64/128 三网格自收敛研究"
PYTHONPATH="${REPO}" python "${CASE_DIR}/src/resolution_study.py" \
  --reference-data "${DATA}" --config "${ARTIFACTS}/resolved_config.yaml" --output "${RESULTS}"
echo "[4/4] 执行自动验收"
python "${CASE_DIR}/src/verify_results.py" "${DATA}" "${RESULTS}"

echo "二维浅水波案例完成：${RESULTS}"
echo "完整控制台日志：${ARTIFACTS}/pipeline.log"
