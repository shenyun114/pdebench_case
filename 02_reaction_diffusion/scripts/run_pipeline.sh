#!/usr/bin/env bash
set -euo pipefail

CASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${PDEBENCH_CASE_DATA:-/home/ubuntu/data}"
WORK_ROOT="${1:-${DATA_ROOT}/pdebench-reacdiff-demo}"
CONFIG="${2:-${CASE_DIR}/configs/default.yaml}"
REPO="${WORK_ROOT}/PDEBench"
ARTIFACTS="${WORK_ROOT}/artifacts"
DATA="${ARTIFACTS}/reaction_diffusion.h5"
RESULTS="${ARTIFACTS}/results"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "错误：请先激活 pdebench-reacdiff（或兼容环境）。" >&2
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
echo "[1/4] 运行 PDEBench/SciPy 二维反应–扩散求解器"
PYTHONPATH="${REPO}" python "${CASE_DIR}/src/simulate_reaction_diffusion.py" \
  --output "${DATA}" --config "${ARTIFACTS}/resolved_config.yaml" --repo "${REPO}"
echo "[2/4] 使用与求解器一致的离散算子生成诊断图和 GIF"
python "${CASE_DIR}/src/analyze_and_visualize.py" \
  --data "${DATA}" --output "${RESULTS}"
echo "[3/4] 执行投影初值的 32/64/128 网格一致性研究"
PYTHONPATH="${REPO}" python "${CASE_DIR}/src/resolution_study.py" \
  --reference-data "${DATA}" --config "${ARTIFACTS}/resolved_config.yaml" --output "${RESULTS}"
echo "[4/4] 执行自动验收"
python "${CASE_DIR}/src/verify_results.py" "${DATA}" "${RESULTS}"
echo "二维反应–扩散案例完成：${RESULTS}"
echo "完整控制台日志：${ARTIFACTS}/pipeline.log"
