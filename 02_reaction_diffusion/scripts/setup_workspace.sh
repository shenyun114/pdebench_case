#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${PDEBENCH_CASE_DATA:-/home/ubuntu/data}"
WORK_ROOT="${1:-${DATA_ROOT}/pdebench-reacdiff-demo}"
REPO="${WORK_ROOT}/PDEBench"
COMMIT="4ff3e3a4aa1561721b5571fa3a048a0a463e0568"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "错误：请先激活 pdebench-reacdiff（或兼容环境）。" >&2
  exit 2
fi
mkdir -p "${WORK_ROOT}"
if [[ ! -d "${REPO}/.git" ]]; then
  git clone https://github.com/pdebench/PDEBench.git "${REPO}"
fi
if ! git -C "${REPO}" diff --quiet || ! git -C "${REPO}" diff --cached --quiet; then
  echo "错误：${REPO} 存在未提交改动，请使用新的工作目录。" >&2
  exit 3
fi
git -C "${REPO}" fetch --quiet origin "${COMMIT}"
git -C "${REPO}" checkout --quiet --detach "${COMMIT}"
mkdir -p "${WORK_ROOT}/artifacts"
PYTHONPATH="${REPO}" python - <<'PY'
from pdebench.data_gen.src.sim_diff_react import Simulator
print("PDEBench reaction-diffusion simulator import: OK")
PY
echo "工作区就绪：${WORK_ROOT}"
echo "PDEBench 提交：$(git -C "${REPO}" rev-parse HEAD)"
