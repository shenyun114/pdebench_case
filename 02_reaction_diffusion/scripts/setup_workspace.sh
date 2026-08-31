#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${PDEBENCH_CASE_DATA:-/home/ubuntu/data}"
WORK_ROOT="${1:-${DATA_ROOT}/pdebench-reacdiff-demo}"
REPO="${PDEBENCH_ROOT:-${DATA_ROOT}/pdebench-upstream/PDEBench}"
COMMIT="4ff3e3a4aa1561721b5571fa3a048a0a463e0568"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "错误：请先激活 pdebench-reacdiff（或兼容环境）。" >&2
  exit 2
fi
mkdir -p "${WORK_ROOT}" "$(dirname "${REPO}")"
if [[ ! -d "${REPO}/.git" ]]; then
  git clone https://github.com/pdebench/PDEBench.git "${REPO}"
fi
if ! git -C "${REPO}" diff --quiet || ! git -C "${REPO}" diff --cached --quiet; then
  echo "错误：共享源码 ${REPO} 存在未提交改动，请清理改动或指定新的 PDEBENCH_ROOT。" >&2
  exit 3
fi
if ! git -C "${REPO}" cat-file -e "${COMMIT}^{commit}" 2>/dev/null; then
  git -C "${REPO}" fetch --quiet origin "${COMMIT}"
fi
git -C "${REPO}" checkout --quiet --detach "${COMMIT}"
mkdir -p "${WORK_ROOT}/artifacts"
PYTHONPATH="${REPO}" python - <<'PY'
from pdebench.data_gen.src.sim_diff_react import Simulator
print("PDEBench reaction-diffusion simulator import: OK")
PY
echo "工作区就绪：${WORK_ROOT}"
echo "共享 PDEBench 源码：${REPO}"
echo "PDEBench 提交：$(git -C "${REPO}" rev-parse HEAD)"
