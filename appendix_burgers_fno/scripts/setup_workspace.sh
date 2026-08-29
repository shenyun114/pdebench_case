#!/usr/bin/env bash
set -euo pipefail

CASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${PDEBENCH_CASE_DATA:-/home/ubuntu/data}"
WORK_ROOT="${1:-${DATA_ROOT}/pdebench-fno-demo}"
REPO="${WORK_ROOT}/PDEBench"
COMMIT="4ff3e3a4aa1561721b5571fa3a048a0a463e0568"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "错误：请先执行 conda activate pdebench-fno" >&2
  exit 2
fi

mkdir -p "${WORK_ROOT}"
if [[ ! -d "${REPO}/.git" ]]; then
  git clone https://github.com/pdebench/PDEBench.git "${REPO}"
fi

git -C "${REPO}" fetch --quiet origin "${COMMIT}"
git -C "${REPO}" checkout --quiet --detach "${COMMIT}"

# 该上游文件采用 CRLF，先做两个精确且幂等的 JAX API 替换。
UTILS="${REPO}/pdebench/data_gen/data_gen_NLE/utils.py"
sed -i 's/uL\.loc\[/uL.at[/g; s/uR\.loc\[/uR.at[/g' "${UTILS}"

PATCH="${CASE_DIR}/patches/pdebench-main-compat.patch"
if git -C "${REPO}" apply --check "${PATCH}" 2>/dev/null; then
  git -C "${REPO}" apply "${PATCH}"
elif git -C "${REPO}" apply --reverse --check "${PATCH}" 2>/dev/null; then
  echo "兼容补丁已经应用。"
else
  echo "错误：仓库有非预期改动，或兼容补丁无法应用；请换一个工作目录。" >&2
  exit 4
fi

python -m pip install --no-deps --editable "${REPO}"
mkdir -p "${WORK_ROOT}/artifacts/raw" "${WORK_ROOT}/artifacts/results"

echo "工作区就绪：${WORK_ROOT}"
echo "PDEBench 提交：$(git -C "${REPO}" rev-parse HEAD)"
