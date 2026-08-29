#!/usr/bin/env bash
set -euo pipefail

CASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${PDEBENCH_CASE_DATA:-/home/ubuntu/data}"
WORK_ROOT="${1:-${DATA_ROOT}/pdebench-fno-demo}"
REPO="${WORK_ROOT}/PDEBench"
RAW="${WORK_ROOT}/artifacts/raw"
RESULTS="${WORK_ROOT}/artifacts/results"
EPOCHS="${PDEBENCH_EPOCHS:-50}"
SAMPLES="${PDEBENCH_SAMPLES:-96}"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "错误：请先执行 conda activate pdebench-fno" >&2
  exit 2
fi
if [[ ! -d "${REPO}/pdebench" ]]; then
  echo "错误：请先运行 scripts/setup_workspace.sh ${WORK_ROOT}" >&2
  exit 3
fi

mkdir -p "${RAW}" "${RESULTS}"
if compgen -G "${RAW}/*.npy" >/dev/null || compgen -G "${RAW}/*.hdf5" >/dev/null; then
  echo "错误：${RAW} 已有数据。为保证独立复现，请使用新的 WORK_ROOT。" >&2
  exit 4
fi

cd "${REPO}"
python "${CASE_DIR}/src/collect_system_info.py" --output "${RESULTS}/system_info.json"

BURGERS_DIR="${REPO}/pdebench/data_gen/data_gen_NLE/BurgersEq"
SAVE_REL="$(realpath --relative-to="${BURGERS_DIR}" "${RAW}")/"
cd "${BURGERS_DIR}"
PYTHONPATH="${REPO}" python burgers_multi_solution_Hydra.py \
  +multi=1e-2.yaml \
  multi.numbers="${SAMPLES}" \
  multi.nx=128 \
  multi.fin_time=1.0 \
  multi.dt_save=0.025 \
  multi.show_steps=100 \
  "multi.save=${SAVE_REL}"

cd "${REPO}"
PYTHONPATH="${REPO}" python pdebench/data_gen/data_gen_NLE/Data_Merge.py \
  args.type=burgers args.dim=1 "args.savedir=${RAW}"

DATA="${RAW}/1D_Burgers_Sols_Nu0.01.hdf5"
python "${CASE_DIR}/src/validate_data.py" "${DATA}" \
  --output "${RESULTS}/data_validation.json"
PYTHONPATH="${REPO}" python "${CASE_DIR}/src/train_and_evaluate.py" \
  --data "${DATA}" --output "${RESULTS}" --epochs "${EPOCHS}"
python "${CASE_DIR}/src/create_extra_visuals.py" \
  --predictions "${RESULTS}/predictions.npz" --output "${RESULTS}"
python "${CASE_DIR}/src/verify_results.py" "${RESULTS}"

echo "流水线完成。结果目录：${RESULTS}"
