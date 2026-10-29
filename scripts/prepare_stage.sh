#!/bin/bash

set -euo pipefail

FRONTIERDRIVE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-${FRONTIERDRIVE_ROOT}}
export NAVSIM_DEVKIT_ROOT=${NAVSIM_DEVKIT_ROOT:-${FRONTIERDRIVE_ROOT}}
export NAVSIM_EXP_ROOT=${NAVSIM_EXP_ROOT:-${FRONTIERDRIVE_ROOT}/outputs}
PYTHON=${PYTHON:-python}
CPU_LIMIT=${SLURM_CPUS_PER_TASK:-7}
export OMP_NUM_THREADS="${CPU_LIMIT}"
export MKL_NUM_THREADS="${CPU_LIMIT}"
export OPENBLAS_NUM_THREADS="${CPU_LIMIT}"
export NUMEXPR_NUM_THREADS="${CPU_LIMIT}"

STAGE=${STAGE:-1}
if [ "${STAGE}" -lt 1 ]; then
  echo "ERROR: prepare stage must be >= 1, got STAGE=${STAGE}" >&2
  exit 2
fi

PREV_STAGE=$((STAGE - 1))
TBAT_RUN_NAME=${TBAT_RUN_NAME:-${MEANFUSER_RUN_NAME:-${LTF_RUN_NAME:-${DIFFUSIONDRIVE_RUN_NAME:-${CBAT_RUN_NAME:-tbat}}}}}
if [ -z "${TBAT_EVAL_NAME_PREFIX:-}" ]; then
  if [ -n "${CBAT_EVAL_NAME_PREFIX:-}" ]; then
    TBAT_EVAL_NAME_PREFIX=${CBAT_EVAL_NAME_PREFIX}
  elif [ -n "${MEANFUSER_RUN_NAME:-}" ] || [ "${TBAT_RUN_NAME}" = "meanfuser" ]; then
    TBAT_EVAL_NAME_PREFIX=${MEANFUSER_EVAL_NAME_PREFIX:-meanfuser_eval}
  elif [ -n "${DIFFUSIONDRIVE_RUN_NAME:-}" ] || [ "${TBAT_RUN_NAME}" = "diffusiondrive" ]; then
    TBAT_EVAL_NAME_PREFIX=${DIFFUSIONDRIVE_EVAL_NAME_PREFIX:-diffusiondrive_eval}
  elif [ -n "${LTF_RUN_NAME:-}" ]; then
    TBAT_EVAL_NAME_PREFIX=${LTF_EVAL_NAME_PREFIX:-${LTF_RUN_NAME}_eval}
  else
    TBAT_EVAL_NAME_PREFIX=${TBAT_RUN_NAME}_eval
  fi
fi
TBAT_LAYOUT=${TBAT_LAYOUT:-structured}
RUN_ROOT=${NAVSIM_EXP_ROOT}/${TBAT_RUN_NAME}
MODEL_PHASE=warmup
if [ "${STAGE}" -gt 1 ]; then
  MODEL_PHASE=stage${PREV_STAGE}
fi

case "${TBAT_LAYOUT}" in
  structured)
    EVAL_STAGE_ROOT=${RUN_ROOT}/evaluate/stage${PREV_STAGE}
    STAGE_DIR_DEFAULT=${RUN_ROOT}/evaluate/stage${STAGE}
    PREVIOUS_CHECKPOINT_DEFAULT=${RUN_ROOT}/training/${MODEL_PHASE}/models/last.ckpt
    WARMUP_EPOCHS=${WARMUP_EPOCHS:-10}
    ;;
  legacy)
    EVAL_STAGE_ROOT=${RUN_ROOT}/stage${PREV_STAGE}
    STAGE_DIR_DEFAULT=${RUN_ROOT}/stage${STAGE}
    PREVIOUS_CHECKPOINT_DEFAULT=${RUN_ROOT}/models/${MODEL_PHASE}/last.ckpt
    WARMUP_EPOCHS=${WARMUP_EPOCHS:-20}
    ;;
  *)
    echo "ERROR: TBAT_LAYOUT must be structured or legacy, got ${TBAT_LAYOUT}" >&2
    exit 2
    ;;
esac

CBAT_RUN_NAME=${TBAT_RUN_NAME}
CBAT_EVAL_NAME_PREFIX=${TBAT_EVAL_NAME_PREFIX}
EVAL_HISTORY_ROOT=${RUN_ROOT}
EVAL_CSV=${EVAL_CSV:-${EVAL_STAGE_ROOT}/${TBAT_EVAL_NAME_PREFIX}_stage${PREV_STAGE}_navtest/metrics_results.csv}
FEATURE_DIR=${FEATURE_DIR:-${NAVSIM_EXP_ROOT}/cbs_features/v21_pca12_whiten}
STAGE_DIR=${STAGE_DIR:-${STAGE_DIR_DEFAULT}}
EVAL_DIR=${STAGE_DIR}/eval
BOUNDARY_DIR=${STAGE_DIR}/boundary
RETRIEVAL_DIR=${STAGE_DIR}/retrieval

TAU=${TAU:-0.95}
ZPD_P_LO=${ZPD_P_LO:-0.20}
ZPD_P_HI=${ZPD_P_HI:-0.60}
ZPD_N_MIN=${ZPD_N_MIN:-300}
BAND_EXPAND_STEP=${BAND_EXPAND_STEP:-0.05}
IRT_L2=${IRT_L2:-0.02}
IRT_STEPS=${IRT_STEPS:-600}
IRT_LR=${IRT_LR:-0.15}
MIN_AGENT_DISTANCE=${MIN_AGENT_DISTANCE:-1.0}
MAX_DRAC=${MAX_DRAC:-30.0}
ANNEAL_RATE=${ANNEAL_RATE:-0.15}
MIN_ANNEAL=${MIN_ANNEAL:-0.65}
RETRIEVAL_CLUSTERS=${RETRIEVAL_CLUSTERS:-}
RADIUS_SCALE=${RADIUS_SCALE:-1.5}
RADIUS_KNN=${RADIUS_KNN:-5}
DIFFICULTY_GAIN=${DIFFICULTY_GAIN:-0.35}
BOUNDARY_MASS=${BOUNDARY_MASS:-0.50}
MIN_TRAIN_PER_SPHERE=${MIN_TRAIN_PER_SPHERE:-300}
MERGE_DISTANCE_SCALE=${MERGE_DISTANCE_SCALE:-1.0}
ISOLATED_BOUNDARY_COUNT=${ISOLATED_BOUNDARY_COUNT:-2}
MAX_FLOOR_RADIUS_RATIO=${MAX_FLOOR_RADIUS_RATIO:-3.0}
MAX_SPHERE_RADIUS_SCALE=${MAX_SPHERE_RADIUS_SCALE:-1.0}
POINT_LIKE_RADIUS_SCALE=${POINT_LIKE_RADIUS_SCALE:-1.0}
POINT_LIKE_TRAIN_COUNT=${POINT_LIKE_TRAIN_COUNT:-10}
MAX_REVISION_ROUNDS=${MAX_REVISION_ROUNDS:-3}
AUDIT_AGENT_ENABLED=${AUDIT_AGENT_ENABLED:-0}
STAGE_EPOCHS=${STAGE_EPOCHS:-10}
TBAT_ROTATION_FRACTION=${TBAT_ROTATION_FRACTION:-0.85}
TBAT_EPOCH_SIZE=${TBAT_EPOCH_SIZE:--1}
TBAT_SAMPLER_SEED=${TBAT_SAMPLER_SEED:-42}
MIN_BOUNDARY_WEIGHT=${MIN_BOUNDARY_WEIGHT:-0.30}
MAX_BOUNDARY_WEIGHT=${MAX_BOUNDARY_WEIGHT:-0.70}
MAX_REGION_WEIGHT=${MAX_REGION_WEIGHT:-0.65}
DEEPSEEK_MODEL=${DEEPSEEK_MODEL:-deepseek-v4-flash}
DEEPSEEK_BASE_URL=${DEEPSEEK_BASE_URL:-https://api.deepseek.com}
DEEPSEEK_API_KEY_ENV=${DEEPSEEK_API_KEY_ENV:-DEEPSEEK_API_KEY}
DEEPSEEK_TEMPERATURE=${DEEPSEEK_TEMPERATURE:-0.2}
DEEPSEEK_TIMEOUT=${DEEPSEEK_TIMEOUT:-120}
DEEPSEEK_MAX_RETRIES=${DEEPSEEK_MAX_RETRIES:-2}
DEEPSEEK_MAX_TOKENS=${DEEPSEEK_MAX_TOKENS:-16384}

PREVIOUS_CHECKPOINT=${PREVIOUS_CHECKPOINT:-${PREVIOUS_CHECKPOINT_DEFAULT}}

SCRIPT_DIR=${FRONTIERDRIVE_ROOT}/frontierdrive
AGENT_LOOP_SCRIPT=${SCRIPT_DIR}/curriculum_controller.py
MEMORY_PATH=${MEMORY_PATH:-${NAVSIM_EXP_ROOT}/${CBAT_RUN_NAME}/experiment_memory.json}

mkdir -p "${EVAL_DIR}" "${BOUNDARY_DIR}" "${RETRIEVAL_DIR}"

echo "[Stage ${STAGE}] Memory-augmented multi-agent boundary preparation"
echo "  TBAT_RUN_NAME=${TBAT_RUN_NAME}"
echo "  TBAT_EVAL_NAME_PREFIX=${TBAT_EVAL_NAME_PREFIX}"
echo "  TBAT_LAYOUT=${TBAT_LAYOUT}"
echo "  EVAL_CSV=${EVAL_CSV}"
echo "  PREVIOUS_CHECKPOINT=${PREVIOUS_CHECKPOINT}"
echo "  FEATURE_DIR=${FEATURE_DIR}"
echo "  STAGE_DIR=${STAGE_DIR}"
echo "  MEMORY_PATH=${MEMORY_PATH}"
echo "  BOUNDARY_MASS=${BOUNDARY_MASS}"
echo "  RETRIEVAL_CLUSTERS=${RETRIEVAL_CLUSTERS:-auto}"
echo "  MAX_REVISION_ROUNDS=${MAX_REVISION_ROUNDS}"
echo "  AUDIT_AGENT_ENABLED=${AUDIT_AGENT_ENABLED}"
echo "  DEEPSEEK_MODEL=${DEEPSEEK_MODEL}"
echo "  DEEPSEEK_BASE_URL=${DEEPSEEK_BASE_URL}"
echo "  DEEPSEEK_API_KEY_ENV=${DEEPSEEK_API_KEY_ENV}"

AGENT_LOOP_ARGS=(
  --project-root "${PROJECT_ROOT}"
  --navsim-devkit-root "${NAVSIM_DEVKIT_ROOT}"
  --navsim-exp-root "${NAVSIM_EXP_ROOT}"
  --python "${PYTHON}"
  --stage "${STAGE}"
  --run-name "${TBAT_RUN_NAME}"
  --eval-name-prefix "${TBAT_EVAL_NAME_PREFIX}"
  --feature-dir "${FEATURE_DIR}"
  --stage-dir "${STAGE_DIR}"
  --eval-csv "${EVAL_CSV}"
  --previous-checkpoint "${PREVIOUS_CHECKPOINT}"
  --memory-path "${MEMORY_PATH}"
  --max-revision-rounds "${MAX_REVISION_ROUNDS}"
  --warmup-epochs "${WARMUP_EPOCHS}"
  --stage-epochs "${STAGE_EPOCHS}"
  --tbat-rotation-fraction "${TBAT_ROTATION_FRACTION}"
  --tbat-epoch-size "${TBAT_EPOCH_SIZE}"
  --tbat-sampler-seed "${TBAT_SAMPLER_SEED}"
  --min-boundary-weight "${MIN_BOUNDARY_WEIGHT}"
  --max-boundary-weight "${MAX_BOUNDARY_WEIGHT}"
  --max-region-weight "${MAX_REGION_WEIGHT}"
  --deepseek-model "${DEEPSEEK_MODEL}"
  --deepseek-base-url "${DEEPSEEK_BASE_URL}"
  --deepseek-api-key-env "${DEEPSEEK_API_KEY_ENV}"
  --deepseek-temperature "${DEEPSEEK_TEMPERATURE}"
  --deepseek-timeout "${DEEPSEEK_TIMEOUT}"
  --deepseek-max-retries "${DEEPSEEK_MAX_RETRIES}"
  --deepseek-max-tokens "${DEEPSEEK_MAX_TOKENS}"
  --tau "${TAU}"
  --p-lo "${ZPD_P_LO}"
  --p-hi "${ZPD_P_HI}"
  --n-min "${ZPD_N_MIN}"
  --band-expand-step "${BAND_EXPAND_STEP}"
  --irt-l2 "${IRT_L2}"
  --irt-steps "${IRT_STEPS}"
  --irt-lr "${IRT_LR}"
  --min-agent-distance "${MIN_AGENT_DISTANCE}"
  --max-drac "${MAX_DRAC}"
  --anneal-rate "${ANNEAL_RATE}"
  --min-anneal "${MIN_ANNEAL}"
  --radius-scale "${RADIUS_SCALE}"
  --radius-knn "${RADIUS_KNN}"
  --difficulty-gain "${DIFFICULTY_GAIN}"
  --boundary-mass "${BOUNDARY_MASS}"
  --min-train-per-sphere "${MIN_TRAIN_PER_SPHERE}"
  --merge-distance-scale "${MERGE_DISTANCE_SCALE}"
  --isolated-boundary-count "${ISOLATED_BOUNDARY_COUNT}"
  --max-floor-radius-ratio "${MAX_FLOOR_RADIUS_RATIO}"
  --max-sphere-radius-scale "${MAX_SPHERE_RADIUS_SCALE}"
  --point-like-radius-scale "${POINT_LIKE_RADIUS_SCALE}"
  --point-like-train-count "${POINT_LIKE_TRAIN_COUNT}"
)

if [ -n "${RETRIEVAL_CLUSTERS}" ]; then
  AGENT_LOOP_ARGS+=(--clusters "${RETRIEVAL_CLUSTERS}")
fi

if [ "${AUDIT_AGENT_ENABLED}" = "0" ]; then
  AGENT_LOOP_ARGS+=(--disable-audit-agent)
fi

if [ -n "${MAX_EPOCHS:-}" ]; then
  AGENT_LOOP_ARGS+=(--max-epochs "${MAX_EPOCHS}")
fi

"${PYTHON}" -u "${AGENT_LOOP_SCRIPT}" "${AGENT_LOOP_ARGS[@]}"

test -s "${RETRIEVAL_DIR}/train_manifest.csv"
test -s "${RETRIEVAL_DIR}/curriculum_manifest.csv"
test -s "${STAGE_DIR}/agent/training_proposal.json"
test -s "${STAGE_DIR}/agent/training_proposal.md"
test -s "${STAGE_DIR}/agent/training_proposal.env"

echo "[DONE] TBAT stage ${STAGE} boundary preparation proposal finished."
echo "Manifest: ${RETRIEVAL_DIR}/train_manifest.csv"
echo "Curriculum manifest: ${RETRIEVAL_DIR}/curriculum_manifest.csv"
echo "Proposal: ${STAGE_DIR}/agent/training_proposal.md"
echo "This prepare job does not submit or start training/evaluation."
