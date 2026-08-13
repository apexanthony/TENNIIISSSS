#!/usr/bin/env bash
set -Eeuo pipefail

# Linux/NVIDIA launcher for the five one-seed V3.8 paper ablations.
# Environment variables may override every server-dependent setting.
SEED="${SEED:-37}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-8}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EVAL_WORKERS="${EVAL_WORKERS:-8}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_ROOT="${OUTPUT_ROOT:-exps/v38_essay_ablation}"
TRACKNET_ROOT="${TRACKNET_ROOT:-datasets/trackNet}"
TRAIN_CSV="${TRAIN_CSV:-datasets/tracknet_v38_match_split/train.csv}"
VALID_CSV="${VALID_CSV:-datasets/tracknet_v38_match_split/valid.csv}"
TEST_CSV="${TEST_CSV:-datasets/tracknet_v38_match_split/test.csv}"
MAPPED_CSV="${MAPPED_CSV:-datasets/tennis_all_v4i_mapped/annotations_hardneg_cleaned_strict.csv}"

mkdir -p "${OUTPUT_ROOT}"

for variant in baseline ca hardneg aux full; do
  run_name="${variant}_seed${SEED}"
  run_dir="${OUTPUT_ROOT}/${run_name}"
  state_path="${run_dir}/training_state.pt"
  mkdir -p "${run_dir}"

  resume_args=()
  if [[ -f "${state_path}" ]]; then
    resume_args=(--resume "${state_path}")
  fi

  echo "=== Training ${run_name} ==="
  python -u versions/v3_lightweight/v3_8_coordinate_attention/train_v38_ablation.py \
    --variant "${variant}" \
    --run-name "${run_name}" \
    --output-root "${OUTPUT_ROOT}" \
    --tracknet-root "${TRACKNET_ROOT}" \
    --train-csv "${TRAIN_CSV}" \
    --valid-csv "${VALID_CSV}" \
    --hardneg-mapped-csv "${MAPPED_CSV}" \
    --aux-mapped-csv "${MAPPED_CSV}" \
    --seed "${SEED}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --val-batch-size "${VAL_BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --device "${DEVICE}" \
    --pin-memory \
    "${resume_args[@]}" \
    2>&1 | tee -a "${run_dir}/train.log"

  echo "=== Evaluating ${run_name} ==="
  python -u versions/v3_lightweight/v3_8_coordinate_attention/evaluate_v38.py \
    --run-dir "${run_dir}" \
    --valid-csv "${VALID_CSV}" \
    --test-csv "${TEST_CSV}" \
    --batch-size "${VAL_BATCH_SIZE}" \
    --num-workers "${EVAL_WORKERS}" \
    --device "${DEVICE}" \
    2>&1 | tee "${run_dir}/evaluate.log"
done

python -u versions/v3_lightweight/v3_8_coordinate_attention/summarize_ablation.py \
  --root "${OUTPUT_ROOT}" --seed "${SEED}" \
  2>&1 | tee "${OUTPUT_ROOT}/summary.log"
