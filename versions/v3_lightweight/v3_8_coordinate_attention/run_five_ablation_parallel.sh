#!/usr/bin/env bash
set -Eeuo pipefail

# Linux/NVIDIA launcher that runs independent ablations concurrently while
# preserving each variant's batch size, seed, optimizer, and validation rules.
SEED="${SEED:-37}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-8}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_ROOT="${OUTPUT_ROOT:-exps/v38_essay_ablation}"
TRACKNET_ROOT="${TRACKNET_ROOT:-datasets/trackNet}"
TRAIN_CSV="${TRAIN_CSV:-datasets/tracknet_v38_match_split/train.csv}"
VALID_CSV="${VALID_CSV:-datasets/tracknet_v38_match_split/valid.csv}"
TEST_CSV="${TEST_CSV:-datasets/tracknet_v38_match_split/test.csv}"
MAPPED_CSV="${MAPPED_CSV:-datasets/tennis_all_v4i_mapped/annotations_hardneg_cleaned_strict.csv}"

if (( MAX_PARALLEL < 1 || MAX_PARALLEL > 3 )); then
  echo "MAX_PARALLEL must be between 1 and 3 for the 24 GB A10 profile" >&2
  exit 2
fi

if (( MAX_PARALLEL == 3 && VAL_BATCH_SIZE > BATCH_SIZE )); then
  echo "For three concurrent A10 workers, VAL_BATCH_SIZE (${VAL_BATCH_SIZE}) must not exceed BATCH_SIZE (${BATCH_SIZE})." >&2
  echo "Use VAL_BATCH_SIZE=${BATCH_SIZE}; validation batch size does not change model optimization." >&2
  exit 2
fi

# Concurrent validation temporarily changes each process's activation footprint.
# Expandable segments reduce allocator fragmentation across those transitions.
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${OUTPUT_ROOT}"

run_variant() {
  local variant="$1"
  local run_name="${variant}_seed${SEED}"
  local run_dir="${OUTPUT_ROOT}/${run_name}"
  local state_path="${run_dir}/training_state.pt"
  local resume_args=()
  mkdir -p "${run_dir}"

  if [[ -f "${state_path}" ]]; then
    resume_args=(--resume "${state_path}")
  fi

  echo "=== Training ${run_name} (parallel worker $$) ==="
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
}

variants=(baseline ca hardneg aux full)
active_pids=()
active_names=()

wait_for_oldest() {
  local pid="${active_pids[0]}"
  local name="${active_names[0]}"
  if ! wait "${pid}"; then
    echo "Variant ${name} failed; stopping the parallel launcher." >&2
    exit 1
  fi
  active_pids=("${active_pids[@]:1}")
  active_names=("${active_names[@]:1}")
}

for variant in "${variants[@]}"; do
  while (( ${#active_pids[@]} >= MAX_PARALLEL )); do
    wait_for_oldest
  done
  run_variant "${variant}" &
  active_pids+=("$!")
  active_names+=("${variant}")
done

while (( ${#active_pids[@]} > 0 )); do
  wait_for_oldest
done

python -u versions/v3_lightweight/v3_8_coordinate_attention/summarize_ablation.py \
  --root "${OUTPUT_ROOT}" --seed "${SEED}" \
  2>&1 | tee "${OUTPUT_ROOT}/summary.log"
