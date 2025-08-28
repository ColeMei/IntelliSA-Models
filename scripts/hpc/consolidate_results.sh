#!/bin/bash
# Usage:
#   ./scripts/hpc/consolidate_results.sh <jobid> [--tar]
# Or inside a job:
#   ./scripts/hpc/consolidate_results.sh "$SLURM_JOB_ID" --tar

set -euo pipefail

JOBID="${1:-${SLURM_JOB_ID:-}}"
if [[ -z "${JOBID}" ]]; then
  echo "ERROR: JOBID not provided and SLURM_JOB_ID not set."
  exit 1
fi

DO_TAR=false
if [[ "${2:-}" == "--tar" ]]; then
  DO_TAR=true
fi

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
RUN_DIR="results/run_${JOBID}_${TIMESTAMP}"
mkdir -p "${RUN_DIR}"

echo "Consolidating results for Job ${JOBID} into ${RUN_DIR} ..."

# Git provenance (best effort)
GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"

# Copy training log (if present)
cp "logs/training/training_${JOBID}.log" "${RUN_DIR}/training.log" 2>/dev/null || true

# Copy SLURM outputs (if present). Keep the newest if multiple.
latest_out="$(ls -1t logs/slurm_outputs/${JOBID}*.out 2>/dev/null | head -n1 || true)"
latest_err="$(ls -1t logs/slurm_outputs/${JOBID}*.err 2>/dev/null | head -n1 || true)"
[[ -n "${latest_out}" ]] && cp "${latest_out}" "${RUN_DIR}/slurm.out"
[[ -n "${latest_err}" ]] && cp "${latest_err}" "${RUN_DIR}/slurm.err"

# Config used (heuristics; adjust as needed)
if [[ -f "configs/config_${JOBID}.yaml" ]]; then
  cp "configs/config_${JOBID}.yaml" "${RUN_DIR}/config_used.yaml"
elif [[ -f "configs/generative_config.yaml" ]]; then
  cp "configs/generative_config.yaml" "${RUN_DIR}/config_used.yaml"
elif [[ -f "configs/encoder_config.yaml" ]]; then
  cp "configs/encoder_config.yaml" "${RUN_DIR}/config_used.yaml"
fi

# Metrics (if produced)
if [[ -f "logs/evaluation/metrics_${JOBID}.json" ]]; then
  cp "logs/evaluation/metrics_${JOBID}.json" "${RUN_DIR}/metrics.json"
fi

# Model artifacts: prefer final root files; fall back to last checkpoint
mkdir -p "${RUN_DIR}/checkpoints"
model_root=""

# Heuristic: most recently updated model directory among common names or matching job id
model_root=$(find models -maxdepth 2 -type d \( -name "*${JOBID}*" -o -name "encoder" -o -name "generative" \) -printf "%T@ %p\n" 2>/dev/null | sort -nr | awk 'NR==1{print $2}') || true

copy_if_exists() { [[ -f "$1" ]] && cp "$1" "${RUN_DIR}/checkpoints/"; }

if [[ -n "${model_root}" && -d "${model_root}" ]]; then
  copy_if_exists "${model_root}/model.safetensors"
  copy_if_exists "${model_root}/adapter_model.safetensors"
  copy_if_exists "${model_root}/training_args.bin"
  copy_if_exists "${model_root}/tokenizer.json"
  copy_if_exists "${model_root}/tokenizer_config.json"
  copy_if_exists "${model_root}/special_tokens_map.json"
  copy_if_exists "${model_root}/merges.txt"
  copy_if_exists "${model_root}/vocab.json"

  # Fallback: copy the latest checkpoint directory
  if [[ -z "$(ls -A "${RUN_DIR}/checkpoints" 2>/dev/null || true)" ]]; then
    last_ckpt="$(find "${model_root}" -maxdepth 1 -type d -name "checkpoint-*" -printf "%T@ %p\n" 2>/dev/null | sort -nr | awk 'NR==1{print $2}')"
    if [[ -n "${last_ckpt}" && -d "${last_ckpt}" ]]; then
      cp -r "${last_ckpt}" "${RUN_DIR}/checkpoints/"
    fi
  fi
fi

# Manifest for provenance
cat > "${RUN_DIR}/manifest.json" <<EOF
{
  "job_id": "${JOBID}",
  "timestamp": "${TIMESTAMP}",
  "git_commit": "${GIT_COMMIT}",
  "git_branch": "${GIT_BRANCH}",
  "host": "$(hostname)",
  "user": "$(whoami)",
  "training_log": "$( [[ -f ${RUN_DIR}/training.log ]] && echo training.log || echo null )",
  "slurm_out": "$( [[ -f ${RUN_DIR}/slurm.out ]] && echo slurm.out || echo null )",
  "slurm_err": "$( [[ -f ${RUN_DIR}/slurm.err ]] && echo slurm.err || echo null )",
  "config_used": "$( [[ -f ${RUN_DIR}/config_used.yaml ]] && echo config_used.yaml || echo null )",
  "metrics": "$( [[ -f ${RUN_DIR}/metrics.json ]] && echo metrics.json || echo null )"
}
EOF

echo "Done: ${RUN_DIR}"

if ${DO_TAR}; then
  tar_path="${RUN_DIR}.tar.gz"
  tar -czf "${tar_path}" -C "results" "$(basename "${RUN_DIR}")"
  echo "Created archive: ${tar_path}"
fi


