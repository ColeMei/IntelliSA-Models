#!/bin/bash
# hpc_sync_models_encoder.sh
# Sync ONLY the selected encoder run from HPC to local models directory.

set -euo pipefail

HPC_USER="qmmei"
HPC_HOST="spartan.hpc.unimelb.edu.au"
HPC_BASE="/data/gpfs/projects/punim2518/LLM-IaC-SecEval-Models"

LOCAL_MODELS="./models/experiments/encoder"
SELECTED_JSON="local/champion_check/selected_champion.json"

usage() {
  echo "Usage: $0 [options] [RUN_NAME]"
  echo "  If RUN_NAME not provided, it will be read from ${SELECTED_JSON}."
  echo ""
  echo "Options:"
  echo "  --dry-run               Show what would be done without transferring"
  echo "  --include-checkpoints   Also sync checkpoint-* directories (large)"
  echo "  --only-metadata         Sync only small metadata (skip model weights)"
  echo "  --user <user>           Override HPC user (default: ${HPC_USER})"
  echo "  --host <host>           Override HPC host (default: ${HPC_HOST})"
}

DRY_RUN=0
INCLUDE_CHECKPOINTS=0
ONLY_METADATA=0
RUN_NAME_ARG=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --include-checkpoints)
      INCLUDE_CHECKPOINTS=1
      shift
      ;;
    --only-metadata)
      ONLY_METADATA=1
      shift
      ;;
    --user)
      HPC_USER="$2"
      shift 2 || true
      ;;
    --host)
      HPC_HOST="$2"
      shift 2 || true
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      # First non-flag arg is RUN_NAME
      if [[ -z "${RUN_NAME_ARG}" ]]; then
        RUN_NAME_ARG="$1"
        shift
      else
        echo "Unknown argument: $1" >&2
        usage
        exit 1
      fi
      ;;
  esac
done

# Resolve RUN_NAME
RUN_NAME=""
if [[ -n "${RUN_NAME_ARG}" ]]; then
  RUN_NAME="${RUN_NAME_ARG}"
elif [[ -f "${SELECTED_JSON}" ]]; then
  if command -v jq >/dev/null 2>&1; then
    RUN_NAME=$(jq -r '.run_name // empty' "${SELECTED_JSON}") || RUN_NAME=""
  fi
  if [[ -z "${RUN_NAME}" ]]; then
    # Python fallback if jq not available
    if [[ -x ./environments/venv/bin/python ]]; then
      RUN_NAME=$(./environments/venv/bin/python - "$SELECTED_JSON" << 'PY' || true
import json, sys
path = sys.argv[1]
try:
    with open(path) as f:
        print(json.load(f).get('run_name',''))
except Exception:
    print('')
PY
)
    fi
  fi
fi

if [[ -z "${RUN_NAME}" ]]; then
  echo "ERROR: RUN_NAME not provided and could not be read from ${SELECTED_JSON}." >&2
  usage
  exit 1
fi

REMOTE_PATH="${HPC_BASE}/models/experiments/encoder/${RUN_NAME}/"
LOCAL_PATH="${LOCAL_MODELS}/${RUN_NAME}/"

echo "=== Syncing selected encoder run from HPC to local ==="
echo "User      : ${HPC_USER}"
echo "Host      : ${HPC_HOST}"
echo "Run name  : ${RUN_NAME}"
echo "Remote    : ${REMOTE_PATH}"
echo "Local     : ${LOCAL_PATH}"
echo "Checkpoints: $([[ ${INCLUDE_CHECKPOINTS} -eq 1 ]] && echo include || echo exclude)"
echo "Only metadata: $([[ ${ONLY_METADATA} -eq 1 ]] && echo yes || echo no)"

mkdir -p "${LOCAL_PATH}"

RSYNC_CMD=(rsync -avh --progress -L)
if [[ ${DRY_RUN} -eq 1 ]]; then
  RSYNC_CMD+=(--dry-run)
fi

# Build include/exclude filters
# Always include directories to allow traversal, then add specific files to include.
RSYNC_FILTERS=(
  "--include=*/"
  "--include=config.json"
  "--include=config_used.yaml"
  "--include=training_args.bin"
  "--include=merges.txt"
  "--include=vocab.json"
  "--include=tokenizer.json"
  "--include=tokenizer_config.json"
  "--include=special_tokens_map.json"
  "--include=added_tokens.json"
  "--include=spiece.model"
  "--include=generation_config.json"
  "--include=threshold_sweep_results.json"
)

if [[ ${ONLY_METADATA} -eq 0 ]]; then
  # Include model weights
  RSYNC_FILTERS+=(
    "--include=model.safetensors"
    "--include=model.safetensors.index.json"
    "--include=pytorch_model.bin"
    "--include=pytorch_model.bin.index.json"
  )
fi

if [[ ${INCLUDE_CHECKPOINTS} -eq 1 ]]; then
  # Include checkpoints (can be very large)
  RSYNC_FILTERS+=(
    "--include=checkpoint-*/"
    "--include=checkpoint-*/pytorch_model.bin*"
    "--include=checkpoint-*/model.safetensors*"
    "--include=checkpoint-*/optimizer.pt"
    "--include=checkpoint-*/rng_state.pth"
    "--include=checkpoint-*/trainer_state.json"
  )
else
  # Explicitly exclude checkpoints
  RSYNC_FILTERS+=("--exclude=checkpoint-*")
fi

# Exclude everything else by default
RSYNC_FILTERS+=("--exclude=*")

RSYNC_CMD+=("${RSYNC_FILTERS[@]}")

RSYNC_CMD+=("${HPC_USER}@${HPC_HOST}:${REMOTE_PATH}" "${LOCAL_PATH}")

echo "Running: ${RSYNC_CMD[*]}"
"${RSYNC_CMD[@]}"

echo "Sync complete."


