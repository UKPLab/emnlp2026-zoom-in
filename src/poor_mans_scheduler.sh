#!/usr/bin/env bash
set -euo pipefail

#############################################
# Configuration
#############################################

# Which GPUs must be free?
# - "all"  => require every visible GPU to have no compute processes
# - "0,1"  => require only these GPU indices to be free
GPUS_REQUIRED="all"

# Poll interval (seconds). 10 minutes = 600
INTERVAL_SECONDS=600

# Need N consecutive "free" checks before starting
CONSECUTIVE_FREE_REQUIRED=2

# Training command to run once GPUs are free enough.
# Put your actual command here.
TRAIN_CMD=(python train_scheduler.py)

# Where to write logs/state
WORKDIR="/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/poor_mans_scheduler/gpu_waiter"
LOG_FILE="${WORKDIR}/gpu_waiter.log"
STATE_FILE="${WORKDIR}/gpu_waiter.state"
LOCK_FILE="${WORKDIR}/gpu_waiter_2.lock"

#############################################
# CLI / mode parsing
#############################################

DRY_RUN="${DRY_RUN:-0}"
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  INTERVAL_SECONDS=10
fi

#############################################
# Helpers
#############################################

mkdir -p "${WORKDIR}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${LOG_FILE}"
}

# Returns 0 if "free enough", 1 otherwise.
gpus_are_free() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "ERROR: nvidia-smi not found in PATH."
    return 1
  fi

  # Build the set of GPUs we care about.
  local gpu_list=()
  if [[ "${GPUS_REQUIRED}" == "all" ]]; then
    # Query all GPU indices present
    mapfile -t gpu_list < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
  else
    IFS=',' read -r -a gpu_list <<< "${GPUS_REQUIRED}"
  fi

  # For each required GPU, check compute processes.
  local gpu
  for gpu in "${gpu_list[@]}"; do
    # If any compute pid exists on that GPU, it's occupied.
    # (nounits avoids weird formatting; empty output means no compute apps)
    local pids
    pids="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^\s*$/d' || true)"
    if [[ -n "${pids}" ]]; then
      log "GPU ${gpu} is occupied (compute PIDs: $(echo "${pids}" | paste -sd',' -))."
      return 1
    fi
  done

  log "Required GPUs appear free: ${GPUS_REQUIRED}"
  return 0
}

read_state() {
  # file format: a single integer "consecutive_free"
  if [[ -f "${STATE_FILE}" ]]; then
    cat "${STATE_FILE}" 2>/dev/null || echo "0"
  else
    echo "0"
  fi
}

write_state() {
  local val="$1"
  printf '%s\n' "${val}" > "${STATE_FILE}"
}

start_training() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    log "DRY-RUN: would start training now: ${TRAIN_CMD[*]}"
    log "DRY-RUN: exiting without executing TRAIN_CMD."
    exit 0
  fi

  log "Starting training (foreground exec): ${TRAIN_CMD[*]}"
  exec "${TRAIN_CMD[@]}"
}

#############################################
# Main loop with lock to prevent double-start
#############################################

# Single-instance lock
exec 200>"${LOCK_FILE}"
if ! flock -n 200; then
  log "Another instance is running (lock: ${LOCK_FILE}). Exiting."
  exit 0
fi

log "GPU waiter started. Interval=${INTERVAL_SECONDS}s, consecutive_free_required=${CONSECUTIVE_FREE_REQUIRED}, GPUs=${GPUS_REQUIRED}"
consecutive_free="$(read_state)"
log "Initial consecutive_free=${consecutive_free}"

while true; do
  if gpus_are_free; then
    consecutive_free=$((consecutive_free + 1))
    write_state "${consecutive_free}"
    log "Free check PASS (${consecutive_free}/${CONSECUTIVE_FREE_REQUIRED})."

    if (( consecutive_free >= CONSECUTIVE_FREE_REQUIRED )); then
      log "Condition met: GPUs free for ${CONSECUTIVE_FREE_REQUIRED} consecutive checks."
      start_training
      log "Exiting after starting training."
      exit 0
    fi
  else
    consecutive_free=0
    write_state "${consecutive_free}"
    log "Free check FAIL. Reset consecutive_free=0."
  fi

  sleep "${INTERVAL_SECONDS}"
done