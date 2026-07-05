#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/runpod_idle_guard.sh [--env PATH] [--dry-run]
  scripts/runpod_idle_guard.sh --create [--env PATH] [--dry-run]
  scripts/runpod_idle_guard.sh --bootstrap [--env PATH]
  scripts/runpod_idle_guard.sh --sync-up [--env PATH]
  scripts/runpod_idle_guard.sh --resolve-ssh [--env PATH]
  scripts/runpod_idle_guard.sh --terminate [--env PATH] [--dry-run]

Mac-side RunPod lifecycle guard.

Default mode:
  - Resolves the active pod's SSH host/port from the RunPod REST API.
  - Detects remote inactivity from process state + watched file mtimes.
  - After RUNPOD_IDLE_STOP_SECONDS, rsyncs remote data locally and stops the pod.
  - After RUNPOD_IDLE_EMAIL_SECONDS, sends one email reminder to terminate the pod.

Create mode:
  - Prints the requested A100/H100 + 300GB pod spec.
  - Requires interactive confirmation.
  - Creates the pod, waits for SSH, runs apt setup, and syncs local code to GPU.

Terminate mode:
  - Resolves the pod from the API.
  - Requires typing DELETE <pod_id>.
  - Syncs remote data down when SSH is available, then stops and deletes the pod.
EOF
}

MODE="guard"
DRY_RUN=0
ENV_FILE="${RUNPOD_IDLE_GUARD_ENV:-}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --create)
      MODE="create"
      shift
      ;;
    --bootstrap)
      MODE="bootstrap"
      shift
      ;;
    --sync-up)
      MODE="sync-up"
      shift
      ;;
    --resolve-ssh)
      MODE="resolve-ssh"
      shift
      ;;
    --terminate|--destroy)
      MODE="terminate"
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --env)
      ENV_FILE="${2:?missing --env path}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -n "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required env: $name" >&2
    exit 2
  fi
}

require_env RUNPOD_API_KEY
require_env RUNPOD_SSH_KEY

RUNPOD_POD_NAME="${RUNPOD_POD_NAME:-kv-cache-headroom}"
export RUNPOD_POD_NAME
REMOTE_WATCH_PATHS="${RUNPOD_REMOTE_WATCH_PATHS:-.}"
ACTIVE_PROCESS_REGEX="${RUNPOD_ACTIVE_PROCESS_REGEX:-python|nsys|vllm|VLLM::EngineCore|run_.*headroom|run_oracle0|run_baseline|run_marginal}"
IDLE_STOP_SECONDS="${RUNPOD_IDLE_STOP_SECONDS:-3600}"
IDLE_EMAIL_SECONDS="${RUNPOD_IDLE_EMAIL_SECONDS:-7200}"
STATE_DIR="${RUNPOD_STATE_DIR:-$HOME/.runpod_idle_guard}"
STATE_KEY="${RUNPOD_POD_ID:-$RUNPOD_POD_NAME}"
STATE_FILE="$STATE_DIR/${STATE_KEY}.state"
LOCK_DIR="$STATE_DIR/${STATE_KEY}.lock"
LOG_FILE="$STATE_DIR/${STATE_KEY}.log"
LOCAL_REPO_ROOT="${RUNPOD_LOCAL_REPO_ROOT:-$(pwd)}"
REMOTE_ROOT="${RUNPOD_REMOTE_ROOT:-/root/kv_cache_research}"
LOCAL_SYNC_ROOT="${RUNPOD_LOCAL_SYNC_ROOT:-$LOCAL_REPO_ROOT/studies/results/runpod_guard_sync}"
SYNC_UP_PATHS="${RUNPOD_SYNC_UP_PATHS:-AGENTS.md CLAUDE.md README.md benchmarking docs scripts studies/specs datasets/synthetic datasets/processed/empirical_headroom}"

mkdir -p "$STATE_DIR"

log() {
  echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG_FILE" >&2
}

load_state() {
  INACTIVE_SINCE=""
  STOPPED_AT=""
  EMAILED_AT=""
  LAST_SYNC_AT=""
  ACTIVE_POD_ID=""
  if [[ -f "$STATE_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$STATE_FILE"
  fi
}

save_state() {
  {
    printf 'INACTIVE_SINCE=%q\n' "${INACTIVE_SINCE:-}"
    printf 'STOPPED_AT=%q\n' "${STOPPED_AT:-}"
    printf 'EMAILED_AT=%q\n' "${EMAILED_AT:-}"
    printf 'LAST_SYNC_AT=%q\n' "${LAST_SYNC_AT:-}"
    printf 'ACTIVE_POD_ID=%q\n' "${RUNPOD_POD_ID:-${ACTIVE_POD_ID:-}}"
  } > "$STATE_FILE"
}

api_request() {
  local method="$1"
  local path="$2"
  local data="${3:-}"
  local args=(
    curl --fail --silent --show-error
    --request "$method"
    --url "https://rest.runpod.io/v1/$path"
    --header "Authorization: Bearer $RUNPOD_API_KEY"
  )
  if [[ -n "$data" ]]; then
    args+=(--header "Content-Type: application/json" --data "$data")
  fi
  "${args[@]}"
}

extract_pod_summary() {
  python3 - "$RUNPOD_POD_NAME" <<'PY'
import json
import sys

preferred_name = sys.argv[1]
data = json.load(sys.stdin)

if isinstance(data, list):
    pods = data
elif isinstance(data, dict):
    if "id" in data:
        pods = [data]
    else:
        pods = data.get("pods") or data.get("data") or data.get("items") or []
else:
    pods = []

def status(pod):
    return str(pod.get("desiredStatus") or pod.get("status") or "").upper()

def is_live(pod):
    return status(pod) not in {"TERMINATED", "DELETED"}

if len(pods) != 1:
    named = [pod for pod in pods if pod.get("name") == preferred_name and is_live(pod)]
    if len(named) == 1:
        pods = named
    elif len(named) > 1:
        print(
            f"Multiple live pods match RUNPOD_POD_NAME={preferred_name!r}; set RUNPOD_POD_ID.",
            file=sys.stderr,
        )
        sys.exit(4)
    else:
        live = [pod for pod in pods if is_live(pod)]
        if len(live) == 1:
            pods = live
        else:
            print(
                f"Could not uniquely resolve pod by RUNPOD_POD_NAME={preferred_name!r}; set RUNPOD_POD_ID.",
                file=sys.stderr,
            )
            sys.exit(4)

if not pods:
    sys.exit(3)

pod = pods[0]
port_mappings = pod.get("portMappings") or {}
ssh_port = port_mappings.get("22") or port_mappings.get(22) or ""
print(f"id={pod.get('id') or ''}")
print(f"name={pod.get('name') or ''}")
print(f"status={status(pod)}")
print(f"public_ip={pod.get('publicIp') or ''}")
print(f"ssh_port={ssh_port}")
print(f"cost_per_hr={pod.get('costPerHr') or pod.get('adjustedCostPerHr') or ''}")
print(f"gpu={((pod.get('gpu') or {}).get('displayName')) or ((pod.get('gpu') or {}).get('id')) or ''}")
PY
}

resolve_pod() {
  load_state
  local pod_json
  local pod_id="${RUNPOD_POD_ID:-${ACTIVE_POD_ID:-}}"
  if [[ -n "$pod_id" ]]; then
    pod_json="$(api_request GET "pods/$pod_id")"
  else
    pod_json="$(api_request GET "pods")"
  fi

  local summary
  summary="$(printf '%s' "$pod_json" | extract_pod_summary)"
  RUNPOD_POD_ID="$(awk -F= '$1=="id"{print $2}' <<< "$summary")"
  RUNPOD_SSH_HOST="$(awk -F= '$1=="public_ip"{print $2}' <<< "$summary")"
  RUNPOD_SSH_PORT="$(awk -F= '$1=="ssh_port"{print $2}' <<< "$summary")"
  RUNPOD_POD_STATUS="$(awk -F= '$1=="status"{print $2}' <<< "$summary")"
  RUNPOD_POD_GPU="$(awk -F= '$1=="gpu"{print $2}' <<< "$summary")"
  RUNPOD_POD_COST_PER_HR="$(awk -F= '$1=="cost_per_hr"{print $2}' <<< "$summary")"
  if [[ -z "$RUNPOD_POD_ID" ]]; then
    echo "Could not resolve RunPod pod id." >&2
    exit 1
  fi
  save_state
}

require_ssh_endpoint() {
  resolve_pod
  if [[ -z "${RUNPOD_SSH_HOST:-}" || -z "${RUNPOD_SSH_PORT:-}" ]]; then
    echo "Pod $RUNPOD_POD_ID has no SSH endpoint yet. status=${RUNPOD_POD_STATUS:-unknown}" >&2
    exit 1
  fi
}

ssh_base() {
  ssh \
    -o BatchMode=yes \
    -o ConnectTimeout=15 \
    -o ServerAliveInterval=10 \
    -o ServerAliveCountMax=2 \
    -o StrictHostKeyChecking=accept-new \
    -i "$RUNPOD_SSH_KEY" \
    -p "$RUNPOD_SSH_PORT" \
    "root@$RUNPOD_SSH_HOST" \
    "$@"
}

rsync_ssh_arg() {
  printf "ssh -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new -i %q -p %q" "$RUNPOD_SSH_KEY" "$RUNPOD_SSH_PORT"
}

remote_probe_script='
set -euo pipefail
root="$1"
regex="$2"
shift 2
now="$(date +%s)"
active_count="$(ps -eo pid=,stat=,comm=,args= | grep -E "$regex" | grep -v grep | grep -v runpod_idle_guard | wc -l | tr -d " ")"
latest=0
for rel in "$@"; do
  if [[ "$rel" = /* ]]; then
    path="$rel"
  else
    path="$root/$rel"
  fi
  [[ -e "$path" ]] || continue
  mtime="$(find "$path" -xdev -type f -not -path "*/.git/*" -printf "%T@\n" 2>/dev/null | sort -nr | head -1 | cut -d. -f1 || true)"
  if [[ -n "$mtime" && "$mtime" -gt "$latest" ]]; then
    latest="$mtime"
  fi
done
if [[ "$latest" -eq 0 ]]; then
  idle_age=0
else
  idle_age=$((now - latest))
fi
printf "active_count=%s\nlatest_mtime=%s\nidle_age=%s\nnow=%s\n" "$active_count" "$latest" "$idle_age" "$now"
'

probe_remote() {
  # Word splitting is intended for REMOTE_WATCH_PATHS.
  # shellcheck disable=SC2086
  ssh_base "bash -s" -- "$REMOTE_ROOT" "$ACTIVE_PROCESS_REGEX" $REMOTE_WATCH_PATHS <<< "$remote_probe_script"
}

sync_remote_to_local() {
  mkdir -p "$LOCAL_SYNC_ROOT"
  local dest="$LOCAL_SYNC_ROOT/$RUNPOD_POD_ID"
  mkdir -p "$dest"
  log "syncing root@$RUNPOD_SSH_HOST:$REMOTE_ROOT/ -> $dest/"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "dry-run: skipped rsync down"
    return
  fi
  rsync -az --delete-delay --partial \
    --exclude '.git/' \
    --exclude '.venv*/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    -e "$(rsync_ssh_arg)" \
    "root@$RUNPOD_SSH_HOST:$REMOTE_ROOT/" "$dest/"
}

sync_local_to_remote() {
  require_ssh_endpoint
  log "syncing local paths from $LOCAL_REPO_ROOT -> root@$RUNPOD_SSH_HOST:$REMOTE_ROOT"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "dry-run: skipped rsync up"
    return
  fi
  ssh_base "mkdir -p '$REMOTE_ROOT'"
  local path
  for path in $SYNC_UP_PATHS; do
    if [[ -e "$LOCAL_REPO_ROOT/$path" ]]; then
      log "sync-up path: $path"
      rsync -az --delete-delay --partial \
        --exclude '.git/' \
        --exclude '.venv*/' \
        --exclude '__pycache__/' \
        --exclude '*.pyc' \
        -e "$(rsync_ssh_arg)" \
        "$LOCAL_REPO_ROOT/$path" "root@$RUNPOD_SSH_HOST:$REMOTE_ROOT/$(dirname "$path")/"
    else
      log "sync-up skipped missing path: $path"
    fi
  done
}

stop_pod() {
  log "stopping RunPod pod $RUNPOD_POD_ID"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "dry-run: skipped RunPod stop API"
    return
  fi
  api_request POST "pods/$RUNPOD_POD_ID/stop" >/dev/null
}

delete_pod() {
  log "deleting RunPod pod $RUNPOD_POD_ID"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "dry-run: skipped RunPod delete API"
    return
  fi
  api_request DELETE "pods/$RUNPOD_POD_ID" >/dev/null
}

send_email() {
  require_env RUNPOD_NOTIFY_EMAIL
  local subject="$1"
  local body="$2"
  log "sending email reminder to $RUNPOD_NOTIFY_EMAIL"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "dry-run: skipped email: $subject"
    return
  fi
  if [[ -n "${RUNPOD_EMAIL_COMMAND:-}" ]]; then
    printf '%s\n' "$body" | "$RUNPOD_EMAIL_COMMAND" "$subject"
  elif command -v mail >/dev/null 2>&1; then
    printf '%s\n' "$body" | mail -s "$subject" "$RUNPOD_NOTIFY_EMAIL"
  else
    log "mail command not found; cannot send email"
    return 1
  fi
}

create_payload() {
  export RUNPOD_CREATE_GPU_TYPES="${RUNPOD_CREATE_GPU_TYPES:-NVIDIA A100 80GB PCIe,NVIDIA A100-SXM4-80GB,NVIDIA H100 PCIe,NVIDIA H100 80GB HBM3}"
  export RUNPOD_CREATE_VOLUME_GB="${RUNPOD_CREATE_VOLUME_GB:-300}"
  export RUNPOD_CREATE_CONTAINER_DISK_GB="${RUNPOD_CREATE_CONTAINER_DISK_GB:-80}"
  export RUNPOD_CREATE_IMAGE="${RUNPOD_CREATE_IMAGE:-runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04}"
  export RUNPOD_CREATE_CLOUD_TYPE="${RUNPOD_CREATE_CLOUD_TYPE:-SECURE}"
  export RUNPOD_CREATE_INTERRUPTIBLE="${RUNPOD_CREATE_INTERRUPTIBLE:-false}"
  export RUNPOD_CREATE_MIN_VCPU_PER_GPU="${RUNPOD_CREATE_MIN_VCPU_PER_GPU:-8}"
  export RUNPOD_CREATE_MIN_RAM_PER_GPU="${RUNPOD_CREATE_MIN_RAM_PER_GPU:-32}"
  python3 - <<'PY'
import json
import os

gpu_types = [item.strip() for item in os.environ["RUNPOD_CREATE_GPU_TYPES"].split(",") if item.strip()]
payload = {
    "name": os.environ.get("RUNPOD_POD_NAME", "kv-cache-headroom"),
    "cloudType": os.environ.get("RUNPOD_CREATE_CLOUD_TYPE", "SECURE"),
    "computeType": "GPU",
    "gpuCount": 1,
    "gpuTypeIds": gpu_types,
    "gpuTypePriority": "custom",
    "imageName": os.environ["RUNPOD_CREATE_IMAGE"],
    "containerDiskInGb": int(os.environ["RUNPOD_CREATE_CONTAINER_DISK_GB"]),
    "volumeInGb": int(os.environ["RUNPOD_CREATE_VOLUME_GB"]),
    "volumeMountPath": "/workspace",
    "ports": ["22/tcp", "8888/http"],
    "supportPublicIp": True,
    "interruptible": os.environ.get("RUNPOD_CREATE_INTERRUPTIBLE", "false").lower() == "true",
    "locked": False,
    "minVCPUPerGPU": int(os.environ["RUNPOD_CREATE_MIN_VCPU_PER_GPU"]),
    "minRAMPerGPU": int(os.environ["RUNPOD_CREATE_MIN_RAM_PER_GPU"]),
}
if os.environ.get("RUNPOD_NETWORK_VOLUME_ID"):
    payload["networkVolumeId"] = os.environ["RUNPOD_NETWORK_VOLUME_ID"]
print(json.dumps(payload, indent=2, sort_keys=True))
PY
}

extract_created_pod_id() {
  python3 - <<'PY'
import json
import sys
data = json.load(sys.stdin)
pod_id = data.get("id") if isinstance(data, dict) else None
if not pod_id:
    raise SystemExit("RunPod create response did not include pod id")
print(pod_id)
PY
}

wait_for_ssh() {
  local deadline="$(( $(date +%s) + ${RUNPOD_CREATE_WAIT_SECONDS:-900} ))"
  while [[ "$(date +%s)" -lt "$deadline" ]]; do
    resolve_pod || true
    if [[ -n "${RUNPOD_SSH_HOST:-}" && -n "${RUNPOD_SSH_PORT:-}" ]]; then
      log "probing SSH root@$RUNPOD_SSH_HOST -p $RUNPOD_SSH_PORT"
      if ssh_base "true" >/dev/null 2>&1; then
        log "SSH ready for pod $RUNPOD_POD_ID"
        return 0
      fi
    else
      log "waiting for publicIp/portMappings. status=${RUNPOD_POD_STATUS:-unknown}"
    fi
    sleep 15
  done
  echo "Timed out waiting for SSH for pod $RUNPOD_POD_ID" >&2
  exit 1
}

bootstrap_remote() {
  require_ssh_endpoint
  log "running remote apt bootstrap"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "dry-run: skipped remote apt bootstrap"
    return
  fi
  ssh_base "export DEBIAN_FRONTEND=noninteractive; apt-get update && apt-get upgrade -y && apt-get install -y rsync sudo"
}

create_pod() {
  local payload
  payload="$(create_payload)"
  echo "About to create this RunPod pod:"
  echo "$payload"
  echo
  echo "This will start billing immediately if creation succeeds."
  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "dry-run: skipped create"
    return
  fi
  if [[ ! -t 0 ]]; then
    echo "--create requires an interactive terminal for confirmation." >&2
    exit 2
  fi
  read -r -p "Type CREATE to create this pod: " answer
  if [[ "$answer" != "CREATE" ]]; then
    echo "Creation cancelled."
    exit 1
  fi
  local response
  response="$(api_request POST "pods" "$payload")"
  RUNPOD_POD_ID="$(printf '%s' "$response" | extract_created_pod_id)"
  ACTIVE_POD_ID="$RUNPOD_POD_ID"
  save_state
  log "created pod $RUNPOD_POD_ID"
  wait_for_ssh
  bootstrap_remote
  sync_local_to_remote
  echo "Created and bootstrapped pod $RUNPOD_POD_ID"
  echo "SSH: ssh root@$RUNPOD_SSH_HOST -p $RUNPOD_SSH_PORT -i $RUNPOD_SSH_KEY"
}

terminate_pod() {
  resolve_pod

  echo "About to permanently delete this RunPod pod:"
  printf '  pod_id: %s\n' "$RUNPOD_POD_ID"
  printf '  status: %s\n' "${RUNPOD_POD_STATUS:-unknown}"
  printf '  gpu: %s\n' "${RUNPOD_POD_GPU:-unknown}"
  printf '  cost_per_hr: %s\n' "${RUNPOD_POD_COST_PER_HR:-unknown}"
  if [[ -n "${RUNPOD_SSH_HOST:-}" && -n "${RUNPOD_SSH_PORT:-}" ]]; then
    printf '  ssh: root@%s -p %s\n' "$RUNPOD_SSH_HOST" "$RUNPOD_SSH_PORT"
  else
    printf '  ssh: unavailable\n'
  fi
  echo
  echo "This is destructive. RunPod deletion can remove pod-attached data that was not synced or stored on a network volume."
  echo "Expected confirmation: DELETE $RUNPOD_POD_ID"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "dry-run: skipped terminate confirmation, sync, stop, and delete"
    return
  fi
  if [[ ! -t 0 ]]; then
    echo "--terminate requires an interactive terminal for confirmation." >&2
    exit 2
  fi

  local answer
  read -r -p "Type DELETE $RUNPOD_POD_ID to terminate this pod: " answer
  if [[ "$answer" != "DELETE $RUNPOD_POD_ID" ]]; then
    echo "Termination cancelled."
    exit 1
  fi

  local now
  now="$(date +%s)"
  if [[ "${RUNPOD_TERMINATE_SYNC_BEFORE_DELETE:-1}" != "0" ]]; then
    if [[ -n "${RUNPOD_SSH_HOST:-}" && -n "${RUNPOD_SSH_PORT:-}" ]]; then
      if sync_remote_to_local; then
        LAST_SYNC_AT="$now"
        save_state
      else
        echo "Sync failed; refusing to delete. Set RUNPOD_TERMINATE_SYNC_BEFORE_DELETE=0 only if you accept data loss risk." >&2
        exit 1
      fi
    else
      echo "No SSH endpoint is available, so sync-before-delete cannot run." >&2
      echo "Set RUNPOD_TERMINATE_SYNC_BEFORE_DELETE=0 and rerun if you still want to delete this stopped/unreachable pod." >&2
      exit 1
    fi
  fi

  stop_pod || true
  delete_pod
  rm -f "$STATE_FILE"
  echo "Deleted RunPod pod $RUNPOD_POD_ID"
}

guard_mode() {
  require_env RUNPOD_NOTIFY_EMAIL
  mkdir -p "$LOCAL_SYNC_ROOT"
  if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    log "another guard is already running"
    exit 0
  fi
  trap 'rmdir "$LOCK_DIR"' EXIT

  load_state
  local now probe active_count idle_age inactive_for
  now="$(date +%s)"

  if ! resolve_pod 2>>"$LOG_FILE"; then
    if [[ -n "${STOPPED_AT:-}" && -n "${INACTIVE_SINCE:-}" ]]; then
      active_count="0"
      idle_age="$((now - INACTIVE_SINCE))"
      log "could not resolve stopped pod; using local idle state idle_age=${idle_age}s"
    else
      log "could not resolve pod before stop; leaving state unchanged"
      exit 1
    fi
  elif [[ -z "${RUNPOD_SSH_HOST:-}" || -z "${RUNPOD_SSH_PORT:-}" ]]; then
    if [[ -n "${STOPPED_AT:-}" && -n "${INACTIVE_SINCE:-}" ]]; then
      active_count="0"
      idle_age="$((now - INACTIVE_SINCE))"
      log "pod has no SSH endpoint after stop; using local idle state idle_age=${idle_age}s"
    else
      log "pod has no SSH endpoint yet. status=${RUNPOD_POD_STATUS:-unknown}"
      exit 1
    fi
  elif probe="$(probe_remote 2>>"$LOG_FILE")"; then
    active_count="$(awk -F= '$1=="active_count"{print $2}' <<< "$probe")"
    idle_age="$(awk -F= '$1=="idle_age"{print $2}' <<< "$probe")"
    now="$(awk -F= '$1=="now"{print $2}' <<< "$probe")"
    log "probe pod=$RUNPOD_POD_ID active_count=$active_count idle_age=${idle_age}s"
  else
    if [[ -n "${STOPPED_AT:-}" && -n "${INACTIVE_SINCE:-}" ]]; then
      active_count="0"
      idle_age="$((now - INACTIVE_SINCE))"
      log "probe failed after pod stop; using local idle state idle_age=${idle_age}s"
    else
      log "probe failed before pod was stopped; leaving state unchanged"
      exit 1
    fi
  fi

  if [[ "$active_count" != "0" ]]; then
    INACTIVE_SINCE=""
    STOPPED_AT=""
    EMAILED_AT=""
    save_state
    log "active processes found; reset idle state"
    exit 0
  fi

  if [[ -z "${INACTIVE_SINCE:-}" ]]; then
    INACTIVE_SINCE="$((now - idle_age))"
  fi

  inactive_for="$((now - INACTIVE_SINCE))"
  log "inactive_for=${inactive_for}s stop_threshold=${IDLE_STOP_SECONDS}s email_threshold=${IDLE_EMAIL_SECONDS}s"

  if [[ -z "${STOPPED_AT:-}" && "$inactive_for" -ge "$IDLE_STOP_SECONDS" ]]; then
    sync_remote_to_local
    LAST_SYNC_AT="$now"
    stop_pod
    STOPPED_AT="$now"
    save_state
  fi

  if [[ -z "${EMAILED_AT:-}" && "$inactive_for" -ge "$IDLE_EMAIL_SECONDS" ]]; then
    local body
    body="RunPod pod $RUNPOD_POD_ID has been inactive for $((inactive_for / 60)) minutes.

I already attempted to sync $REMOTE_ROOT to:
  $LOCAL_SYNC_ROOT/$RUNPOD_POD_ID

I also attempted to stop the pod after $((IDLE_STOP_SECONDS / 60)) minutes.

Please open RunPod and terminate the pod if you no longer need it. Stopped pods can still accrue storage charges."
    send_email "Terminate idle RunPod pod $RUNPOD_POD_ID" "$body"
    EMAILED_AT="$now"
    save_state
  fi

  save_state
}

case "$MODE" in
  create)
    create_pod
    ;;
  bootstrap)
    bootstrap_remote
    ;;
  sync-up)
    sync_local_to_remote
    ;;
  resolve-ssh)
    require_ssh_endpoint
    printf 'RUNPOD_POD_ID=%s\nRUNPOD_SSH_HOST=%s\nRUNPOD_SSH_PORT=%s\nRUNPOD_POD_STATUS=%s\nRUNPOD_POD_GPU=%s\nRUNPOD_POD_COST_PER_HR=%s\n' \
      "$RUNPOD_POD_ID" "$RUNPOD_SSH_HOST" "$RUNPOD_SSH_PORT" "${RUNPOD_POD_STATUS:-}" "${RUNPOD_POD_GPU:-}" "${RUNPOD_POD_COST_PER_HR:-}"
    ;;
  terminate)
    terminate_pod
    ;;
  guard)
    guard_mode
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    exit 2
    ;;
esac
