#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/runpod_idle_guard.sh [--dry-run] [--env PATH]

Cron-friendly Mac-side RunPod idle guard.

Behavior:
  - Detects remote inactivity from process state + file mtimes.
  - After RUNPOD_IDLE_STOP_SECONDS, rsyncs remote data locally and stops the pod.
  - After RUNPOD_IDLE_EMAIL_SECONDS, sends one email reminder to terminate the pod.

Required env:
  RUNPOD_API_KEY
  RUNPOD_POD_ID
  RUNPOD_SSH_HOST
  RUNPOD_SSH_PORT
  RUNPOD_SSH_KEY
  RUNPOD_REMOTE_ROOT
  RUNPOD_LOCAL_SYNC_ROOT
  RUNPOD_NOTIFY_EMAIL

Optional env:
  RUNPOD_REMOTE_WATCH_PATHS      Space-separated paths relative to RUNPOD_REMOTE_ROOT.
                                 Default: "."
  RUNPOD_ACTIVE_PROCESS_REGEX    Default matches python/vllm/nsys/headroom jobs.
  RUNPOD_IDLE_STOP_SECONDS       Default: 3600
  RUNPOD_IDLE_EMAIL_SECONDS      Default: 7200
  RUNPOD_STATE_DIR               Default: "$HOME/.runpod_idle_guard"
  RUNPOD_EMAIL_COMMAND           Optional command. Receives subject as $1 and body on stdin.
EOF
}

DRY_RUN=0
ENV_FILE="${RUNPOD_IDLE_GUARD_ENV:-}"
while [[ $# -gt 0 ]]; do
  case "$1" in
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

for name in \
  RUNPOD_API_KEY \
  RUNPOD_POD_ID \
  RUNPOD_SSH_HOST \
  RUNPOD_SSH_PORT \
  RUNPOD_SSH_KEY \
  RUNPOD_REMOTE_ROOT \
  RUNPOD_LOCAL_SYNC_ROOT \
  RUNPOD_NOTIFY_EMAIL
do
  require_env "$name"
done

REMOTE_WATCH_PATHS="${RUNPOD_REMOTE_WATCH_PATHS:-.}"
ACTIVE_PROCESS_REGEX="${RUNPOD_ACTIVE_PROCESS_REGEX:-python|nsys|vllm|VLLM::EngineCore|run_.*headroom|run_oracle0|run_baseline|run_marginal}"
IDLE_STOP_SECONDS="${RUNPOD_IDLE_STOP_SECONDS:-3600}"
IDLE_EMAIL_SECONDS="${RUNPOD_IDLE_EMAIL_SECONDS:-7200}"
STATE_DIR="${RUNPOD_STATE_DIR:-$HOME/.runpod_idle_guard}"
STATE_FILE="$STATE_DIR/${RUNPOD_POD_ID}.state"
LOCK_DIR="$STATE_DIR/${RUNPOD_POD_ID}.lock"
LOG_FILE="$STATE_DIR/${RUNPOD_POD_ID}.log"

mkdir -p "$STATE_DIR" "$RUNPOD_LOCAL_SYNC_ROOT"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "$(date -u +%FT%TZ) another guard is already running" >> "$LOG_FILE"
  exit 0
fi
trap 'rmdir "$LOCK_DIR"' EXIT

log() {
  echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG_FILE"
}

ssh_base=(
  ssh
  -o BatchMode=yes
  -o ConnectTimeout=15
  -o ServerAliveInterval=10
  -o ServerAliveCountMax=2
  -o StrictHostKeyChecking=accept-new
  -i "$RUNPOD_SSH_KEY"
  -p "$RUNPOD_SSH_PORT"
  "root@$RUNPOD_SSH_HOST"
)

rsync_base=(
  rsync
  -az
  --delete-delay
  --partial
  --exclude '.git/'
  --exclude '.venv*/'
  --exclude '__pycache__/'
  --exclude '*.pyc'
  -e "ssh -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new -i '$RUNPOD_SSH_KEY' -p '$RUNPOD_SSH_PORT'"
)

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
  "${ssh_base[@]}" "bash -s" -- "$RUNPOD_REMOTE_ROOT" "$ACTIVE_PROCESS_REGEX" $REMOTE_WATCH_PATHS <<< "$remote_probe_script"
}

load_state() {
  INACTIVE_SINCE=""
  STOPPED_AT=""
  EMAILED_AT=""
  LAST_SYNC_AT=""
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
  } > "$STATE_FILE"
}

sync_remote() {
  local dest="$RUNPOD_LOCAL_SYNC_ROOT/${RUNPOD_POD_ID}"
  mkdir -p "$dest"
  log "syncing root@$RUNPOD_SSH_HOST:$RUNPOD_REMOTE_ROOT/ -> $dest/"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "dry-run: skipped rsync"
    return
  fi
  "${rsync_base[@]}" "root@$RUNPOD_SSH_HOST:$RUNPOD_REMOTE_ROOT/" "$dest/"
}

stop_pod() {
  log "stopping RunPod pod $RUNPOD_POD_ID"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "dry-run: skipped RunPod stop API"
    return
  fi
  curl --fail --silent --show-error \
    --request POST \
    --url "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID/stop" \
    --header "Authorization: Bearer $RUNPOD_API_KEY" >/dev/null
}

send_email() {
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

load_state
now="$(date +%s)"
if probe="$(probe_remote 2>>"$LOG_FILE")"; then
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
  sync_remote
  LAST_SYNC_AT="$now"
  stop_pod
  STOPPED_AT="$now"
  save_state
fi

if [[ -z "${EMAILED_AT:-}" && "$inactive_for" -ge "$IDLE_EMAIL_SECONDS" ]]; then
  body="RunPod pod $RUNPOD_POD_ID has been inactive for $((inactive_for / 60)) minutes.

I already attempted to sync $RUNPOD_REMOTE_ROOT to:
  $RUNPOD_LOCAL_SYNC_ROOT/$RUNPOD_POD_ID

I also attempted to stop the pod after $((IDLE_STOP_SECONDS / 60)) minutes.

Please open RunPod and terminate the pod if you no longer need it. Stopped pods can still accrue storage charges."
  send_email "Terminate idle RunPod pod $RUNPOD_POD_ID" "$body"
  EMAILED_AT="$now"
  save_state
fi

save_state
