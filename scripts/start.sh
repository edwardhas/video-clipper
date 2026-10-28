#!/usr/bin/env bash
# Start Local Clipper (web server + worker). Runs in the foreground by default (Ctrl+C stops it);
# --detach runs it in the background and returns once it answers. Secrets are read by the
# app itself from .env; this script never reads or prints them.
set -euo pipefail
# shellcheck source=scripts/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

usage() {
  cat <<'EOF'
Usage: scripts/start.sh [--detach] [--no-browser]

  --detach      run in the background (logs in data/logs/); stop with scripts/stop.sh
  --no-browser  do not open a browser window
EOF
}

DETACH=0
SUP_ARGS=()
for arg in "$@"; do
  case "$arg" in
    -d | --detach) DETACH=1 ;;
    --no-browser) SUP_ARGS+=(--no-browser) ;;
    -h | --help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done

require_venv
export PYTHONPATH="$CLIPPER_ROOT/backend${PYTHONPATH:+:$PYTHONPATH}"

if ((!DETACH)); then
  exec "$CLIPPER_VENV_PY" -P -m clipper.supervisor ${SUP_ARGS[@]+"${SUP_ARGS[@]}"}
fi

load_paths || die "$CONFIG_UNREADABLE"
existing="$(read_pid_file "$PID_FILE")"
if [[ -n "$existing" ]] && proc_matches "$existing" clipper.supervisor; then
  echo "Local Clipper is already running at $APP_URL (process $existing)."
  echo "Stop with: scripts/stop.sh"
  exit 0
fi
# A supervisor that was killed (crash, OOM) leaves its api and worker running, and that api still
# answers. They are listed in this run dir: stop them, as the supervisor's own start does.
stop_leftover_children
# Something answers although the PID file of the current settings does not point at it, e.g. a
# copy started before CLIPPER_DATA_DIR was changed. Never report that copy as this start.
if clipper_cli health --quiet; then
  die "Another copy of Local Clipper is already answering at $APP_URL, but $PID_FILE does not point to it.
Stop that copy first. If CLIPPER_DATA_DIR was changed while it was running, set it back, run scripts/stop.sh, then change it again."
fi

mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/supervisor.log"
nohup "$CLIPPER_VENV_PY" -P -m clipper.supervisor --detached ${SUP_ARGS[@]+"${SUP_ARGS[@]}"} \
  >>"$LOG" 2>&1 </dev/null &
SUP_PID=$!
echo "Starting Local Clipper in the background (process $SUP_PID)…"

# Success only when the new supervisor is the one answering: it is still alive and wrote its
# PID file (it does so before starting the web server).
if clipper_cli health --wait 45 --pid "$SUP_PID" --quiet && kill -0 "$SUP_PID" 2>/dev/null &&
  [[ "$(read_pid_file "$PID_FILE")" == "$SUP_PID" ]]; then
  echo "Local Clipper is running at $APP_URL"
  echo "Stop with: scripts/stop.sh"
  echo "Logs: $LOG_DIR"
  exit 0
fi

echo "Local Clipper did not start. Last lines of $LOG:" >&2
tail -n 25 "$LOG" >&2 || true
if kill -0 "$SUP_PID" 2>/dev/null; then
  kill -TERM "$SUP_PID" 2>/dev/null || true
fi
exit 1
