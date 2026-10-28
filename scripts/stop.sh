#!/usr/bin/env bash
# Stop Local Clipper started with scripts/start.sh. Idempotent: safe when it is not running.
# Only signals processes whose command line shows they belong to Local Clipper, so a stale
# PID file (e.g. after a reboot) never stops an unrelated program.
set -euo pipefail
# shellcheck source=scripts/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

STOP_WAIT_SECONDS=30

case "${1:-}" in
  "") ;;
  -h | --help) echo "Usage: scripts/stop.sh"; exit 0 ;;
  *) echo "Usage: scripts/stop.sh" >&2; exit 2 ;;
esac

require_venv
load_paths || die "$CONFIG_UNREADABLE"

pid="$(read_pid_file "$PID_FILE")"
if [[ -z "$pid" ]] || ! proc_matches "$pid" clipper.supervisor; then
  [[ -f "$PID_FILE" ]] && rm -f "$PID_FILE"
  stop_leftover_children
  # The run files follow the current settings: a copy started before CLIPPER_DATA_DIR was
  # changed keeps answering but is not recorded here. Never call that "not running".
  if clipper_cli health --quiet; then
    echo "No running Local Clipper is recorded in $PID_FILE, but one is still answering at $APP_URL." >&2
    echo "Was CLIPPER_DATA_DIR changed while it was running? Set it back to the folder it was started with and run scripts/stop.sh again." >&2
    exit 1
  fi
  echo "Local Clipper is not running."
  exit 0
fi

echo "Stopping Local Clipper (process $pid)… (the worker may take up to 20 s to pause its job)"
kill -TERM "$pid" 2>/dev/null || true
if ! wait_gone "$STOP_WAIT_SECONDS" "$pid"; then
  echo "It did not stop within $STOP_WAIT_SECONDS s; forcing it to stop." >&2
  pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
  if [[ "$pgid" == "$pid" ]]; then
    kill -KILL -- "-$pid" 2>/dev/null || true # its own process group (started with --detach)
  else
    kill -KILL "$pid" 2>/dev/null || true
  fi
  wait_gone 5 "$pid" || true
fi
stop_leftover_children
if [[ "$(read_pid_file "$PID_FILE")" == "$pid" ]]; then rm -f "$PID_FILE"; fi
echo "Local Clipper stopped."
