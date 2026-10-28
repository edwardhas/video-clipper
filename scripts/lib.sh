# shellcheck shell=bash
# Shared helpers for scripts/*.sh. Sourced, never executed directly.
# Compatible with macOS's bash 3.2 and Linux bash 4/5. Never prints secrets.

# An exported CDPATH (e.g. ".:~") makes `cd scripts/..` print the directory as well, which
# would corrupt CLIPPER_ROOT; none of these scripts relies on it.
unset CDPATH
CLIPPER_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CLIPPER_VENV_PY="$CLIPPER_ROOT/backend/.venv/bin/python"

die() {
  printf '%s\n' "$*" >&2
  exit 1
}

require_venv() {
  [[ -x "$CLIPPER_VENV_PY" ]] || die "The Python environment (backend/.venv) is missing. Run scripts/setup.sh first."
}

# Run the venv's Python with the backend on the import path. -P: never import from the
# current directory, so the scripts can be run from anywhere with relative path arguments.
clipper_py() {
  PYTHONPATH="$CLIPPER_ROOT/backend${PYTHONPATH:+:$PYTHONPATH}" "$CLIPPER_VENV_PY" -P "$@"
}

clipper_cli() {
  clipper_py -m clipper.cli "$@"
}

# Sets DATA_DIR, LOG_DIR, RUN_DIR, PID_FILE, CHILDREN_FILE and APP_URL from the configuration.
# Returns non-zero when the configuration cannot be read.
load_paths() {
  local key value
  while IFS='=' read -r key value; do
    case "$key" in
      data_dir) DATA_DIR="$value" ;;
      log_dir) LOG_DIR="$value" ;;
      run_dir) RUN_DIR="$value" ;;
      pid_file) PID_FILE="$value" ;;
      children_file) CHILDREN_FILE="$value" ;;
      url) APP_URL="$value" ;;
    esac
  done < <(clipper_cli paths)
  [[ -n "${PID_FILE:-}" && -n "${DATA_DIR:-}" ]]
}

CONFIG_UNREADABLE="Could not read the configuration (.env). Run scripts/doctor.sh for details."

# True when $1 is a live process whose command line contains $2. Without `ps`
# (minimal containers) a live process is assumed to match.
proc_matches() {
  local pid="$1" pattern="$2" cmd
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  command -v ps >/dev/null 2>&1 || return 0
  cmd="$(ps -o command= -p "$pid" 2>/dev/null || true)"
  [[ "$cmd" == *"$pattern"* ]]
}

read_pid_file() {
  [[ -f "$1" ]] || return 0
  tr -dc '0-9' <"$1"
}

wait_gone() { # wait_gone SECONDS PID...
  local seconds="$1" i pid alive
  shift
  for ((i = 0; i < seconds * 2; i++)); do
    alive=0
    for pid in "$@"; do
      if kill -0 "$pid" 2>/dev/null; then alive=1; fi
    done
    ((alive)) || return 0
    sleep 0.5
  done
  return 1
}

# Children left behind by a supervisor that was killed (each leads its own process group).
# Needs load_paths. Only signals PIDs listed in this run dir's children.pids whose command line
# shows they belong to Local Clipper; call it only when no supervisor of this run dir is alive.
stop_leftover_children() {
  [[ -f "$CHILDREN_FILE" ]] || return 0
  local name pid
  local -a pids=()
  while read -r name pid; do
    if proc_matches "${pid:-}" clipper; then
      echo "Stopping leftover $name process ($pid)…"
      kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
      pids+=("$pid")
    fi
  done <"$CHILDREN_FILE"
  if ((${#pids[@]})) && ! wait_gone 25 "${pids[@]}"; then
    for pid in "${pids[@]}"; do
      kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
    done
  fi
  rm -f "$CHILDREN_FILE"
}

sha256_of() {
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  elif command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    clipper_py -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$1"
  fi
}
