#!/usr/bin/env bash
# Install or update everything Local Clipper needs, then run the doctor.
# Safe to run again at any time (idempotent). Never uses sudo or changes system settings.
set -euo pipefail
# shellcheck source=scripts/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

UV_VERSION="0.12.23"
YUNET_SHA256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
VENV="$CLIPPER_ROOT/backend/.venv"

usage() {
  cat <<'EOF'
Usage: scripts/setup.sh [--prod] [--skip-frontend]

  --prod            install runtime dependencies only (backend/requirements.lock)
  --skip-frontend   do not install or build the web interface

Environment: CLIPPER_PYTHON=/path/to/python3.12 to choose the Python interpreter.
EOF
}

PROD=0
SKIP_FRONTEND=0
for arg in "$@"; do
  case "$arg" in
    --prod) PROD=1 ;;
    --skip-frontend) SKIP_FRONTEND=1 ;;
    -h | --help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done

step() { printf '\n==> %s\n' "$*"; }

is_python312() {
  "$1" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)' >/dev/null 2>&1
}

find_python312() {
  local candidate
  local -a candidates=()
  if [[ -n "${CLIPPER_PYTHON:-}" ]]; then candidates+=("$CLIPPER_PYTHON"); fi
  if command -v python3.12 >/dev/null 2>&1; then candidates+=("$(command -v python3.12)"); fi
  for candidate in "$HOME"/.pyenv/versions/3.12.*/bin/python3.12; do
    [[ -x "$candidate" ]] && candidates+=("$candidate")
  done
  for candidate in ${candidates[@]+"${candidates[@]}"}; do
    if is_python312 "$candidate"; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

# --------------------------------------------------------------------------- Python
step "Python environment"
if [[ -x "$VENV/bin/python" ]]; then
  is_python312 "$VENV/bin/python" ||
    die "backend/.venv uses a Python other than 3.12. Move it away (mv backend/.venv backend/.venv.old) and run this again."
fi
# Reuse the venv only when it can install packages. A venv left half-created by an earlier run
# (Debian/Ubuntu without python3.12-venv: bin/python exists, pip does not) is recreated; nothing
# is lost because the locked sync below makes its contents match the lock file exactly.
if [[ -x "$VENV/bin/python" ]] && { [[ -x "$VENV/bin/uv" ]] || "$VENV/bin/python" -m pip --version >/dev/null 2>&1; }; then
  echo "Using the existing backend/.venv ($("$VENV/bin/python" -V 2>&1))"
else
  PYTHON312="$(find_python312)" ||
    die "Python 3.12 was not found. Install it (pyenv install 3.12, or brew install python@3.12) or set CLIPPER_PYTHON=/path/to/python3.12, then run this again."
  if [[ -e "$VENV" ]]; then
    echo "backend/.venv is incomplete (no pip or uv); recreating it with $("$PYTHON312" -V 2>&1)"
  else
    echo "Creating backend/.venv with $("$PYTHON312" -V 2>&1)"
  fi
  "$PYTHON312" -m venv --clear "$VENV" ||
    die "Could not create backend/.venv. On Debian/Ubuntu install the venv module first (as root: apt install python3.12-venv), then run this again."
fi

if [[ ! -x "$VENV/bin/uv" ]]; then
  echo "Installing uv $UV_VERSION into backend/.venv"
  "$VENV/bin/python" -m pip install --disable-pip-version-check --quiet "uv==$UV_VERSION"
fi

LOCK="requirements-dev.lock"
if ((PROD)); then LOCK="requirements.lock"; fi
echo "Installing locked dependencies from backend/$LOCK (hash-checked)"
"$VENV/bin/uv" pip sync --quiet --python "$VENV/bin/python" --require-hashes "$CLIPPER_ROOT/backend/$LOCK"

# --------------------------------------------------------------------------- frontend
if ((SKIP_FRONTEND)); then
  step "Web interface (skipped)"
else
  step "Web interface"
  command -v node >/dev/null 2>&1 && command -v npm >/dev/null 2>&1 ||
    die "Node.js and npm are needed to build the web interface (Node 20.19+ or 22.12+; macOS: brew install node)."
  node -e 'const [a, b] = process.versions.node.split(".").map(Number);
           process.exit((a === 20 && b >= 19) || (a === 22 && b >= 12) || a >= 23 ? 0 : 1)' ||
    die "Node.js $(node -v) is too old for the build tools; install Node 22 LTS or newer."
  (
    cd "$CLIPPER_ROOT/frontend"
    if [[ -f package-lock.json ]]; then
      npm ci --no-audit --no-fund
    else
      npm install --no-audit --no-fund
    fi
    npm run build
  )
fi

# --------------------------------------------------------------------------- assets
step "Bundled assets"
MODEL="$CLIPPER_ROOT/backend/clipper/assets/models/face_detection_yunet_2023mar.onnx"
[[ -f "$MODEL" ]] || die "The face detection model is missing (backend/clipper/assets/models). Restore it from the repository."
[[ "$(sha256_of "$MODEL")" == "$YUNET_SHA256" ]] ||
  die "The face detection model has an unexpected checksum (damaged or replaced). Restore it from the repository."
echo "Face detection model: checksum OK"
for font in Montserrat-ExtraBold.ttf Montserrat-Black.ttf OFL.txt; do
  [[ -f "$CLIPPER_ROOT/backend/clipper/assets/fonts/$font" ]] ||
    die "Caption font file $font is missing (backend/clipper/assets/fonts). Restore it from the repository."
done
echo "Caption fonts: present"

# --------------------------------------------------------------------------- config + data
step "Configuration"
if [[ -f "$CLIPPER_ROOT/.env" ]]; then
  echo "Keeping your existing .env (not modified)."
else
  cp "$CLIPPER_ROOT/.env.example" "$CLIPPER_ROOT/.env"
  chmod 600 "$CLIPPER_ROOT/.env"
  echo "Created .env from .env.example. Add your OPENAI_API_KEY there before analysing videos."
fi

if load_paths 2>/dev/null; then
  mkdir -p "$DATA_DIR"
  echo "Data folder: $DATA_DIR"
else
  echo "Could not read the configuration yet; the doctor below explains why."
fi

# --------------------------------------------------------------------------- doctor
step "Doctor"
if "$CLIPPER_ROOT/scripts/doctor.sh"; then
  printf '\nSetup complete. Start Local Clipper with: scripts/start.sh\n'
else
  printf '\nSetup finished, but the doctor reported problems (FAIL lines above). Fix them and run scripts/doctor.sh.\n' >&2
  exit 1
fi
