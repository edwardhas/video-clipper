#!/usr/bin/env bash
# Check that everything Local Clipper needs is installed and configured (PASS/WARN/FAIL).
# Exits non-zero when any check FAILs.
set -euo pipefail
# shellcheck source=scripts/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
require_venv
clipper_cli doctor "$@"
