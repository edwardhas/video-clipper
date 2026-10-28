#!/usr/bin/env bash
# Back up the database and projects: scripts/backup.sh --out /path/to/backups [--archive]
# [--include-uploads] [--include-cache] [--force]. Run it while no job is processing or rendering.
set -euo pipefail
# shellcheck source=scripts/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
require_venv
clipper_cli backup "$@"
