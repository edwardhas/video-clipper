#!/usr/bin/env bash
# Restore a backup: scripts/restore.sh --from /path/to/clipper-backup-YYYYmmdd-HHMMSS[.tar.gz] [--yes]
# Refuses while the app runs; the current data is moved to data.bak-<timestamp>, never deleted.
set -euo pipefail
# shellcheck source=scripts/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
require_venv
clipper_cli restore "$@"
