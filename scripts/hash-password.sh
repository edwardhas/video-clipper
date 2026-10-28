#!/usr/bin/env bash
# Print an argon2 hash of a new owner password for CLIPPER_OWNER_PASSWORD_HASH.
# Asks twice (input hidden). For automation: printf '%s\n' "$PASSWORD" | scripts/hash-password.sh --stdin
set -euo pipefail
# shellcheck source=scripts/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
require_venv
clipper_cli hash-password "$@"
