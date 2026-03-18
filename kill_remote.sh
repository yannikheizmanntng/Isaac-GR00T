#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/jobs_remote.sh"

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 <job_name>"
  exit 1
fi

kill_screen_remote "$1"
