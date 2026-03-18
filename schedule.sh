#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/jobs_local.sh"
source "$SCRIPT_DIR/jobs_remote.sh"

mkdir -p "$IG_LOGS"

# --------------------------
# Paths
# --------------------------

EXPERIMENT_1="/home/innovation-hacking/heizmany/ur5_chess/source/ur5_chess/ur5_chess/evaluation/benchmarks/eval/experiments/experiment_1"
DATASETS="/home/innovation-hacking/heizmany/ur5_chess/datasets"
MODELS="/home/innovation-hacking/heizmany/Isaac-GR00T/models"

# --------------------------
# Runs
# --------------------------

run_finetune_remote "test_finetune" false 80000 "$REMOTE_DATASETS/dataset_20260311_131933/lerobot"
wait_screen_gone_remote "test_finetune"


echo "[scheduler] all runs complete."


# screen -S scheduler bash -lc "./schedule.sh 2>&1 | tee logs/scheduler.log"
