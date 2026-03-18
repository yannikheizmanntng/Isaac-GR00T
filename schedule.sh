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
REMOTE_DATASETS="/home/innovation-hacking/heizmany/datasets"
REMOTE_MODELS="/home/innovation-hacking/heizmany/models"

# --------------------------
# Runs
# --------------------------

run_datagen "datagen_a_exp1" "$EXPERIMENT_1/eval_a_short_experiment_1.yaml" "$DATASETS/eval_a_short_experiment_1"
# run_datagen "datagen_b_exp1" "$EXPERIMENT_1/eval_b_short_experiment_1.yaml" "$DATASETS/eval_b_short_experiment_1" "datagen_a_exp1"
# run_datagen "datagen_d_exp1" "$EXPERIMENT_1/eval_d_short_experiment_1.yaml" "$DATASETS/eval_d_short_experiment_1" "datagen_b_exp1"
wait_screen_gone "datagen_a_exp1"


echo "[scheduler] all runs complete."


# screen -S scheduler bash -lc "./schedule.sh 2>&1 | tee logs/scheduler.log"
