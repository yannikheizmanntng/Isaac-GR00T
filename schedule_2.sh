#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/jobs_local.sh"
source "$SCRIPT_DIR/jobs_remote.sh"

mkdir -p "$IG_LOGS"

# --------------------------
# Paths
# --------------------------

DATASETS_LOCAL="/home/innovation-hacking/heizmany/ur5_chess/datasets"

# --------------------------
# Pre-launch datagen_2 immediately — waits internally for datagen to disappear.
# --------------------------

run_datagen "datagen_3" "$DATASETS_LOCAL/dataset_3" "datagen_2"

# --------------------------
# Step 1: wait for datagen_1, then copy dataset_1 to remote
# --------------------------

wait_screen_gone "datagen_3"

#rsync_to_remote "$DATASETS_LOCAL/dataset_1" "~/heizmany/datasets/dataset_1"

# --------------------------
# Step 2: start finetune_1 (dataset_1, resume from model_1)
# --------------------------

# wait_for_job_server
# run_finetune_remote "finetune_1" false 40000 \
#   "$REMOTE_DATASETS/dataset_1/lerobot" \
#   "$REMOTE_MODELS/model_1" \
#   "" \
#   "save_steps=10000"

# --------------------------
# Step 3: wait for datagen_2, copy dataset_2, start datagen_3
# # --------------------------

# wait_screen_gone "datagen_2"
# # run_datagen "datagen_3" "$DATASETS_LOCAL/dataset_3"
# rsync_to_remote "$DATASETS_LOCAL/dataset_2" "~/heizmany/datasets/dataset_2"

# # --------------------------
# # Step 4: wait for finetune_1, start finetune_2 (dataset_2, resume from model_2)
# # --------------------------

# # wait_screen_gone_remote "finetune_1"

# wait_for_job_server
# run_finetune_remote "finetune_2" false 40000 \
#   "$REMOTE_DATASETS/dataset_2/lerobot" \
#   "$REMOTE_MODELS/model_2" \
#   "" \
#   "save_steps=10000" \
#   "1"

# # --------------------------
# # Step 5: wait for datagen_3
# # --------------------------

# wait_screen_gone "datagen_3"

echo "[scheduler] all runs complete."


# screen -S scheduler bash -lc "./schedule.sh 2>&1 | tee logs/scheduler.log"

# curl http://localhost:8765/jobs
