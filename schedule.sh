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

run_datagen "datagen_a_exp1" "$EXPERIMENT_1/eval_a_short_experiment_1.yaml" "$DATASETS/eval_a_short_experiment_1"
run_datagen "datagen_b_exp1" "$EXPERIMENT_1/eval_b_short_experiment_1.yaml" "$DATASETS/eval_b_short_experiment_1" "datagen_a_exp1"
# run_datagen "datagen_d_exp1" "$EXPERIMENT_1/eval_d_short_experiment_1.yaml" "$DATASETS/eval_d_short_experiment_1" "datagen_b_exp1"
wait_screen_gone "datagen_b_exp1"


# --------------------------
# Old / archived runs
# --------------------------

# run_benchmark "a" "PartAShort" "$MODELS/GR00T-N1.5-3B_20260316-110000" "$EXPERIMENT_1" "inf"
# run_benchmark "b" "PartBShort" "$MODELS/GR00T-N1.5-3B_20260316-110000" "$EXPERIMENT_1" "inf" "a"
# run_benchmark "d" "PartDShort" "$MODELS/GR00T-N1.5-3B_20260316-110000" "$EXPERIMENT_1" "inf" "b"
#
# run_finetune "single_prompt" false 80000 \
#   "$DATASETS/dataset_20260316_100000/lerobot" \
#   "$MODELS/GR00T-N1.5-3B_20260316-110000"
# wait_screen_gone "single_prompt"
#
# run_inference_server "inf" "$MODELS/GR00T-N1.5-3B_20260316-110000"
# wait_screen_gone "d"
# kill_screen "inf" 60

# run_benchmark "a_ia1a"  "ShortFieldIdHomoBlack" "$MODELS/GR00T-N1.5-3B_20260304-200000" "" "inf_a"
# run_benchmark "a_ia1b"  "ShortFieldIdHomoMixed" "$MODELS/GR00T-N1.5-3B_20260304-200000" "" "inf_a" "a_ia1a"
# run_benchmark "b_ia1a"  "ShortFieldIdHomoBlack" "$MODELS/GR00T-N1.5-3B_20260308-101133" "" "inf_b"
# run_benchmark "b_ia1b"  "ShortFieldIdHomoMixed" "$MODELS/GR00T-N1.5-3B_20260308-101133" "" "inf_b" "b_ia1a"
# run_inference_server "inf_a" "$MODELS/GR00T-N1.5-3B_20260304-200000"
# wait_screen_gone "a_ia1b"
# kill_screen "inf_a" 60
# run_inference_server "inf_b" "$MODELS/GR00T-N1.5-3B_20260308-101133"
# wait_screen_gone "b_ia1a"
# kill_screen "inf_b" 60

# run_finetune "pieces" false 80000 \
#   "$DATASETS/dataset_20260311_131933/lerobot" \
#   "$MODELS/GR00T-N1.5-3B_20260311-120000"
# wait_screen_gone "pieces"

# run_finetune "pieces_visual" true 120000 \
#   "$DATASETS/dataset_20260311_131933/lerobot" \
#   "$MODELS/GR00T-N1.5-3B_20260313-165333"
# wait_screen_gone "pieces_visual"

echo "[scheduler] all runs complete."


# screen -S scheduler bash -lc "./schedule.sh 2>&1 | tee logs/scheduler.log"
