#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/jobs_local.sh"
source "$SCRIPT_DIR/jobs_remote.sh"


#run_datagen "datagen" "/home/innovation-hacking/heizmany/ur5_chess/datasets/eval_a_2" "/home/innovation-hacking/heizmany/ur5_chess/source/ur5_chess/ur5_chess/evaluation/benchmarks/eval/experiments/experiment_2/eval_a_exp_2.yaml" 
#run_datagen "datagen2" "/home/innovation-hacking/heizmany/ur5_chess/datasets/eval_b_2" "/home/innovation-hacking/heizmany/ur5_chess/source/ur5_chess/ur5_chess/evaluation/benchmarks/eval/experiments/experiment_2/eval_b_exp_2.yaml" "datagen"
run_datagen "datagen3" "/home/innovation-hacking/heizmany/ur5_chess/datasets/eval_d_2" "/home/innovation-hacking/heizmany/ur5_chess/source/ur5_chess/ur5_chess/evaluation/benchmarks/eval/experiments/experiment_2/eval_d_exp_2.yaml" "datagen2"
#run_datagen "datagen4" "/home/innovation-hacking/heizmany/ur5_chess/datasets/eval_c_2" "/home/innovation-hacking/heizmany/ur5_chess/source/ur5_chess/ur5_chess/evaluation/benchmarks/eval/experiments/experiment_2/eval_c_exp_2.yaml" 

wait_screen_gone "datagen3"

# screen -S scheduler bash -lc "./schedule.sh 2>&1 | tee logs/scheduler.log"