#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/jobs_local.sh"

mkdir -p "$IL_LOGS"

MODEL="/home/innovation-hacking/heizmany/Isaac-GR00T/models/model_6"
EXPERIMENT="/home/innovation-hacking/heizmany/ur5_chess/source/ur5_chess/ur5_chess/evaluation/benchmarks/eval/experiments/experiment_6"

$PY_IL -u $MAIN_OFFLINE \
  --model-path    "$MODEL" \
  --experiment-path "$EXPERIMENT" \
  --parts a b c d \
  --action-horizon 8 \
  --workers 2 #\
  #--max-episodes 30


# screen -S offline_eval bash -lc "cd /home/innovation-hacking/heizmany/Isaac-GR00T && ./offline_eval.sh"
