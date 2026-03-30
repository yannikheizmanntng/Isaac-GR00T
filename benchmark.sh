#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/jobs_local.sh"
source "$SCRIPT_DIR/jobs_remote.sh"

mkdir -p "$IG_LOGS"

REMOTE_MODEL_1="/workspace/models/model_1"
LOCAL_MODEL="/home/innovation-hacking/data/heizmany/models/model_1"
EXPERIMENT="/home/innovation-hacking/heizmany/ur5_chess/source/ur5_chess/ur5_chess/evaluation/benchmarks/eval/experiments/experiment_1"

run_benchmark "benchmark_a" "PartA" "$LOCAL_MODEL" "$EXPERIMENT" "" ""
run_benchmark "benchmark_b" "PartB" "$LOCAL_MODEL" "$EXPERIMENT" "" "benchmark_a"
run_benchmark "benchmark_d" "PartD" "$LOCAL_MODEL" "$EXPERIMENT" "" "benchmark_b"
wait_screen_gone "benchmark_d"


# curl http://localhost:8765/jobs