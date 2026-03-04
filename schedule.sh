#!/usr/bin/env bash
set -euo pipefail
mkdir -p logs
ts () { date +"%Y%m%d-%H%M%S"; }

PY_IG="/home/innovation-hacking/heizmany/Isaac-GR00T/.venv_ig/bin/python"
MAIN_IG="/home/innovation-hacking/heizmany/Isaac-GR00T/extension/main.py"

PY_IL="/home/innovation-hacking/heizmany/ur5_chess/.venv/bin/python"
MAIN_IL="/home/innovation-hacking/heizmany/ur5_chess/source/ur5_chess/ur5_chess/main.py"


run_finetune () {
  local name="$1"
  local num_steps="$2"
  local dataset_path="$3"   
  local resume_from="${4-}" 

  local log="/home/innovation-hacking/heizmany/Isaac-GR00T/logs/${name}_$(ts).log"

  local resume_arg=""
  if [[ -n "$resume_from" ]]; then
    resume_arg="--model_args resume_from=$resume_from"
  fi

  echo "[scheduler] launching ${name}"
  screen -dmS "$name" bash -lc "
    $PY_IG -u $MAIN_IG \
      --mode run_finetune \
      $resume_arg \
      --training_args max_steps=$num_steps \
      --dataset_args dataset_path=$dataset_path \
    2>&1 | tee '${log}'
  "
}


run_datagen () {
  local name="$1"
  local sampling_profile="$2"

  local log="/home/innovation-hacking/heizmany/Isaac-GR00T/logs/${name}_$(ts).log"
  local venv_activate="/home/innovation-hacking/heizmany/ur5_chess/.venv/bin/activate"

  echo "[scheduler] launching ${name}"
  screen -dmS "$name" bash -lc "
    # ensure predictable working dir
    cd /home/innovation-hacking/heizmany/Isaac-GR00T

    source '$venv_activate'

    # 1) setup conda env (relative to Isaac-GR00T)
    source ../IsaacSim/setup_conda_env.sh

    # 2) Vulkan ICD for NVIDIA
    export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

    # run datagen + log
    python -u $MAIN_IL \
      --mode record_dataset \
      --sim_args sampling_profile_yaml="$sampling_profile" \
    2>&1 | tee '$log'
  "
}



run_inference_server () {
  local name="$1"
  local model_path="$2"

  local log="/home/innovation-hacking/heizmany/Isaac-GR00T/logs/${name}_$(ts).log"

  echo "[scheduler] launching inference server ${name}"
  screen -dmS "$name" bash -lc "
    $PY_IG -u $MAIN_IG \
      --mode start_inference_server \
      --inference_args model_path=$model_path \
    2>&1 | tee '${log}'
  "
}


run_benchmark () {
  local name="$1"
  local benchmark="$2"
  local model_path="$3"
  local wait_for="${4-}"
  local wait_for_gone="${5-}"

  local log="/home/innovation-hacking/heizmany/Isaac-GR00T/logs/${name}_$(ts).log"
  local venv_activate="/home/innovation-hacking/heizmany/ur5_chess/.venv/bin/activate"

  local wait_arg=""
  [[ -n "$wait_for" ]] && wait_arg="--wait_for_screen $wait_for"

  local wait_gone_arg=""
  [[ -n "$wait_for_gone" ]] && wait_gone_arg="--wait_for_screen_gone $wait_for_gone"

  echo "[scheduler] launching benchmark ${name}"
  screen -dmS "$name" bash -lc "
    cd /home/innovation-hacking/heizmany/Isaac-GR00T

    source '$venv_activate'
    source ../IsaacSim/setup_conda_env.sh
    export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

    python -u $MAIN_IL \
      --mode run_benchmark \
      $wait_arg \
      $wait_gone_arg \
      --benchmark_args benchmark=$benchmark model_path=$model_path \
    2>&1 | tee '$log'
  "
}


wait_screen_gone () {
  local name="$1"
  echo "[scheduler] waiting for screen session '$name' to finish..."
  while screen -list | grep -q "\.${name}[[:space:]]"; do
    sleep 30
  done
  echo "[scheduler] '$name' finished."
}


kill_screen () {
  local name="$1"
  local wait_s="${2:-30}"
  if screen -list | grep -q "\.${name}[[:space:]]"; then
    screen -S "$name" -X quit
    echo "[scheduler] killed screen session '$name', waiting ${wait_s}s for GPU resources to free..."
    sleep "$wait_s"
  fi
}



# --------------------------
# Runs 
# --------------------------

BASE="/home/innovation-hacking/heizmany/Isaac-GR00T/models"

# All benchmark screens launched immediately — Python processes freeze code state at this point.
# Each benchmark screen waits inside Python until its inference server screen appears.
# Inference servers use bash-level sequencing: the scheduler creates each one at the right time.

run_benchmark "labels_ia1a"  "ShortFieldIdHomoBlack" "$BASE/GR00T-N1.5-3B_20260304-180000" "labels"
run_benchmark "labels_ia1b"  "ShortFieldIdHomoMixed" "$BASE/GR00T-N1.5-3B_20260304-180000" "labels" "labels_ia1a"

run_inference_server "labels" "$BASE/GR00T-N1.5-3B_20260304-180000"
wait_screen_gone "labels_ia1b"
kill_screen "labels" 60

echo "[scheduler] all runs complete."



# """
# screen -S scheduler bash -lc "./schedule.sh 2>&1 | tee logs/scheduler.log"
# """