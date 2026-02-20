#!/usr/bin/env bash
set -euo pipefail
mkdir -p logs
ts () { date +"%Y%m%d-%H%M%S"; }

PY_IG="/home/innovation-hacking/heizmany/Isaac-GR00T/.venv_ig/bin/python"
MAIN_IG="/home/innovation-hacking/heizmany/Isaac-GR00T/extension/main.py"

PY_IL="/home/innovation-hacking/heizmany/ur5_chess/.venv_il/bin/python"
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

  local log="/home/innovation-hacking/heizmany/Isaac-GR00T/logs/${name}_$(ts).log"
  local venv_activate="/home/innovation-hacking/heizmany/ur5_chess/.venv_il/bin/activate"

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





# --------------------------
# Runs 
# --------------------------
wait_screen_gone "datagen"

run_finetune "pp" 50000 "/home/innovation-hacking/heizmany/ur5_chess/datasets/dataset_20260218_192632/lerobot" 
wait_screen_gone "pp"


run_finetune "tobi_pp" 50000 "/home/innovation-hacking/heizmany/ur5_chess/datasets/dataset_20260218_192632/lerobot" "/home/innovation-hacking/heizmany/Isaac-GR00T/models/GR00T-N1.5-3B_20260218-160000"
wait_screen_gone "tobi_pp"


echo "[scheduler] all runs complete."


# """
# screen -S scheduler bash -lc "./schedule.sh 2>&1 | tee logs/scheduler.log"
# """