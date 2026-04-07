#!/usr/bin/env bash
# Local job functions — source this file, do not execute directly.

PY_IG="/home/innovation-hacking/heizmany/Isaac-GR00T/.venv_ig/bin/python"
MAIN_IG="/home/innovation-hacking/heizmany/Isaac-GR00T/extension/main.py"

PY_IL="/home/innovation-hacking/heizmany/ur5_chess/.venv/bin/python"
MAIN_IL="/home/innovation-hacking/heizmany/ur5_chess/source/ur5_chess/ur5_chess/main.py"
MAIN_OFFLINE="/home/innovation-hacking/heizmany/ur5_chess/source/ur5_chess/ur5_chess/evaluation/offline/main.py"

IG_LOGS="/home/innovation-hacking/heizmany/Isaac-GR00T/logs"
IL_LOGS="/home/innovation-hacking/heizmany/ur5_chess/logs"

ts () { date +"%Y%m%d-%H%M%S"; }


run_finetune () {
  local name="$1"
  local tune_visual="$2"
  local num_steps="$3"
  local dataset_path="$4"
  local resume_from="${5-}"
  local output_name="${6-}"
  local training_args_extra="${7-}"

  local log="$IG_LOGS/${name}_$(ts).log"

  local model_args="tune_visual=$tune_visual"
  [[ -n "$resume_from" ]] && model_args+=" resume_from=$resume_from"
  [[ -n "$output_name" ]] && model_args+=" output_name=$output_name"

  local training_args="max_steps=$num_steps"
  [[ -n "$training_args_extra" ]] && training_args+=" $training_args_extra"

  local dataset_args_items=""
  IFS=',' read -ra dataset_paths <<< "$dataset_path"
  for p in "${dataset_paths[@]}"; do
    dataset_args_items+="dataset_path=${p} "
  done

  echo "[scheduler] launching ${name}"
  screen -dmS "$name" bash -lc "
    $PY_IG -u $MAIN_IG \
      --mode run_finetune \
      --model_args $model_args \
      --training_args $training_args \
      --dataset_args $dataset_args_items \
    2>&1 | tee '${log}'
  "
}


run_datagen () {
  local name="$1"
  local output_path="$2"
  local cfg_path="$3"
  local wait_for_gone="${4-}"

  local log="$IG_LOGS/${name}_$(ts).log"
  local venv_activate="/home/innovation-hacking/heizmany/ur5_chess/.venv/bin/activate"

  local wait_gone_arg=""
  [[ -n "$wait_for_gone" ]] && wait_gone_arg="--wait_for_screen_gone $wait_for_gone"

  echo "[scheduler] launching ${name}"
  screen -dmS "$name" bash -lc "
    source '$venv_activate'
    source /home/innovation-hacking/heizmany/IsaacSim/setup_conda_env.sh
    export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

    python -u $MAIN_IL \
      --mode record_dataset \
      $wait_gone_arg \
      --sim_args dataset_cfg_yaml=$cfg_path\
      --rec_args dataset_output_path=$output_path \
    2>&1 | tee '$log'
  "
}


run_inference_server () {
  local name="$1"
  local model_path="$2"

  local log="$IG_LOGS/${name}_$(ts).log"

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
  local experiment_path="$4"
  local wait_for="${5-}"
  local wait_for_gone="${6-}"

  local log="$IG_LOGS/${name}_$(ts).log"
  local venv_activate="/home/innovation-hacking/heizmany/ur5_chess/.venv/bin/activate"

  local wait_arg=""
  [[ -n "$wait_for" ]] && wait_arg="--wait_for_screen $wait_for"

  local wait_gone_arg=""
  [[ -n "$wait_for_gone" ]] && wait_gone_arg="--wait_for_screen_gone $wait_for_gone"

  echo "[scheduler] launching benchmark ${name}"
  screen -dmS "$name" bash -lc "
    source '$venv_activate'
    source /home/innovation-hacking/heizmany/IsaacSim/setup_conda_env.sh
    export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

    python -u $MAIN_IL \
      --mode run_benchmark \
      $wait_arg \
      $wait_gone_arg \
      --benchmark_args benchmark=$benchmark model_path=$model_path experiment_path=$experiment_path 
    2>&1 | tee '$log'
  "
}



wait_screen_gone () {
  local name="$1"
  local pattern="\.${name}[^[:alnum:]_]"
  echo "[scheduler] waiting for screen session '$name' to finish..."
  while screen -list | grep -q "$pattern"; do
    if screen -list | grep "$pattern" | grep -qi "dead"; then
      echo "[scheduler] screen '$name' is dead (process crashed), wiping and treating as finished."
      screen -wipe > /dev/null 2>&1
      break
    fi
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
