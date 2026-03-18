#!/usr/bin/env bash
# Remote job functions — source this file, do not execute directly.

REMOTE_HOST="innovation-hacking@bigboi.ihack.host"
REMOTE_JOB_SERVER="http://bigboi.ihack.host:8765"
REMOTE_INFERENCE_HOST="bigboi.ihack.host"
REMOTE_MODELS="/workspace/models"
REMOTE_DATASETS="/workspace/datasets"


_remote_run () {
  local name="$1"
  local args_str="$2"
  curl -sf -X POST "$REMOTE_JOB_SERVER/run" \
    -H "Content-Type: application/json" \
    -d "{\"name\":\"$name\",\"args_str\":\"$args_str\"}"
}


run_finetune_remote () {
  local name="$1"
  local tune_visual="$2"
  local num_steps="$3"
  local dataset_path="$4"
  local resume_from="${5-}"

  local model_args="tune_visual=$tune_visual"
  [[ -n "$resume_from" ]] && model_args+=" resume_from=$resume_from"

  local dataset_args_items=""
  IFS=',' read -ra dataset_paths <<< "$dataset_path"
  for p in "${dataset_paths[@]}"; do
    dataset_args_items+="dataset_path=${p} "
  done

  echo "[scheduler] launching ${name} on remote"
  _remote_run "$name" \
    "--mode run_finetune --model_args $model_args --training_args max_steps=$num_steps --dataset_args $dataset_args_items"
}


run_inference_server_remote () {
  local name="$1"
  local model_path="$2"    # relative to REMOTE_MODELS on the remote machine

  echo "[scheduler] launching inference server ${name} on remote"
  _remote_run "$name" \
    "--mode start_inference_server --inference_args model_path=$REMOTE_MODELS/$model_path"
}


wait_screen_gone_remote () {
  local name="$1"
  echo "[scheduler] $(date '+%H:%M:%S') waiting for '${name}' to finish on remote..."
  while true; do
    local response gone exit_code
    response=$(curl -sf "$REMOTE_JOB_SERVER/screen_gone/$name" 2>&1) || {
      echo "[scheduler] $(date '+%H:%M:%S') WARNING: could not reach job server, retrying..."
      sleep 30
      continue
    }
    gone=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin)['gone'])")
    if [[ "$gone" == "True" ]]; then
      exit_code=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('exit_code','?'))")
      if [[ "$exit_code" == "0" ]]; then
        echo "[scheduler] $(date '+%H:%M:%S') '${name}' finished successfully."
      else
        echo "[scheduler] $(date '+%H:%M:%S') '${name}' FAILED (exit ${exit_code}) — check remote logs."
      fi
      break
    fi
    sleep 30
  done
}


kill_screen_remote () {
  local name="$1"
  local wait_s="${2:-30}"
  echo "[scheduler] killing '${name}' on remote..."
  curl -sf -X POST "$REMOTE_JOB_SERVER/kill_screen" \
    -H "Content-Type: application/json" \
    -d "{\"name\":\"$name\"}"
  echo "[scheduler] killed '$name', waiting ${wait_s}s for GPU resources to free..."
  sleep "$wait_s"
}
