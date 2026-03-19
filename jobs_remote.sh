#!/usr/bin/env bash
# Remote job functions — source this file, do not execute directly.

REMOTE_HOST="innovation-hacking@bigboi.ihack.host"
REMOTE_JOB_SERVER="http://bigboi.ihack.host:8765"
REMOTE_INFERENCE_HOST="bigboi.ihack.host"
REMOTE_MODELS="/workspace/models"
REMOTE_DATASETS="/workspace/datasets"


wait_for_job_server () {
  echo "[scheduler] $(date '+%H:%M:%S') waiting for job server at $REMOTE_JOB_SERVER..."
  until curl -sf "$REMOTE_JOB_SERVER/jobs" > /dev/null 2>&1; do
    echo "[scheduler] $(date '+%H:%M:%S') job server not reachable, retrying in 30s..."
    sleep 30
  done
  echo "[scheduler] $(date '+%H:%M:%S') job server is up."
}


_remote_run () {
  local name="$1"
  local args_str="$2"
  local response
  until response=$(curl -sf -X POST "$REMOTE_JOB_SERVER/run" \
    -H "Content-Type: application/json" \
    -d "{\"name\":\"$name\",\"args_str\":\"$args_str\"}" 2>&1); do
    echo "[scheduler] $(date '+%H:%M:%S') ERROR: failed to post job '$name', retrying in 30s... ($response)"
    sleep 30
  done
  echo "$response"
}


run_finetune_remote () {
  local name="$1"
  local tune_visual="$2"
  local num_steps="$3"
  local dataset_path="$4"
  local resume_from="${5-}"
  local output_name="${6-}"
  local training_args_extra="${7-}"

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

  echo "[scheduler] launching ${name} on remote"
  _remote_run "$name" \
    "--mode run_finetune --model_args $model_args --training_args $training_args --dataset_args $dataset_args_items"
}


rsync_to_remote () {
  local src="$1"
  local dest_path="$2"
  echo "[scheduler] $(date '+%H:%M:%S') rsyncing $src → $REMOTE_HOST:$dest_path"
  until rsync -av "$src/" "$REMOTE_HOST:$dest_path/"; do
    echo "[scheduler] $(date '+%H:%M:%S') rsync failed (remote unreachable?), retrying in 60s..."
    sleep 60
  done
  echo "[scheduler] $(date '+%H:%M:%S') rsync done."
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
