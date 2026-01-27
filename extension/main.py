from __future__ import annotations
import os
import signal
import subprocess
import sys
from pathlib import Path
from huggingface_hub import HfApi, upload_folder
import time

from extension.config.secrets import HF_TOKEN
from extension.config.args import Args
from extension.utils.argparsing import ArgsParser


class Main:
    @classmethod
    def startup(cls, args: Args):
        if args.training_args.report_to == "wandb" and args.mode == "run_finetune":
            os.environ.setdefault("WANDB_PROJECT", "ur5-chess")
            os.environ["WANDB_GROUP"] = f"{args.model_args.base_model_path}"

    @classmethod
    def shutdown(cls, args: Args, proc: subprocess.Popen):
        if args.model_args.push_to_hub:
            api = HfApi(token=HF_TOKEN)
            repo_id = f"{args.model_args.hf_user}/gr00t_{time.strftime('%Y%m%d-%H%M%S')}"
            output_dir = f"{args.model_args.model_output_dir}/{args.model_args.base_model_path.split('/')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}" 
            api.create_repo(
                repo_id=repo_id,
                repo_type="model",
                private=True,
                exist_ok=True,
            )
            upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=output_dir,
                commit_message=f"Upload GR00T fine-tune output: {Path(output_dir).name}",
            )

        try:
            returncode = proc.wait()
        except KeyboardInterrupt:
            os.killpg(proc.pid, signal.SIGINT)
            try:
                returncode = proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    returncode = proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    returncode = 1
        return returncode

    @classmethod
    def _finetune_script_path(cls) -> Path:
        repo_root = Path(__file__).resolve().parent.parent
        return repo_root / "scripts" / "gr00t_finetune.py"
    
    @classmethod
    def _inference_script_path(cls) -> Path:
        repo_root = Path(__file__).resolve().parent.parent
        return repo_root / "scripts" / "inference_service.py"

    @classmethod
    def run_finetune(cls, args: Args) -> subprocess.Popen:
        script_path = cls._finetune_script_path()
        if not script_path.exists():
            raise FileNotFoundError(f"Could not find finetune script at: {script_path}")

        output_dir = f"{args.model_args.model_output_dir}/{args.model_args.base_model_path.split('/')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}" 
        os.makedirs(output_dir, exist_ok=True)
        args.save(output_dir)

        os.environ["PYTHONWARNINGS"] = "ignore::UserWarning:torchvision.io._video_deprecation_warning"
        cmd = [sys.executable, str(script_path), *args.to_gr00t_finetune_argv()]
        proc = subprocess.Popen(cmd, preexec_fn=os.setsid)

        print("\n" + "=" * 60)
        print("GR00T WRAPPER: launching finetune script")
        print("Command:")
        print(" ".join(cmd))
        print("=" * 60 + "\n")
        return proc
    
    @classmethod
    def start_inference_server(cls, args: Args) -> subprocess.Popen:
        script_path = cls._inference_script_path()
        if not script_path.exists():
            raise FileNotFoundError(f"Could not find inference script at: {script_path}")

        os.environ["PYTHONWARNINGS"] = "ignore::UserWarning:torchvision.io._video_deprecation_warning"
        cmd = [sys.executable, str(script_path), *args.to_gr00t_inference_argv()]
        proc = subprocess.Popen(cmd, preexec_fn=os.setsid)

        print("\n" + "=" * 60)
        print("GR00T WRAPPER: launching inference server")
        print("Command:")
        print(" ".join(cmd))
        print("=" * 60 + "\n")
        return proc

    @classmethod
    def run(cls, args: Args) -> None:
        cls.startup(args)
        print(args, flush=True)
        match args.mode:
            case "run_finetune":
                proc = cls.run_finetune(args)
            case "start_inference_server":
                proc = cls.start_inference_server(args)
        cls.shutdown(args, proc)


if __name__ == "__main__":
    args: Args = ArgsParser(Args).parse()
    Main.run(args)


# start named session running your command + tee log
# screen -S finetune bash -lc "/home/innovation-hacking/heizmany/Isaac-GR00T/.venv_ig/bin/python extension/main.py 2>&1 | tee finetune.log"


"""
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json




screen -S server bash -lc "/home/innovation-hacking/heizmany/Isaac-GR00T/.venv_ig/bin/python extension/main.py 2>&1 | tee server.log"

screen -S finetune bash -lc "/home/innovation-hacking/heizmany/Isaac-GR00T/.venv_ig/bin/python extension/main.py 2>&1 | tee finetune.log"
"""
# # detach
# # Ctrl + a, then d

# # list sessions
# screen -ls

# # attach
# screen -r finetune

# # kill session
# screen -S finetune -X quit


# screen -S scheduler bash -lc "./schedule.sh 2>&1 | tee -a logs/scheduler.log"
