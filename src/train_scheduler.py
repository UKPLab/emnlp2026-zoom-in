#!/usr/bin/env python3
import os
import subprocess
import time
import requests
import signal
import sys
import re
import datetime
import json

def screen_exists(screen_name):
    """Check if a screen session exists."""
    try:
        result = subprocess.run(
            ['screen', '-list'], 
            capture_output=True, 
            text=True, 
            check=True
        )
        return any(re.search(rf"\b{re.escape(screen_name)}\b", line) for line in result.stdout.splitlines())
    except subprocess.CalledProcessError:
        return False

def start_or_resume_screen(screen_name, log_file, command=None, env_vars=None):
    """Start a new screen session or resume an existing one."""
    
    if screen_exists(screen_name):
        print(f"Resuming existing screen {screen_name}")
        if command:
            # Attach to screen, run command, then detach
            env_cmd = ""
            if env_vars:
                env_cmd = " ".join([f"export {k}={v};" for k, v in env_vars.items()])

            full_cmd = f"screen -S {screen_name} -X stuff '{env_cmd} {command} >> {log_file} 2>&1\n'"
            subprocess.run(full_cmd, shell=True, check=True)
    else:
        print(f"Creating new screen {screen_name}")
        # Create a new detached screen session
        env_string = ""
        if env_vars:
            env_string = " ".join([f"{k}={v}" for k, v in env_vars.items()])

        if command:
            screen_cmd = f"{env_string} screen -dmS {screen_name} bash -c '{command} >> {log_file} 2>&1'"
        else:
            screen_cmd = f"{env_string} screen -dmLS {screen_name} -L -Logfile {log_file}"

        print(f"Running cmd in screen: {screen_cmd}")
        subprocess.run(screen_cmd, shell=True, check=True)

def stop_screen(screen_name):
    """Stop a screen session."""
    if screen_exists(screen_name):
        print(f"Stopping screen {screen_name}")
        try:
            # Try to gracefully terminate processes in the screen
            subprocess.run(f"screen -S {screen_name} -X stuff $'\003'", shell=True, check=False)
            time.sleep(2)  # Give some time for processes to terminate
            subprocess.run(f"screen -S {screen_name} -X quit", shell=True, check=True)
            print(f"Screen {screen_name} stopped successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error stopping screen {screen_name}: {e}")
            return False
    else:
        print(f"Screen {screen_name} does not exist")
        return True

def check_server_health(url="http://localhost:8000/health/", max_attempts=30, delay=10):
    """Check if the VLLM server is healthy by polling the /health/ endpoint."""
    print(f"Waiting for VLLM server to be ready at {url}")
    for attempt in range(max_attempts):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print("VLLM server is ready!")
                return True
        except requests.RequestException:
            pass
        
        print(f"Server not ready, checking again in {delay} seconds... (attempt {attempt+1}/{max_attempts})")
        time.sleep(delay)
    
    print("Server health check timed out!")
    return False

def get_available_gpus():
    """Get a list of available CUDA device IDs."""
    try:
        # Try using nvidia-smi to get GPU count
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            check=True
        )
        gpu_indices = [line.strip() for line in result.stdout.splitlines()]
        return gpu_indices
    except (subprocess.SubprocessError, FileNotFoundError):
        print("Warning: Could not determine available GPUs automatically.")
        return []

def is_screen_active(screen_name):
    """Check if a screen session is still running."""
    return screen_exists(screen_name)

def run_training_pipeline(vllm_screen_name, train_screen_name, vllm_command, train_command, output_dir):
    """Run the complete training pipeline."""

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    vllm_log_file = os.path.join(output_dir, 'vllm_log.txt')
    run_log_file = os.path.join(output_dir, 'run_log.txt')

    # Create empty log files
    open(vllm_log_file, 'w').close()
    open(run_log_file, 'w').close()

    # Step 1 & 2: Start or resume VLLM server screen with CUDA_VISIBLE_DEVICES=0
    #vllm_command = f"VLLM_USE_V1=0 trl vllm-serve --model Qwen/Qwen2.5-VL-3B-Instruct --limit_image_per_prompt {images_per_prompt}"
    vllm_env = {"CUDA_VISIBLE_DEVICES": "0"}
    
    start_or_resume_screen(vllm_screen_name, vllm_log_file, vllm_command, vllm_env)
    
    # Step 4: Wait for the VLLM server to be ready
    if not check_server_health():
        print("VLLM server failed to start properly. Exiting.")
        return False
    
    # Step 5 & 6: Start or resume training screen with CUDA_VISIBLE_DEVICES set to all devices except 0
    # Get available GPUs and exclude GPU 0
    available_gpus = get_available_gpus()
    if available_gpus and "0" in available_gpus:
        available_gpus.remove("0")
    
    # Set CUDA_VISIBLE_DEVICES for training to all GPUs except 0
    cuda_devices = ",".join(available_gpus) if available_gpus else "1"  # Default to 1 if we can't detect GPUs
    train_env = {"CUDA_VISIBLE_DEVICES": cuda_devices}

    # Step 7: Launch the training
    train_command = (f"cd /pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/helm/focusreason/src "
                     f"&& {train_command}")
    
    print(f"Starting training on GPUs: {cuda_devices}")
    start_or_resume_screen(train_screen_name, run_log_file, train_command, train_env)
    
    print("Training pipeline started successfully!")
    print("Monitoring training process... Press Ctrl+C to stop monitoring but keep training running.")
    
    try:
        # Continuously check if training has finished
        while is_screen_active(train_screen_name):
            print(f"Training is still running. Checking again in 60 seconds...")
            time.sleep(60)
        
        print("Training has completed!")
    except KeyboardInterrupt:
        print("\nStopped monitoring training, but training continues to run in the background.")
        print("The VLLM server will remain active as training may still be in progress.")
        print("To terminate the VLLM server manually later, run:")
        print(f"  screen -S {vllm_screen_name} -X quit")
        return True
    
    # Training has completed, now stop the VLLM server
    print("Training has finished. Terminating VLLM server...")
    if stop_screen(vllm_screen_name):
        print("VLLM server has been terminated.")
    else:
        print("Failed to terminate VLLM server. You may need to stop it manually.")
    
    return True

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\nScript interrupted by user. Exiting monitoring mode...")
    sys.exit(0)

def get_commands(hparams: dict, run_name: str):

    resume = False
    if "output_dir" in hparams["train_params"]:
        resume = True
        output_dir = hparams["train_params"]["output_dir"]
        assert "aim_run_hash" in hparams["train_params"] and hparams["train_params"]["aim_run_hash"] is not None
        hparams["train_params"]["model_name_or_path"] = os.path.join(hparams["train_params"]["output_dir"],
                                                                     f"checkpoint-{hparams["train_params"]["resume_from_checkpoint"]}")
        hparams["train_params"].pop("resume_from_checkpoint")


    hf_cmd = "torchrun --nproc_per_node=7 --nnodes=1 grpo_jsonl_top.py"
    for name, value in hparams["train_params"].items():
        if isinstance(value, list):
            inferred_value = " ".join([str(v) for v in value])
            inferred_name = name
        elif name == "output_dir_prefix":
            if resume:
                continue
            inferred_value = os.path.join(f"{value}", f"{run_name}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}")
            output_dir = inferred_value
            inferred_name = "output_dir"
        else:
            inferred_name = name
            inferred_value = value
        hf_cmd = hf_cmd + f" --{inferred_name} {inferred_value}"
    hf_cmd = hf_cmd + f" --run_name {run_name}"

    hf_cmd = hf_cmd + f" --training_mode singlenode"

    vllm_cmd = "trl vllm-serve"
    for name, value in hparams["vllm_params"].items():
        if value == "infer":
            if name == "model":
                inferred_value = hparams["train_params"]["model_name_or_path"]
            else:
                inferred_value = hparams["train_params"][name]
        else:
            inferred_value = value

        vllm_cmd = vllm_cmd + f" --{name} {inferred_value}"



    return vllm_cmd, hf_cmd, output_dir

if __name__ == "__main__":
    # Register signal handler for graceful exit
    signal.signal(signal.SIGINT, signal_handler)

    script_dir = f"/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/helm/focusreason/src/scripts"


    runs = [
        {
            "json_name": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_5k_image_tokens_min_image_500_no_tool.json",
            "shell_number": 1,
            "path": "",
            "state": "running"},

        {
            "json_name": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_5k_image_tokens_min_image_500_mi_tool_box.json",
            "shell_number": 4,
            "path": "",
            "state": "running"},
        {
            "json_name": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_5k_image_tokens_min_image_500_mi_tool_box.json",
            "shell_number": 6,
            "path": "",
            "state": "running"},
        {
            "json_name": "Qwen_2p5_7B_pr_data_cold_relative_pixels_5k_image_tokens_min_image_500.json",
            "shell_number": 5,
            "path": "",
            "state": "running"},
        {
            "json_name": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_5k_image_tokens_min_image_500.json",
            "shell_number": 2,
            "path": "",
            "state": "running"
        },
        {
            "json_name": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_5k_image_tokens_min_image_500.json",
            "shell_number": 3,
            "path": "",
            "state": "running"
        },
        {
            "json_name": "Qwen_2p5_7B_pr_data_cold_relative_pixels_5k_image_tokens_min_image_500.json",
            "shell_number": 4,
            "path": "",
            "state": "to_be_launched"
        },
        {
            "json_name": "Qwen_2p5_7B_pr_data_warm_relative_pixels_5k_image_tokens_min_image_500.json",
            "shell_number": 6,
            "path": "",
            "state": "running"
        }

    ]

    for run in runs:
        if run["state"] == "to_be_launched":
            vllm_screen_name = f"{run['shell_number']}_auto_vllm_new2"
            train_screen_name = f"{run['shell_number']}_auto_run_new2"

            # Run the training pipeline
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            full_hparams = json.load(open(os.path.join(script_dir, run["json_name"]), "r"))

            run_name = run["json_name"].removesuffix(".json")

            vllm_command, train_command, output_dir = get_commands(full_hparams, run_name)
            print(f"starting run: {run_name}")
            print(f"vllm_command: {vllm_command}")
            print(f"hf_command: {train_command}")

            run_training_pipeline(vllm_screen_name, train_screen_name, vllm_command, train_command, output_dir=output_dir)
