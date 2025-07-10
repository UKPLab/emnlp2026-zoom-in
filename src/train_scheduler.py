#!/usr/bin/env python3
import os
import subprocess
import time
import requests
import signal
import sys
import re
import datetime

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

def run_training_pipeline(vllm_screen_name, train_screen_name, train_script, train_script_kwargs, images_per_prompt=1):
    """Run the complete training pipeline."""

    # Create output directory if it doesn't exist
    os.makedirs(train_script_kwargs['output_dir'], exist_ok=True)

    vllm_log_file = os.path.join(train_script_kwargs['output_dir'], 'vllm_log.txt')
    run_log_file = os.path.join(train_script_kwargs['output_dir'], 'run_log.txt')

    # Create empty log files
    open(vllm_log_file, 'w').close()
    open(run_log_file, 'w').close()

    # Step 1 & 2: Start or resume VLLM server screen with CUDA_VISIBLE_DEVICES=0
    vllm_command = f"VLLM_USE_V1=0 trl vllm-serve --model Qwen/Qwen2.5-VL-3B-Instruct --limit_image_per_prompt {images_per_prompt}"
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

    kwarg_string = " ".join([f"--{k} {v}" for k, v in train_script_kwargs.items()])

    # Step 7: Launch the training
    train_command = (f"cd /pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/helm/focusreason/src/scripts "
                     f"&& bash {train_script} {kwarg_string}")
    
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

if __name__ == "__main__":
    # Register signal handler for graceful exit
    signal.signal(signal.SIGINT, signal_handler)
    debug = False

    run_dir = f"/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/helm/focusreason/runs"


    if debug:
        vllm_screen_name = "2_auto_vllm"
        train_screen_name = "2_auto_run"

        runs = [
            {"script": "run_training_standard_test.sh", "images": 1},
            {"script": "run_training_rethink_text_test.sh", "images": 1},
            {"script": "run_training_rethink_image_test.sh", "images": 2},
        ]

    else:
        vllm_screen_name = "8_auto_vllm"
        train_screen_name = "8_auto_run"

        runs = [
            {"run_name": "run_training_rethink_image", "images": 2},
            #{"script": "run_training_rethink_text.sh", "images": 1},
            #{"script": "run_training_standard.sh", "images": 1},
        ]

    for run in runs:
        # Run the training pipeline
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        run_training_pipeline(vllm_screen_name, train_screen_name, train_script=f'{run["run_name"]}.sh',
                              train_script_kwargs={"logging":True,
                                                   "output_dir": os.path.join(run_dir, f"{run['run_name']}_{current_time}"),
                                                   #"data_subset": "7500:8000"
                                                   },
                              images_per_prompt=run["images"])
        time.sleep(60)