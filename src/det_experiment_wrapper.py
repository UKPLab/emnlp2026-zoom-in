import time
import os
import subprocess
import requests

def start_vllm_if_rank0(vllm_devices: list[int]):
    node_rank = int(os.environ.get("GROUP_RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    print(f"node_rank: {node_rank}, local_rank: {local_rank}")

    non_vllm_devices = [idx for idx in range(8) if idx not in vllm_devices]

    if node_rank != 0:
        #time.sleep(200)

        return None, None
    else:
        if local_rank in non_vllm_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(idx) for idx in non_vllm_devices])
            #time.sleep(200)
            return None, None
        else:


            #if rank != 0 or local_rank != 0:
            #    return None, None

            # Choose the GPUs vLLM should use on the chief node.
            # Example: dedicate GPU 0 to vLLM on the chief node (adjust as needed).

            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(idx) for idx in non_vllm_devices])

            full_env = os.environ.copy()

            dist_keys = [
                "RANK",
                "LOCAL_RANK",
                "WORLD_SIZE",
                "LOCAL_WORLD_SIZE",
                "GROUP_RANK",
                "ROLE_RANK",
                "ROLE_NAME",
                "OMP_NUM_THREADS",
                "MASTER_ADDR",
                "MASTER_PORT",
                "TORCHELASTIC_USE_AGENT_STORE",
                "TORCHELASTIC_MAX_RESTARTS",
                "TORCHELASTIC_RUN_ID",
                "TORCH_NCCL_ASYNC_ERROR_HANDLING",
                "TORCHELASTIC_ERROR_FILE",
            ]

            for dist_key in dist_keys:
                del os.environ[dist_key]

            reduced_env = os.environ.copy()

            #reduced_env["CUDA_VISIBLE_DEVICES"] = ",".join(vllm_devices)

            port = int(os.environ.get("VLLM_PORT", "8000"))
            args = [
                "trl", "vllm-serve",
                "--host", "0.0.0.0",
                "--port", str(port),
                "--model", "Qwen/Qwen2.5-VL-7B-Instruct",
                "--tensor_parallel", str(len(vllm_devices)),
                "--limit_image_per_prompt", "3",
                "--max_pixels", "784000"
            ]
            proc = subprocess.Popen(
                args,
                stdout=None,
                stderr=None,
                env=reduced_env,
                bufsize=0
            )


            os.environ = full_env

            return proc, port

def get_vllm_base_url(port: int) -> str:
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    return f"http://{master_addr}:{port}"




if __name__ == "__main__":
    vllm_proc, vllm_port = start_vllm_if_rank0(vllm_devices=[0])
    base_url = get_vllm_base_url(vllm_port or int(os.environ.get("VLLM_PORT", "8000")))
    print(f"base_url: {base_url}")

    #time.sleep(30)
    #subprocess.Popen(["nvidia-smi"])
    subprocess.Popen(["bash", "scripts/multinode_test.sh"])

