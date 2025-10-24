import argparse
import os
import subprocess
import determined as det
import multiprocessing as mp
import time
import requests

def start_vllm_if_rank0_node_wise(vllm_gpus:int):
    info = det.get_cluster_info()
    print(f"det info gpu uuids: {info.gpu_uuids}")
    print(f"det info container rank: {info.container_rank}")
    print(f"det info container slot counts: {info.container_slot_counts}")
    print(f"det info container addrs: {info.container_addrs}")
    print(f"det info trial: {info.trial}")
    print(f"det info master url: {info.master_url}")
    print(f"det info agent id: {info.agent_id}")
    print(f"det info slot ids: {info.slot_ids}")

    vllm_ip = info.container_addrs[info.container_rank]

    vllm_env = {k:v for k,v in os.environ.items()}
    print(f"vllm env vars: {vllm_env}")

    dist_keys = [
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "NODE_RANK",
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
        "TORCHELASTIC_RESTART_COUNT",
        "ETCD_HOST",
        "ETCD_PORT",
        "ETCD_PROTOCOL",
        "ETCD_PREFIX",
        "ELASTIC_C10D_RDZV_ID",
        "C10D_PORT",
        "DET_MASTER",
        "DET_MASTER_ADDR",
    ]

    for dist_key in dist_keys:
        if dist_key in vllm_env.keys():
            vllm_env.pop(dist_key)

    vllm_env["CUDA_VISIBLE_DEVICES"] = ",".join([str(idx) for idx in range(vllm_gpus)])

    vllm_env["MASTER_ADDR"] = "127.0.0.1"
    vllm_env["MASTER_PORT"] = "29511"


    host = "0.0.0.0"
    port = 8000
    args = [
        "trl", "vllm-serve",
        "--host", host,
        "--port", str(port),
        "--model", "TIGER-Lab/PixelReasoner-WarmStart",
        "--tensor_parallel", str(vllm_gpus),
        "--limit_image_per_prompt", "3",
        "--max_pixels", "3920000"
    ]
    #print(f"starting vllm server on node {vllm_node}={node_rank} with IP address: {info.container_addrs[node_rank]}")

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    proc = subprocess.Popen(
        args,
        stdout=None,
        stderr=None,
        env=vllm_env,
        bufsize=0
    )
    print("after subprocess")

    try:
        print(f"health/ : {requests.get(f"http://127.0.0.1:8000/health/")}")
    except Exception as e:
        print(f"health/ : {e}")

    try:
        print(f"/ : {requests.get(f"http://127.0.0.1:8000/")}")
    except Exception as e:
        print(f"/ : {e}")

    try:
        print(f"health : {requests.get(f"http://127.0.0.1:8000/health")}")
    except Exception as e:
        print(f"health : {e}")

    print("start to sleep")

    time.sleep(100)
    print("sleep end")

    try:
        print(f"health/ : {requests.get(f"http://127.0.0.1:8000/health/")}")
    except Exception as e:
        print(f"health/ : {e}")
    try:
        print(f"/ : {requests.get(f"http://127.0.0.1:8000/")}")
    except Exception as e:
        print(f"/ : {e}")
    try:
        print(f"health : {requests.get(f"http://127.0.0.1:8000/health")}")
    except Exception as e:
        print(f"health : {e}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vllm_gpus", type=int)
    args = parser.parse_args()
    start_vllm_if_rank0_node_wise(args.vllm_gpus)