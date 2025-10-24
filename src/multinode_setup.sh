NNODES=""
VLLM_GPUS=""
SCRIPT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --nnodes) NNODES="$2"; shift 2 ;;
    --nnodes=*) NNODES="${1#*=}"; shift ;;
    --vllm_gpus) VLLM_GPUS="$2"; shift 2 ;;
    --vllm_gpus=*) VLLM_GPUS="${1#*=}"; shift ;;
    --script) SCRIPT="$2"; shift 2 ;;
    --script=*) SCRIPT="${1#*=}"; shift ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 --nnodes N --vllm_gpus GPU_LIST --script SCRIPT_FILENAME"
      exit 2
      ;;
  esac
done

if [[ -z "$NNODES" || -z "$VLLM_GPUS" || -z "$SCRIPT" ]]; then
  echo "Usage: $0 --nnodes N --vllm_gpus GPU_LIST --script SCRIPT_FILENAME"
  exit 2
fi

#----------------------------------------------------------------------------------

VLLM_PORT="8000"

VLLM_RANK=$(($NNODES-1))
TRAIN_NODES=$(($NNODES-1))

echo "Vllm rank: ${VLLM_RANK}"

echo "Train nodes: ${TRAIN_NODES}"

RANK=$(python -c "import determined as det; print(det.get_cluster_info().container_rank)")
echo "Rank: ${RANK}"
VLLM_HOST=$(python -c "import determined as det; print(det.get_cluster_info().container_addrs[${VLLM_RANK}])")
echo "vllm host: ${VLLM_HOST}"
CONTAINER_ADDRS=$(python -c "import determined as det; print(det.get_cluster_info().container_addrs)")
echo "container addresses: ${CONTAINER_ADDRS}"

if [ "$RANK" -eq "$VLLM_RANK" ]; then
  echo "Starting vllm_launcher on rank ${RANK}"
  python vllm_launcher.py --vllm_gpus "${VLLM_GPUS}"
fi

success=0
echo "Waiting for vLLM on ${VLLM_HOST}:${VLLM_PORT} ..."
  for i in $(seq 1 300); do
    if curl -fsS "http://${VLLM_HOST}:${VLLM_PORT}/health/" >/dev/null 2>&1; then
      echo "vLLM is healthy"
      success=1
      break
    fi
    sleep 1
    if (( i % 10 == 0)); then
      echo "curl after ${i} seconds: $(curl -v "http://${VLLM_HOST}:${VLLM_PORT}/health" 2>&1)"
    fi
  done

if [[ $success -ne 1 ]]; then
  echo "vllm engine startup failed"
  exit 1
fi

if [ "$RANK" -eq "$VLLM_RANK" ]; then
  tail -f /dev/null
fi


if [ "$RANK" -ne "$VLLM_RANK" ]; then
  source "scripts/${SCRIPT}" multinode
  eval "$RUN_ARGS_DUMP"
  #RUN_ARGS["run_name"] = SCRIPT[:-3]
  #RUN_ARGS["gradient_accumulation_steps"] = $((288 / (TRAIN_NODES * 8 * RUN_ARGS["per_device_train_batch_size"])))

  python -m determined.launch.torch_distributed --nproc_per_node=8 --nnodes="${TRAIN_NODES}" -- python grpo_jsonl_top.py "${RUN_ARGS[@]}"
fi