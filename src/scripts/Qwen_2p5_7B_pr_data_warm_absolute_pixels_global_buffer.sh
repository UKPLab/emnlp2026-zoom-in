run_name=$(basename "$0" .sh)

RUN_ARGS=(
    --run_name ${run_name} \
    --output_dir /pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/helm/focusreason/runs/${run_name}_$(date +%Y%m%d_%H%M%S) \
    --model_name_or_path TIGER-Lab/PixelReasoner-WarmStart \
    --deepspeed /pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/helm/focusreason/src/scripts/zero3.json \
    --dataset_name pixel_reasoner \
    --data_file_paths /pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/helm/datasets/pixel_reasoner/RL_data_without_video/train.jsonl \
    --image_folders /pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/helm/datasets/pixel_reasoner/RL_data_without_video \
    --max_completion_length 256 \
    --ds3_gather_for_generation True \
    --use_vllm True \
    --vllm_device cuda:0 \
    --beta 0.04 \
    --num_generations 8 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 5 \
    --num_iterations 2 \
    --gradient_checkpointing True \
    --torch_dtype bfloat16 \
    --attn_implementation flash_attention_2 \
    --bf16 True \
    --logging_steps 1 \
    --logging True \
    --num_train_epochs 3 \
    --temperature 1.0 \
    --multi_turn tool \
    --reward_funcs accuracy pr_penalty curiosity \
    --reward_func_weights 1 0.05 0.5 \
    --tool_use_penalty_threshold 1 \
    --pixel_reasoning_threshold 0.3 \
    --chat_template /pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/helm/focusreason/src/qwen_chat_template_tool.json \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-6 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --max_tool_uses 2 \
    --max_pixels 784000 \
    --epsilon 0.2 \
    --tool_config PR_zoom_in_old \
    --global_buffer True \
    --save_strategy epoch \
    --save_only_model True \
    --steps_per_generation 5 \
    --generation_batch_size 280
)

echo "mode is '$1'"

RUN_ARGS+=(--training_mode "$1")

echo "run args is: ${RUN_ARGS[@]}"

if [ "$1" = multinode  ]; then
  #export run_name
  #printf -v RUN_ARGS_STR '%q ' "${RUN_ARGS[@]}"
  #export RUN_ARGS_STR
  export RUN_ARGS_DUMP
  RUN_ARGS_DUMP="$(declare -p RUN_ARGS)"
fi

if [ "$1" = singlenode  ]; then
  torchrun --nproc_per_node=7 --nnodes=1 \
    ../grpo_jsonl_top.py \
    "${RUN_ARGS[@]}"
fi
