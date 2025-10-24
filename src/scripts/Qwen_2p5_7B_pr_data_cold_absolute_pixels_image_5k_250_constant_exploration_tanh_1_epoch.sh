run_name=$(basename "$0" .sh)

RUN_ARGS=(
    --run_name ${run_name}
    --output_dir /pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/helm/focusreason/runs/${run_name}_$(date +%Y%m%d_%H%M%S)
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct
    --deepspeed /pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/helm/focusreason/src/scripts/zero3.json
    --dataset_name pixel_reasoner
    --data_file_paths /pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/helm/datasets/pixel_reasoner/RL_data_without_video/train.jsonl
    --image_folders /pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/helm/datasets/pixel_reasoner/RL_data_without_video
    --max_completion_length 256
    --ds3_gather_for_generation True
    --use_vllm True
    --beta 0.04
    --num_generations 8
    --per_device_train_batch_size 2
    --scoring_batch_size_multiplier 1
    --gradient_accumulation_steps 20
    --num_iterations 2
    --steps_per_generation 20
    --generation_batch_size 280
    --gradient_checkpointing True
    --torch_dtype bfloat16
    --attn_implementation flash_attention_2
    --bf16 True
    --logging_steps 1
    --logging True
    --num_train_epochs 1
    --temperature 1.0
    --multi_turn tool
    --reward_funcs accuracy constant_exploration
    --reward_func_weights 1 0.1
    --chat_template /pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/helm/focusreason/src/qwen_chat_template_tool.json
    --per_device_eval_batch_size 4
    --learning_rate 1e-6
    --lr_scheduler_type cosine
    --warmup_ratio 0.03
    --max_tool_uses 1
    --max_pixels 3920000
    --min_pixels 196000
    --epsilon 0.2
    --tool_config PR_zoom_in_old
    --global_buffer True
    --save_strategy epoch
    --save_only_model False
    --vllm_server_timeout 600.0
    --shuffle_dataset True
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
