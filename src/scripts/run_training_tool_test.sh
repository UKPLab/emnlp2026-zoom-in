run_name="Qwen_2p5_3B_tool_test"

RUN_ARGS=(
    --run_name ${run_name} \
    --output_dir /pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/focusreason/runs/${run_name}_$(date +%Y%m%d_%H%M%S) \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --deepspeed /pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/focusreason/src/scripts/zero3_for_testing.json \
    --dataset_name chartqa \
    --data_file_paths /pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/focusreason/chartqa_original/train_full/train_augmented_GRPO_format.jsonl \
    --image_folders /pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/focusreason/chartqa_original/train_full/png \
    --max_completion_length 256 \
    --max_prompt_length 1024 \
    --ds3_gather_for_generation True \
    --use_vllm True \
    --vllm_device cuda:0 \
    --beta 0.04 \
    --num_generations 2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --num_iterations 2 \
    --gradient_checkpointing True \
    --torch_dtype bfloat16 \
    --attn_implementation flash_attention_2 \
    --bf16 True \
    --logging_steps 1 \
    --logging False \
    --max_steps 10 \
    --save_steps 750 \
    --temperature 1.0 \
    --multi_turn tool \
    --reward_funcs accuracy format_no_think pr_penalty curiosity \
    --reward_func_weights 1 1 0.05 0.5 \
    --tool_use_penalty_threshold 1 \
    --pixel_reasoning_threshold 0.3 \
    --chat_template /pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/focusreason/src/qwen_chat_template_tool.json \
    --learning_rate 1e-6 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.2 \
    --max_tool_uses 1 \
    --max_pixels 784000 \
    --tool_config PR_zoom_in_old \
    --global_buffer True \
    --shuffle_dataset True \
    --steps_per_generation 2 \
    --generation_batch_size 8
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
  torchrun --nproc_per_node=2 --nnodes=1 \
    ../grpo_jsonl_top.py \
    "${RUN_ARGS[@]}"
fi
