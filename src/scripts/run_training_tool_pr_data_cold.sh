run_name="Qwen_2p5_7B_tool_pr_data_cold"

torchrun --nproc_per_node="7" \
    --nnodes="1" \
  ../open_r1/grpo_jsonl.py \
    --run_name ${run_name} \
    --output_dir /pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/helm/focusreason/runs/${run_name}_$(date +%Y%m%d_%H%M%S) \
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
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
    --num_iterations 16 \
    --gradient_checkpointing True \
    --torch_dtype bfloat16 \
    --attn_implementation flash_attention_2 \
    --bf16 True \
    --logging_steps 1 \
    --logging True \
    --num_train_epochs 1 \
    --save_steps 750 \
    --temperature 1.0 \
    --multi_turn tool \
    --prompt_type no_think_tool \
    --reward_funcs accuracy format_no_think pr_penalty curiosity \
    --reward_func_weights 1 1 0.05 0.5 \
    --tool_use_penalty_threshold 1 \
    --pixel_reasoning_threshold 0.3 \
    --chat_template /pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/helm/focusreason/src/qwen_chat_template_tool.json \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-6 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --max_tool_uses 2 \
    --max_pixels 784000 # 1000*28*28


