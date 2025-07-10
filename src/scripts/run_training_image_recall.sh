run_name="Qwen_2p5_3B_test_image_recall"

export TRANSFORMERS_CACHE_READ_ONLY=1

torchrun --nproc_per_node="1" \
    --nnodes="1" \
  ../om_ai_lab_vlm_r1/src/open-r1-multimodal/src/open_r1/grpo_jsonl.py \
    --run_name ${run_name} \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset_name chartQA \
    --data_file_paths /pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/helm/datasets/focusreason/chartqa_original/train_full/train_augmented_GRPO_format.jsonl \
    --image_folders /pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/helm/datasets/focusreason/chartqa_original/train_full/png \
    --max_completion_length 1200 \
    --max_prompt_length 1024 \
    --ds3_gather_for_generation True \
    --use_vllm False \
    --beta 0.04 \
    --num_generations 2 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing True \
    --torch_dtype bfloat16 \
    --attn_implementation flash_attention_2 \
    --bf16 True \
    --logging_steps 1 \
    --max_steps 100 \
    --data_subset 5000 \
    --logging False
