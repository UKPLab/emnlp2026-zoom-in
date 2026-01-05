from train_scheduler import start_or_resume_screen

if __name__ == '__main__':
    node = 3

    mapping_list = [
        {"model_path": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_5k_image_tokens_min_image_500_pruning_0p05_20260102_184811", "gpu": 0},
        {"model_path": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_5k_image_tokens_min_image_500_pruning_0p075_20260102_173932", "gpu": 1},
        {"model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_5k_image_tokens_min_image_500_no_tool_20260102_175500", "gpu": 2},
        {"model_path": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_500_5k_image_mi_iou_random_sum_20260102_183643", "gpu": 3},
        {"model_path": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_500_5k_image_mi_iou_long_curr_sum_20260102_183159", "gpu": 4},
        {"model_path": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_5k_image_tokens_min_image_500_pruning_0p025_20260102_185111", "gpu": 5},
    ]

    for mapping in mapping_list:
        gpu = mapping["gpu"]
        print(f"start screen for {mapping}")
        start_or_resume_screen(f"{node}_eval_{gpu}",
                               None, #"/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/helm/focusreason/eval_log.log",
                               f"cd /pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/helm/focusreason/src && python initiate_analysis.py --model_path {mapping['model_path']}",
                               {"CUDA_VISIBLE_DEVICES": f"{gpu}"})
