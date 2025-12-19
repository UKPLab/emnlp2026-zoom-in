from train_scheduler import start_or_resume_screen

if __name__ == '__main__':
    node = 3

    mapping_list = [
        {"model_path": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_500_5k_image_mi_iou_random_20251215_121302", "gpu": 0},
        {"model_path": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_500_5k_image_mi_iou_20251216_010228", "gpu": 1},
        {"model_path": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_500_5k_image_mi_iou_long_curr_20251216_113951", "gpu": 2},
        {"model_path": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_500_5k_image_mi_iou_sum_20251217_102117", "gpu": 3},
    ]

    for mapping in mapping_list:
        gpu = mapping["gpu"]
        print(f"start screen for {mapping}")
        start_or_resume_screen(f"{node}_eval_{gpu}",
                               None, #"/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/helm/focusreason/eval_log.log",
                               f"cd /pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/helm/focusreason/src && python initiate_analysis.py --model_path {mapping['model_path']}",
                               {"CUDA_VISIBLE_DEVICES": f"{gpu}"})
