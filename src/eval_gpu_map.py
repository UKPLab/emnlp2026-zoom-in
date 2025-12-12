from train_scheduler import start_or_resume_screen

if __name__ == '__main__':
    node = 3

    mapping_list = [
        {"model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_5k_image_tokens_min_image_500_20251204_000948", "gpu": 0},
        #{"model_path": "Qwen/Qwen2.5-VL-7B-Instruct", "gpu": 1},
        {"model_path": "TIGER-Lab/PixelReasoner-WarmStart", "gpu": 2},
        #{"model_path": "TIGER-Lab/PixelReasoner-RL-v1", "gpu": 3},
    ]

    for mapping in mapping_list:
        gpu = mapping["gpu"]
        print(f"start screen for {mapping}")
        start_or_resume_screen(f"{node}_eval_{gpu}",
                               None, #"/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/helm/focusreason/eval_log.log",
                               f"cd /pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/helm/focusreason/src && python initiate_analysis.py --model_path {mapping['model_path']}",
                               {"CUDA_VISIBLE_DEVICES": f"{gpu}"})
