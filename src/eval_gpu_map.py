from train_scheduler import start_or_resume_screen

if __name__ == '__main__':
    node = 5

    mapping_list = [

        {
            "model_path":"Qwen_2p5_7B_pr_data_cold_absolute_pixels_5k_image_tokens_min_image_500_1_epoch_const_padding_0p05_20260219_234642",
            "gpu": 1
        },

        {
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_5k_image_tokens_min_image_500_1_epoch_const_padding_0p0_20260219_234130",
            "gpu": 2
        },

        {
            "model_path": "Qwen_2p5_7B_mini_o3_full_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_30_100_17p5_continue_pr_20260218_002110",
            "gpu": 3
        },




    ]

    for mapping in mapping_list:
        gpu = mapping["gpu"]
        print(f"start screen for {mapping}")
        start_or_resume_screen(f"{node}_eval_{gpu}",
                               None, #"/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/focusreason/eval_log.log",
                               f"cd /pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/focusreason/src && python initiate_analysis.py --model_path {mapping['model_path']}",
                               {"CUDA_VISIBLE_DEVICES": f"{gpu}"})
