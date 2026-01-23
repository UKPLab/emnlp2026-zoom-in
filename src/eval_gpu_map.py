from train_scheduler import start_or_resume_screen

if __name__ == '__main__':
    node = 5

    mapping_list = [



        #{
        # "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_long_curr_sum_max_0p6_conditional_20260111_144201",
        #    "gpu": 0},

        # manually started on #0
        #{
        #    "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_long_curr_sum_max_0p6_infonce_20260111_143637",
        #    "gpu": 1},


         #{
         #   "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_long_curr_sum_max_0p6_conditional_infonce_20260112_192315",
         #   "gpu": 1},

        #{
        #    "model_path": "Qwen_2p5_7B_muffin_warm_absolute_pixels_500_5k_iou_reward_20260120_094018",
        #    "gpu": 2

        #}

        #{
        #    "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_5k_image_tokens_min_image_500_1_epoch_const_20260120_224520",
        #    "gpu": 0},

        #{
        #    "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_5k_image_tokens_min_image_500_1_epoch_const_long_warmup_20260120_224700",
        #    "gpu": 1},

        {
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_5k_image_tokens_min_image_500_1_epoch_cosine_long_warmup_20260120_235613",
            "gpu": 2},












    ]

    for mapping in mapping_list:
        gpu = mapping["gpu"]
        print(f"start screen for {mapping}")
        start_or_resume_screen(f"{node}_eval_{gpu}",
                               None, #"/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/focusreason/eval_log.log",
                               f"cd /pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/focusreason/src && python initiate_analysis.py --model_path {mapping['model_path']}",
                               {"CUDA_VISIBLE_DEVICES": f"{gpu}"})
