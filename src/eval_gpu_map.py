from train_scheduler import start_or_resume_screen

if __name__ == '__main__':
    node = 5

    mapping_list = [

        #{
        #    "model_path":"Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_conditional_constant_tool_0p3_20260225_144739",
        #    "gpu": 0
        #},

        #{
        #    "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_1_epoch_30_100_17p5_tool_uses_2_20260225_112558",
        #    "gpu": 1
        #},

        #{
        #    "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_1_epoch_30_100_17p5_negatives_2_20260225_100616",
        #    "gpu": 2
        #},

        #########################

        {
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_5k_image_tokens_min_image_500_no_tool_20260227_014456",
            "gpu": 0
        },

        #{
        #    "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_1_epoch_30_100_17p5_tool_params_20260227_003116",
        #    "gpu": 1
        #},




        #{
        #    "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_5k_image_tokens_min_image_500_1_epoch_const_20260120_224520",
        #    "gpu": 3
        #},

        #{
        #    "model_path": "Mini-o3/Mini-o3-7B-v1",
        #    "gpu": 4
        #},

    ]

    for mapping in mapping_list:
        gpu = mapping["gpu"]
        print(f"start screen for {mapping}")
        start_or_resume_screen(f"{node}_eval_{gpu}",
                               None, #"/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/focusreason/eval_log.log",
                               f"cd /pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/focusreason/src && python initiate_analysis.py --model_path {mapping['model_path']}",
                               {"CUDA_VISIBLE_DEVICES": f"{gpu}"})
