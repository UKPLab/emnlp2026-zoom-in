from train_scheduler import start_or_resume_screen

if __name__ == '__main__':
    node = 5

    mapping_list = [

        {
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_1_epoch_30_100_17p5_sample_recall_20260302_042226",
                "gpu": 2
        },
        #{
        #    "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_1_epoch_30_100_15_20260301_145725",
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
