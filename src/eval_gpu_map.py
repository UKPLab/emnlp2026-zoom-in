import base64

from train_scheduler import start_or_resume_screen
import json
import shlex

if __name__ == '__main__':
    node = 4

    mapping_list = []

    for gpu_id in range(8):

        if gpu_id == 0:
            mode = "single_cell_query"
            grid_size = [1,2]
        if gpu_id == 1:
            mode = "single_cell_query"
            grid_size = [4]
        if gpu_id == 2:
            mode = "single_cell_query"
            grid_size = [8]
        if gpu_id == 3:
            mode = "single_cell_query"
            grid_size = [16]

        if gpu_id == 4:
            mode = "find_outlier"
            grid_size = [1,2]
        if gpu_id == 5:
            mode = "find_outlier"
            grid_size = [4]
        if gpu_id == 6:
            mode = "find_outlier"
            grid_size = [8]
        if gpu_id == 7:
            mode = "find_outlier"
            grid_size = [16]

        dataset_dict = {"name": "muffin_chihuahua",
            "grid_pixels": [1, 2, 4, 8],
            "gridsize": grid_size,
            "mode": [mode]}

        mapping_list.append({
            "model_path": "Qwen_2p5_7B_mini_o3_full_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_30_100_17p5_continue_pr_no_exploration_reward_20260313_003046",
            "gpu": gpu_id,
            "dataset_name": base64.b64encode(json.dumps(dataset_dict).encode()).decode(),
            "tool_config_type": "zoom_in_absolute"
        })

    #for gpu_id in range(8):

    #    mapping_list.append({
    #        "model_path": "Qwen_2p5_7B_mini_o3_full_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_30_100_17p5_continue_pr_no_exploration_reward_20260313_003046",
    #        "gpu": gpu_id,
    #        "dataset_name": f"mme_shard_{gpu_id}",
    #        "tool_config_type": "zoom_in_absolute"
    #    })

    """
    mapping_list = [
        # 4 JAIF: 0,1: no tool, 2,3: absolute
        {
            "model_path": ,
                "gpu": 2
        },
    
    ]
    """

    for mapping in mapping_list:
        gpu = mapping["gpu"]
        print(f"start screen for {mapping}")
        start_or_resume_screen(f"{node}_eval_{gpu}",
                               None, #"/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/focusreason/eval_log.log",
                               f"cd /pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/focusreason/src && python initiate_analysis.py "
                               f"--model_path {mapping['model_path']} "
                               f"--dataset_name {mapping['dataset_name']} "
                               f"--tool_config_type {mapping['tool_config_type']}",
                               {"CUDA_VISIBLE_DEVICES": f"{gpu}"})
