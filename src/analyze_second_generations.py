import numpy as np
import os
import re

path_prefix = "/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/focusreason/runs/"
# total_matches = 2674
# "Qwen_2p5_7B_pr_data_warm_absolute_pixels_500_5k_image_mi_iou_sum_20251217_102117", 2674, rise between 100 and 500
#"Qwen_2p5_7B_pr_data_warm_absolute_pixels_500_5k_image_mi_iou_random_20251215_121302" 2674, drop between 300 and 400, i.e. 0.25-0.35
#"Qwen_2p5_7B_pr_data_warm_absolute_pixels_500_5k_image_mi_iou_long_curr_20251216_113951"
"Qwen_2p5_7B_pr_data_warm_absolute_pixels_500_5k_image_mi_iou_random_sum_20260102_183643" # 2247
"Qwen_2p5_7B_pr_data_warm_absolute_pixels_500_5k_image_mi_iou_long_curr_sum_20260102_183159" # 2247
"Qwen_2p5_7B_pr_data_warm_absolute_pixels_500_5k_image_mi_iou_random_sum_20260102_183643" # 1337

run_path = "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_1_epoch_30_100_17p5_no_tanh_20260128_142853"

filepath = os.path.join(path_prefix, run_path, "run_log.txt")
with open(filepath) as f:
    lines = f.readlines()



search_string = f"model_generations for reward calculation:"

too_longs = []
print(len(lines))
matches = 0.0
total_matches = 1337
targets = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9])
for line in lines:
    if search_string in line:
        matches += 1.0
        if not np.min(np.abs(targets - matches /total_matches)) < 0.0005:
            #print(np.min(np.abs(targets - matches /2674.0)))
            #print(np.abs(targets - matches /2674.0))
            continue
        print(f"use match: {matches /total_matches}")
        #print("found!")
        #print(line)
        try:
            print(line)
            splitted = re.split(r"(?<!\\)'", line)
            second_gen = splitted[1].split("</tool_call>")[-1]
            print(second_gen)
        except Exception:
            pass
            print(f"could not be parsed, skipping")

        #try:
        #    lst = ast.literal_eval(line)
        #    print(lst[0])
        #except Exception:
        #    print(f"could not be parsed, skipping")
print(f"total matches: {matches}")


