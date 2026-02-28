import numpy as np
import os
import re
from typing import List
import ast

_MODEL_GENS_RE = re.compile(r"\bmodel_generations\b.*?:\s*(\[(?:.|\n)*\])\s*$")

def parse_model_generations_from_log_line(line: str) -> List[str]:
    """
    Returns the Python list of generations (strings) from a single log line.

    Assumptions:
    - The line contains "... model_generations ...: [<python-literal-list>]"
    - The list is written using Python repr (single quotes, backslash escapes, etc.)
    """
    m = _MODEL_GENS_RE.search(line)
    if not m:
        raise ValueError("Could not find a model_generations Python-list literal in the line.")

    payload = m.group(1)
    try:
        gens = ast.literal_eval(payload)
    except (SyntaxError, ValueError) as e:
        raise ValueError(f"Failed to literal-eval model_generations payload: {e}") from e

    if not isinstance(gens, list) or not all(isinstance(x, str) for x in gens):
        raise ValueError(f"Expected List[str], got: {type(gens)} with elements {[type(x) for x in gens[:3]]}")
    return gens

path_prefix = "/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/focusreason/runs/"
# total_matches = 2674
# "Qwen_2p5_7B_pr_data_warm_absolute_pixels_500_5k_image_mi_iou_sum_20251217_102117", 2674, rise between 100 and 500
#"Qwen_2p5_7B_pr_data_warm_absolute_pixels_500_5k_image_mi_iou_random_20251215_121302" 2674, drop between 300 and 400, i.e. 0.25-0.35
#"Qwen_2p5_7B_pr_data_warm_absolute_pixels_500_5k_image_mi_iou_long_curr_20251216_113951"
"Qwen_2p5_7B_pr_data_warm_absolute_pixels_500_5k_image_mi_iou_random_sum_20260102_183643" # 2247
"Qwen_2p5_7B_pr_data_warm_absolute_pixels_500_5k_image_mi_iou_long_curr_sum_20260102_183159" # 2247
"Qwen_2p5_7B_pr_data_warm_absolute_pixels_500_5k_image_mi_iou_random_sum_20260102_183643" # 1337
"Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_1_epoch_30_100_17p5_no_tanh_20260128_142853" # 1337
"Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_30_100_17p5_first_20_mean_20260210_123020" # 1337
"Qwen_2p5_7B_mini_o3_full_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_30_100_17p5_continue_pr_20260218_002110" # 1939



run_path = "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_30_100_17p5_first_20_mean_roadmap_20260223_161244" # 1337
filepath = os.path.join(path_prefix, run_path, "run_log.txt")
with open(filepath) as f:
    lines = f.readlines()

search_string = f"model_generations for reward calculation:"

too_longs = []
print(len(lines))
matches = 0.0
total_matches = 1337
targets = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9])
#targets += 0.01
#targets = np.array([0.7, 0.73, 0.76, 0.8, 0.83, 0.86, 0.9, 0.93, 0.96])

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
            #print(f"original line")
            #print(line)
            #print(f"\n\nparsed line")
            generation_list = parse_model_generations_from_log_line(line)
            for gen in generation_list:
                if "</tool_call>" in gen:
                    second_gen = gen.split("</tool_call>")[-1]
                    print(second_gen)
                    break
            #splitted = re.split(r"(?<!\\)'", line)
            #second_gen = splitted[1].split("</tool_call>")[-1]
            #print(second_gen)
        except Exception:
            pass
            print(f"could not be parsed, skipping")

        #try:
        #    lst = ast.literal_eval(line)
        #    print(lst[0])
        #except Exception:
        #    print(f"could not be parsed, skipping")
print(f"total matches: {matches}")


