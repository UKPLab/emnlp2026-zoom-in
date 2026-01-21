
import json
import matplotlib.pyplot as plt

filepath = "/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/focusreason/runs/Qwen_2p5_7B_pr_data_warm_absolute_pixels_5k_image_tokens_min_image_500_strict_20251203_235005/run_log.txt"
with open(filepath) as f:
    lines = f.readlines()

round = 1
max_completion_length = 512

search_string = f"completion_ids from conv round {round}: "

too_longs = []
for line in lines:
    if search_string in line:
        list_as_str = line.split(search_string)[-1]
        lst = json.loads(list_as_str)

        too_longs.append(sum(1 for comp in lst if len(comp) >= max_completion_length)/280)

plt.plot(too_longs)
plt.show()