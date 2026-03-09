import json

path = "/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/focusreason/runs/Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_1_epoch_30_100_17p5_20260123_184044/checkpoint-382/eval/dataset_pixel_reasoner_vstar_prompt_zoom_in_absolute_max_pixels_3920000_min_pixels_392000_padding_0.1_temp_1.0_samples_2/full_results.json"

data = json.load(open(path, "r"))

print(data.keys())

for key in data.keys():
    print(key, len(data[key]))
print(data["accuracy"])
print(data["solution"][:6])

#print(data["query"][:6])
print(data["model_answer"][:6])
#print(data["model_answer"][1])
#print(len(data["accuracy"]))

both_correct = 0
both_incorrect = 0
mixed = 0
for idx in range(0, len(data["accuracy"]), 2):
    if data["accuracy"][idx] == data["accuracy"][idx+1] == 1:
        both_correct += 1
    if data["accuracy"][idx] == data["accuracy"][idx + 1] == 0:
        both_incorrect += 1
    if data["accuracy"][idx] != data["accuracy"][idx + 1]:
        mixed += 1

print(both_correct)
print(both_incorrect)
print(mixed)