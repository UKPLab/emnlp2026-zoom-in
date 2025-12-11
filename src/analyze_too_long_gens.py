
filepath = "/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/helm/focusreason/runs/Qwen_2p5_7B_pr_data_warm_absolute_pixels_5k_image_tokens_min_image_500_strict_20251203_235005/run_log.txt"
with open(filepath) as f:
    lines = f.readlines()

print(lines[0])
