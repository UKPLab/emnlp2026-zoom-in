import os

def find_eval(eval_path):
    if os.path.exists(eval_path):
        for eval_dir in os.listdir(eval_path):
            results_exist = False
            for file in os.listdir(os.path.join(eval_path, eval_dir)):
                if file == "full_results.json":
                    results_exist = True
            if results_exist:
                print(f"\t\tfound finished eval: {eval_dir}")
            else:
                print(f"\t\tfound unfinished eval: {eval_dir}")

def overview(run_path):


    if os.path.exists(run_path):
        print(f"found run: {os.path.basename(os.path.normpath(run_path))}")
        for dir in os.listdir(run_path):
            if dir.startswith("checkpoint-"):
                chkp_type = "light"
                for chkp_dir in os.listdir(os.path.join(run_path, dir)):
                    if chkp_dir.startswith("global_step"):
                        chkp_type = "full"
                print(f"\tfound {chkp_type} checkpoint: {dir.removeprefix('checkpoint-')}")
                eval_path = os.path.join(run_path, dir, "eval")
                find_eval(eval_path)
            if dir.startswith("eval"):
                print(f"\tfound eval: {dir}")
                find_eval(os.path.join(run_path, dir))


if __name__ == "__main__":
    path_prefix = "/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/helm/focusreason/runs/"


    run_paths = ["Qwen_2p5_7B_tool_pr_data_cold_2_iterations_box_20250725_141909",
                 "Qwen_2p5_7B_tool_pr_data_warm_20250717_114421",
                 "Qwen_2p5_7B_tool_pr_data_warm_2_iterations_box_20250718_161414",
                 "Qwen_2p5_7B_tool_pr_data_warm_2_iterations_box_3_epochs_20250720_222037",
                 "Qwen_2p5_7B_tool_pr_data_warm_2_iterations_box_2k_tokens_20250720_222203",
                 "Qwen_2p5_3B_no_train",
                 "Qwen_2p5_7B_no_train",
                 "PixelReasoner_original_after_SFT",
                 "PixelReasoner_original_after_RL",
                 "Qwen_2p5_7B_pr_data_warm_absolute_pixels_20250806_135021",
                 "Qwen_2p5_7B_pr_data_warm_global_buffer_20250806_140735",
                 "Qwen_2p5_7B_pr_data_warm_global_buffer_top_p_95_20250806_145202"
                 ]
    for run_path in run_paths:
        overview(os.path.join(path_prefix, run_path))