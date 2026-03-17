# merge_results.py
import argparse
import json
import os


def merge_results(shard_dirs: list[str], output_path: str, results_filename: str = "full_results.json"):
    merged = None
    found = 0

    for shard_dir in shard_dirs:
        result_file = os.path.join(shard_dir, results_filename)
        if not os.path.exists(result_file):
            print(f"WARNING: {result_file} not found, skipping")
            continue

        with open(result_file, "r") as f:
            shard_results = json.load(f)
        found += 1

        if merged is None:
            merged = shard_results
        else:
            for key in merged:
                if isinstance(merged[key], list) and isinstance(shard_results.get(key), list):
                    merged[key].extend(shard_results[key])

    if merged is None:
        print("ERROR: No result files found in any shard directory")
        return

    os.makedirs(output_path, exist_ok=True)
    out_file = os.path.join(output_path, results_filename)
    with open(out_file, "w") as f:
        json.dump(merged, f)

    # Print a quick summary
    total_samples = None
    for key, val in merged.items():
        if isinstance(val, list):
            if total_samples is None:
                total_samples = len(val)
            print(f"  {key}: {len(val)} entries")

    print(f"\nMerged {found} shards -> {out_file}  ({total_samples} total samples)")


if __name__ == "__main__":

    """parser = argparse.ArgumentParser(description="Merge full_results.json from multiple evaluation shards")
    parser.add_argument("--shard_dirs", type=str, nargs="+", required=True,
                        help="Paths to shard output directories (space-separated)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Directory to write the merged result into")
    parser.add_argument("--results_filename", type=str, default="full_results.json",
                        help="Name of the results file in each shard dir (default: full_results.json)")
    args = parser.parse_args()

    merge_results(args.shard_dirs, args.output_path, args.results_filename)"""

    base_path = "/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/focusreason/runs"
    model_path = "Qwen_2p5_7B_mini_o3_full_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_30_100_17p5_continue_pr_no_exploration_reward_20260313_003046/checkpoint-554/eval"

    tool_type = "zoom_in_absolute"
    shard_dirs = []
    for shard in range(8):
        dataset_path = f"dataset_mme_shard_{shard}_prompt_{tool_type}_max_pixels_3920000_min_pixels_392000_padding_0.1"
        shard_dirs.append(os.path.join(base_path, model_path, dataset_path))
    new_dataset_path = f"dataset_mme_prompt_{tool_type}_max_pixels_3920000_min_pixels_392000_padding_0.1"
    output_path = os.path.join(base_path, model_path, new_dataset_path)
    results_filename = "full_results.json"

    merge_results(shard_dirs, output_path, results_filename)
