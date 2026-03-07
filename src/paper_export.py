import pandas as pd

from initiate_analysis import analyze

def main_results_table():
    models = [{"model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_5k_image_tokens_min_image_500_no_tool_20260227_014456",
               "short_name": "no tool"},
              {"model_path": "Qwen/Qwen2.5-VL-7B-Instruct",
                "short_name": "Qwen no train"},
              {"model_path": "TIGER-Lab/PixelReasoner-RL-v1",
               "short_name": "PixelReasoner"},
              {"model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_5k_image_tokens_min_image_500_1_epoch_const_20260120_224520",
               "short_name": "Curiosity",
               "tool_padding": [0.1]},
              {"model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_1_epoch_30_100_17p5_20260123_184044",
               "short_name": "Ours",
               "tool_padding": [0.1]}

              ]
    datasets = ["V*", "hrb_4k", "hrb_8k", "mme"]#, "InfoVQA"]
    datasets_map = {"V*": "pixel_reasoner_vstar",
        "InfoVQA": "pixel_reasoner_infovqa",
        "hrb_4k":"hr_bench_4k",
        "hrb_8k": "hr_bench_8k",
        # "mme_lite",
        "mme": "mme"}
    tool_config_types = [
        "no_tool",
        "PR_crop_image_normalized,select_frames",
        "zoom_in_absolute",
    ]
    metrics = ["accuracy"]
    model_paths = [model["model_path"] for model in models]
    short_names = [model["short_name"] for model in models]
    paddings = [model["tool_padding"] if "tool_padding" in model and model["tool_padding"] is not None else None for model in models]

    results_df = analyze(model_paths=model_paths, tool_config_types=tool_config_types,
                         datasets_for_analysis=[datasets_map[d] for d in datasets],
                         do_print=False, tool_paddings = paddings, metrics=metrics)

    cols = []
    for dataset in datasets:
        cols.append((dataset, "no_tool", "accuracy"))
        cols.append((dataset, "tool", "accuracy"))

    print(f"after analyze: {results_df.head()}")
    pruned_df = results_df[cols]

    pruned_df.rename(mapper=dict(zip(range(len(short_names)), short_names)), axis=0, inplace=True)

    aggregate_cols_no_tool = []
    aggregate_cols_tool = []
    for dataset in datasets:
        aggregate_cols_no_tool.append((f"{dataset}", "no_tool", "accuracy"))
        aggregate_cols_tool.append((f"{dataset}", "tool", "accuracy"))

    pruned_df[("overall", "no tool", "avg")] = pruned_df[aggregate_cols_no_tool].mean(axis=1)
    pruned_df[("overall", "tool", "avg")] = pruned_df[aggregate_cols_tool].mean(axis=1)

    #pruned_df = pruned_df.round(2)
    pruned_df = pruned_df.to_latex(float_format="%.2f")
    print(f"final: {pruned_df}")

def acc_at_t():

        models = [
            {"model_path": "Mini-o3/Mini-o3-7B-v1",
             "short_name": "Mini o3",
             "tool_padding": ['32_turns_sample_1']},
        ]
        datasets = ["mme", ]  # , , "hrb_8k", "mme", "V*"
        datasets_map = {"V*": "pixel_reasoner_vstar",
                        "InfoVQA": "pixel_reasoner_infovqa",
                        "hrb_4k": "hr_bench_4k",
                        "hrb_8k": "hr_bench_8k",
                        # "mme_lite",
                        "mme": "mme"}
        tool_config_types = [
            "zoom_in_relative",
        ]
        metrics = [#"accuracy@0", "accuracy@1", "accuracy@2", "accuracy@4",
                   "accuracy@6",
                   #"accuracy@8", "accuracy@16", "accuracy@32"
            ]
        model_paths = [model["model_path"] for model in models]
        short_names = [model["short_name"] for model in models]
        paddings = [model["tool_padding"] if "tool_padding" in model and model["tool_padding"] is not None else None for
                    model in models]

        results_df = analyze(model_paths=model_paths, tool_config_types=tool_config_types,
                             datasets_for_analysis=[datasets_map[d] for d in datasets],
                             do_print=False, tool_paddings=paddings, metrics=metrics)

        cols = []
        for dataset in datasets:
            for metric in metrics:
                cols.append((dataset, "tool", metric))

        print(f"after analyze: {results_df.head()}")
        pruned_df = results_df[cols]

        pruned_df.rename(mapper=dict(zip(range(len(short_names)), short_names)), axis=0, inplace=True)

        #aggregate_cols_tool = []
        #for dataset in datasets:
        #    aggregate_cols_tool.append((f"{dataset}", "tool", "accuracy"))

        #pruned_df[("overall", "tool", "avg")] = pruned_df[aggregate_cols_tool].mean(axis=1)

        # pruned_df = pruned_df.round(2)
        pruned_df = pruned_df.to_latex(float_format="%.2f")
        print(f"final: {pruned_df}")


def muffin_main(task:str):
    models = [
        {"model_path": "Qwen/Qwen2.5-VL-7B-Instruct",
         "short_name": "Qwen no train"},

        {"model_path": "TIGER-Lab/PixelReasoner-RL-v1",
         "short_name": "PixelReasoner"},

        {"model_path": "Mini-o3/Mini-o3-7B-v1",
         "short_name": "Mini o3",
         "tool_padding": ['32_turns_sample_4']},

        #wait for eval
        {"model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_5k_image_tokens_min_image_500_no_tool_20260227_014456",
         "short_name": "no tool"},

        {"model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_5k_image_tokens_min_image_500_1_epoch_const_20260120_224520",
         "short_name": "Curiosity",
         "tool_padding": [0.1]},

        {"model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_1_epoch_30_100_17p5_20260123_184044",
         "short_name": "Ours",
         "tool_padding": [0.1]},

        ]

    datasets = [{"name": "muffin_chihuahua",
                              "grid_pixels": [1, 2, 4, 8],
                              "gridsize": [1, 2, 4, 8, 16],
                              "mode": ["single_cell_query", "find_outlier"]}]
    tool_config_types = [
        "no_tool",
        "PR_crop_image_normalized,select_frames",
        "zoom_in_absolute",
        "zoom_in_relative"
    ]
    metrics = ["accuracy"]
    model_paths = [model["model_path"] for model in models]
    short_names = [model["short_name"] for model in models]
    paddings = [model["tool_padding"] if "tool_padding" in model and model["tool_padding"] is not None else None for
                model in models]

    results_df = analyze(model_paths=model_paths, datasets_for_analysis=datasets,
                         do_print=False, tool_paddings=paddings, tool_config_types=tool_config_types,
                         metrics=metrics)

    results_df.rename(mapper=dict(zip(range(len(short_names)), short_names)), axis=0, inplace=True)

    print(f"results_df columns: {results_df.columns}")

    final_df = pd.DataFrame(index=results_df.index)

    if task == "scq":
        grid_sizes = [1,2,4,8,16]
    elif task == "fo":
        grid_sizes = [2,4,8,16]
    else:
        raise ValueError(f"Unsupported task: {task}")

    aggregate_cols_no_tool = []
    aggregate_cols_tool = []
    for grid_size in grid_sizes:
        cols_for_no_tool_mean = []
        cols_for_tool_mean = []
        for img_size in [1, 2, 4, 8]:
            # 1k_1_scq
            cols_for_no_tool_mean.append((f"{img_size}k_{grid_size}_{task}", "no_tool", "accuracy"))
            cols_for_tool_mean.append((f"{img_size}k_{grid_size}_{task}", "tool", "accuracy"))

        aggregate_col_no_tool = (f"{grid_size}x{grid_size}", "no tool", "avg")
        aggregate_cols_no_tool.append(aggregate_col_no_tool)
        final_df[aggregate_col_no_tool] = results_df[cols_for_no_tool_mean].mean(axis=1)
        aggregate_col_tool = (f"{grid_size}x{grid_size}", "tool", "avg")
        aggregate_cols_tool.append(aggregate_col_tool)
        final_df[(f"{grid_size}x{grid_size}", "tool", "avg")] = results_df[cols_for_tool_mean].mean(axis=1)
    final_df[("overall", "no tool", "avg")] = final_df[aggregate_cols_no_tool].mean(axis=1)
    final_df[("overall", "tool", "avg")] = final_df[aggregate_cols_tool].mean(axis=1)


    final_df.columns = pd.MultiIndex.from_tuples(final_df.columns, names=["grid", "tool", "metric"])
    final_df = final_df.to_latex(float_format="%.2f")
    print(final_df)

def mini_o3_no_answer():
    models = [

        {"model_path": "Mini-o3/Mini-o3-7B-v1",
         "short_name": "Mini o3",
         "tool_padding": ['32_turns_sample_4']},

    ]

    datasets = [{"name": "muffin_chihuahua",
                 "grid_pixels": [1, 2, 4, 8],
                 "gridsize": [1, 2, 4, 8, 16],
                 "mode": ["single_cell_query", "find_outlier"]}]

    tool_config_types = [
        "zoom_in_relative"
    ]
    metrics = ["no_answer"]
    model_paths = [model["model_path"] for model in models]
    short_names = [model["short_name"] for model in models]
    paddings = [model["tool_padding"] if "tool_padding" in model and model["tool_padding"] is not None else None for
                model in models]

    results_df = analyze(model_paths=model_paths, datasets_for_analysis=datasets,
                         do_print=False, tool_paddings=paddings, tool_config_types=tool_config_types,
                         metrics=metrics)

    results_df.rename(mapper=dict(zip(range(len(short_names)), short_names)), axis=0, inplace=True)
    cols_to_keep = [col for col in results_df.columns if col[2] == "no_answer"]
    results_df = results_df[cols_to_keep]

    task_dfs = []
    for task in ["scq", "fo"]:
        task_df = pd.DataFrame()
        if task == "scq":
            grid_sizes = [1,2,4,8,16]
        elif task == "fo":
            grid_sizes = [2,4,8,16]
        else:
            raise ValueError(f"Unsupported task: {task}")

        aggregate_cols_tool = []
        for grid_size in grid_sizes:
            cols_for_tool_mean = []
            for img_size in [1, 2, 4, 8]:
                # 1k_1_scq
                cols_for_tool_mean.append((f"{img_size}k_{grid_size}_{task}", "tool", "no_answer"))

            aggregate_col_tool = (f"{grid_size}x{grid_size}", "tool", "avg")
            aggregate_cols_tool.append(aggregate_col_tool)
            task_df[f"{grid_size}x{grid_size}"] = results_df[cols_for_tool_mean].mean(axis=1)
        task_df.rename(mapper={"Mini o3": task}, axis=0, inplace=True)
        task_dfs.append(task_df)
        #final_df[("overall", "no tool", "avg")] = final_df[aggregate_cols_no_tool].mean(axis=1)
        #final_df[("overall", "tool", "avg")] = final_df[aggregate_cols_tool].mean(axis=1)
    final_df = pd.concat(task_dfs, axis=0)
    #print(final_df)
    print(final_df.to_latex(float_format="%.2f"))

def muffin_overlap_metrics(task):
    models = [

        #{"model_path": "TIGER-Lab/PixelReasoner-RL-v1",
        # "short_name": "PixelReasoner"},

        #{"model_path": "Mini-o3/Mini-o3-7B-v1",
        # "short_name": "Mini o3",
        # "tool_padding": ['32_turns']},

        {"model_path": "Mini-o3/Mini-o3-7B-v1",
         "short_name": "Mini o3",
         "tool_padding": ['32_turns_sample_4']},

        #{
        #    "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_5k_image_tokens_min_image_500_1_epoch_const_20260120_224520",
        #    "short_name": "Curiosity",
        #    "tool_padding": [0.1]},

        #{
        #    "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_1_epoch_30_100_17p5_20260123_184044",
        #    "short_name": "Ours",
        #    "tool_padding": [0.1]},
    ]

    datasets = [{"name": "muffin_chihuahua",
                 "grid_pixels": [1, 2, 4, 8],
                 "gridsize": [1, 2, 4, 8, 16],
                 "mode": ["single_cell_query", "find_outlier"]}]

    tool_config_types = [
        "PR_crop_image_normalized,select_frames",
        "zoom_in_absolute",
        "zoom_in_relative"
    ]

    metrics = ["accuracy", "precision", "precision_std", "recall", "recall_std", "iou_std", "no_answer"]#, "no_answer"]
    model_paths = [model["model_path"] for model in models]
    short_names = [model["short_name"] for model in models]
    paddings = [model["tool_padding"] if "tool_padding" in model and model["tool_padding"] is not None else None for
                model in models]

    results_df = analyze(model_paths=model_paths, datasets_for_analysis=datasets,
                         do_print=False, tool_paddings=paddings, tool_config_types=tool_config_types,
                         metrics=metrics)

    results_df.rename(mapper=dict(zip(range(len(short_names)), short_names)), axis=0, inplace=True)
    print(results_df.columns)

    if task == "scq":
        grid_sizes = [2,4,8,16]
    elif task == "fo":
        grid_sizes = [2,4,8,16]
    else:
        raise ValueError(f"Unsupported task: {task}")

    final_df = pd.DataFrame(index=results_df.index)

    metrics.insert(-1, "ious")

    aggregate_cols = {metric:[] for metric in metrics}
    for grid_size in grid_sizes:
        for img_size in [1, 2, 4, 8]:
            for metric in metrics:
                # 1k_1_scq
                aggregate_cols[metric].append((f"{img_size}k_{grid_size}_{task}", "tool", metric))


    for metric in metrics:
        if "_" in metric:
            metric_display_name = (metric.split("_")[0], "std", "value")
        else:
            metric_display_name = (metric, "mean", "value")
        final_df[metric_display_name] = results_df[aggregate_cols[metric]].mean(axis=1)
        if metric != "accuracy":
            acc_df = results_df[aggregate_cols["accuracy"]].copy()
            other_df = results_df[aggregate_cols[metric]].copy()
            acc_df.columns = acc_df.columns.droplevel(-1)
            other_df.columns = other_df.columns.droplevel(-1)

            metric_display_name = (metric_display_name[0], metric_display_name[1], "pearson")
            final_df[metric_display_name] = acc_df.corrwith(other_df, axis=1, method="pearson")


    #for corr in ["pearson", "spearman"]:
    #    corr_row = {}
    #    for metric in metrics:
    #        if metric != "accuracy":
    #            corr_row[metric] = final_df[["accuracy", metric]].corr(method=corr).iloc[0,1]
    #    final_df.loc[corr] = corr_row
    print(final_df)
    final_df.columns = pd.MultiIndex.from_tuples(final_df.columns, names=["overlap metric", "X", "XX"])
    print(final_df.to_latex(float_format="%.2f"))



if __name__ == "__main__":
    #main_results_table()
    #acc_at_t()
    #muffin_main(task="fo")
    #mini_o3_no_answer()
    muffin_overlap_metrics(task="scq")