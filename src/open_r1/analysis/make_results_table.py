import os
import json
from json import JSONDecodeError
import numpy as np
import pandas as pd

from open_r1.utils.logger import get_logger, setup_project_logging
from open_r1.utils.tools import TOOL_CONFIGS
from open_r1.utils.rewards import accuracy_reward
from scipy.stats import pearsonr


logger = get_logger(__name__)

def get_results(path:str, metrics: list[str]):
    try:
        data = json.load(open(path, "r"))
    except FileNotFoundError:
        logger.info(f"File not found: {path}")
        return None
    except JSONDecodeError:
        logger.info(f"Could not decode JSON (empty file?): {path}")
        return None
    result = {}
    uncropped = []
    #logger.info(data)
    for metric in metrics:
        try:
            if metric == "accuracy":
                result[metric] = np.mean(np.array(data["accuracy"]))*100
            if metric == "avg_pixel_reasoning":
                result[metric] = np.mean(np.array(data["tool_use"]))
            if metric == "pixel_reasoning_distr":
                distr = []
                for i in range(7):
                    distr.append(np.mean(np.array(data["tool_use"]) == i).round(4)*100)
                result[metric] = distr
            if metric == "avg_image_size":
                pixels = np.array([sample[0][0] * sample[0][1] for idx, sample in enumerate(data["image_sizes"]) if idx not in uncropped])
                result[metric] = np.mean(pixels)
            if metric == "max_image_size":
                pixels = np.array([sample[0][0] * sample[0][1] for idx, sample in enumerate(data["image_sizes"]) if
                                   idx not in uncropped])
                result[metric] = np.max(pixels)
            if metric == "min_image_size":
                pixels = np.array([sample[0][0] * sample[0][1] for idx, sample in enumerate(data["image_sizes"]) if
                                   idx not in uncropped])
                result[metric] = np.min(pixels)
            if metric == "median_image_size":
                pixels = np.array([sample[0][0] * sample[0][1] for idx, sample in enumerate(data["image_sizes"]) if
                                   idx not in uncropped])
                result[metric] = np.median(pixels)
            if metric == "zoom_in_fraction_avg":
                image_fractions = np.array([(sample[1][0] * sample[1][1])/(sample[0][0] * sample[0][1]) for idx, sample in enumerate(data["image_sizes"]) if len(sample) > 1])
                result[metric] = np.mean(image_fractions)
            if metric == "zoom_in_fraction_median":
                image_fractions = np.array([(sample[1][0] * sample[1][1])/(sample[0][0] * sample[0][1]) for idx, sample in enumerate(data["image_sizes"]) if len(sample) > 1])
                result[metric] = np.median(image_fractions)
            if metric == "avg_total_completion_len":
                result[metric] = np.mean(np.array([sum(sample) for sample in data["completion_len"]]))
            if metric == "avg_first_completion_len":
                result[metric] = np.mean(np.array([sample[0] for sample in data["completion_len"]]))
            if metric == "avg_second_completion_len":
                result[metric] = np.mean(np.array([sample[1] for sample in data["completion_len"] if len(sample) > 1]))
            if metric == "image_size_x_performance_correlation":
                pixels = np.array([sample[0][0] * sample[0][1] for idx, sample in enumerate(data["image_sizes"]) if
                                   idx not in uncropped])
                accuracies = np.array([sample for idx, sample in enumerate(data["accuracy"]) if idx not in uncropped])
                correlation, p_value = pearsonr(pixels, accuracies)
                result[metric] = correlation #f"{correlation},{p_value}"
            if metric == "accuracy_if_tool_used":
                tool_use_indices = [idx for idx, sample in enumerate(data["tool_use"]) if sample > 0]
                result[metric] = np.mean(np.array([data["accuracy"][idx] for idx in tool_use_indices]))
            if metric == "accuracy_if_tool_not_used":
                non_tool_use_indices = [idx for idx, sample in enumerate(data["tool_use"]) if sample == 0]
                result[metric] = np.mean(np.array([data["accuracy"][idx] for idx in non_tool_use_indices]))
            if metric == "tool_success_rate":
                tool_use_array = np.array(data["tool_use"], dtype=np.float16)
                attempted_tool_use_array = np.array(data["attempted_tool_use"], dtype=np.float16)
                attempts_mask = attempted_tool_use_array != 0
                result[metric] = np.mean(tool_use_array[attempts_mask] / attempted_tool_use_array[attempts_mask])
            if isinstance(metric, dict):
                metric_name = metric["metric_name"]

                accu_reward_method = [metric_name for _ in range(len(data["model_answer"]))]

                new_metric = accuracy_reward(completions=data["model_answer"], solution=data["solution"],
                                             accu_reward_method=accu_reward_method,
                                             **metric)
                result[metric["metric_short_name"]] = np.mean(np.array(new_metric))




        except KeyError as e:
            logger.info(f"KeyError for metric {metric} in {path}. Existing keys are {data.keys()}. {e}")
            result[metric] = None

    return result

def compute_new_metric(save_path, metrics: list[dict[str, str]], exist_ok=False):
    logger.info(f"metrics: {metrics}")
    try:
        with open(save_path, "r") as f:
            data = json.load(f)
        #data = json.load(open(os.path.join(save_path, "full_results.json"))
    except FileNotFoundError:
        logger.info(f"Could not add metric, File not found: {save_path}")
        return
    logger.info(f"old metric: {np.mean(np.array(data['accuracy']))}")
    for metric in metrics:
        metric_name = metric["metric_name"]
        if metric_name in data.keys():
            if exist_ok:
                logger.info(f"Metric already exists, will overwrite.")
            else:
                logger.info(f"Metric already exists, Skipping. If you want to overwrite, please set exist_ok=True.")
                return

        accu_reward_method = [metric_name for _ in range(len(data["model_answer"]))]

        new_metric = accuracy_reward(completions=data["model_answer"], solution=data["solution"],
                        accu_reward_method=accu_reward_method,
                        **metric)

        logger.info(f"new metric: {np.mean(np.array(new_metric))}")
        #data[metric_name] = new_metric
    # with open(save_path, "w") as f:
    #    json.dump(data, f, indent=4)





