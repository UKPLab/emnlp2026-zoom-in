import os

import pandas as pd
import subprocess
from open_r1.utils.logger import setup_project_logging
from open_r1.analysis.make_results_table import get_results, compute_new_metric
from open_r1.utils.rewards import accuracy_reward
import sys
import itertools
import json
import numpy as np

from open_r1.analysis.analyze_PMI import do_pmi_analysis

logger = setup_project_logging(log_file=None)

class ModelParams:

    def __init__(self, short_name, model_path, checkpoint, model_class, output_path=None,
                 dir_prefix="/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/helm/focusreason/runs/",
                 dir_suffix="eval", **kwargs):
        self.dir_prefix = dir_prefix
        self.dir_suffix = dir_suffix


        self.short_name = short_name
        self.model_path = model_path

        self.checkpoint = checkpoint
        self.output_path = output_path

        if self.checkpoint is not None:
            self.full_model_path = os.path.join(dir_prefix, self.model_path, f"checkpoint-{checkpoint}")
            self.full_output_path = self.full_model_path
        else:
            self.full_output_path = os.path.join(dir_prefix, self.output_path)
            self.full_model_path = self.model_path


        self.model_class = model_class


class DatasetParams:
    def __init__(self, dataset_name, **kwargs):
        self.dataset_name = dataset_name
        if self.dataset_name == "pixel_reasoner":
            self.data_files = '/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/helm/datasets/pixel_reasoner/RL_data_without_video/train.jsonl'
            self.image_folders = '/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/helm/datasets/pixel_reasoner/RL_data_without_video'
        elif self.dataset_name == "pixel_reasoner_vstar":
            self.data_files = "/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/helm/datasets/pixel_reasoner/eval/V_Star/test.jsonl"
            self.image_folders = "/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/helm/datasets/pixel_reasoner/eval/V_Star"
        elif self.dataset_name == "pixel_reasoner_infovqa":
            self.data_files = "/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/helm/datasets/pixel_reasoner/eval/Infographics_VQA/test.jsonl"
            self.image_folders = "/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/helm/datasets/pixel_reasoner/eval/Infographics_VQA"
        else:
            raise NotImplementedError(f"dataset {dataset_name} not implemented!")



class EvalParams:
    def __init__(self, batch_size=200, tensor_parallel_size=1, enforce_eager=True, no_vllm=False, dry_run=False,
                 min_pixels=None, max_pixels=None, tool_config_type = "no_tool", image_limit=6, max_tool_uses=5, **kwargs):
        # technical
        self.batch_size = batch_size
        self.tensor_parallel_size = tensor_parallel_size
        self.enforce_eager = enforce_eager
        self.no_vllm = no_vllm
        self.dry_run = dry_run

        # outcome relevant
        if max_pixels is None:
            self.max_pixels = 1024*16 * 28 * 28
        else:
            self.max_pixels = max_pixels

        if min_pixels is None:
            self.min_pixels = 4*28*28
        else:
            self.min_pixels = min_pixels

        self.tool_config_type = tool_config_type
        self.image_limit = image_limit
        self.max_tool_uses = max_tool_uses

        if self.dry_run:
            self.batch_size = 2
            self.tensor_parallel_size = 1
            self.enforce_eager = True



class SingleEval:
    def __init__(self, model_params: ModelParams, dataset_params: DatasetParams, eval_params: EvalParams):

        self.model_params = model_params
        self.dataset_params = dataset_params
        self.eval_params = eval_params


    def get_command(self, save_path ):
        cmd = []
        cmd += [
            'python', '-m', 'open_r1.evaluation.evaluate',
            '--model_path', f'{self.model_params.full_model_path}',
            '--model_class', f'{self.model_params.model_class}',
            '--output_path', f'{save_path}',

            '--dataset_name', self.dataset_params.dataset_name,
            # '--prompt_type', f"{prompt_type}",
            '--data_filepath', self.dataset_params.data_files,
            '--image_filepath', self.dataset_params.image_folders,

            '--tensor_parallel_size', f'{self.eval_params.tensor_parallel_size}',
            '--image_limit', f'{self.eval_params.image_limit}',
            '--max_tool_uses', f'{self.eval_params.max_tool_uses}',
            '--batch_size', f'{self.eval_params.batch_size}',
            '--tool_config', f'{self.eval_params.tool_config_type}',
            '--max_pixels', f'{self.eval_params.max_pixels}',
            '--min_pixels', f'{self.eval_params.min_pixels}',
        ]
        if self.eval_params.no_vllm:
            cmd.append('--no_vllm')
        if self.eval_params.enforce_eager:
            cmd.append('--enforce_eager')

        return cmd

    def get_eval_name(self):
        eval_name = f"{self.model_params.short_name}_{self.model_params.checkpoint}_{self.dataset_params.dataset_name}_{self.eval_params.tool_config_type}_{self.eval_params.max_pixels}_{self.eval_params.min_pixels}"
        return eval_name

    def get_save_path(self):
        save_path = os.path.join(self.model_params.full_output_path, self.model_params.dir_suffix,
                                 f"dataset_{self.dataset_params.dataset_name}_prompt_{self.eval_params.tool_config_type}_max_pixels_{self.eval_params.max_pixels}_min_pixels_{self.eval_params.min_pixels}")
        return save_path


    def do_eval(self, exist_ok=False, no_vllm=False, batch_size=None):
        if no_vllm:
            self.eval_params.no_vllm = True
        if batch_size is not None:
            self.eval_params.batch_size = batch_size

        eval_name = self.get_eval_name()
        save_path = self.get_save_path()

        #if not exist_ok and os.path.exists(save_path):
        #    logger.info(f"Path {save_path} exists. If you want to overwrite, please set exist_ok=True.")
        #    return

        try:
            os.makedirs(save_path, exist_ok=exist_ok)
        except OSError as e:
            logger.info(f"Path {save_path} exists. If you want to overwrite, please set exist_ok=True.")
            #raise e



        os.makedirs(os.path.join(save_path, "tool_calls", "generated_images"), exist_ok=True)


        logger.info(f"Evaluating {eval_name}")
        cmd_list = self.get_command(save_path = save_path)
        #logger.info(f"cmd list: {cmd_list}")
        cmd = " ".join(cmd_list)
        #logger.info(f"Running command: {cmd}")

        with open(os.path.join(save_path, "command.txt"), "w") as f:
            f.write(cmd)
        process = subprocess.Popen(cmd, shell=True)
        return_code = process.wait()


class Evals:
    def __init__(self, input: list[dict]):
        self.input = input

        self.classes = {"model_params": ModelParams, "dataset_params": DatasetParams, "eval_params": EvalParams}

        self.evals = self.split_input()


    def split_input(self) -> list[SingleEval]:
        result = []
        for entry in self.input:
            for key, value in entry.items():
                if isinstance(value, list):
                    entry[key] = value
                else:
                    entry[key] = [value]

            # Get all parameter names and their corresponding value lists
            param_names = list(entry.keys())
            param_values = list(entry.values())

            # Generate all combinations using Cartesian product
            combinations = itertools.product(*param_values)

            # Create objects for each combination
            for combo in combinations:
                # Create kwargs dict from parameter names and current combination
                kwargs = dict(zip(param_names, combo))
                instances = tuple(cls(**kwargs) for cls in self.classes.values())
                single_eval = SingleEval(**dict(zip(self.classes.keys(), instances)))
                result.append(single_eval)
        return result

    def __repr__(self):
        return "\n".join([f"'{eval.get_eval_name()}'," for eval in self.evals])

    def do_evals(self, names=list[str], exist_ok=False, no_vllm=False, batch_size=None):
        if names is None:
            eval_list = self.evals
        else:
            eval_list = [eval for eval in self.evals if eval.get_eval_name() in names]

        eval_list = sorted(eval_list, key=lambda x: names.index(x.get_eval_name()))
        for eval in eval_list:
            eval.do_eval(exist_ok, no_vllm, batch_size)
            if eval.eval_params.dry_run:
                break

    def wrap_compute_new_metric(self, names, metrics: list[dict[str, str]], exist_ok=False):
        if names is None:
            eval_list = self.evals
        else:
            eval_list = [eval for eval in self.evals if eval.get_eval_name() in names]

        eval_list = sorted(eval_list, key=lambda x: names.index(x.get_eval_name()))
        for eval in eval_list:
            save_path = eval.get_save_path()
            file_path = os.path.join(save_path, "full_results.json")
            compute_new_metric(file_path, metrics.copy(), exist_ok)

    def filter(self, names: list[str], fixed_params: dict = None):
        if names is None:
            eval_list = self.evals
        else:
            eval_list = []
            for eval in self.evals:
                should_be_added = False
                if eval.get_eval_name() in names:
                    if fixed_params is not None:
                        should_be_added = True
                        for param_key, param_value in fixed_params.items():
                            if not ((param_key in eval.dataset_params.__dict__.keys() and eval.dataset_params.__dict__[param_key] == param_value) or (param_key in eval.eval_params.__dict__.keys() and eval.eval_params.__dict__[param_key] == param_value)):
                                should_be_added = False
                    else:
                        should_be_added = True

                if should_be_added:
                    eval_list.append(eval)


        eval_list = sorted(eval_list, key=lambda x: names.index(x.get_eval_name()))

        return eval_list

    def make_results_table(self, names: list[str], metrics: list[str], fixed_params: dict = None):
        results = []

        eval_list = self.filter(names, fixed_params)


        for eval in eval_list:
            save_path = eval.get_save_path()
            print(f"save_path: {save_path}")

            result = get_results(os.path.join(save_path, "full_results.json"), metrics)
            if result is not None:
                result["name"] = eval.get_eval_name()
                results.append(result)

        df = pd.DataFrame(results, columns=["name"] + [metric if isinstance(metric, str) else metric["metric_short_name"] for metric in metrics])

        return df


if __name__ == "__main__":


    models_input = [
        {
            "short_name": "3B_no_train",
            "model_path": "Qwen/Qwen2.5-VL-3B-Instruct",
            "checkpoint": None,
            "model_class": "Qwen/Qwen2.5-VL-3B-Instruct",
            "output_path": "Qwen_2p5_3B_no_train",
            "tool_config_type": "no_tool",
            "dataset_name": ["pixel_reasoner_vstar"],
            "evaluate": False,
            "analyze": False
        },
        {
            "short_name": "7B_no_train",
            "model_path": "Qwen/Qwen2.5-VL-7B-Instruct",
            "checkpoint": None,
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "output_path": "Qwen_2p5_7B_no_train",
            "tool_config_type": "no_tool",
            "dataset_name": ["pixel_reasoner_vstar"],
            "evaluate": False,
            "analyze": False
        },
        {
            "short_name": "PixelReasoner_original_after_SFT",
            "model_path": "TIGER-Lab/PixelReasoner-WarmStart",
            "checkpoint": None,
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "output_path": "PixelReasoner_original_after_SFT",
            "tool_config_type": "no_tool",
            "dataset_name": ["pixel_reasoner_vstar"],
            "evaluate": False,
            "analyze": False
        },
        {
            "short_name": "PixelReasoner_original_after_RL",
            "model_path": "TIGER-Lab/PixelReasoner-RL-v1",
            "checkpoint": None,
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "output_path": "PixelReasoner_original_after_RL",
            "tool_config_type": ["no_tool", "PR_crop", "select_frames,PR_crop_original", "PR_crop_original,select_frames"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [None, 1000 * 28 * 28, 5120 * 28 * 28],
            "min_pixels": [None, 512 * 28 * 28],
            "evaluate": False,
            "analyze": False
        },
        {
            "short_name": "cold",
            "model_path": "Qwen_2p5_7B_tool_pr_data_cold_2_iterations_box_20250725_141909",
            "checkpoint": [382],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": "first_own_training",
            "dataset_name": ["pixel_reasoner_vstar"],
            "evaluate": False,
            "analyze": False
        },
        {
            "short_name": "warm_answer",
            "model_path": "Qwen_2p5_7B_tool_pr_data_warm_20250717_114421",
            "checkpoint": [382],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": "first_own_training",
            "dataset_name": ["pixel_reasoner_vstar"],
            "evaluate": False,
            "analyze": False
        },
        {
            "short_name": "warm",
            "model_path": "Qwen_2p5_7B_tool_pr_data_warm_2_iterations_box_20250718_161414",
            "checkpoint": [382],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": "first_own_training",
            "dataset_name": ["pixel_reasoner_vstar"],
            "evaluate": False,
            "analyze": False
        },
        {
            "short_name": "3_epochs",
            "model_path": "Qwen_2p5_7B_tool_pr_data_warm_2_iterations_box_3_epochs_20250720_222037",
            "checkpoint": [375],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": "first_own_training",
            "dataset_name": ["pixel_reasoner_vstar"],
            "evaluate": False,
            "analyze": False
        },
        {
            "short_name": "2k_tokens",
            "model_path": "Qwen_2p5_7B_tool_pr_data_warm_2_iterations_box_2k_tokens_20250720_222203",
            "checkpoint": [446],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": "first_own_training",
            "dataset_name": ["pixel_reasoner_vstar"],
            "evaluate": False,
            "analyze": False
        },
        {
            "short_name": "absolute_pixels",
            "model_path": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_20250806_135021",
            "checkpoint": [382, 764, 1146],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["no_tool", "PR_zoom_in_old"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [None, 1000 * 28 * 28],
            "evaluate": False,
            "analyze": False
        },

        {
            "short_name": "global_buffer",
            "model_path": "Qwen_2p5_7B_pr_data_warm_global_buffer_20250806_140735",
            "checkpoint": [382, 764, 1146],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": "PR_crop",
            "dataset_name": ["pixel_reasoner_vstar"],
            "evaluate": False,
            "analyze": False
        },

        {
            "short_name": "global_buffer_top_p_95",
            "model_path": "Qwen_2p5_7B_pr_data_warm_global_buffer_top_p_95_20250806_145202",
            "checkpoint": [382, 764, 1146],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": "PR_crop",
            "dataset_name": ["pixel_reasoner_vstar"],
            "evaluate": False,
            "analyze": False
        },
        {
            "short_name": "absolute_pixels_global_buffer",
            "model_path": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_global_buffer_20250808_175150",
            "checkpoint": [382, 764, 1146],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["PR_zoom_in_old", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [None, 1000 * 28 * 28],
            "evaluate": False,
            "analyze": False
        },
        {
            "short_name": "relative_pixels_local_buffer",
            "model_path": "Qwen_2p5_7B_pr_data_warm_local_buffer_top_p_95_20250808_181626",
            "checkpoint": [382, 764, 1146],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["PR_crop", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [None, 1000 * 28 * 28],
            "evaluate": False,
            "analyze": False
        },
        {
            "short_name": "absolute_pixels_long",
            "model_path": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_6_epochs_20250808_175506",
            "checkpoint": [382, 764, 1146, 1528, 1910, 2292],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["PR_zoom_in_old", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [None, 1000 * 28 * 28],
            "evaluate": False,
            "analyze": False
        },
        {
            "short_name": "absolute_pixels_cold",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_global_buffer_20250812_215006",
            "checkpoint": [382, 764, 1146],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["PR_zoom_in_old", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [None, 1000 * 28 * 28],
            "evaluate": False,
            "analyze": False
        },
        {
            "short_name": "absolute_pixels_global_buffer_beta_1e3",
            "model_path": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_global_buffer_beta_1e3_20250812_222202",
            "checkpoint": [382, 764, 1146],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["PR_zoom_in_old", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [None, 1000 * 28 * 28],
            "evaluate": False,
            "analyze": False
        },
        {
            "short_name": "absolute_pixels_global_buffer_N2",
            "model_path": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_global_buffer_N2_20250815_171829",
            "checkpoint": [382, 764, 1146],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["PR_zoom_in_old", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [None, 1000 * 28 * 28],
            "evaluate": False,
            "analyze": False
        },
        {
            "short_name": "absolute_pixels_global_buffer_2k_image_tokens",
            "model_path": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_global_buffer_2k_images_20250817_162447",
            "checkpoint": [372, 744, 1116],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["PR_zoom_in_old", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [None, 2000 * 28 * 28],
            "evaluate": False,
            "analyze": False
        },
        {
            "short_name": "absolute_pixels_global_buffer_4k_image_tokens",
            "model_path": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_global_buffer_4k_images_20250822_170521",
            "checkpoint": [382, 764],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["PR_zoom_in_old", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [None, 4000 * 28 * 28],
            "evaluate": False,
            "analyze": False
        },

        {
            "short_name": "absolute_pixels_global_buffer_500_image_tokens",
            "model_path": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_global_buffer_500_images_20250826_215959",
            "checkpoint": [382, 764, 1146],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["PR_zoom_in_old", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [None],
            "evaluate": False,
            "analyze": False
        },

        {
            "short_name": "absolute_pixels_global_buffer_750_image_tokens",
            "model_path": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_global_buffer_750_images_20250827_221000",
            "checkpoint": [382, 764, 1146],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["PR_zoom_in_old", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [None, 28*28*2000],
            "evaluate": False,
            "analyze": False
        },

        {
            "short_name": "absolute_pixels_global_buffer_cold_beta_1e3",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_global_buffer_beta_1e3_20250829_073837",
            "checkpoint": [382, 764, 1146],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["PR_zoom_in_old", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [None],
            "evaluate": False,
            "analyze": False
        },
        #
        {
            "short_name": "absolute_pixels_global_buffer_cold_beta_1e3_new",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_global_buffer_beta_1e3_20250831_111115",
            "checkpoint": [382, 764, 1146],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["PR_zoom_in_old", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [None],
            "evaluate": False,
            "analyze": False
        },

        {
            "short_name": "absolute_pixels_global_buffer_warm_beta_1e3_4k_images",
            "model_path": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_global_buffer_beta_1e3_4k_images_20250830_130907",
            "checkpoint": [334, 668, 1002],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["PR_zoom_in_old", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [None, 28*28*4000],
            "evaluate": False,
            "analyze": False
        },

        {
            "short_name": "absolute_pixels_global_buffer_warm_mi_gamma_1p5_old_zoom",
            "model_path": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_mi_gamma_1p5_20250902_173134",
            "checkpoint": [382],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["PR_zoom_in_old", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [None],
            "evaluate": False,
            "analyze": False
        },

        {
            "short_name": "absolute_pixels_global_buffer_warm_mi_delta_0p1_old_zoom",
            "model_path": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_mi_delta_0p1_20250902_171349",
            "checkpoint": [382, 764],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["PR_zoom_in_old", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [None],
            "evaluate": False,
            "analyze": False
        },
        {
            "short_name": "absolute_pixels_global_buffer_warm_mi_delta_0",
            "model_path": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_mi_delta_0_20250903_151622",
            "checkpoint": [382, 764, 1146],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["PR_zoom_in_old", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [None, 1000*28*28],
            "evaluate": False,
            "analyze": False
        },
        {
            "short_name": "absolute_pixels_global_buffer_warm_mi_gamma_1p5",
            "model_path": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_mi_gamma_1p5_20250903_151207",
            "checkpoint": [382, 764, 1146],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["PR_zoom_in_old", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [None, 1000 * 28 * 28],
            "evaluate": False,
            "analyze": False
        },

        {
            "short_name": "absolute_pixels_global_buffer_warm_mi_delta_0p1",
            "model_path": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_mi_delta_0p1_20250903_214257",
            "checkpoint": [382, 764, 1146],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["PR_zoom_in_old", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [None, 1000 * 28 * 28],
            "evaluate": False,
            "analyze": False
        },

        {
            "short_name": "absolute_pixels_warm",
            "model_path": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_global_buffer_20250905_055347",
            "checkpoint": [382, 764, 1146],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["PR_zoom_in_old", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [None, 1000 * 28 * 28],
            "evaluate": False,
            "analyze": False
        },
        {
            "short_name": "absolute_pixels_warm_mi_gamma_1p5_len_linear",
            "model_path": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_mi_gamma_1p5_len_linear_20250905_163008",
            "checkpoint": [382, 764],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["PR_zoom_in_old", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [None, 1000 * 28 * 28],
            "evaluate": False,
            "analyze": False
        },

        {
            "short_name": "absolute_pixels_warm_mi_gamma_1p5_len_root",
            "model_path": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_mi_gamma_1p5_len_root_20250905_162715",
            "checkpoint": [382, 764, 1146],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["PR_zoom_in_old", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [None, 1000 * 28 * 28],
            "evaluate": False,
            "analyze": False
        },

        {
            "short_name": "absolute_pixels_warm_mi_gamma_1p5_len_linear_full",
            "model_path": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_mi_gamma_1p5_len_linear_20250907_171955",
            "checkpoint": [382, 764, 1146],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["PR_zoom_in_old", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [None, 1000 * 28 * 28],
            "evaluate": False,
            "analyze": False
        },

        {
            "short_name": "absolute_pixels_warm_mi_gamma_1p5_len_linear_threshold_0p5",
            "model_path": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_mi_gamma_1p5_len_linear_threshold_0p5_20250919_105738",
            "checkpoint": [372, 744, 1116],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["PR_zoom_in_old", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [None, 1000 * 28 * 28],
            "evaluate": False,
            "analyze": False
        },

        {
            "short_name": "absolute_pixels_warm_mi_gamma_1p5_len_linear_threshold_0p3",
            "model_path": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_mi_gamma_1p5_len_linear_threshold_0p3_20250919_145850",
            "checkpoint": [372, 744, 1116],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["PR_zoom_in_old", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [None, 1000 * 28 * 28],
            "evaluate": False,
            "analyze": False
        },


        {
            "short_name": "absolute_pixels_warm_mi_gamma_1p5_len_linear_threshold_0p1",
            "model_path": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_mi_gamma_1p5_len_linear_threshold_0p1_20250919_104633",
            "checkpoint": [372, 744, 1116],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["PR_zoom_in_old", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [None, 1000 * 28 * 28],
            "evaluate": False,
            "analyze": False
        },

        {
            "short_name": "absolute_pixels_cold_new",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_global_buffer_20250918_095942",
            "checkpoint": [382, 764],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["PR_zoom_in_old", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [None, 1000 * 28 * 28],
            "evaluate": False,
            "analyze": False
        },

        {
            "short_name": "absolute_pixels_cold_mi_len_linear",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_mi_gamma_1p5_len_linear_20250919_083416",
            "checkpoint": [382, 764],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["PR_zoom_in_old", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [None, 1000 * 28 * 28],
            "evaluate": False,
            "analyze": False
        },

        {
            "short_name": "absolute_pixels_warm_mi_len_linear_threshold_0p25",
            "model_path": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_mi_gamma_1p5_len_linear_threshold_0p25_20250921_163346",
            "checkpoint": [372, 744, 1116],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["PR_zoom_in_old", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [None, 1000 * 28 * 28],
            "evaluate": False,
            "analyze": False
        },

        {
            "short_name": "absolute_pixels_warm_mi_len_linear_threshold_0p2",
            "model_path": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_mi_gamma_1p5_len_linear_threshold_0p2_20250921_163432",
            "checkpoint": [372, 744],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["PR_zoom_in_old", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [None, 1000 * 28 * 28],
            "evaluate": False,
            "analyze": False
        },
        {
            "short_name": "absolute_pixels_warm_mi_len_linear_threshold_0p15",
            "model_path": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_mi_gamma_1p5_len_linear_threshold_0p15_20250921_163434",
            "checkpoint": [372, 744, 1116],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["PR_zoom_in_old", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [None, 1000 * 28 * 28],
            "evaluate": False,
            "analyze": False
        },
        {
            "short_name": "absolute_pixels_warm_mi_len_linear_threshold_0p15_discretize",
            "model_path": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_mi_gamma_1p5_len_linear_threshold_0p15_discretize_20250923_141826",
            "checkpoint": [372, 744, 1116],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["PR_zoom_in_old", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [None, 1000 * 28 * 28],
            "evaluate": False,
            "analyze": False
        },

        {
            "short_name": "absolute_pixels_warm_mi_len_linear_threshold_0p2_discretize",
            "model_path": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_mi_gamma_1p5_len_linear_threshold_0p2_discretize_20250923_215418",
            "checkpoint": [372, 744, 1116],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["PR_zoom_in_old", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [None, 1000 * 28 * 28],
            "evaluate": False,
            "analyze": False
        },
        {
            "short_name": "absolute_pixels_warm_5k_500",
            "model_path": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_5k_image_tokens_min_image_500_20251007_122857",
            "checkpoint": [382],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["PR_zoom_in_old", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500*28*28],
            "evaluate": False,
            "analyze": False,
            "PMI_analysis": True,
            "run_finished": True,
            "contains_full_chkp": False
        },
        {
            "short_name": "absolute_pixels_warm_5k_1_epoch",
            "model_path": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_5k_image_tokens_1_epoch_20251007_224129",
            "checkpoint": [382],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["PR_zoom_in_old", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [5000 * 28 * 28],
            "evaluate": False,
            "analyze": False,
            "run_finished": True,
            "contains_full_chkp": False
        },
        {
            "short_name": "absolute_pixels_warm_5k",
            "model_path": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_5k_image_tokens_20251008_102353",
            "checkpoint": [382, 764, 1146],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["PR_zoom_in_old", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [5000 * 28 * 28],
            "evaluate": False,
            "analyze": False,#True
            "PMI_analysis": False,
            "contains_full_chkp": False,
            "run_finished": True,
        },
        {# check min pixels
            "short_name": "absolute_pixels_warm_5k_min_250",
            "model_path": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_5k_image_tokens_min_image_250_20251007_132821",
            "checkpoint": [382, 764],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["PR_zoom_in_old", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [250 * 28 * 28],#3136 accidental eval. slightly better for 382, equal for 764
            "evaluate": False,
            "analyze": False,
            "contains_full_chkp": False,
            "run_finished": True
        },
        {
            "short_name": "absolute_pixels_cold_5k_min_250",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_5k_image_tokens_min_image_250_20251009_215535",
            "checkpoint": [382, 764, 1146],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["PR_zoom_in_old", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [250*28*28],
            "evaluate": False,
            "analyze": False,
            "contains_full_chkp": False,
            "run_finished": True
        },
        {
            "short_name": "absolute_pixels_warm_5k_min_500",
            "model_path": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_5k_image_tokens_min_image_500_20251009_220656",
            "checkpoint": [382, 764, 1146],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["PR_zoom_in_old", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            "evaluate": False,
            "analyze": False,
            "contains_full_chkp": False,
            "run_finished": True
        },
        { ### flawed bc of vLLM min_image=500 != training min_image=250
            "short_name": "absolute_pixels_cold_5k_min_250",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_5k_image_tokens_min_image_250_20251012_162724",
            "checkpoint": [382, 764],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["PR_zoom_in_old", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [250 * 28 * 28, 500*28*28], # same results with both min_pixel values
            "evaluate": False,
            "analyze": False,#True
            "contains_full_chkp": False,
            "run_finished": True
        },
        {
            "short_name": "absolute_pixels_cold_5k",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_5k_image_tokens_20251012_163824",
            "checkpoint": [382, 764, 1146],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["PR_zoom_in_old", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [5000 * 28 * 28],
            "evaluate": False,
            "analyze": False,#True
            "contains_full_chkp": False,
            "run_finished": True
        },
        {# check 382, 764 for double eval
            "short_name": "absolute_pixels_warm_5k_250",
            "model_path": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_5k_image_tokens_min_image_250_20251012_160857",
            "checkpoint": [382, 764, 1146],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["PR_zoom_in_old", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [250 * 28 * 28],
            "evaluate": False,
            "analyze": False,#True
            "contains_full_chkp": False,
            "run_finished": True
        },
        {
            "short_name": "absolute_pixels_cold_5k_250_mi_median",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_5k_image_tokens_min_image_250_mi_median_20251015_111539",
            "checkpoint": [382],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["PR_zoom_in_old", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [250 * 28 * 28],
            "evaluate": False,
            "analyze": False,#True
            "contains_full_chkp": False,
            "run_finished": True
        },
        {
            "short_name": "absolute_pixels_cold_5k_250_mi_median_ref",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_5k_image_tokens_min_image_250_mi_median_ref_20251015_181343",
            "checkpoint": [382],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["PR_zoom_in_old", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [250 * 28 * 28],
            "evaluate": False,
            "analyze": False,#True
            "contains_full_chkp": True,
            "run_finished": False
        },

        {
            "short_name": "absolute_pixels_cold_5k_250_mi_tanh_1_epoch",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_image_5k_250_exploration_tanh_1_epoch_20251020_163159",
            "checkpoint": [382],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["PR_zoom_in_old", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [250 * 28 * 28],
            "evaluate": False,
            "analyze": False,  # True
            "contains_full_chkp": True,
            "run_finished": True
        },

        {
            "short_name": "absolute_pixels_cold_5k_250_constant_1_epoch",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_image_5k_250_constant_exploration_tanh_1_epoch_20251020_163202",
            "checkpoint": [382],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["no_tool"], #["PR_zoom_in_old", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar"], #["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [250 * 28 * 28],
            "evaluate": False,
            "analyze": False,  # True
            "contains_full_chkp": False,
            "run_finished": True
        },
        {
            "short_name": "absolute_pixels_cold_5k_250_mi_tanh_1_epoch_constant_lr",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_image_5k_250_exploration_tanh_1_epoch_constant_lr_20251022_121241",
            "checkpoint": [382],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["PR_zoom_in_old", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [250 * 28 * 28],
            "evaluate": False,
            "analyze": False,  # True
            "contains_full_chkp": True,
            "run_finished": True
        },

        {
            "short_name": "absolute_pixels_cold_5k_250_only_exploration_tanh_1_epoch_constant_lr",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_image_5k_250_only_exploration_tanh_1_epoch_20251022_125452",
            "checkpoint": [382],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["PR_zoom_in_old", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [250 * 28 * 28],
            "evaluate": False,
            "analyze": False,  # True
            "contains_full_chkp": True,
            "run_finished": True
        },

        {
            "short_name": "absolute_pixels_cold_5k_250_only_exploration_tanh_1_epoch_mask_image",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_image_5k_250_tanh_1_epoch_mask_image_20251022_135223",
            "checkpoint": [382],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["PR_zoom_in_old", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [250 * 28 * 28],
            "evaluate": False,
            "analyze": False,  # True
            "contains_full_chkp": True,
            "run_finished": True
        },
        {
            "short_name": "absolute_pixels_cold_5k_250_only_exploration_tanh_1_epoch_mask_image_pad",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_image_5k_250_tanh_1_epoch_mask_image_pad_20251022_140733",
            "checkpoint": [382],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["PR_zoom_in_old", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [250 * 28 * 28],
            "evaluate": False,
            "analyze": False,  # True
            "contains_full_chkp": True,
            "run_finished": True
        },
        {
            "short_name": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_5k_image_tokens_min_image_500",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_5k_image_tokens_min_image_500_20251031_103105",
            "checkpoint": [382, 764, 1146],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["PR_zoom_in_old", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            "evaluate": False,
            "analyze": True,  # True
            "contains_full_chkp": False,
            "run_finished": True
        },
        {
            "short_name": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_5k_image_tokens_min_image_250",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_5k_image_tokens_min_image_250_20251031_110627",
            "checkpoint": [382, 764, 1146],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["PR_zoom_in_old", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [250 * 28 * 28],
            "evaluate": False,
            "analyze": True,  # True
            "contains_full_chkp": False,
            "run_finished": True
        },
        {
            "short_name": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_5k_image_tokens_min_image_4",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_5k_image_tokens_min_image_4_20251031_131631",
            "checkpoint": [382, 764, 1146],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["PR_zoom_in_old", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [4 * 28 * 28],
            "evaluate": False,
            "analyze": True,  # True
            "contains_full_chkp": False,
            "run_finished": True
        },
        {
            "short_name": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_5k_image_tokens_min_image_500",
            "model_path": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_5k_image_tokens_min_image_500_20251031_111228",
            "checkpoint": [382, 764, 1146],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["PR_zoom_in_old", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            "evaluate": True,# should stay false
            "analyze": False,  # True
            "contains_full_chkp": True,
            "run_finished": True # last chkp is missing
        },
        {
            "short_name": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_5k_image_tokens_min_image_250",
            "model_path": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_5k_image_tokens_min_image_250_20251031_111820",
            "checkpoint": [382, 764, 1146],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["PR_zoom_in_old", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [250 * 28 * 28],
            "evaluate": False,
            "analyze": True,  # True
            "contains_full_chkp": False,
            "run_finished": True
        },
        {
            "short_name": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_5k_image_tokens_min_image_4",
            "model_path": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_5k_image_tokens_min_image_4_20251103_112855",
            "checkpoint": [382, 764, 1146],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["PR_zoom_in_old", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [4 * 28 * 28],
            "evaluate": False,
            "analyze": True,  # True
            "contains_full_chkp": False,
            "run_finished": True
        },
        {
            "short_name": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_5k_image_tokens_min_image_4_no_tool",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_5k_image_tokens_min_image_4_no_tool_20251105_231603",
            "checkpoint": [382, 764, 1146],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [4 * 28 * 28],
            "evaluate": False,
            "analyze": True,  # True
            "contains_full_chkp": False,
            "run_finished": True
        },
        {
            "short_name": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_5k_image_tokens_min_image_250_no_tool",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_5k_image_tokens_min_image_250_no_tool_20251105_232334",
            "checkpoint": [382, 764, 1146],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [250 * 28 * 28],
            "evaluate": False,
            "analyze": True,  # True
            "contains_full_chkp": False,
            "run_finished": True
        },
        {
            "short_name": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_5k_image_tokens_min_image_500_no_tool",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_5k_image_tokens_min_image_500_no_tool_20251105_232404",
            "checkpoint": [382, 764, 1146],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            "evaluate": False,
            "analyze": True,  # True
            "contains_full_chkp": False,
            "run_finished": True
        },



    ]

    do_eval = True
    batch_eval = True

    do_analysis = False
    batch_analyze = True

    do_metric_comparison = False

    PMI_analysis = False



    # "absolute_pixels_382_pixel_reasoner_infovqa_no_tool_12845056",
    # "absolute_pixels_global_buffer_382_pixel_reasoner_infovqa_PR_zoom_in_old_12845056"],
    if do_metric_comparison:
        evals.wrap_compute_new_metric(names = ['absolute_pixels_global_buffer_cold_beta_1e3_new_1146_pixel_reasoner_infovqa_no_tool_12845056_0'],
                             metrics=[{"metric_name": "default", "cutoff": 1.0},
                                      {"metric_name": "default", "cutoff": 0.0},
                                      {"metric_name": "string_matching", "match_type": "exact_match", "preprocess": None, "cutoff": None},
                                      {"metric_name": "string_matching", "match_type": "exact_match", "preprocess": "lowercase", "cutoff": None},
                                      {"metric_name": "string_matching", "match_type": "levenshtein_similarity", "preprocess": None, "cutoff": None},
                                      {"metric_name": "string_matching", "match_type": "levenshtein_similarity", "preprocess": "lowercase", "cutoff": None},
                                      {"metric_name": "string_matching", "match_type": "levenshtein_similarity", "preprocess": "lowercase", "cutoff": 0.5}])
    """
        compute_new_metric("/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/helm/focusreason/runs/PixelReasoner_original_after_RL/eval/dataset_pixel_reasoner_infovqa_prompt_pr_adapted/full_results.json",
                       metrics=[
         {"metric_name": "default", "cutoff": 1.0},
        {"metric_name": "default", "cutoff": 0.0},
         {"metric_name": "string_matching", "match_type": "exact_match", "preprocess": None, "cutoff": None},
         #{"metric_name": "string_matching", "match_type": "exact_match", "preprocess": "lowercase", "cutoff": None},
         {"metric_name": "string_matching", "match_type": "levenshtein_similarity", "preprocess": None, "cutoff": None},
         #{"metric_name": "string_matching", "match_type": "levenshtein_similarity", "preprocess": "lowercase",
         # "cutoff": None},
         {"metric_name": "string_matching", "match_type": "levenshtein_similarity", "preprocess": None,
          "cutoff": 0.5}
         ]
    )"""


    if do_eval:
        if batch_eval:
            all_evals = Evals([model for model in models_input if model["evaluate"] is True])
            order = [eval.get_eval_name() for eval in all_evals.evals]
        else:
            all_evals = Evals(models_input)
            logger.info(all_evals)

            order = [

            ]
        all_evals.do_evals(names=order, exist_ok=True, no_vllm=False, batch_size=None)

    if do_analysis:
        metrics = [
            "accuracy",
            "avg_pixel_reasoning",
            "pixel_reasoning_distr",
            "avg_first_completion_len",
            "avg_second_completion_len",
            "avg_total_completion_len",
            #"image_size_x_performance_correlation",
            "zoom_in_fraction_median",
            #"avg_image_size",
            #"accuracy_if_tool_used",
            #"accuracy_if_tool_not_used",
            #"tool_success_rate",
            {"metric_name": "string_matching", "metric_short_name": "ANLS",
             "match_type": "levenshtein_similarity", "preprocess": "lowercase", "cutoff": 0.5}
            # "max_image_size", "min_image_size", "median_image_size"
        ]

        if batch_analyze:
            all_evals = Evals([model for model in models_input if model["analyze"] is True])
            order = [eval.get_eval_name() for eval in all_evals.evals]
        else:
            all_evals = Evals(models_input)
            logger.info(all_evals)

            order = [
            ]


        df = all_evals.make_results_table(names=order, metrics=metrics,
                                          fixed_params={"dataset_name": "pixel_reasoner_vstar",
                                                                                      "max_pixels": 5000*28*28,#1024*16*28*28, #1000*28*28, #1024*16*28*28,# 1024*16*28*28, #,# 1024*16*28*28, #1000*28*28,# 1024*16*28*28,# ,# , #1024*16*28*28,
                                                                                      #"min_pixels": 500 * 28 * 28,
                                                                                      "tool_config_type": "PR_zoom_in_old"#"no_tool"#"PR_zoom_in_old"#"no_tool"# "no_tool"#"PR_zoom_in_old"#"no_tool",#"PR_zoom_in_old"#"no_tool"## #  #"PR_zoom_in_old"# "no_tool" #"PR_zoom_in_old" #"no_tool" ## #"no_tool", #""#
                                                                                  }
                                          )
        pd.set_option('display.max_columns', 10)
        print(df)
        metrics_to_drop = [m for m in metrics if not(isinstance(m, dict) or m in ["accuracy", "ANLS"])] + ["name"]
        print(metrics_to_drop)
        print(df.drop(metrics_to_drop, axis=1).to_csv(sep=",", index=False))
    if PMI_analysis:
        all_evals = Evals([model for model in models_input if "PMI_analysis" in model.keys() and model["PMI_analysis"] is True])
        order = [eval.get_eval_name() for eval in all_evals.evals]
        full_paths = [eval.get_save_path() for eval in all_evals.evals]

        logger.info(f"order: {order}")

        #full_path = ["/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/helm/focusreason/runs/Qwen_2p5_7B_pr_data_warm_absolute_pixels_5k_image_tokens_20251008_102353/checkpoint-382/eval/dataset_pixel_reasoner_vstar_prompt_PR_zoom_in_old_max_pixels_3920000_min_pixels_3136"]
        fixed_params = {"dataset_name": "pixel_reasoner_vstar",
                        "max_pixels": 5000 * 28 * 28,
                        "min_pixels": 250*28*28,
                        "tool_config_type": "PR_zoom_in_old"#"no_tool"#
                        }
        filtered_evals = all_evals.filter(order, fixed_params=fixed_params)
        do_pmi_analysis([eval.get_save_path() for eval in filtered_evals],
                        fixed_params=fixed_params)