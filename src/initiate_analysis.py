import argparse
import base64
import copy
import os
import shutil
import signal
from pathlib import Path
from typing import Union

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

def str_or_dict(value):
    try:
        parsed = json.loads(base64.b64decode(value).decode())
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    return value  # fallback: treat as plain string

class ModelParams:

    def __init__(self, short_name, model_path, checkpoint, model_class, output_path=None,
                 dir_prefix="/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/focusreason/runs/",
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
    def __init__(self, dataset_name, verl_eval=False, **kwargs):
        self.dataset_name = dataset_name
        if self.dataset_name == "pixel_reasoner":
            self.data_files = '/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/pixel_reasoner/RL_data_without_video/train.jsonl'
            self.image_folders = '/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/pixel_reasoner/RL_data_without_video'
        elif self.dataset_name == "pixel_reasoner_vstar":
            if verl_eval:
                task = "test_verl.json"
                img_path = "/images"
            else:
                task = "test.jsonl"
                img_path = ""
            self.data_files = f"/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/pixel_reasoner/eval/V_Star/{task}"
            self.image_folders = f"/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/pixel_reasoner/eval/V_Star{img_path}"
            self.default_num_generations = 32

        elif self.dataset_name == "pixel_reasoner_infovqa":
            self.data_files = "/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/pixel_reasoner/eval/Infographics_VQA/test.jsonl"
            self.image_folders = "/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/pixel_reasoner/eval/Infographics_VQA"
            self.default_num_generations = 2
        elif self.dataset_name == "hr_bench_4k":
            if verl_eval:
                task = "test_verl.json"
            else:
                task = "test.jsonl"
            self.data_files = f"/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/focusreason/HR_Bench_4k/{task}"
            self.image_folders = "/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/focusreason/HR_Bench_4k/images"
            self.default_num_generations = 8
        elif self.dataset_name == "hr_bench_8k":
            if verl_eval:
                task = "test_verl.json"
            else:
                task = "test.jsonl"
            self.data_files = f"/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/focusreason/HR_Bench_8k/{task}"
            self.image_folders = "/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/focusreason/HR_Bench_8k/images"
            self.default_num_generations = 8
        elif self.dataset_name == "mme":
            if verl_eval:
                task = "test_verl.json"
            else:
                task = "test.jsonl"
            self.data_files = f"/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/focusreason/MME-RealWorld/{task}"
            self.image_folders = "/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/focusreason/MME-RealWorld/images"
            self.default_num_generations = 1
        elif self.dataset_name.startswith("mme_shard_"):
            shard_number = self.dataset_name.removeprefix("mme_shard_")
            self.data_files = f"/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/focusreason/MME-RealWorld/test_shard_{shard_number}.jsonl"
            self.image_folders = "/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/focusreason/MME-RealWorld/images"
            self.default_num_generations = 1

        elif self.dataset_name == "mme_lite":
            self.data_files = "/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/focusreason/MME-RealWorld-lite/test.jsonl"
            self.image_folders = "/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/focusreason/MME-RealWorld-lite/images"
        elif self.dataset_name.startswith("muffin_chihuahua"):
            if self.dataset_name.endswith("class_0_1") and "gridsize_1_" not in self.dataset_name:
                task = "find_outlier"
            else:
                task = "single_cell_query"

            if verl_eval:
                task += "_extended_prompt_verl.json"
            else:
                task += ".jsonl"

            self.data_files = f"/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/focusreason/muffin_chihuahua/grid/{dataset_name.removeprefix('muffin_chihuahua_')}/{task}"
            self.image_folders = f"/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/focusreason/muffin_chihuahua/grid/{dataset_name.removeprefix('muffin_chihuahua_')}/images"
            self.default_num_generations = 4
        else:
            raise NotImplementedError(f"dataset {dataset_name} not implemented!")



class EvalParams:
    def __init__(self, batch_size=200, tensor_parallel_size=1, enforce_eager=True, no_vllm=False, dry_run=False,
                 min_pixels=None, max_pixels=None, tool_config_type = "no_tool", image_limit=6, max_tool_uses=5,
                 bbox_type=None, strict_tool_extraction=False,max_tokens_per_reply=256,
                 tool_padding=0.1, tool_adaptive_padding_threshold=600, verl_eval:bool=False, qwen_3p5_eval:bool=False,
                 port:int=8000,
                 temperature=0.0, num_generations=1, **kwargs):
        # technical
        self.batch_size = batch_size
        self.tensor_parallel_size = tensor_parallel_size
        self.enforce_eager = enforce_eager
        self.no_vllm = no_vllm
        self.dry_run = dry_run
        self.verl_eval = verl_eval
        self.qwen_3p5_eval = qwen_3p5_eval
        self.temperature = temperature
        self.num_generations = num_generations
        self.port = port

        # outcome relevant
        if max_pixels is None:
            if self.qwen_3p5_eval:
                self.max_pixels = 1024 * 16*32*32
            else:
                self.max_pixels = 1024*16 * 28 * 28
        else:
            self.max_pixels = max_pixels

        if min_pixels is None:
            if self.qwen_3p5_eval:
                self.min_pixels = 64*32*32
            else:
                self.min_pixels = 4 * 28 * 28
        else:
            self.min_pixels = min_pixels

        self.tool_config_type = tool_config_type
        self.image_limit = image_limit
        self.max_tool_uses = max_tool_uses

        self.bbox_type = bbox_type
        self.strict_tool_extraction = strict_tool_extraction
        self.tool_padding = tool_padding
        self.tool_adaptive_padding_threshold = tool_adaptive_padding_threshold
        self.max_tokens_per_reply = max_tokens_per_reply


        if self.dry_run:
            self.batch_size = 2
            self.tensor_parallel_size = 1
            self.enforce_eager = True



class SingleEval:
    def __init__(self, model_params: ModelParams, dataset_params: DatasetParams, eval_params: EvalParams):

        self.model_params = model_params
        self.dataset_params = dataset_params
        self.eval_params = eval_params

        if self.eval_params.temperature == 0:
            self.eval_params_num_generations = 1
        else:
            if self.eval_params.num_generations == "dataset_dependent":
                self.eval_params.num_generations = self.dataset_params.default_num_generations


    def get_command(self, save_path ):

        cmd = []

        if self.eval_params.verl_eval:
            if self.dataset_params.data_files == "/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/focusreason/MME-RealWorld/test_verl.json":
                eval_script = "run_eval_mme.sh"
                cuda_devices = "0,1"
            elif self.dataset_params.data_files in ["/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/focusreason/HR_Bench_4k/test_verl.json"]:

                eval_script = "run_eval_hrb.sh"
                cuda_devices = "0,1,2,3,4,5,6,7"
            elif self.dataset_params.data_files in ["/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/focusreason/HR_Bench_8k/test_verl.json"]:
                eval_script = "run_eval_hrb_8k.sh"
                cuda_devices = "0,1,2,3,4,5,6,7"
            else:
                eval_script = "run_eval.sh"
                cuda_devices = "0,1"

            if "seed_" in save_path:
                #save_path.index("seed_")[]
                seed = int(save_path.split("_")[-1])
            else:
                seed = 0
            cmd += [
                'VLLM_USE_V1=1',
                f'CUDA_VISIBLE_DEVICES={cuda_devices}',
                'bash', eval_script,
                '--model_path', f'{self.model_params.full_model_path}',
                '--save_path', f'{save_path}',
                '--base_image_dir', str(Path(self.dataset_params.image_folders).parent),
                '--data_path', self.dataset_params.data_files,
                '--seed', f'{seed}'
                #'--data_path', f'{self.dataset_params.data_files.removesuffix(".jsonl")}_verl.json'
            ]
            working_directory = '/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/mini_o3/Mini-o3'

        else:
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
                '--bbox_type', f'{self.eval_params.bbox_type}',
                '--strict_tool_extraction', f'{self.eval_params.strict_tool_extraction}',
                '--tool_padding', f'{self.eval_params.tool_padding}',
                '--tool_adaptive_padding_threshold', f'{self.eval_params.tool_adaptive_padding_threshold}',
                '--max_tokens_per_reply', f'{self.eval_params.max_tokens_per_reply}',
                '--num_generations', f'{self.eval_params.num_generations}',
                '--temperature', f'{self.eval_params.temperature}',
                '--port', f'{self.eval_params.port}'
            ]
            if self.eval_params.no_vllm:
                cmd.append('--no_vllm')
            if self.eval_params.enforce_eager:
                cmd.append('--enforce_eager')
            if self.eval_params.qwen_3p5_eval:
                cmd.append('--qwen_3p5_eval')
            working_directory = None

        return cmd, working_directory

    def get_eval_name(self):
        if self.eval_params.temperature > 0:
            eval_name = f"{self.model_params.short_name}_{self.model_params.checkpoint}_{self.dataset_params.dataset_name}_{self.eval_params.tool_config_type}_{self.eval_params.max_pixels}_{self.eval_params.min_pixels}_{self.eval_params.tool_padding}_{self.eval_params.temperature}_{self.eval_params.num_generations}"
        else:
            eval_name = f"{self.model_params.short_name}_{self.model_params.checkpoint}_{self.dataset_params.dataset_name}_{self.eval_params.tool_config_type}_{self.eval_params.max_pixels}_{self.eval_params.min_pixels}_{self.eval_params.tool_padding}"
        return eval_name

    def get_save_path(self, backward_comp_mode=False):
        if backward_comp_mode:
            save_path = os.path.join(self.model_params.full_output_path, self.model_params.dir_suffix,
                                     f"dataset_{self.dataset_params.dataset_name}_prompt_{self.eval_params.tool_config_type}_max_pixels_{self.eval_params.max_pixels}_min_pixels_{self.eval_params.min_pixels}")
        else:
            if self.eval_params.temperature > 0:
                save_path = os.path.join(self.model_params.full_output_path, self.model_params.dir_suffix,
                                         f"dataset_{self.dataset_params.dataset_name}_prompt_{self.eval_params.tool_config_type}_max_pixels_{self.eval_params.max_pixels}_min_pixels_{self.eval_params.min_pixels}_padding_{self.eval_params.tool_padding}_temp_{self.eval_params.temperature}_samples_{self.eval_params.num_generations}")
            else:
                save_path = os.path.join(self.model_params.full_output_path, self.model_params.dir_suffix,
                                     f"dataset_{self.dataset_params.dataset_name}_prompt_{self.eval_params.tool_config_type}_max_pixels_{self.eval_params.max_pixels}_min_pixels_{self.eval_params.min_pixels}_padding_{self.eval_params.tool_padding}")
        return save_path


    def do_eval(self, exist_ok=False, no_vllm=False, batch_size=None):
        if no_vllm:
            self.eval_params.no_vllm = True
        if batch_size is not None:
            self.eval_params.batch_size = batch_size

        eval_name = self.get_eval_name()
        save_path = self.get_save_path()

        if exist_ok:
            if os.path.exists(save_path):
                logger.info(f"Path {save_path} exists. removing and overwriting")
                shutil.rmtree(save_path)
        else:
            if os.path.exists(save_path):
                logger.info(f"Path {save_path} exists. skipping")
                return

        #if not exist_ok and os.path.exists(save_path):
        #    logger.info(f"Path {save_path} exists. If you want to overwrite, please set exist_ok=True.")
        #    return

        os.makedirs(save_path, exist_ok=True)

        os.makedirs(os.path.join(save_path, "tool_calls", "generated_images"), exist_ok=True)

        logger.info(f"Evaluating {eval_name}")
        cmd_list, working_directory = self.get_command(save_path = save_path)
        logger.info(f"cmd list: {cmd_list}")
        cmd = " ".join(cmd_list)
        logger.info(f"Running command: {cmd}")
        with open(os.path.join(save_path, "command.txt"), "w") as f:
            f.write(cmd)
        p = subprocess.Popen(cmd, shell=True, cwd=working_directory)

        try:
            return p.wait()
        finally:
            # If anything is still alive, terminate the whole group.
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
            # Escalate quickly if needed.
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

def unpack_datasets(list_of_datasets: list[Union[str, dict]]) -> tuple[list[str], list[str]]:
    unpacked_list = []
    short_names = []

    short_name_map = {
        "pixel_reasoner_vstar": "V*",
        "pixel_reasoner_infovqa": "infovqa",
        "hr_bench_4k": "hrb_4k",
        "hr_bench_8k": "hrb_8k",
        "mme_lite": "mme_lite",
        "mme": "mme"
    }
    print(f"in unpack datasets: list_of_datasets={list_of_datasets}")
    for dataset in list_of_datasets:
        if isinstance(dataset, str):
            unpacked_list.append(dataset)
            if dataset in short_name_map:
                short_names.append(short_name_map[dataset])
            else:
                short_names.append(dataset)
        elif isinstance(dataset, dict):
            if dataset["name"] == "muffin_chihuahua":
                for gridsize in dataset["gridsize"]:
                    for grid_pixels in dataset["grid_pixels"]:
                        for mode in dataset["mode"]:
                            if gridsize == 1 and mode == "find_outlier":
                                continue
                            dataset_as_string = f"muffin_chihuahua_grid_pixels_{grid_pixels*1024}_gridsize_{gridsize}_samples_class_0_"
                            if mode == "single_cell_query":
                                dataset_as_string += f"{max(1, int(gridsize**2 / 2))}"
                            elif mode == "find_outlier":
                                dataset_as_string += f"{1}"
                            else:
                                raise ValueError(f"Unknown mode {mode}")
                            unpacked_list.append(dataset_as_string)
                            short_names.append(f"{grid_pixels}k_{gridsize}_{'scq' if mode == 'single_cell_query' else 'fo'}")
            else:
                raise ValueError(f"Unknown dataset for unpacking {dataset}")
        else:
            raise ValueError(f"Datasets have to be either strings or dicts but got {dataset}")
    print(f"unpack datasets to return: unpacked_list={unpacked_list}, short_names={short_names}")
    return unpacked_list, short_names


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

            entry["dataset_name"] = unpack_datasets(entry["dataset_name"])[0]

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
        print(f"Generated {len(result)} evaluations from {self.input}")
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


        #eval_list = sorted(eval_list, key=lambda x: names.index(x.get_eval_name()))

        return eval_list

    def make_results_table(self, names: list[str], metrics: list[str], fixed_params: dict = None):
        results = []
        eval_list = self.filter(names, fixed_params)

        for eval in eval_list:
            save_path = eval.get_save_path()
            print(f"save_path: {save_path}")
            # "full_results_relaxed.json"
            result = get_results(os.path.join(save_path, "full_results.json"), metrics, is_verl=eval.eval_params.verl_eval)

            if result is None and eval.eval_params.tool_padding==0.1:
                save_path = eval.get_save_path(backward_comp_mode = True)
                print(f"trying backward compatible save path: {save_path}")
                result = get_results(os.path.join(save_path, "full_results.json"), metrics)

            if result is not None:
                result["name"] = eval.get_eval_name()
            else:
                result = {"name": eval.get_eval_name()}
                for metric in metrics:
                    if isinstance(metric, str):
                        result[metric] = None
                    else:
                        result[metric["metric_short_name"]] = None
            results.append(result)

        df = pd.DataFrame(results, columns=["name"] + [metric if isinstance(metric, str) else metric["metric_short_name"] for metric in metrics])

        return df

def get_models_input():
    models_input = [
        {
            "short_name": "3B_no_train",
            "model_path": "Qwen/Qwen2.5-VL-3B-Instruct",
            "checkpoint": None,
            "model_class": "Qwen/Qwen2.5-VL-3B-Instruct",
            "output_path": "Qwen_2p5_3B_no_train",
            "tool_config_type": ["zoom_in_absolute", "no_tool"],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            'bbox_type': [None],
            'strict_tool_extraction': [False],
            'max_tokens_per_reply': [1024],
            "evaluate": False,
            "analyze": False
        },
        {
            "short_name": "7B_no_train",
            "model_path": "Qwen/Qwen2.5-VL-7B-Instruct",
            "checkpoint": None,
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "output_path": "Qwen_2p5_7B_no_train",
            "tool_config_type": ["no_tool", "zoom_in_absolute"],  # , "zoom_in_absolute"], #,"zoom_in_absolute",
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa",
                             {"name": "muffin_chihuahua",
                              "grid_pixels": [1, 2, 4, 8],
                              "gridsize": [1, 2, 4, 8, 16],
                              "mode": ["single_cell_query", "find_outlier"]},
                             "hr_bench_4k", "hr_bench_8k",
                             #"mme_lite",
                             "mme"
                             ],
            # ["hr_bench_4k", "hr_bench_8k", "mme_lite", "mme"], #["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            'bbox_type': [None],
            'strict_tool_extraction': [False],
            'tool_padding': [0.1],
            'max_tokens_per_reply': [1024],
            #'temperature': [1.0],
            #'num_generations': ["dataset_dependent"],
            "evaluate": False,
            "analyze": False,
            "paper": True
        },
        {
            "short_name": "PixelReasoner_original_after_SFT",
            "model_path": "TIGER-Lab/PixelReasoner-WarmStart",
            "checkpoint": None,
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "output_path": "PixelReasoner_original_after_SFT",
            "tool_config_type": ["PR_crop_image_normalized,select_frames", "no_tool"],  # ,
            "max_pixels": [5120 * 28 * 28],
            "min_pixels": [512 * 28 * 28],
            "dataset_name": ["hr_bench_4k", "hr_bench_8k", "mme_lite", "mme"],
            'bbox_type': [None],
            'strict_tool_extraction': [False],
            'max_tokens_per_reply': [1024],
            "evaluate": False,
            "analyze": False
        },
        {
            "short_name": "PixelReasoner_original_after_RL",
            "model_path": "TIGER-Lab/PixelReasoner-RL-v1",
            "checkpoint": None,
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "output_path": "PixelReasoner_original_after_RL",
            "tool_config_type": ["no_tool", "PR_crop_image_normalized,select_frames"],  # ],#, "no_tool"],#, ],
            "dataset_name": [
                {"name": "muffin_chihuahua",
                 "grid_pixels": [1, 2, 4, 8],
                 "gridsize": [
                     1,
                     2, 4, 8, 16],
                 "mode": ["single_cell_query", "find_outlier"]
                 },
                "hr_bench_4k", "hr_bench_8k", "mme",  #"mme_lite"
                "pixel_reasoner_infovqa"
            ],
            # ["hr_bench_4k", "hr_bench_8k", "mme_lite", "mme"], #"pixel_reasoner_vstar"],#["pixel_reasoner_vstar", ],#, "pixel_reasoner_infovqa"],
            'bbox_type': [None],
            'strict_tool_extraction': [False],
            'max_tokens_per_reply': [1024],
            #'temperature': [1.0],
            #'num_generations': ["dataset_dependent"],
            "max_pixels": [5120 * 28 * 28],
            "min_pixels": [512 * 28 * 28],
            "evaluate": False,
            "analyze": False,
            "paper": True
        },

        {
            "short_name": "mini_o3_original_after_RL",
            "model_path": "Mini-o3/Mini-o3-7B-v1",
            "checkpoint": None,
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "output_path": "mini_o3_original_after_RL",
            "verl_eval": True,
            "tool_config_type": ["zoom_in_relative"],  # , "no_tool"],
            "dataset_name": [  #"mme" #"pixel_reasoner_vstar",
                #"hr_bench_4k",
                "hr_bench_8k",
                # "mme_lite",
                # "pixel_reasoner_infovqa",
                #{"name": "muffin_chihuahua",
                # "grid_pixels": [
                     #1,
                #                  2,
                #                  4,
                #                 8
                # ],  #
                #"gridsize": [
                #    1,
                #     2,
                #      4,
                #     8,
                #      16
                #],
                #"mode": [
                #    "single_cell_query",
                           #"find_outlier"
               #           ]
               # }
            ],
            "max_pixels": [2000000],
            "min_pixels": [40000],
            'bbox_type': ["relative"],
            'strict_tool_extraction': [True],
            'tool_padding': ['32_turns_sample_1_seed_5', '32_turns_sample_1_seed_6', '32_turns_sample_1_seed_7', '32_turns_sample_1_seed_8'],  # '32_turns'],  '0.0' '32_turns_extended_prompt'
            'max_tokens_per_reply': [8192],
            "evaluate": False,
            "analyze": False,  # later
        },

        {
            "short_name": "Qwen3p5_9B",
            "model_path": "Qwen/Qwen3.5-9B",
            "checkpoint": None,
            "model_class": "Qwen/Qwen3.5-9B",
            "output_path": "Qwen3p5_9B",
            "qwen_3p5_eval": True,
            "tool_config_type": ["zoom_in_absolute_q3p5"],  # , ], ,
            "dataset_name": [  # "mme"
                #"pixel_reasoner_vstar",
                #"hr_bench_4k",
                #"hr_bench_8k",
                "mme",
                # "mme_lite",
                # "pixel_reasoner_infovqa",
                #{"name": "muffin_chihuahua",
                #"grid_pixels": [
                #1,
                #                  2,
                #                  4,
                #                 8
                #],  #
                #"gridsize": [
                #    1,
                #     2,
                #      4,
                #     8,
                #      16
                #],
                #"mode": [
                #    "single_cell_query",
                #"find_outlier"
                #          ]
                #}
            ],
            "max_pixels": None, #[5000*32*32],
            "min_pixels": None, #[500*32*32],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [True],
            'tool_padding': [0.0],  # '32_turns'],  '0.0' '32_turns_extended_prompt'
            'max_tokens_per_reply': [8192*4],
            'max_tool_uses': [32],
            'port': 8011,
            "evaluate": False,
            "analyze":False,  # later
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
            "max_pixels": [None, 28 * 28 * 2000],
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
            "max_pixels": [None, 28 * 28 * 4000],
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
            "max_pixels": [None, 1000 * 28 * 28],
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
            "min_pixels": [500 * 28 * 28],
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
            "analyze": False,  # True
            "PMI_analysis": False,
            "contains_full_chkp": False,
            "run_finished": True,
        },
        {  # check min pixels
            "short_name": "absolute_pixels_warm_5k_min_250",
            "model_path": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_5k_image_tokens_min_image_250_20251007_132821",
            "checkpoint": [382, 764],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["PR_zoom_in_old", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [250 * 28 * 28],  # 3136 accidental eval. slightly better for 382, equal for 764
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
            "min_pixels": [250 * 28 * 28],
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
        {  ### flawed bc of vLLM min_image=500 != training min_image=250
            "short_name": "absolute_pixels_cold_5k_min_250",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_5k_image_tokens_min_image_250_20251012_162724",
            "checkpoint": [382, 764],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["PR_zoom_in_old", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [250 * 28 * 28, 500 * 28 * 28],  # same results with both min_pixel values
            "evaluate": False,
            "analyze": False,  # True
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
            "analyze": False,  # True
            "contains_full_chkp": False,
            "run_finished": True
        },
        {  # check 382, 764 for double eval
            "short_name": "absolute_pixels_warm_5k_250",
            "model_path": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_5k_image_tokens_min_image_250_20251012_160857",
            "checkpoint": [382, 764, 1146],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["PR_zoom_in_old", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [250 * 28 * 28],
            "evaluate": False,
            "analyze": False,  # True
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
            "analyze": False,  # True
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
            "analyze": False,  # True
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
            "tool_config_type": ["no_tool"],  # ["PR_zoom_in_old", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar"],  # ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
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
            "analyze": False,  # True
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
            "analyze": False,  # True
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
            "analyze": False,  # True
            "contains_full_chkp": False,
            "run_finished": True
        },
        {
            "short_name": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_5k_image_tokens_min_image_500",
            "model_path": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_5k_image_tokens_min_image_500_20251031_111228",
            "checkpoint": [382],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["zoom_in_absolute"],
            "dataset_name": ["pixel_reasoner_vstar"],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [True],
            'max_tokens_per_reply': [256],
            "evaluate": False,
            "analyze": False,  # True
            "contains_full_chkp": False,
            "run_finished": True
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
            "analyze": False,  # True
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
            "analyze": False,  # True
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
            "analyze": False,  # True
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
            "analyze": False,  # True
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
            "analyze": False,  # True
            "contains_full_chkp": False,
            "run_finished": True
        },
        {
            "short_name": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_5k_image_tokens_min_image_500_no_tool",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_5k_image_tokens_min_image_500_no_tool_20251115_012947",
            "checkpoint": [382, 764, 1146],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'max_tokens_per_reply': [256],
            "evaluate": False,
            "analyze": False,  # True
            "contains_full_chkp": False,
            "run_finished": True
        },
        {
            "short_name": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_5k_image_tokens_min_image_500_no_tool",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_5k_image_tokens_min_image_500_no_tool_20251115_012947",
            "checkpoint": [382, 764, 1146],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'max_tokens_per_reply': [256],
            "evaluate": False,
            "analyze": False,  # True
            "contains_full_chkp": False,
            "run_finished": True
        },
        {
            "short_name": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_5k_image_tokens_min_image_500_description",
            "model_path": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_5k_image_tokens_min_image_500_description_20251124_110836",
            "checkpoint": [382],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["zoom_in_absolute"],
            "dataset_name": ["pixel_reasoner_vstar"],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [True],
            'max_tokens_per_reply': [256],
            "evaluate": False,
            "analyze": False,  # True
            "contains_full_chkp": False,
            "run_finished": True
        },
        {
            "short_name": "Qwen_2p5_7B_pr_data_cold_relative_pixels_5k_image_tokens_min_image_500",
            "model_path": "Qwen_2p5_7B_pr_data_cold_relative_pixels_5k_image_tokens_min_image_500_20251126_235102",
            "checkpoint": [382, 764, 1146],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["zoom_in_relative", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["relative"],
            'strict_tool_extraction': [False],
            'max_tokens_per_reply': [512],
            "evaluate": False,  # should be True
            "analyze": False,  # True
            "contains_full_chkp": True,
            "run_finished": True
        },
        {
            "short_name": "Qwen_2p5_7B_pr_data_cold_relative_pixels_5k_image_tokens_min_image_500_strict",
            "model_path": "Qwen_2p5_7B_pr_data_cold_relative_pixels_5k_image_tokens_min_image_500_strict_20251126_233213",
            "checkpoint": [382, 764, 1146],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["no_tool", "zoom_in_relative"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["relative"],
            'strict_tool_extraction': [False],
            'max_tokens_per_reply': [512],
            "evaluate": False,  # should be True
            "analyze": False,  # True
            "contains_full_chkp": True,
            "run_finished": True
        },
        {
            "short_name": "Qwen_2p5_7B_pr_data_cold_relative_pixels_5k_image_tokens_min_image_500_strict_append_wrong_tool_call",
            "model_path": "Qwen_2p5_7B_pr_data_cold_relative_pixels_5k_image_tokens_min_image_500_strict_append_wrong_tool_call_20251127_162548",
            "checkpoint": [382, 764, 1146],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["zoom_in_relative", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["relative"],
            'strict_tool_extraction': [False],
            'max_tokens_per_reply': [512],
            "evaluate": False,  # should be True
            "analyze": False,  # True
            "contains_full_chkp": True,
            "run_finished": True
        },
        {
            "short_name": "Qwen_2p5_7B_pr_data_warm_relative_pixels_5k_image_tokens_min_image_500",
            "model_path": "Qwen_2p5_7B_pr_data_warm_relative_pixels_5k_image_tokens_min_image_500_20251126_233728",
            "checkpoint": [382, 764, 1146],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["zoom_in_relative", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["relative"],
            'strict_tool_extraction': [False],
            'max_tokens_per_reply': [512],
            "evaluate": False,  # should be True
            "analyze": False,  # True
            "contains_full_chkp": True,
            "run_finished": True
        },
        {
            "short_name": "Qwen_2p5_7B_pr_data_warm_relative_pixels_5k_image_tokens_min_image_500_strict",
            "model_path": "Qwen_2p5_7B_pr_data_warm_relative_pixels_5k_image_tokens_min_image_500_strict_20251126_233906",
            "checkpoint": [382, 764, 1146],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["zoom_in_relative", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["relative"],
            'strict_tool_extraction': [False],
            'max_tokens_per_reply': [512],
            "evaluate": False,  # should be True
            "analyze": False,  # True
            "contains_full_chkp": True,
            "run_finished": True
        },
        {  # this is strict
            "short_name": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_5k_image_tokens_min_image_500",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_5k_image_tokens_min_image_500_20251128_181804",
            "checkpoint": [382, 764, 1146],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["zoom_in_absolute", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'max_tokens_per_reply': [512],
            "evaluate": False,  # should be True
            "analyze": False,  # True
            "contains_full_chkp": True,
            "run_finished": True
        },
        {  # this is strict
            "short_name": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_5k_image_tokens_min_image_500",
            "model_path": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_5k_image_tokens_min_image_500_20251128_182318",
            "checkpoint": [382, 764, 1146],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["zoom_in_absolute", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'max_tokens_per_reply': [512],
            "evaluate": False,  # should be True
            "analyze": False,  # True
            "contains_full_chkp": True,
            "run_finished": True
        },
        {
            "short_name": "Qwen_2p5_7B_pr_data_cold_relative_pixels_5k_image_tokens_min_image_500_strict",
            "model_path": "Qwen_2p5_7B_pr_data_cold_relative_pixels_5k_image_tokens_min_image_500_strict_20251204_001445",
            "checkpoint": [382, 764, 1146],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["zoom_in_relative", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["relative"],
            'strict_tool_extraction': [False],
            'max_tokens_per_reply': [512],
            "evaluate": False,  # should be True
            "analyze": False,  # True
            "contains_full_chkp": True,
            "run_finished": True
        },
        {
            "short_name": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_5k_image_tokens_min_image_500_strict",
            "model_path": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_5k_image_tokens_min_image_500_strict_20251203_235005",
            "checkpoint": [382, 764, 1146],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["zoom_in_absolute", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'max_tokens_per_reply': [512],
            "evaluate": False,  # should be True
            "analyze": False,  # True
            "contains_full_chkp": True,
            "run_finished": True
        },
        {
            "short_name": "Qwen_2p5_7B_pr_data_warm_relative_pixels_5k_image_tokens_min_image_500_strict",
            "model_path": "Qwen_2p5_7B_pr_data_warm_relative_pixels_5k_image_tokens_min_image_500_strict_20251204_001805",
            "checkpoint": [382, 764, 1146],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["zoom_in_relative", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["relative"],
            'strict_tool_extraction': [False],
            'max_tokens_per_reply': [512],
            "evaluate": False,  # should be True
            "analyze": False,  # True
            "contains_full_chkp": True,
            "run_finished": True
        },
        {
            "short_name": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_5k_image_tokens_min_image_500",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_5k_image_tokens_min_image_500_20251204_000948",
            "checkpoint": [382, 764, 1146],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["zoom_in_absolute", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'max_tokens_per_reply': [512],
            "evaluate": False,  # should be True
            "analyze": False,  # True
            "contains_full_chkp": True,
            "run_finished": True
        },
        {
            "short_name": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_500_5k_image_mi_iou_random",
            "model_path": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_500_5k_image_mi_iou_random_20251215_121302",
            "checkpoint": [382, 764, 1146],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["zoom_in_absolute", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'max_tokens_per_reply': [1024],
            "evaluate": False,
            "analyze": False,
            "contains_full_chkp": True,
            "run_finished": False
        },
        {
            "short_name": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_500_5k_image_mi_iou",
            "model_path": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_500_5k_image_mi_iou_20251216_010228",
            "checkpoint": [382, 764, 1146],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["zoom_in_absolute", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'max_tokens_per_reply': [1024],
            "evaluate": False,
            "analyze": False,
            "contains_full_chkp": True,
            "run_finished": False
        },
        {
            "short_name": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_500_5k_image_mi_iou_long_curr",
            "model_path": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_500_5k_image_mi_iou_long_curr_20251216_113951",
            "checkpoint": [382, 764, 1146],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["zoom_in_absolute", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'max_tokens_per_reply': [1024],
            "evaluate": False,
            "analyze": False,
            "contains_full_chkp": True,
            "run_finished": False
        },
        {
            "short_name": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_500_5k_image_mi_iou_sum",
            "model_path": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_500_5k_image_mi_iou_sum_20251217_102117",
            "checkpoint": [382, 764, 1146],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["zoom_in_absolute", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'max_tokens_per_reply': [1024],
            "evaluate": False,
            "analyze": False,
            "contains_full_chkp": True,
            "run_finished": False
        },
        {
            "short_name": "Qwen_2p5_7B_pr_data_cold_relative_pixels_5k_image_tokens_min_image_500",
            "model_path": "Qwen_2p5_7B_pr_data_cold_relative_pixels_5k_image_tokens_min_image_500_20251219_170942",
            "checkpoint": [382, 764, 1146],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["zoom_in_relative", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["relative"],
            'strict_tool_extraction': [False],
            'max_tokens_per_reply': [1024],
            "evaluate": False,
            "analyze": False,
            "contains_full_chkp": True,
            "run_finished": False
        },
        {
            "short_name": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_5k_image_tokens_min_image_500",
            "model_path": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_5k_image_tokens_min_image_500_20251219_165830",
            "checkpoint": [382],  # , 764, 1146],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["no_tool", "zoom_in_absolute"],  # , "no_tool"],
            "dataset_name": [
                {"name": "muffin_chihuahua",
                 "grid_pixels": [1, 2, 4, 8],
                 "gridsize": [1, 2, 4, 8, 16],
                 "mode": ["single_cell_query", "find_outlier"]}
                # #["hr_bench_4k", "hr_bench_8k", "mme_lite"],
            ],

            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'max_tokens_per_reply': [1024],
            "evaluate": False,
            "analyze": False,  # later
            "contains_full_chkp": True,
            "run_finished": False
        },
        {
            "short_name": "Qwen_2p5_7B_pr_data_warm_relative_pixels_5k_image_tokens_min_image_500",
            "model_path": "Qwen_2p5_7B_pr_data_warm_relative_pixels_5k_image_tokens_min_image_500_20251219_170424",
            "checkpoint": [382, 764, 1146],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["zoom_in_relative", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["relative"],
            'strict_tool_extraction': [False],
            'max_tokens_per_reply': [1024],
            "evaluate": False,
            "analyze": False,
            "contains_full_chkp": True,
            "run_finished": False
        },
        {
            "short_name": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_5k_image_tokens_min_image_500",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_5k_image_tokens_min_image_500_20251220_171430",
            "checkpoint": [382],  # , 764, 1146],#, 764, 1146],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["no_tool", "zoom_in_absolute"],  # , "no_tool"],#, ],
            "dataset_name": [
                {"name": "muffin_chihuahua",
                 "grid_pixels": [1, 2, 4, 8],
                 "gridsize": [1, 2, 4, 8, 16],
                 "mode": ["single_cell_query", "find_outlier"]}

            ],
            # ["hr_bench_4k", "hr_bench_8k", "mme_lite", "mme"],#["mme_lite"],#"hr_bench_8k" "pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'max_tokens_per_reply': [1024],
            "evaluate": False,
            "analyze": False,  # True, # later
            "contains_full_chkp": True,
            "run_finished": False
        },
        {
            "short_name": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_5k_image_tokens_min_image_500_pruning_0p1",
            "model_path": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_5k_image_tokens_min_image_500_pruning_0p1_20251224_113738",
            "checkpoint": [382],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["zoom_in_absolute", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'max_tokens_per_reply': [1024],
            "evaluate": False,
            "analyze": False,
            "contains_full_chkp": True,
            "run_finished": False
        },
        {
            "short_name": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_5k_image_tokens_min_image_500_pruning_0p4",
            "model_path": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_5k_image_tokens_min_image_500_pruning_0p4_20251224_114755",
            "checkpoint": [382],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["zoom_in_absolute", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'max_tokens_per_reply': [1024],
            "evaluate": False,
            "analyze": False,
            "contains_full_chkp": True,
            "run_finished": False
        },
        {
            "short_name": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_5k_image_tokens_min_image_500_pruning_0p7",
            "model_path": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_5k_image_tokens_min_image_500_pruning_0p7_20251224_114832",
            "checkpoint": [382],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["zoom_in_absolute", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'max_tokens_per_reply': [1024],
            "evaluate": False,
            "analyze": False,
            "contains_full_chkp": True,
            "run_finished": False
        },
        {
            "short_name": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_5k_image_tokens_min_image_500_pruning_0p075",
            "model_path": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_5k_image_tokens_min_image_500_pruning_0p075_20260102_173932",
            "checkpoint": [382],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["zoom_in_absolute", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'max_tokens_per_reply': [1024],
            "evaluate": False,
            "analyze": False,
            "contains_full_chkp": True,
            "run_finished": False
        },
        {
            "short_name": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_5k_image_tokens_min_image_500_pruning_0p05",
            "model_path": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_5k_image_tokens_min_image_500_pruning_0p05_20260102_184811",
            "checkpoint": [382],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["zoom_in_absolute", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'max_tokens_per_reply': [1024],
            "evaluate": False,
            "analyze": False,
            "contains_full_chkp": True,
            "run_finished": False
        },
        {
            "short_name": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_5k_image_tokens_min_image_500_pruning_0p025",
            "model_path": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_5k_image_tokens_min_image_500_pruning_0p025_20260102_185111",
            "checkpoint": [382],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["zoom_in_absolute", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'max_tokens_per_reply': [1024],
            "evaluate": False,
            "analyze": False,
            "contains_full_chkp": True,
            "run_finished": False
        },
        {
            "short_name": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_5k_image_tokens_min_image_500_pruning_0p025",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_5k_image_tokens_min_image_500_pruning_0p025_20260104_181946",
            "checkpoint": [382],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["no_tool", "zoom_in_absolute"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'max_tokens_per_reply': [1024],
            "evaluate": False,
            "analyze": False,
            "contains_full_chkp": True,
            "run_finished": False
        },
        {
            "short_name": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_5k_image_tokens_min_image_500_no_tool",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_5k_image_tokens_min_image_500_no_tool_20260102_175500",
            "checkpoint": [382],  # , 764, 1146],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["no_tool", "zoom_in_absolute"],
            "dataset_name": [
                {"name": "muffin_chihuahua",
                 "grid_pixels": [1, 2, 4, 8],
                 "gridsize": [1, 2, 4, 8, 16],
                 "mode": ["single_cell_query", "find_outlier"]}
                # ["hr_bench_4k", "hr_bench_8k", "mme_lite", "mme"], #["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            ],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'max_tokens_per_reply': [1024],
            "evaluate": False,
            "analyze": False,  # True, # later
            "contains_full_chkp": True,
            "run_finished": False
        },
        {
            "short_name": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_500_5k_image_mi_iou_random_sum",
            "model_path": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_500_5k_image_mi_iou_random_sum_20260102_183643",
            "checkpoint": [1146],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["no_tool", "zoom_in_absolute"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'max_tokens_per_reply': [1024],
            "evaluate": False,
            "analyze": False,
            "contains_full_chkp": True,
            "run_finished": False
        },
        {
            "short_name": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_500_5k_image_mi_iou_long_curr_sum",
            "model_path": "Qwen_2p5_7B_pr_data_warm_absolute_pixels_500_5k_image_mi_iou_long_curr_sum_20260102_183159",
            "checkpoint": [1146],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["no_tool", "zoom_in_absolute"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'max_tokens_per_reply': [1024],
            "evaluate": False,
            "analyze": False,
            "contains_full_chkp": True,
            "run_finished": False
        },
        {
            "short_name": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_5k_image_tokens_min_image_500_pruning_if_correct",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_5k_image_tokens_min_image_500_pruning_if_correct_0p025_20260106_105315",
            "checkpoint": [382],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["no_tool", "zoom_in_absolute"],
            "dataset_name": ["pixel_reasoner_vstar", "pixel_reasoner_infovqa"],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'max_tokens_per_reply': [1024],
            "evaluate": False,
            "analyze": False,
            "contains_full_chkp": True,
            "run_finished": False
        },
        {
            "short_name": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_long_curr_sum_max_0p6_conditional",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_long_curr_sum_max_0p6_conditional_20260109_182746",
            "checkpoint": [382],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["no_tool", "zoom_in_absolute"],
            "dataset_name": ["pixel_reasoner_vstar", "hr_bench_4k", "hr_bench_8k", "mme_lite",
                             "pixel_reasoner_infovqa"],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'max_tokens_per_reply': [1024],
            "evaluate": False,
            "analyze": False,
            "contains_full_chkp": True,
            "run_finished": False
        },
        {
            "short_name": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_long_curr_sum_max_0p6_conditional_infonce",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_long_curr_sum_max_0p6_conditional_infonce_20260109_185324",
            "checkpoint": [382],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["no_tool", "zoom_in_absolute"],
            "dataset_name": ["pixel_reasoner_vstar", "hr_bench_4k", "hr_bench_8k", "mme_lite",
                             "pixel_reasoner_infovqa"],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'max_tokens_per_reply': [1024],
            "evaluate": False,
            "analyze": False,
            "contains_full_chkp": True,
            "run_finished": False
        },
        {
            "short_name": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_long_curr_sum_max_0p6_infonce",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_long_curr_sum_max_0p6_infonce_20260109_184252",
            "checkpoint": [382],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["no_tool", "zoom_in_absolute"],
            "dataset_name": ["pixel_reasoner_vstar", "hr_bench_4k", "hr_bench_8k", "mme_lite",
                             "pixel_reasoner_infovqa"],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'max_tokens_per_reply': [1024],
            "evaluate": False,
            "analyze": False,
            "contains_full_chkp": True,
            "run_finished": False
        },
        {
            "short_name": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_long_curr_sum_max_0p6_infonce",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_long_curr_sum_max_0p6_infonce_20260111_143637",
            "checkpoint": [382, 764, 1146],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["no_tool", "zoom_in_absolute"],
            "dataset_name": ["pixel_reasoner_vstar", "hr_bench_4k", "hr_bench_8k", "mme_lite",
                             "pixel_reasoner_infovqa"],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'max_tokens_per_reply': [1024],
            "evaluate": False,
            "analyze": False,  # later
            "contains_full_chkp": True,
            "run_finished": False
        },
        {
            "short_name": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_long_curr_sum_max_0p6",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_long_curr_sum_max_0p6_conditional_20260111_144201",
            "checkpoint": [382, 764, 1146],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["no_tool", "zoom_in_absolute"],
            "dataset_name": ["pixel_reasoner_vstar", "hr_bench_4k", "hr_bench_8k", "mme_lite",
                             "pixel_reasoner_infovqa"],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'max_tokens_per_reply': [1024],
            "evaluate": False,
            "analyze": False,  # later
            "contains_full_chkp": True,
            "run_finished": False
        },
        {
            "short_name": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_long_curr_sum_max_0p6_conditional_infonce",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_long_curr_sum_max_0p6_conditional_infonce_20260112_192315",
            "checkpoint": [382],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["no_tool", "zoom_in_absolute"],
            "dataset_name": ["pixel_reasoner_vstar", "hr_bench_4k", "hr_bench_8k", "mme_lite",
                             "pixel_reasoner_infovqa",
                             "mme",
                             {"name": "muffin_chihuahua",
                              "grid_pixels": [1, 2, 4, 8],
                              "gridsize": [1, 2, 4, 8, 16],
                              "mode": ["single_cell_query", "find_outlier"]}
                             ],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'max_tokens_per_reply': [1024],
            "evaluate": False,
            "analyze": False,  # later
            "contains_full_chkp": True,
            "run_finished": False
        },
        {
            "short_name": "Qwen_2p5_7B_muffin_warm_absolute_pixels_500_5k",
            "model_path": "Qwen_2p5_7B_muffin_warm_absolute_pixels_500_5k_20260119_105444",
            "checkpoint": [80],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["no_tool", "zoom_in_absolute"],
            "dataset_name": ["pixel_reasoner_vstar", "hr_bench_4k", "hr_bench_8k", "mme_lite",
                             "pixel_reasoner_infovqa",
                             {"name": "muffin_chihuahua",
                              "grid_pixels": [1, 2, 4, 8],
                              "gridsize": [1, 2, 4, 8, 16],
                              "mode": ["single_cell_query", "find_outlier"]}
                             ],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'max_tokens_per_reply': [1024],
            "evaluate": False,
            "analyze": False,  # later
            "contains_full_chkp": True,
            "run_finished": False
        },

        {
            "short_name": "Qwen_2p5_7B_muffin_cold_absolute_pixels_500_5k",
            "model_path": "Qwen_2p5_7B_muffin_cold_absolute_pixels_500_5k_20260119_104730",
            "checkpoint": [80],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["no_tool", "zoom_in_absolute"],
            "dataset_name": ["pixel_reasoner_vstar", "hr_bench_4k", "hr_bench_8k", "mme_lite",
                             "pixel_reasoner_infovqa",
                             {"name": "muffin_chihuahua",
                              "grid_pixels": [1, 2, 4, 8],
                              "gridsize": [1, 2, 4, 8, 16],
                              "mode": ["single_cell_query", "find_outlier"]}
                             ],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'max_tokens_per_reply': [1024],
            "evaluate": False,
            "analyze": False,  # later
            "contains_full_chkp": True,
            "run_finished": False
        },
        {
            "short_name": "Qwen_2p5_7B_muffin_warm_absolute_pixels_500_5k_iou_reward",
            "model_path": "Qwen_2p5_7B_muffin_warm_absolute_pixels_500_5k_iou_reward_20260120_094018",
            "checkpoint": [80],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["no_tool", "zoom_in_absolute"],
            "dataset_name": ["pixel_reasoner_vstar", "hr_bench_4k", "hr_bench_8k", "mme_lite",
                             "pixel_reasoner_infovqa",
                             {"name": "muffin_chihuahua",
                              "grid_pixels": [1, 2, 4, 8],
                              "gridsize": [1, 2, 4, 8, 16],
                              "mode": ["single_cell_query", "find_outlier"]}
                             ],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'max_tokens_per_reply': [1024],
            "evaluate": False,
            "analyze": False,  # later
            "contains_full_chkp": True,
            "run_finished": False
        },
        {
            "short_name": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_5k_image_tokens_min_image_500_1_epoch_const",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_5k_image_tokens_min_image_500_1_epoch_const_20260120_224520",
            "checkpoint": [382],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["no_tool", "zoom_in_absolute"],  # , "no_tool"], #,zoom_in_absolute
            "dataset_name": [  "pixel_reasoner_vstar", "hr_bench_4k", "hr_bench_8k",
                #"mme_lite",
                               "mme",
                "pixel_reasoner_infovqa",
                {"name": "muffin_chihuahua",
                 "grid_pixels": [1, 2, 4, 8],
                 "gridsize": [1, 2,
                              4, 8, 16
                              ],
                 "mode": ["single_cell_query", "find_outlier"]}
            ],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'tool_padding': [0.1], #, 0.05, 0.0
            'max_tokens_per_reply': [1024],
            #'temperature': [1.0],
            #'num_generations': ["dataset_dependent"],
            "evaluate": False,
            "analyze": False,  # later+
            "contains_full_chkp": True,
            "run_finished": False,
            "paper": True
        },
        {
            "short_name": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_5k_image_tokens_min_image_500_1_epoch_const_long_warmup",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_5k_image_tokens_min_image_500_1_epoch_const_long_warmup_20260120_224700",
            "checkpoint": [382],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["no_tool", "zoom_in_absolute"],
            "dataset_name": ["pixel_reasoner_vstar", "hr_bench_4k", "hr_bench_8k",
                             # "mme_lite",
                             # "pixel_reasoner_infovqa",
                             # {"name": "muffin_chihuahua",
                             # "grid_pixels": [1, 2, 4, 8],
                             # "gridsize": [1, 2, 4, 8, 16],
                             # "mode": ["single_cell_query", "find_outlier"]}
                             ],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'max_tokens_per_reply': [1024],
            "evaluate": False,
            "analyze": False,  # later
            "contains_full_chkp": True,
            "run_finished": False
        },

        {
            "short_name": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_5k_image_tokens_min_image_500_1_epoch_cosine_long_warmup",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_5k_image_tokens_min_image_500_1_epoch_cosine_long_warmup_20260120_235613",
            "checkpoint": [382],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["no_tool", "zoom_in_absolute"],
            "dataset_name": ["pixel_reasoner_vstar", "hr_bench_4k", "hr_bench_8k",
                             # "mme_lite",
                             # "pixel_reasoner_infovqa",
                             # {"name": "muffin_chihuahua",
                             # "grid_pixels": [1, 2, 4, 8],
                             # "gridsize": [1, 2, 4, 8, 16],
                             # "mode": ["single_cell_query", "find_outlier"]}
                             ],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'max_tokens_per_reply': [1024],
            "evaluate": False,
            "analyze": False,  # later
            "contains_full_chkp": True,
            "run_finished": False
        },
        {
            "short_name": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_1_epoch_100_100_100",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_1_epoch_100_100_100_20260123_184438",
            "checkpoint": [382],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["no_tool", "zoom_in_absolute"],
            "dataset_name": [  # "pixel_reasoner_vstar", "hr_bench_4k", "hr_bench_8k",
                #"mme_lite",
                "mme"
                #"pixel_reasoner_infovqa",
                #{"name": "muffin_chihuahua",
                # "grid_pixels": [1, 2, 4, 8],
                # "gridsize": [1, 2, 4, 8, 16],
                # "mode": ["single_cell_query", "find_outlier"]}
            ],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'max_tokens_per_reply': [1024],
            "evaluate": False,
            "analyze": False,  # later
            "contains_full_chkp": True,
            "run_finished": False
        },
        {
            "short_name": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_1_epoch_10_90_60",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_1_epoch_10_90_60_20260123_183012",
            "checkpoint": [382],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["no_tool", "zoom_in_absolute"],
            "dataset_name": [  # "pixel_reasoner_vstar", "hr_bench_4k", "hr_bench_8k",
                "mme_lite",
                "pixel_reasoner_infovqa",
                {"name": "muffin_chihuahua",
                 "grid_pixels": [1, 2, 4, 8],
                 "gridsize": [1, 2, 4, 8, 16],
                 "mode": ["single_cell_query", "find_outlier"]}
            ],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'max_tokens_per_reply': [1024],
            "evaluate": False,
            "analyze": False,  # later
            "contains_full_chkp": True,
            "run_finished": False
        },
        {
            "short_name": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_1_epoch_30_100_17p5",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_1_epoch_30_100_17p5_20260123_184044",
            "checkpoint": [382],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["zoom_in_absolute", "no_tool"],  # , "no_tool"], #,
            "dataset_name": [  "pixel_reasoner_vstar",
                "hr_bench_4k", "hr_bench_8k",
                # "mme_lite",
                "pixel_reasoner_infovqa",
                {"name": "muffin_chihuahua",
                 "grid_pixels": [1, 2, 4, 8],
                 "gridsize": [1, 2,
                              4, 8, 16
                              ],
                 "mode": ["single_cell_query", "find_outlier"]},
                 "mme"
            ],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'max_tokens_per_reply': [1024],
            'tool_padding': [0.1],#, 0.05, 0.0],
            #'temperature': [1.0],
            #'num_generations': ["dataset_dependent"],
            "evaluate": False,
            "analyze": False,  # later+
            "contains_full_chkp": True,
            "run_finished": False,
            "paper": True
        },
        {
            "short_name": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_1_epoch_30_100_17p5_no_tanh",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_1_epoch_30_100_17p5_no_tanh_20260126_223735",
            "checkpoint": [382],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["no_tool", "zoom_in_absolute"],
            "dataset_name": ["pixel_reasoner_vstar", "hr_bench_4k", "hr_bench_8k",
                             # "mme_lite",
                             # "pixel_reasoner_infovqa",
                             # {"name": "muffin_chihuahua",
                             # "grid_pixels": [1, 2, 4, 8],
                             # "gridsize": [1, 2, 4, 8, 16],
                             # "mode": ["single_cell_query", "find_outlier"]}
                             ],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'max_tokens_per_reply': [1024],
            "evaluate": False,
            "analyze": False,  # later
            "contains_full_chkp": True,
            "run_finished": False
        },
        {
            "short_name": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_1_epoch_30_100_40",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_1_epoch_30_100_40_20260126_224113",
            "checkpoint": [382],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["no_tool", "zoom_in_absolute"],
            "dataset_name": ["pixel_reasoner_vstar", "hr_bench_4k", "hr_bench_8k",
                             # "mme_lite",
                             # "pixel_reasoner_infovqa",
                             # {"name": "muffin_chihuahua",
                             # "grid_pixels": [1, 2, 4, 8],
                             # "gridsize": [1, 2, 4, 8, 16],
                             # "mode": ["single_cell_query", "find_outlier"]}
                             ],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'max_tokens_per_reply': [1024],
            "evaluate": False,
            "analyze": False,  # later
            "contains_full_chkp": True,
            "run_finished": False
        },
        {
            "short_name": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_1_epoch_30_90_17p5",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_1_epoch_30_90_17p5_20260126_224356",
            "checkpoint": [382],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["no_tool", "zoom_in_absolute"],
            "dataset_name": ["pixel_reasoner_vstar", "hr_bench_4k", "hr_bench_8k",
                             # "mme_lite",
                             # "pixel_reasoner_infovqa",
                             # {"name": "muffin_chihuahua",
                             # "grid_pixels": [1, 2, 4, 8],
                             # "gridsize": [1, 2, 4, 8, 16],
                             # "mode": ["single_cell_query", "find_outlier"]}
                             ],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'max_tokens_per_reply': [1024],
            "evaluate": False,
            "analyze": False,  # later
            "contains_full_chkp": True,
            "run_finished": False
        },

        {
            "short_name": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_1_epoch_30_100_17p5_no_tanh",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_1_epoch_30_100_17p5_no_tanh_20260128_142853",
            "checkpoint": [382],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["no_tool", "zoom_in_absolute"],
            "dataset_name": ["pixel_reasoner_vstar", "hr_bench_4k", "hr_bench_8k",
                             # "mme_lite",
                             # "pixel_reasoner_infovqa",
                             # {"name": "muffin_chihuahua",
                             # "grid_pixels": [1, 2, 4, 8],
                             # "gridsize": [1, 2, 4, 8, 16],
                             # "mode": ["single_cell_query", "find_outlier"]}
                             ],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'max_tokens_per_reply': [1024],
            "evaluate": False,
            "analyze": False,  # later
            "contains_full_chkp": True,
            "run_finished": False
        },
        {
            "short_name": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_1_epoch_30_100_17p5_per_seq",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_1_epoch_30_100_17p5_per_seq_20260128_173341",
            "checkpoint": [382],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["no_tool", "zoom_in_absolute"],
            "dataset_name": ["pixel_reasoner_vstar", "hr_bench_4k", "hr_bench_8k",
                             # "mme_lite",
                             # "pixel_reasoner_infovqa",
                             # {"name": "muffin_chihuahua",
                             # "grid_pixels": [1, 2, 4, 8],
                             # "gridsize": [1, 2, 4, 8, 16],
                             # "mode": ["single_cell_query", "find_outlier"]}
                             ],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'max_tokens_per_reply': [1024],
            "evaluate": False,
            "analyze": False,  # later
            "contains_full_chkp": True,
            "run_finished": False
        },
        {
            "short_name": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_1_epoch_30_90_17p5",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_1_epoch_30_90_17p5_20260128_155702",
            "checkpoint": [382],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["no_tool", "zoom_in_absolute"],
            "dataset_name": ["pixel_reasoner_vstar", "hr_bench_4k", "hr_bench_8k",
                             # "mme_lite",
                             # "pixel_reasoner_infovqa",
                             # {"name": "muffin_chihuahua",
                             # "grid_pixels": [1, 2, 4, 8],
                             # "gridsize": [1, 2, 4, 8, 16],
                             # "mode": ["single_cell_query", "find_outlier"]}
                             ],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'max_tokens_per_reply': [1024],
            "evaluate": False,
            "analyze": False,  # later
            "contains_full_chkp": True,
            "run_finished": False
        },

        #{
        #    "short_name": "Mini_o3",
        #    "model_path": "Mini-o3/Mini-o3-7B-v1",
        #    "checkpoint": None,
        #    "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
        #    "output_path": "mini_o3",
        #    "tool_config_type": ["zoom_in_relative"],
        #    "dataset_name": ["pixel_reasoner_vstar"
        #                     # , "hr_bench_4k", "hr_bench_8k",
        #                     # "mme_lite",
        #                     # "pixel_reasoner_infovqa",
        #                     # {"name": "muffin_chihuahua",
        #                     # "grid_pixels": [1, 2, 4, 8],
        #                     # "gridsize": [1, 2, 4, 8, 16],
        #                     # "mode": ["single_cell_query", "find_outlier"]}
        #                     ],
        #    "max_pixels": [2000000],  # 2551 tokens
        #    "min_pixels": [40000],  # 51 tokens
        #    "max_tool_uses": [32],
        #    'bbox_type': ["relative"],
        #    'strict_tool_extraction': [False],
        #    'max_tokens_per_reply': [8192],
        #    "evaluate": False,
        #    "analyze": False,  # later
        #    "contains_full_chkp": True,
        #    "run_finished": False
        #},

        {
            "short_name": "Qwen_2p5_7B_mini_o3_full_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_1_epoch_30_100_17p5_20260201_164304",
            "model_path": "Qwen_2p5_7B_mini_o3_full_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_1_epoch_30_100_17p5_20260201_164304",
            "checkpoint": [554],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["no_tool", "zoom_in_absolute"],
            "dataset_name": ["pixel_reasoner_vstar", "hr_bench_4k", "hr_bench_8k",
                             # "mme_lite",
                             # "pixel_reasoner_infovqa",
                             # {"name": "muffin_chihuahua",
                             # "grid_pixels": [1, 2, 4, 8],
                             # "gridsize": [1, 2, 4, 8, 16],
                             # "mode": ["single_cell_query", "find_outlier"]}
                             ],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'max_tokens_per_reply': [1024],
            "evaluate": False,
            "analyze": False,  # later
            "contains_full_chkp": True,
            "run_finished": False
        },

        {
            "short_name": "Qwen_2p5_7B_mini_o3_full_data_cold_absolute_pixels_5k_image_tokens_min_image_500",
            "model_path": "Qwen_2p5_7B_mini_o3_full_data_cold_absolute_pixels_5k_image_tokens_min_image_500_20260201_164541",
            "checkpoint": [554],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["no_tool", "zoom_in_absolute"],
            "dataset_name": ["pixel_reasoner_vstar", "hr_bench_4k", "hr_bench_8k",
                             # "mme_lite",
                             # "pixel_reasoner_infovqa",
                             # {"name": "muffin_chihuahua",
                             # "grid_pixels": [1, 2, 4, 8],
                             # "gridsize": [1, 2, 4, 8, 16],
                             # "mode": ["single_cell_query", "find_outlier"]}
                             ],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'max_tokens_per_reply': [1024],
            "evaluate": False,
            "analyze": False,  # later
            "contains_full_chkp": True,
            "run_finished": False
        },

        {
            "short_name": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_30_100_17p5_max_10",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_30_100_17p5_max_10_20260203_160210",
            "checkpoint": [382],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["no_tool", "zoom_in_absolute"],
            "dataset_name": ["pixel_reasoner_vstar", "hr_bench_4k", "hr_bench_8k",
                             # "mme_lite",
                             # "pixel_reasoner_infovqa",
                             # {"name": "muffin_chihuahua",
                             # "grid_pixels": [1, 2, 4, 8],
                             # "gridsize": [1, 2, 4, 8, 16],
                             # "mode": ["single_cell_query", "find_outlier"]}
                             ],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'max_tokens_per_reply': [1024],
            "evaluate": False,
            "analyze": False,  # later
            "contains_full_chkp": True,
            "run_finished": False
        },

        {
            "short_name": "Qwen_2p5_7B_mini_o3_data_full_cold_absolute_pixels_5k_image_tokens_min_image_500_no_tool",
            "model_path": "Qwen_2p5_7B_mini_o3_data_full_cold_absolute_pixels_5k_image_tokens_min_image_500_no_tool_20260204_150007",
            "checkpoint": [554],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["zoom_in_absolute"],  # "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar"  # , "hr_bench_4k", "hr_bench_8k",
                             # "mme_lite",
                             # "pixel_reasoner_infovqa",
                             # {"name": "muffin_chihuahua",
                             # "grid_pixels": [1, 2, 4, 8],
                             # "gridsize": [1, 2, 4, 8, 16],
                             # "mode": ["single_cell_query", "find_outlier"]}
                             ],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'max_tokens_per_reply': [1024],
            "evaluate": False,
            "analyze": False,  # later
            "contains_full_chkp": True,
            "run_finished": False
        },

        {
            "short_name": "Qwen_2p5_7B_mini_o3_full_data_cold_absolute_pixels_500_5k_image_mi_iou_uncond_infonce_1_epoch_50_100_17p5",
            "model_path": "Qwen_2p5_7B_mini_o3_full_data_cold_absolute_pixels_500_5k_image_mi_iou_uncond_infonce_1_epoch_50_100_17p5_20260204_104104",
            "checkpoint": [554],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["zoom_in_absolute", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "hr_bench_4k", "hr_bench_8k",
                             # "mme_lite",
                             # "pixel_reasoner_infovqa",
                             # {"name": "muffin_chihuahua",
                             # "grid_pixels": [1, 2, 4, 8],
                             # "gridsize": [1, 2, 4, 8, 16],
                             # "mode": ["single_cell_query", "find_outlier"]}
                             ],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'max_tokens_per_reply': [1024],
            "evaluate": False,
            "analyze": False,  # later
            "contains_full_chkp": True,
            "run_finished": False
        },

        {
            "short_name": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_conditional_constant_tool_0p01",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_conditional_constant_tool_0p01_20260208_203453",
            "checkpoint": [382],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["zoom_in_absolute", "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "hr_bench_4k", "hr_bench_8k",
                             # "mme_lite",
                             # "pixel_reasoner_infovqa",
                             # {"name": "muffin_chihuahua",
                             # "grid_pixels": [1, 2, 4, 8],
                             # "gridsize": [1, 2, 4, 8, 16],
                             # "mode": ["single_cell_query", "find_outlier"]}
                             ],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'max_tokens_per_reply': [1024],
            "evaluate": False,
            "analyze": False,  # later
            "contains_full_chkp": True,
            "run_finished": False
        },

        {
            "short_name": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_30_100_17p5_first_20_mean",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_30_100_17p5_first_20_mean_20260210_123020",
            "checkpoint": [382],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["zoom_in_absolute"],  # , "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar"  # , "hr_bench_4k", "hr_bench_8k",
                             # "mme_lite",
                             # "pixel_reasoner_infovqa",
                             # {"name": "muffin_chihuahua",
                             # "grid_pixels": [1, 2, 4, 8],
                             # "gridsize": [1, 2, 4, 8, 16],
                             # "mode": ["single_cell_query", "find_outlier"]}
                             ],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'max_tokens_per_reply': [1024],
            "evaluate": False,
            "analyze": False,  # later
            "contains_full_chkp": True,
            "run_finished": False
        },
        {
            "short_name": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_1_epoch_30_100_17p5_padding_0p05",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_1_epoch_30_100_17p5_padding_0p05_20260216_164701",
            "checkpoint": [382],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["zoom_in_absolute"],  # , "no_tool"],
            "dataset_name": [  # "mme", "pixel_reasoner_vstar", "hr_bench_4k", "hr_bench_8k",
                # "mme_lite",
                # "pixel_reasoner_infovqa",
                {"name": "muffin_chihuahua",
                 "grid_pixels": [1, 2, 4, 8],
                 "gridsize": [1, 2,
                              4, 8, 16
                              ],
                 "mode": ["single_cell_query", "find_outlier"]}
            ],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'tool_padding': [0.1, 0.05, 0.0],  # , 0.1],
            'max_tokens_per_reply': [1024],
            "evaluate": False,
            "analyze": False,  # later+
            "contains_full_chkp": True,
            "run_finished": False
        },
        {
            "short_name": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_1_epoch_30_100_17p5_padding_0p0",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_1_epoch_30_100_17p5_padding_0p0_20260218_000757",
            "checkpoint": [382],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["zoom_in_absolute"],  # "no_tool"],  # , "no_tool"],
            "dataset_name": [  # "pixel_reasoner_vstar", "hr_bench_4k", "hr_bench_8k",
                # "mme_lite",
                # "pixel_reasoner_infovqa",
                {"name": "muffin_chihuahua",
                 "grid_pixels": [1, 2, 4, 8],
                 "gridsize": [1, 2,
                              4, 8, 16],
                 "mode": ["single_cell_query", "find_outlier"]}
            ],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'tool_padding': [0.1, 0.05, 0.0],  # , 0.1],
            'max_tokens_per_reply': [1024],
            "evaluate": False,
            "analyze": False,  # later+
            "contains_full_chkp": True,
            "run_finished": False
        },
        {
            "short_name": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_conditional_constant_tool_1p0",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_conditional_constant_tool_1p0_20260218_001054",
            "checkpoint": [382],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["no_tool", "zoom_in_absolute"],  # , "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "hr_bench_4k", "hr_bench_8k",
                             # "mme_lite",
                             # "pixel_reasoner_infovqa",
                             # {"name": "muffin_chihuahua",
                             # "grid_pixels": [1, 2, 4, 8],
                             # "gridsize": [1, 2, 4, 8, 16],
                             # "mode": ["single_cell_query", "find_outlier"]}
                             ],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'tool_padding': [0.1],  # , 0.05, 0.1],  # , 0.1],
            'max_tokens_per_reply': [1024],
            "evaluate": False,
            "analyze": False,  # later
            "contains_full_chkp": True,
            "run_finished": False
        },

        {
            "short_name": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_5k_image_tokens_min_image_500_1_epoch_const_padding_0p05",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_5k_image_tokens_min_image_500_1_epoch_const_padding_0p05_20260219_234642",
            "checkpoint": [382],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["no_tool", "zoom_in_absolute"],  # , "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "hr_bench_4k", "hr_bench_8k",
                             # "mme_lite",
                             # "pixel_reasoner_infovqa",
                             # {"name": "muffin_chihuahua",
                             # "grid_pixels": [1, 2, 4, 8],
                             # "gridsize": [1, 2, 4, 8, 16],
                             # "mode": ["single_cell_query", "find_outlier"]}
                             ],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'tool_padding': [0.05],  # , 0.05, 0.1],  # , 0.1],
            'max_tokens_per_reply': [1024],
            "evaluate": False,
            "analyze": False,  # later
            "contains_full_chkp": True,
            "run_finished": False
        },

        {
            "short_name": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_5k_image_tokens_min_image_500_1_epoch_const_padding_0p0",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_5k_image_tokens_min_image_500_1_epoch_const_padding_0p0_20260219_234130",
            "checkpoint": [382],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["no_tool", "zoom_in_absolute"],  # , "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "hr_bench_4k", "hr_bench_8k",
                             # "mme_lite",
                             # "pixel_reasoner_infovqa",
                             # {"name": "muffin_chihuahua",
                             # "grid_pixels": [1, 2, 4, 8],
                             # "gridsize": [1, 2, 4, 8, 16],
                             # "mode": ["single_cell_query", "find_outlier"]}
                             ],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'tool_padding': [0.0],  # , 0.05, 0.1],  # , 0.1],
            'max_tokens_per_reply': [1024],
            "evaluate": False,
            "analyze": False,  # later
            "contains_full_chkp": True,
            "run_finished": False
        },
        {
            "short_name": "Qwen_2p5_7B_mini_o3_full_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_30_100_17p5_continue_pr",
            "model_path": "Qwen_2p5_7B_mini_o3_full_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_30_100_17p5_continue_pr_20260218_002110",
            "checkpoint": [554],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["no_tool", "zoom_in_absolute"],  # , "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "hr_bench_4k", "hr_bench_8k",
                             # "mme_lite",
                             # "pixel_reasoner_infovqa",
                             # {"name": "muffin_chihuahua",
                             # "grid_pixels": [1, 2, 4, 8],
                             # "gridsize": [1, 2, 4, 8, 16],
                             # "mode": ["single_cell_query", "find_outlier"]}
                             ],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'tool_padding': [0.1],  # , 0.05, 0.1],  # , 0.1],
            'max_tokens_per_reply': [1024],
            "evaluate": False,
            "analyze": False,  # later
            "contains_full_chkp": True,
            "run_finished": False
        },

        {
            "short_name": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_30_100_17p5_first_20_mean_roadmap",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_30_100_17p5_first_20_mean_roadmap_20260223_161244",
            "checkpoint": [382],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["no_tool", "zoom_in_absolute"],  # , "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "hr_bench_4k", "hr_bench_8k",
                             # "mme_lite",
                             # "pixel_reasoner_infovqa",
                             # {"name": "muffin_chihuahua",
                             # "grid_pixels": [1, 2, 4, 8],
                             # "gridsize": [1, 2, 4, 8, 16],
                             # "mode": ["single_cell_query", "find_outlier"]}
                             ],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'tool_padding': [0.1],  # , 0.05, 0.1],  # , 0.1],
            'max_tokens_per_reply': [1024],
            "evaluate": False,
            "analyze": False,  # later
            "contains_full_chkp": True,
            "run_finished": False
        },

        {
            "short_name": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_conditional_constant_tool_0p3",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_conditional_constant_tool_0p3_20260225_144739",
            "checkpoint": [382],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["no_tool", "zoom_in_absolute"],  # , "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "hr_bench_4k", "hr_bench_8k",
                             # "mme_lite",
                             # "pixel_reasoner_infovqa",
                             # {"name": "muffin_chihuahua",
                             # "grid_pixels": [1, 2, 4, 8],
                             # "gridsize": [1, 2, 4, 8, 16],
                             # "mode": ["single_cell_query", "find_outlier"]}
                             ],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'tool_padding': [0.1],  # , 0.05, 0.1],  # , 0.1],
            'max_tokens_per_reply': [1024],
            "evaluate": False,
            "analyze": False,  # later
            "contains_full_chkp": True,
            "run_finished": False
        },

        {
            "short_name": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_1_epoch_30_100_17p5_tool_uses_2",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_1_epoch_30_100_17p5_tool_uses_2_20260225_112558",
            "checkpoint": [382],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["no_tool", "zoom_in_absolute"],  # , "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "hr_bench_4k", "hr_bench_8k",
                             # "mme_lite",
                             # "pixel_reasoner_infovqa",
                             # {"name": "muffin_chihuahua",
                             # "grid_pixels": [1, 2, 4, 8],
                             # "gridsize": [1, 2, 4, 8, 16],
                             # "mode": ["single_cell_query", "find_outlier"]}
                             ],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'tool_padding': [0.1],  # , 0.05, 0.1],  # , 0.1],
            'max_tokens_per_reply': [1024],
            "evaluate": False,
            "analyze": False,  # later
            "contains_full_chkp": True,
            "run_finished": False
        },

        {
            "short_name": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_1_epoch_30_100_17p5_negatives_2",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_1_epoch_30_100_17p5_negatives_2_20260225_100616",
            "checkpoint": [382],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["no_tool", "zoom_in_absolute"],  # , "no_tool"],
            "dataset_name": [#"pixel_reasoner_vstar", "hr_bench_4k", "hr_bench_8k",
                             "mme"
                             # "mme_lite",
                             # "pixel_reasoner_infovqa",
                             # {"name": "muffin_chihuahua",
                             # "grid_pixels": [1, 2, 4, 8],
                             # "gridsize": [1, 2, 4, 8, 16],
                             # "mode": ["single_cell_query", "find_outlier"]}
                             ],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'tool_padding': [0.1],  # , 0.05, 0.1],  # , 0.1],
            'max_tokens_per_reply': [1024],
            "evaluate": False,
            "analyze": False,  # later
            "contains_full_chkp": True,
            "run_finished": False
        },

        {
            "short_name": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_1_epoch_30_100_17p5_tool_params",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_1_epoch_30_100_17p5_tool_params_20260227_003116",
            "checkpoint": [382],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["no_tool", "zoom_in_absolute"],  # , "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "hr_bench_4k", "hr_bench_8k",
                             # "mme_lite",
                             # "pixel_reasoner_infovqa",
                             # {"name": "muffin_chihuahua",
                             # "grid_pixels": [1, 2, 4, 8],
                             # "gridsize": [1, 2, 4, 8, 16],
                             # "mode": ["single_cell_query", "find_outlier"]}
                             ],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'tool_padding': [0.1],  # , 0.05, 0.1],  # , 0.1],
            'max_tokens_per_reply': [1024],
            "evaluate": False,
            "analyze": False,  # later
            "contains_full_chkp": True,
            "run_finished": False
        },

        {
            "short_name": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_5k_image_tokens_min_image_500_no_tool",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_5k_image_tokens_min_image_500_no_tool_20260227_014456",
            "checkpoint": [382],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["no_tool", "zoom_in_absolute"],  # , "no_tool"],
            "dataset_name": [  "pixel_reasoner_vstar", "hr_bench_4k", "hr_bench_8k",
                "mme",
                # "mme_lite",
                "pixel_reasoner_infovqa",
                {"name": "muffin_chihuahua",
                 "grid_pixels": [1, 2, 4, 8],
                 "gridsize": [1, 2, 4, 8, 16],
                 "mode": ["single_cell_query", "find_outlier"]}
            ],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'tool_padding': [0.1],  # , 0.05, 0.1],  # , 0.1],
            'max_tokens_per_reply': [1024],
            #'temperature': [1.0],
            #'num_generations': ["dataset_dependent"],
            "evaluate": False,
            "analyze": False,  # later
            "contains_full_chkp": True,
            "run_finished": False,
            "paper": True
        },

        {
            "short_name": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_conditional_constant_tool_0p03",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_conditional_constant_tool_0p03_20260228_133940",
            "checkpoint": [382],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["no_tool", "zoom_in_absolute"],  # , "no_tool"],
            "dataset_name": [  "pixel_reasoner_vstar", "hr_bench_4k", "hr_bench_8k",
                # "mme",
                # "mme_lite",
                #"pixel_reasoner_infovqa",
                # {"name": "muffin_chihuahua",
                # "grid_pixels": [1, 2, 4, 8],
                # "gridsize": [1, 2, 4, 8, 16],
                # "mode": ["single_cell_query", "find_outlier"]}
            ],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'tool_padding': [0.1],  # , 0.05, 0.1],  # , 0.1],
            'max_tokens_per_reply': [1024],
            "evaluate": False,
            "analyze": False,  # later
            "contains_full_chkp": True,
            "run_finished": False
        },

        {
            "short_name": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_1_epoch_30_100_20",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_1_epoch_30_100_20_20260228_150728",
            "checkpoint": [382],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["zoom_in_absolute", "no_tool"],#, "zoom_in_absolute"],  # , "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "hr_bench_4k", "hr_bench_8k",
                             "mme",
                             # "mme_lite",
                             # "pixel_reasoner_infovqa",
                             # {"name": "muffin_chihuahua",
                             # "grid_pixels": [1, 2, 4, 8],
                             # "gridsize": [1, 2, 4, 8, 16],
                             # "mode": ["single_cell_query", "find_outlier"]}
                             ],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'tool_padding': [0.1],  # , 0.05, 0.1],  # , 0.1],
            'max_tokens_per_reply': [1024],
            "evaluate": False,
            "analyze": True,  # later
            "contains_full_chkp": True,
            "run_finished": False
        },
        {
            "short_name": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_1_epoch_30_100_17p5_chkps",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_1_epoch_30_100_17p5_chkps_20260302_012106",
            "checkpoint": [300],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["no_tool", "zoom_in_absolute"],  # , "no_tool"],
            "dataset_name": [#"pixel_reasoner_vstar", "hr_bench_4k", "hr_bench_8k",
                             # "mme",
                             # "mme_lite",
                             # "pixel_reasoner_infovqa",
                             {"name": "muffin_chihuahua",
                             "grid_pixels": [1, 2, 4, 8],
                             "gridsize": [1, 2, 4, 8, 16],
                             "mode": ["single_cell_query", "find_outlier"]}
                             ],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'tool_padding': [0.1],  # , 0.05, 0.1],  # , 0.1],
            'max_tokens_per_reply': [1024],
            "evaluate": False,
            "analyze": False,  # later
            "contains_full_chkp": True,
            "run_finished": False
        },
        {
            "short_name": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_1_epoch_30_100_15",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_1_epoch_30_100_15_20260301_145725",
            "checkpoint": [382],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["zoom_in_absolute", "no_tool"],#, "zoom_in_absolute"],  # , "no_tool"],
            "dataset_name": [  "pixel_reasoner_vstar", "hr_bench_4k", "hr_bench_8k",
                "mme",
                # "mme_lite",
                # "pixel_reasoner_infovqa",
                #{"name": "muffin_chihuahua",
                # "grid_pixels": [1, 2, 4, 8],
                # "gridsize": [1, 2, 4, 8, 16],
                # "mode": ["single_cell_query", "find_outlier"]}
            ],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'tool_padding': [0.1],  # , 0.05, 0.1],  # , 0.1],
            'max_tokens_per_reply': [1024],
            "evaluate": False,
            "analyze": True,  # later
            "contains_full_chkp": True,
            "run_finished": False
        },
        {
            "short_name": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_1_epoch_30_100_17p5_sample_recall",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_1_epoch_30_100_17p5_sample_recall_20260302_042226",
            "checkpoint": [382],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["no_tool", "zoom_in_absolute"],  # , "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "hr_bench_4k", "hr_bench_8k",
                             # "mme",
                             # "mme_lite",
                             # "pixel_reasoner_infovqa",
                             # {"name": "muffin_chihuahua",
                             # "grid_pixels": [1, 2, 4, 8],
                             # "gridsize": [1, 2, 4, 8, 16],
                             # "mode": ["single_cell_query", "find_outlier"]}
                             ],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'tool_padding': [0.1],  # , 0.05, 0.1],  # , 0.1],
            'max_tokens_per_reply': [1024],
            "evaluate": False,
            "analyze": False,  # later
            "contains_full_chkp": True,
            "run_finished": False
        },
        #"_20260302_042226"
        #"Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_1_epoch_30_100_17p5_sample_recall_20260302_042226"
        {
            "short_name": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_1_epoch_30_100_17p5_no_clamp",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_1_epoch_30_100_17p5_no_clamp_20260307_180324",
            "checkpoint": [382],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["no_tool", "zoom_in_absolute"],  # , "no_tool"],
            "dataset_name": ["pixel_reasoner_vstar", "hr_bench_4k", "hr_bench_8k",
                             # "mme",
                             # "mme_lite",
                             # "pixel_reasoner_infovqa",
                             # {"name": "muffin_chihuahua",
                             # "grid_pixels": [1, 2, 4, 8],
                             # "gridsize": [1, 2, 4, 8, 16],
                             # "mode": ["single_cell_query", "find_outlier"]}
                             ],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'tool_padding': [0.1],  # , 0.05, 0.1],  # , 0.1],
            'max_tokens_per_reply': [1024],
            "evaluate": False,
            "analyze": False,  # later
            "contains_full_chkp": True,
            "run_finished": False
        },
        {
            "short_name": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_1_epoch_30_100_17p5_per_seq",
            "model_path": "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_1_epoch_30_100_17p5_per_seq_20260309_110313",
            "checkpoint": [382],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["no_tool", "zoom_in_absolute"],  # , "no_tool"],
            "dataset_name": [#"pixel_reasoner_vstar", "hr_bench_4k", "hr_bench_8k",
                             "mme",
                             # "mme_lite",
                             # "pixel_reasoner_infovqa",
                             # {"name": "muffin_chihuahua",
                             # "grid_pixels": [1, 2, 4, 8],
                             # "gridsize": [1, 2, 4, 8, 16],
                             # "mode": ["single_cell_query", "find_outlier"]}
                             ],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'tool_padding': [0.1],  # , 0.05, 0.1],  # , 0.1],
            'max_tokens_per_reply': [1024],
            "evaluate": False,
            "analyze": False,  # later
            "contains_full_chkp": True,
            "run_finished": False
        },
        {
            "short_name": "Qwen_2p5_7B_mini_o3_full_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_30_100_17p5_continue_pr_no_exploration_reward",
            "model_path": "Qwen_2p5_7B_mini_o3_full_data_cold_absolute_pixels_500_5k_image_mi_iou_cond_infonce_30_100_17p5_continue_pr_no_exploration_reward_20260313_003046",
            "checkpoint": [554],
            "model_class": "Qwen/Qwen2.5-VL-7B-Instruct",
            "tool_config_type": ["no_tool"], #["no_tool", "zoom_in_absolute"],  # , "no_tool"],
            "dataset_name": [  #"pixel_reasoner_vstar", "hr_bench_4k", "hr_bench_8k",
                "mme_shard_0"
                # "mme_lite",
                # "pixel_reasoner_infovqa",
                # {"name": "muffin_chihuahua",
                # "grid_pixels": [1, 2, 4, 8],
                # "gridsize": [1, 2, 4, 8, 16],
                # "mode": ["single_cell_query", "find_outlier"]}
            ],
            "max_pixels": [5000 * 28 * 28],
            "min_pixels": [500 * 28 * 28],
            'bbox_type': ["absolute"],
            'strict_tool_extraction': [False],
            'tool_padding': [0.1],  # , 0.05, 0.1],  # , 0.1],
            'max_tokens_per_reply': [1024],
            "evaluate": True,
            "analyze": True,  # later
            "contains_full_chkp": True,
            "run_finished": False
        },


    ]

    return models_input

def analyze(model_paths = None, metrics = None, datasets_for_analysis=None, do_print=None, tool_config_types=None,
            tool_paddings=None, num_generations=None):
    if metrics is None:
        metrics = [
        "accuracy",

        # "no_answer",

        "avg_pixel_reasoning",
        "pixel_reasoning_distr",
        "avg_first_completion_len",
        # "avg_second_completion_len",
        # "avg_total_completion_len",

        # "image_size_x_performance_correlation",
        "zoom_in_fraction_median",
        # "avg_image_size",
        # "max_image_size",
        # "min_image_size",
        # "median_image_size",
        #"accuracy_if_tool_used",
        #"accuracy_if_tool_not_used",
        # "tool_success_rate",

        # "iou_std",

        # "precision",
        # "precision_std",#

        # "recall",
        # "recall_std"

        ]
        metrics_to_keep_for_export = []
    else:
        metrics_to_keep_for_export = metrics.copy()
    if datasets_for_analysis is None:
        datasets_for_analysis = [  # {"name": "muffin_chihuahua",
        # "grid_pixels": [1,
        #                 2, 4,
        #                8],
        # "gridsize": [1, 2, 4, 8, 16],
        # "mode": ["single_cell_query",
        #          "find_outlier"
        #          ]
        # },
        "pixel_reasoner_vstar",
        # "pixel_reasoner_infovqa",
        "hr_bench_4k",
        "hr_bench_8k",
        # "mme_lite",
         "mme"
        ]
    if do_print is None:
        do_print = True

    if tool_config_types is None:
        tool_config_types = [
            "no_tool",
            #"PR_crop_image_normalized,select_frames",
            "zoom_in_absolute"
        ]

    models_input = get_models_input()

    if model_paths is not None:
        mapping = {model["model_path"]:model for model in models_input}
        models_input = [mapping[model_path] for model_path in model_paths]

        for idx, model in enumerate(models_input):
            model["analyze"] = True
            model["dataset_name"] = datasets_for_analysis
            model["tool_config_type"] = tool_config_types
            if tool_paddings is not None and tool_paddings[idx] is not None:
                model["tool_padding"] = tool_paddings[idx]
            if num_generations is not None and num_generations[idx] is not None:
                model["num_generations"] = num_generations[idx]
            #model["evaluate"] = False


    filtered_models = [model for model in models_input if model["analyze"] is True]

    all_evals = Evals(filtered_models)

    order = [eval.get_eval_name() for eval in all_evals.evals]

    unpacked_datasets_for_analysis, short_names = unpack_datasets(datasets_for_analysis)


    metrics_to_keep_for_export += ["accuracy",

                                  "no_answer",
                                  # "ANLS",
                                  "avg_pixel_reasoning",
                                  "zoom_in_fraction_median",
                                   #"accuracy_if_tool_used",
                                   #"accuracy_if_tool_not_used",
                                  "ious",
                                  "iou_std",

                                  "precision",
                                  "precision_std",

                                  "recall",
                                  "recall_std"
                                  ]

    metrics_to_keep_for_export_dedup = []
    for metric in metrics_to_keep_for_export:
        if metric not in metrics_to_keep_for_export_dedup:
            metrics_to_keep_for_export_dedup.append(metric)
    metrics_to_keep_for_export = metrics_to_keep_for_export_dedup

    dfs_dict = {}
    dfs_dict_for_export = {}
    pd.set_option('display.max_columns', 10)
    for dataset, short_name in zip(unpacked_datasets_for_analysis, short_names):
        if dataset == "pixel_reasoner_infovqa":
            metrics.append({"metric_name": "string_matching", "metric_short_name": "ANLS",
                            "match_type": "levenshtein_similarity", "preprocess": "lowercase", "cutoff": 0.5})
        else:
            metrics = [m for m in metrics if not isinstance(m, dict)]
        for tool_config_type in tool_config_types:
            full_metrics = copy.deepcopy(metrics)
            if dataset.startswith("muffin") and tool_config_type != "no_tool":
                full_metrics.append("ious")
            df = all_evals.make_results_table(names=order, metrics=full_metrics,
                                              fixed_params={"dataset_name": dataset,
                                                            # "max_pixels":
                                                            # "min_pixels": 500 * 28 * 28,
                                                            "tool_config_type": tool_config_type,
                                                            # 'bbox_type': "absolute",
                                                            # 'strict_tool_extraction': True,
                                                            # 'max_tokens_per_reply': 256
                                                            }
                                              )
            #print(f"df after make_results_table: {df}")
            # print(df.head())
            metrics_to_drop = [m for m in metrics if not (m in metrics_to_keep_for_export or isinstance(m, dict))]
            metrics_to_drop.append("name")
            move_to_front = None
            if dataset == "pixel_reasoner_infovqa":
                metrics_to_drop.append("accuracy")
                move_to_front = "ANLS"

            # if not dataset.startswith("muffin"):
            # metrics_to_drop.append("ious")

            if tool_config_type == "no_tool":
                for m in ["avg_pixel_reasoning", "pixel_reasoning_distr", "zoom_in_fraction_median", "ious", "iou_std"]:
                    if m in metrics:
                        metrics_to_drop.append(m)

            metrics_to_drop = list(set(metrics_to_drop))
            print(f"metrics_to_drop: {metrics_to_drop}")
            # metrics_to_drop += [m for m in metrics if isinstance(m, dict)]

            if move_to_front:
                first_col = df.pop(move_to_front)
                df.insert(0, move_to_front, first_col)

            tool_config_export_name = "tool" if tool_config_type != "no_tool" else "no_tool"

            if (short_name, tool_config_export_name) not in dfs_dict_for_export:
                dfs_dict_for_export[(short_name, tool_config_export_name)] = df.drop(metrics_to_drop, axis=1)
            else:
                dfs_dict_for_export[(short_name, tool_config_export_name)] = dfs_dict_for_export[(short_name, tool_config_export_name)].combine_first(df.drop(metrics_to_drop, axis=1))
            # print(df)
            dfs_dict[(dataset, tool_config_type)] = df
    final_df = pd.concat(dfs_dict, axis=1)
    #print(final_df)
    final_df_for_export = pd.concat(dfs_dict_for_export, axis=1)
        #print(f"final_df_for_export: {final_df_for_export}")
    if do_print:
        print(final_df_for_export.to_csv(sep=",", index=False))
    else:
        return final_df_for_export

if __name__ == "__main__":

    models_input = get_models_input()

    parser = argparse.ArgumentParser(description='Evaluate VLM model performance')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the model')
    parser.add_argument('--dataset_name', type=str_or_dict, default=None, help='Name of the dataset to evaluate')
    parser.add_argument('--tool_config_type', type=str, default=None, help='Type of tool configuration to evaluate')

    args = parser.parse_args()

    if args.model_path is not None:
        found_one = False
        for m in models_input:
            if m["model_path"] == args.model_path:
                m["evaluate"] = True
                m["dataset_name"] = [args.dataset_name]
                m["tool_config_type"] = [args.tool_config_type]
                if found_one:
                    raise ValueError(f"multiple models with the same name!")
                else:
                    found_one = True
            else:
                m["evaluate"] = False
        do_eval = True
        batch_eval = True
        exist_ok = True

        do_analysis = False
        batch_analyze = False

        do_metric_comparison = False

        PMI_analysis = False
    else:
        do_eval = True
        batch_eval = True
        exist_ok = True

        do_analysis = False
        batch_analyze = True

        do_metric_comparison = False

        PMI_analysis = False


    if do_eval:
        if batch_eval:
            all_evals = Evals([model for model in models_input if model["evaluate"] is True])
            order = [eval.get_eval_name() for eval in all_evals.evals]
        else:
            all_evals = Evals(models_input)
            logger.info(all_evals)

            order = [

            ]
        all_evals.do_evals(names=order, exist_ok=exist_ok, no_vllm=False, batch_size=None)

    if do_analysis:
        analyze()


    if PMI_analysis:
        all_evals = Evals([model for model in models_input if "PMI_analysis" in model.keys() and model["PMI_analysis"] is True])
        order = [eval.get_eval_name() for eval in all_evals.evals]
        full_paths = [eval.get_save_path() for eval in all_evals.evals]

        logger.info(f"order: {order}")

        #full_path = ["/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/focusreason/runs/Qwen_2p5_7B_pr_data_warm_absolute_pixels_5k_image_tokens_20251008_102353/checkpoint-382/eval/dataset_pixel_reasoner_vstar_prompt_PR_zoom_in_old_max_pixels_3920000_min_pixels_3136"]
        fixed_params = {"dataset_name": "pixel_reasoner_vstar",
                        "max_pixels": 5000 * 28 * 28,
                        "min_pixels": 250*28*28,
                        "tool_config_type": "PR_zoom_in_old"#"no_tool"#
                        }
        filtered_evals = all_evals.filter(order, fixed_params=fixed_params)
        do_pmi_analysis([eval.get_save_path() for eval in filtered_evals],
                        fixed_params=fixed_params)