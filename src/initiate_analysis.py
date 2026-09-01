import argparse
import os
import shutil
import signal
import itertools
import json
import pandas as pd
import subprocess

from tqdm import tqdm
from typing import Union
from open_r1.utils.logger import setup_project_logging
from open_r1.analysis.make_results_table import get_results, compute_new_metric

logger = setup_project_logging(log_file=None)

def unpack_datasets(list_of_datasets: list[dict]) -> tuple[list[str], list[str], list[str]]:
    """Expand the config's ``dataset_name`` entries into aligned (name, short_name,
    path) lists. Every entry is a dict ``{"name", "path", ...}``; ``muffin_chihuahua``
    expands over its grid configs (all sharing the same path), every other dataset
    maps to a single entry. The three returned lists are aligned element-wise."""
    unpacked_list = []
    short_names = []
    paths = []

    short_name_map = {
        "pixel_reasoner": "pr",
        "pixel_reasoner_vstar": "V*",
        "hr_bench_4k": "hrb_4k",
        "hr_bench_8k": "hrb_8k",
        "mme_lite": "mme_lite",
        "mme": "mme"
    }
    for dataset in list_of_datasets:
        if not isinstance(dataset, dict):
            raise ValueError(
                f"Each dataset must be a dict with 'name' and 'path', but got {dataset!r}")
        if "name" not in dataset or "path" not in dataset:
            raise ValueError(f"dataset entry {dataset!r} needs both 'name' and 'path'")
        name, path = dataset["name"], dataset["path"]

        if name == "muffin_chihuahua":
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
                        paths.append(path)
        else:
            unpacked_list.append(name)
            short_names.append(short_name_map.get(name, name))
            paths.append(path)
    return unpacked_list, short_names, paths

class ModelParams:

    def __init__(self, short_name, model_path, model_class, output_path=None, dir_suffix="eval", **kwargs):

        self.dir_suffix = dir_suffix

        self.short_name = short_name
        self.full_model_path = model_path

        self.output_path = output_path

        self.full_output_path = model_path if output_path is None else output_path

        self.model_class = model_class

class DatasetParams:
    def __init__(self, dataset_name, dataset_path, **kwargs):
        self.dataset_name = dataset_name
        self.default_num_generations = 4

        if self.dataset_name.startswith("muffin_chihuahua"):
            if self.dataset_name.endswith("class_0_1") and "gridsize_1_" not in self.dataset_name:
                task = "find_outlier"
            else:
                task = "single_cell_query"

            subdir = dataset_name.removeprefix("muffin_chihuahua_")
            self.data_files = os.path.join(dataset_path, subdir, f"{task}.jsonl")
            self.image_folders = os.path.join(dataset_path, subdir, "images")
        else:
            # Every other dataset follows the download_data.py output convention:
            # <path>/test.jsonl, with image paths written relative to <path>.
            self.data_files = os.path.join(dataset_path, "test.jsonl")
            self.image_folders = dataset_path

class EvalParams:
    def __init__(self, batch_size=200, tensor_parallel_size=1, enforce_eager=True, no_vllm=False, dry_run=False,
                 min_pixels=None, max_pixels=None, tool_config_type = "no_tool", image_limit=6, max_tool_uses=5,
                 bbox_type=None, strict_tool_extraction=False, max_tokens_per_reply=256,
                 tool_padding=0.1, tool_adaptive_padding_threshold=600, port:int=8000,
                 temperature=0.0, num_generations=1, **kwargs):
        # technical
        self.batch_size = batch_size
        self.tensor_parallel_size = tensor_parallel_size
        self.enforce_eager = enforce_eager
        self.no_vllm = no_vllm
        self.dry_run = dry_run
        self.temperature = temperature
        self.num_generations = num_generations
        self.port = port

        # outcome relevant
        if max_pixels is None:
            self.max_pixels = 1024*16 * 28 * 28
        else:
            self.max_pixels = max_pixels

        if min_pixels is None:
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

        cmd += [
            'python', '-m', 'open_r1.evaluation.evaluate',
            '--model_path', f'{self.model_params.full_model_path}',
            '--model_class', f'{self.model_params.model_class}',
            '--output_path', f'{save_path}',

            '--dataset_name', self.dataset_params.dataset_name,

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
        working_directory = None

        return cmd, working_directory

    def get_eval_name(self):
        if self.eval_params.temperature > 0:
            eval_name = f"{self.model_params.short_name}_{self.dataset_params.dataset_name}_{self.eval_params.tool_config_type}_{self.eval_params.max_pixels}_{self.eval_params.min_pixels}_{self.eval_params.tool_padding}_{self.eval_params.temperature}_{self.eval_params.num_generations}"
        else:
            eval_name = f"{self.model_params.short_name}_{self.dataset_params.dataset_name}_{self.eval_params.tool_config_type}_{self.eval_params.max_pixels}_{self.eval_params.min_pixels}_{self.eval_params.tool_padding}"
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


    def do_eval(self, exist_behaviour:str="raise", no_vllm=False, batch_size=None):
        if no_vllm:
            self.eval_params.no_vllm = True
        if batch_size is not None:
            self.eval_params.batch_size = batch_size

        eval_name = self.get_eval_name()
        save_path = self.get_save_path()

        if os.path.exists(save_path):
            if exist_behaviour == "raise":
                raise RuntimeError(f"Path {save_path} exists and exist_behaviour = {exist_behaviour}.")
            elif exist_behaviour == "overwrite":
                logger.info(f"Path {save_path} exists. removing and overwriting")
                shutil.rmtree(save_path)
            elif exist_behaviour == "skip":
                logger.info(f"Path {save_path} exists. skipping")
                return
            else:
                raise ValueError(f"exist_behaviour = {exist_behaviour} is not supported")

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

class Evals:
    def __init__(self, input: list[dict]):
        self.input = input

        self.classes = {"model_params": ModelParams, "dataset_params": DatasetParams, "eval_params": EvalParams}

        self.evals = self.split_input()


    def split_input(self) -> list[SingleEval]:
        result = []
        for entry in self.input:
            # Unpack datasets first so name and path stay aligned, then treat each
            # (dataset_name, dataset_path) pair as a single axis -- otherwise the
            # cartesian product below would mismatch names with paths.
            raw_datasets = entry.pop("dataset_name")
            if isinstance(raw_datasets, dict):
                raw_datasets = [raw_datasets]
            names, _, paths = unpack_datasets(raw_datasets)
            dataset_pairs = list(zip(names, paths))

            # Wrap every remaining scalar into a 1-element list.
            for key, value in entry.items():
                if not isinstance(value, list):
                    entry[key] = [value]

            param_names = list(entry.keys()) + ["__dataset__"]
            param_values = list(entry.values()) + [dataset_pairs]

            # Create objects for each combination of the cartesian product.
            for combo in itertools.product(*param_values):
                kwargs = dict(zip(param_names, combo))
                kwargs["dataset_name"], kwargs["dataset_path"] = kwargs.pop("__dataset__")
                instances = tuple(cls(**kwargs) for cls in self.classes.values())
                single_eval = SingleEval(**dict(zip(self.classes.keys(), instances)))
                result.append(single_eval)
        print(f"Generated {len(result)} evaluations from {self.input}")
        return result

    def __repr__(self):
        return "\n".join([f"'{eval.get_eval_name()}'," for eval in self.evals])

    def do_evals(self, names=list[str], exist_behaviour="raise", no_vllm=False, batch_size=None):
        if names is None:
            eval_list = self.evals
        else:
            eval_list = [eval for eval in self.evals if eval.get_eval_name() in names]

        eval_list = sorted(eval_list, key=lambda x: names.index(x.get_eval_name()))
        for eval in tqdm(eval_list):
            eval.do_eval(exist_behaviour, no_vllm, batch_size)
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

        return eval_list

    def make_results_table(self, names: list[str], metrics: list[str], fixed_params: dict = None):
        results = []
        eval_list = self.filter(names, fixed_params)

        for eval in eval_list:
            save_path = eval.get_save_path()
            print(f"save_path: {save_path}")
            # "full_results_relaxed.json"
            result = get_results(os.path.join(save_path, "full_results.json"), metrics)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate VLM model performance', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--config_path', type=str, default=None,
                        help='''Path to config json. The following settings are supported:
                        short_name: the displayed name,
                        model_path: path to model,
                        output_path: path to save the output (same as model path if not given),
                        model_class: only "Qwen/Qwen2.5-VL-7B-Instruct" is currently supported,
                        tool_config_type*: choose from "no_tool", "zoom_in_absolute", 
                                          needed for Pixel Reasoner evaluation: "PR_crop_image_normalized,select_frames"
                        dataset_name: list of dicts with keys "name" and "path". For non-M&C datasets,
                                                 "path" is the download_data.py --out_dir (holding test.jsonl + images).
                                                 For M&C ("muffin_chihuahua"), "path" is the generated splits dir plus
                                                 "grid_pixels"*, "gridsize"* and "mode"* ("single_cell_query"/"find_outlier"),
                        max_pixels*: maximum number of pixels per image,
                        min_pixels*: minimum number of pixels per image,
                        bbox_type*: if not given, absolute or relative bboxes are allowed if "absolute" then only absolute bboxes are allowed
                        strict_tool_extraction*: requires that there is a single tool_start, a single tool_end and tool_end is at the end of the generation
                        tool_padding*: how much the tool call region is extended in all directions
                        max_tokens_per_reply*: maximum number of tokens the model is allowed to generate per turn
                        
                        all values marked with * should be given as a list. All combinations from the cartesian product of the lists are evaluated.
                        '''
                        )
    parser.add_argument('--evaluate', action='store_true', help='Whether to evaluate model performance')
    parser.add_argument('--analyse', action='store_true', help='Whether to display model performance')
    parser.add_argument('--metrics', nargs='+', type=str, default=None,
                        help='''The metrics to be displayed. The following metrics are supported:
                             accuracy, 
                             accuracy_if_tool_used,
                             accuracy_if_tool_not_used,
                             avg_pixel_reasoning: average tool use,
                             pixel_reasoning_distr: normalized histogram of tool uses,
                             avg_first_completion_len: number of tokens before tool use,
                             avg_second_completion_len: number of tokens after tool use,
                             avg_total_completion_len: total number of model-generated tokens,
                             image_size_x_performance_correlation: pearson correlation between image size in pixel and performance in accuracy,
                             zoom_in_fraction_median: which ratio of the image was zoomed into,
                             tool_success_rate: ratio between succeeded and attempted tool calls,
                             ious: mean intersection-over-union of zoom-in area and ground truth,
                             iou_std: standard deviation of intersection-over-union of zoom-in area and ground truth,
                             precision: mean precision of zoom-in area and ground truth,
                             precision_std: standard deviation of precision of zoom-in area and ground truth,
                             recall: mean recall of zoom-in area and ground truth,
                             recall_std: standard deviation of recall of zoom-in area and ground truth'''
                        )
    parser.add_argument('--exist_behaviour', type=str, default="raise", help="What to do if the eval save path exists. "
                                                                             "Choose from 'raise', 'overwrite' or 'skip'")

    args = parser.parse_args()

    models_input = [json.load(open(args.config_path))]

    print(models_input)

    all_evals = Evals([model for model in models_input])
    names = [eval.get_eval_name() for eval in all_evals.evals]

    if args.evaluate:
        all_evals.do_evals(names=names, exist_behaviour=args.exist_behaviour, no_vllm=False)

    if args.analyse:
        metrics = ["accuracy",
                "avg_pixel_reasoning",
                "pixel_reasoning_distr",
                "avg_first_completion_len",
                "zoom_in_fraction_median"
        ] if args.metrics is None else args.metrics


        results_df = all_evals.make_results_table(names=names, metrics=metrics)
        print(results_df.to_csv(sep=",", index=False))

