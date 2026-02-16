# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import json
from dataclasses import dataclass, field, asdict
from typing import Optional, Union
from functools import partial
from aim.hugging_face import AimCallback
from trl import GRPOConfig, ModelConfig, ScriptArguments, TrlParser, get_peft_config

import determined as det

from open_r1.utils.logger import setup_project_logging
from open_r1.utils.tools import Tool, Message, TOOL_CONFIGS
from open_r1.utils.rewards import curiosity_reward, pr_penalty_reward, format_reward, accuracy_reward, \
    format_reward_only_answer, mutual_information_reward, constant_exploration, iou_reward
from open_r1.utils.utils import basic_iou_target_fn
from open_r1.trainer import UpdatedVLMGRPOTrainerVLLM
from open_r1.preprocess_data import prepare_data
from open_r1.utils.prompts import get_question_template
from open_r1.vlm_modules import *


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.
    """
    data_file_paths: str = field(
        default=None,
        metadata={"help": "Paths to data files, separated by ':'"},
    )
    image_folders: str = field(
        default=None,
        metadata={"help": "Paths to image folders, separated by ':'"},
    )
    arrow_cache_dir: str = field(
        default=None,
        metadata={"help": "Path to arrow cache directory"},
    )
    val_split_ratio: float = field(
        default=0.0,
        metadata={"help": "Ratio of validation split, default 0.0"},
    )
    data_subset: Optional[str] = field(
        default=None,
        metadata={"help": "Slice notation for dataset subset (e.g. '0:100', '::2'). None uses full dataset"},
    )
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    reward_func_weights: list[float] = field(
        default_factory=lambda: [1.0],
        metadata={"help": "Weights for reward functions. Must have the same length as reward_funcs"},
    )
    reward_func_conditionals: list[bool] = field(
        default=None,
        metadata={"help": "whether this reward function should be conditioned on accuracy reward"},
    )
    reward_func_usage: list[str] = field(
        default=None,
        metadata={"help": "specifies for each reward function if it should be used for 'reward', 'sampling_weights' or 'both'"}
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image (for QwenVL)"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image (for QwenVL)"},
    )
    max_anyres_num: Optional[int] = field(
        default=12,
        metadata={"help": "Maximum number of anyres blocks for the image (for InternVL)"},
    )
    reward_method: Optional[str] = field(
        default=None,
        metadata={
            "help": "Choose reward method: 'default', 'mcp', ..."
        },
    )
    logging: Optional[bool] = field(
        default=None,
        metadata={
            "help": "if True, saves config and checkpoints to local folder and logs to aim"
        }
    )
    multi_turn: Optional[str] = field(
        default= None,
        metadata={
            "help": "which version of multi_turn should be used. Options are none, 'text', 'image'"
        }
    )
    chat_template: Optional[str] = field(
        default=None,
        metadata={
            "help": "path to chat template json file. If None, uses the chat template from the model "
        }
    )
    prompt_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "which type of initial prompt to use. supported are 'default' and 'no_think'"
        }
    )
    tool_use_penalty_threshold: Optional[float] = field(
        default=None,
        metadata={
            "help": "threshold for penalizing tool use. Must be set if 'pr_penalty' should be used in reward_funcs "
        }
    )
    pixel_reasoning_threshold: Optional[float] = field(
        default=None,
        metadata={
            "help": "threshold for penalizing tool use. Must be set if 'curiosity' should be used in reward_funcs "
        }
    )
    max_tool_uses: Optional[int] = field(
        default=None,
        metadata={
            "help": "maximum number of tool uses allowed. This is a technical hyperparameter preventing us from running into OOM."
                    "When setting up vLLM engine, limit_image_per_prompt >= max_tool_uses + input_image_count "
        }
    )
    strict_tool_extraction: Optional[bool] = field(
        default=None,
        metadata={
            "help": "whether to use strict tool extraction. this means we only allow a single tool_start, "
                    "a single tool_end and tool_end must be at the end of the generated seq"
        }
    )
    finish_after_wrong_tool_call: Optional[bool] = field(
        default=True,
        metadata={
            "help": "whether to stop the generation after a wrong tool call or give the model the chance to correct itself"
        }
    )
    tool_config: Optional[str] = field(
        default=None,
        metadata = {
            "help": "pre-configured tool configuration to use. comma-separated list of tool configurations"
        }
    )
    global_buffer: Optional[bool] = field(
        default=False,
        metadata = {
            "help": "whether to share the buffer across processes"
        }
    )
    mutual_information_clip_value: Optional[float] = field(
    default = None,
    metadata = {
        "help": "the clip value for the mean-like mutual information reward. mean(clip(contrast, +-gamma)) "
    })
    mutual_information_threshold: Optional[float] = field(
        default=None,
        metadata={
            "help": "the minimal threshold for the median-like mutual information reward. #{contrast > delta}/#contrast"
        }

    )
    mutual_information_len_exponent: Optional[float] = field(
        default=None,
        metadata={
            "help": "the exponent alpha for length-adaptive reward: (len/len_scaling)^alpha"
        }
    )

    mutual_information_len_scaling: Optional[float] = field(
        default=None,
        metadata={
            "help": "the scale factor for length-adaptive reward: (len/len_scaling)^alpha."
        }
    )
    mutual_information_length_factor: Optional[Union[float, str]] = field(
        default=None,
        metadata={
            "help": "the length factor to alter the full diff: 1/lf sum d_i . if 'len' then it is the length of the contrasted area"
        }
    )
    mutual_information_mean_threshold: Optional[float] = field(
        default=None,
        metadata={
            "help": "the mean threshold for MI reward: mean(clip(PMI) - threshold)."
        }
    )

    mutual_information_discretize: Optional[bool] = field(
        default = False,
        metadata = {
            "help": "whether to discretize the reward at the end: reward = -1 if reward < 0, else +1"
        }
    )

    mutual_information_quantile: Optional[float] = field(
        default = None,
        metadata={
            "help": "use the q-th quantile: quantile(clip(PMI, gamma) - threshold)"
        }
    )

    mutual_information_select_k: Optional[int] = field(
        default = None,
        metadata={
            "help": "only use k values for further processing. they are determined by select_k_type"
        }
    )

    mutual_information_select_k_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "how to choose the k values of select_k. choose from 'first', 'max'"
        }
    )

    mi_tanh: Optional[bool] = field(
        default = False,
        metadata={
            "help": "whether to use a tanh function: torch.tanh(torch.clamp(contrast_diff, min=-gamma, max=gamma) * length_factor)"
        }
    )

    mi_alternative_action: Optional[str] = field(
        default=None,
        metadata={
            "help": 'the alternative action to use. choose from "second_model_generation", "double_newline", "double_newline,the,answer,is"'
        }
    )

    mi_answer_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "answer type: 'ground_truth', 'full_generation' or 'entropy' (first token) or 'alternative_tool_parameters'"
        }
    )

    mi_use_advantages_directly: Optional[bool] = field(
        default=False,
        metadata={
            "help": "whether to use mi score as advantages directly and not as reward"
        }
    )

    mi_custom_advantage_position: Optional[str] = field(
        default=None,
        metadata={
            "help": "the position of the custom advantage reward. choose from 'state'."
        }
    )

    mi_importance_sampling: Optional[bool] = field(
        default=False,
        metadata={
            "help": "whether to use importance sampling in the MI diff: full - IS * short"
        }
    )

    ignored_prefix_len: Optional[int] = field(
        default = None,
        metadata={
            "help": "deprecated. use mi_contrasted_area instead"
        }
    )

    #mi_contrasted_area: Optional[str] = field(
    #    default = None,
    #    metadata={
    #        "help": "which part of the sequence should be contrasted. choose from 'first_box_entry', 'first_box_entry_to_end"
    #    }
    #)

    #mi_removed_area: Optional[str] = field(
    #    default=None,
    #    metadata={
    #        "help": "which part of the sequence should be removed, so the contrast is not trivial. choose from 'tool_to_box'"
    #    }
    #)

    mi_contrasted_score: Optional[str] = field(
        default="log_probs",
        metadata={
            "help": "what to contrast. choose from ['log_probs', 'entropy', 'probs']"
        }
    )

    mi_contrasted_score_type: Optional[str] = field(
        default="per_token",
        metadata={
            "help": "whether to contrast the sequence on token or sequence level (i.e. summing over token logps). choose from ['per_token', 'per_sequence']"
        }
    )

    mi_use_info_nce: Optional[bool] = field(
        default=False,
        metadata={
            "help": "whether to use log((full + full) / (full + short)) instead of log(full/short)"
        }
    )

    mi_tool_turn_selection: Optional[str] = field(
        default='last',
        metadata={
            "help": "tool selection strategy. choose from ['random', 'last', 'first']"
        }
    )

    #mi_short_bridge: Optional[str] = field(
    #    default=None,
    #    metadata={
    #        "help": "what to insert after the removed area and before the contrasted area. Choose from 'double_newline', 'double_newline,the,answer,is' or None"
    #    }
    #)

    iou_reward_aggregate: Optional[str] = field(
        default="last",
        metadata={
            "help": "how to aggregate the iou scores per conversation. Only makes a difference for multiple successful tool uses."
                    " choose from 'first', 'last', 'mean'."
        }
    )

    training_mode: Optional[str] = field(
        default=None,
        metadata={
            "help": "run mode. 'singlenode' or 'multinode'"
        }
    )

    vllm_devices: Optional[int] = field(
        default=1,
        metadata={
            "help": "the number of devices for the vllm server. will be mapped to range(vllm_devices) on node 0"
        }
    )

    mi_masked_vision_forward_model: Optional[str] = field(
        default=None,
        metadata={
            "help": "which model to use to calculate the masked vision forward pass for MI reward. "
                    "values can be 'self' or 'reference'"
        }
    )

    mi_full_forward_model: Optional[str] = field(
        default=None,
        metadata={
            "help": "which model to use to calculate the full forward pass for MI reward. "
                    "values can be 'self' or 'reference'"
        }
    )



    scoring_batch_size_multiplier: Optional[int] = field(
        default=1,
        metadata={
            "help": "this value is multiplied to the per_device_train_batch_size for forward passes. "
                    "(per_device_train_batch_size * scoring_batch_size_multiplier) must divide "
                    "(generation_batch_size / num_gpus) evenly"
        }
    )
    # tool_use_rate_threshold = 0.5
    # exploration_threshold = 100
    eps_tool_use_rate_threshold: Optional[float] = field(
        default = None,
        metadata={
            "help": "if the tool use rate is above this threshold for a group, we will increase the exploration counter "
        }
    )

    eps_exploration_threshold: Optional[int] = field(
        default=None,
        metadata={
            "help": "if the exploration counter is above this threshold, we will switch into pruning mode"
        }
    )

    aim_run_hash: Optional[str] = field(
        default = None,
        metadata={
            "help": "if not None, set it to be the aim run hash"
        }
    )

    constant_exploration: Optional[float] = field(
        default = None,
        metadata={
            "help": "the constant exploration reward for the tool use"
        }
    )

    tool_bbox_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "the type of bbox coordinates to accept. choose from 'absolute' (int) or 'relative' (float). If None, both are accepted."
        }
    )

    tool_padding: Optional[float] = field(
        default=None,
        metadata={
            "help": "how much padding to add to the bbox. ratio of the original image size"
        }
    )

    tool_adaptive_padding_threshold: Optional[int] = field(
        default=None,
        metadata={
            "help": "upper bound (in px) for the padding. only relevant if this value is smaller than tool_padding*height or tool_padding*width"
                    "special values: None -> 600 px, -1 -> None"
        }
    )

    iou_target_zero: Optional[float] = field(
        default=None,
        metadata={
            "help": "fraction at beginning of training where iou_target is zero"
        }
    )

    iou_target_one: Optional[float] = field(
        default=None,
        metadata={
            "help": "after which fraction of training the iou_target is one"
        }
    )

    iou_target_increase: Optional[str] = field(
        default=None,
        metadata={
            "help": "how the iou_target should be increased between iou_target_zero and iou_target_one. Supported: linear"
        }
    )

    iou_target_max_value: Optional[float] = field(
        default=None,
        metadata={
            "help": "maximum value of iou_target, default is 1.0. after iou_target_one it will jump to 1"
        }
    )

    dummy_vllm_generation: Optional[str] = field(
        default=None,
        metadata={
            "help": "dummy_vllm_generation: save, load, None. "
                    "save: save the generated multi-turn and tool use to disk. "
                    "load: load the generated multi-turn and tool use from disk. "
                    "None: do not save or load"
                    "can be used for debugging to skip the time-consuming vllm generation"
        }
    )

@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False

#SYSTEM_PROMPT = (
#    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
#    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
#    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
#    "<think> reasoning process here </think><answer> answer here </answer>"
#)


def get_vlm_module(model_name_or_path):
    if "qwen" in model_name_or_path.lower() or "pixelreasoner" in model_name_or_path.lower():
        return Qwen2VLModule
    elif "internvl" in model_name_or_path.lower():
        return InvernVLModule
    else:
        raise ValueError(f"Unsupported model: {model_name_or_path}")

def main(script_args, training_args, model_args):
    if script_args.logging is True:
        if script_args.aim_run_hash is None:
            os.makedirs(training_args.output_dir, exist_ok=True)
            json.dump(asdict(model_args), open(os.path.join(training_args.output_dir, "model_args.json"), "w"))
            json.dump(asdict(script_args), open(os.path.join(training_args.output_dir, "script_args.json"), "w"))
            json.dump(asdict(training_args), open(os.path.join(training_args.output_dir, "training_args.json"), "w"))
        logger = setup_project_logging(log_file=os.path.join(training_args.output_dir, "log.log"))


        logger.info(f"Model args: {model_args}")
        logger.info(f"Script args: {script_args}")
        logger.info(f"Training args: {training_args}")
        aim_callback = [AimCallback(experiment=training_args.run_name)]
        if script_args.aim_run_hash is not None:
            aim_callback[0]._run_hash = script_args.aim_run_hash
    else:
        aim_callback = None
        logger = setup_project_logging(log_file=None)

    if script_args.training_mode == "singlenode":
        logger.info(f"single node training!")
        vllm_address = None
    elif script_args.training_mode == "multinode":
        logger.info(f"multi node training!")
        info = det.get_cluster_info()
        container_addrs = info.container_addrs
        vllm_address = container_addrs[-1]
        #start_vllm_if_rank0(list(range(script_args.vllm_devices)))
    else:
        raise ValueError(f"Unsupported mode: {script_args.training_mode}")

    processor_init_kwargs = {
        "max_pixels": script_args.max_pixels,
        "min_pixels": script_args.min_pixels
    }

    reward_funcs_registry = {
        "accuracy": {"func": accuracy_reward, "type": "per_completion", "name": "accuracy"},
        "format": {"func": format_reward, "type": "per_completion", "name": "format"},
        "format_no_think": {"func": format_reward_only_answer, "type": "per_completion", "name": "format_no_think"},
        "curiosity": {"func": partial(curiosity_reward,
                                      pixel_reasoning_threshold=script_args.pixel_reasoning_threshold),
                      "type": "per_group",
                      "name": "curiosity"},
        "pr_penalty": {"func": partial(pr_penalty_reward,
                                       tool_use_penalty_threshold=script_args.tool_use_penalty_threshold),
                       "type": "per_group",
                       "name": "pr_penalty"},
        "mutual_information": {"func": partial(mutual_information_reward,
                                               gamma=script_args.mutual_information_clip_value,
                                               delta=script_args.mutual_information_threshold,
                                               alpha=script_args.mutual_information_len_exponent,
                                               length_factor_scaling=script_args.mutual_information_len_scaling,
                                               tau=script_args.mutual_information_mean_threshold,
                                               discretize=script_args.mutual_information_discretize,
                                               q=script_args.mutual_information_quantile,
                                               ignored_prefix_len=script_args.ignored_prefix_len,
                                               tanh=script_args.mi_tanh,
                                               length_factor=script_args.mutual_information_length_factor,
                                               select_k=script_args.mutual_information_select_k,
                                               select_k_type=script_args.mutual_information_select_k_type),
                               "type": "per_completion",
                               "name": "mutual_information"},
        "constant_exploration": {"func": constant_exploration,
                                 "type": "per_group",
                                 "name":"constant_exploration"},
        "iou": {"func": partial(iou_reward, aggregate_over_conv=script_args.iou_reward_aggregate), "type": "per_completion", "name": "iou"}
    }

    # todo: refactor this to allow for variable dataset size
    if script_args.iou_target_zero is None or script_args.iou_target_one is None or script_args.iou_target_increase is None:
        iou_target_fn = None
    else:
        iou_target_fn = partial(basic_iou_target_fn, start=script_args.iou_target_zero,
                            end=script_args.iou_target_one, increase = script_args.iou_target_increase,
                                max_value=script_args.iou_target_max_value)

    # Load the VLM module
    vlm_module_cls = get_vlm_module(model_args.model_name_or_path)
    logger.info(f"using vlm module: {vlm_module_cls.__name__}")

    #tool_args = TOOL_CONFIGS[script_args.tool_config if script_args.tool_config is not None else "no_tool"]
    tool_args = TOOL_CONFIGS["no_tool"] if script_args.tool_config is None else [TOOL_CONFIGS[tool_config] for tool_config in script_args.tool_config.split(",")]

    if script_args.tool_config != "no_tool":
        adaptive_padding_threshold = script_args.tool_adaptive_padding_threshold
        if adaptive_padding_threshold is None:
            adaptive_padding_threshold = 600
        if adaptive_padding_threshold < 0:
            adaptive_padding_threshold = None
        tools = [Tool(name=tool_arg["tool_name"],
                     template_name=tool_arg["tool_template"],
                     json_customization=tool_arg["tool_json_customization"],

                     message=Message(tool_arg["tool_message_image_pos"],
                             tool_arg["tool_message_text_message"],
                             tool_arg["tool_message_text_fillers"]),

                     tool_hparams={"max_pixels": script_args.max_pixels,
                                   "min_pixels": script_args.min_pixels,
                                   "bbox_type": script_args.tool_bbox_type,
                                   "padding": (script_args.tool_padding,script_args.tool_padding),
                                   "adaptive_padding_threshold": adaptive_padding_threshold
                                   })
                 for tool_arg in tool_args]
    else:
        tools = None

    #prompt_type = script_args.prompt_type + "_tool" if script_args.multi_turn == "tool" else script_args.prompt_type
    #question_prompt = vlm_module_cls.get_question_template(task_type=script_args.prompt_type)
    prompt_type = script_args.prompt_type if script_args.prompt_type is not None else tool_args[0]["prompt_type"]
    question_prompt = get_question_template(task_type=prompt_type)
    if tools is not None:
        if len(tools) == 1:
            question_prompt = question_prompt.replace("{tool_name}", tools[0].name)
        else:
            for idx, tool in enumerate(tools):
                replacement_string = f"tool_name{idx + 1}"
                tool_name = "crop_image" if tool.name == "crop_image_normalized" else tool.name
                question_prompt = question_prompt.replace(f"{{{replacement_string}}}", tool_name)


    #reward_funcs = script_args.reward_funcs
    #if script_args.prompt_type == "no_think":
    #    reward_funcs.append("no_think")
    # Get reward functions
    reward_funcs = []
    for idx, func in enumerate(script_args.reward_funcs):
        rf = reward_funcs_registry[func]
        if script_args.reward_func_conditionals is not None:
            rf["is_conditional"] = script_args.reward_func_conditionals[idx]
        else:
            rf["is_conditional"] = False
        reward_funcs.append(rf)
    assert len(reward_funcs) == len(script_args.reward_func_weights), f"the number of reward functions {reward_funcs} must be equal to the number of reward function weights {script_args.reward_func_weights}"
    logger.info(f"!!! reward_funcs !!!:\n {reward_funcs} \n!!! reward_funcs !!!")
    if script_args.reward_func_usage is None:
        reward_func_usage = ["both" for _ in range(len(reward_funcs))]
        logger.info(f"as reward_func_usage is None, we assume that it is 'both' for all reward_funcs")
    else:
        reward_func_usage = script_args.reward_func_usage
    assert len(reward_funcs) == len(reward_func_usage), f"the number of reward functions {reward_funcs} must be equal to the number of reward function usages {reward_func_usage}"

    dataset_names = script_args.dataset_name.split(":")
    data_files = script_args.data_file_paths.split(":")
    image_folders = script_args.image_folders.split(":")

    # reward_method was ..._args.reward_method but it was never used anyway
    dataset = prepare_data(dataset_names, data_files, image_folders, question_prompt, None)

    # Split dataset for validation if requested
    splits = {'train': dataset}
    if script_args.val_split_ratio > 0:
        train_val_split = dataset.train_test_split(
            test_size=script_args.val_split_ratio
        )
        splits['train'] = train_val_split['train']
        splits['validation'] = train_val_split['test']

    if training_args.generation_batch_size is None:
        training_args.generation_batch_size = training_args.per_device_train_batch_size

    # Select trainer class based on vlm_trainer argument
    if training_args.use_vllm:
        #trainer_cls = Qwen2VLGRPOVLLMTrainer
        #trainer_cls = VLMGRPOTrainerVLLM
        trainer_cls = UpdatedVLMGRPOTrainerVLLM
        logger.info("Using vllm only works with Qwen 2.5 or 2 as of now!")
    else:
        raise NotImplementedError("generation with hf is deprecated, use vllm only")
    logger.info(f"using trainer: {trainer_cls.__name__}")

    data_subset = script_args.data_subset
    if not data_subset in [None, "none", "None"]:
        data_subset = data_subset.split(":")
        if data_subset[0] == "":
            left = 0
        else:
            left = int(data_subset[0])
        if data_subset[1] == "":
            raise ValueError("Right boundary of dataset can not be inferred yet")
        else:
            right = int(data_subset[1])
        logger.info(f"Training from {left} to {right}")
        data_range = range(left, right)
    else:
        data_range = None

    if script_args.chat_template is not None:
        with open(script_args.chat_template, 'r', encoding='utf-8') as f:
            chat_template = json.load(f)
    else:
        chat_template = None


    mi_mode = {
               "contrasted_score": script_args.mi_contrasted_score,
               "contrasted_score_type": script_args.mi_contrasted_score_type,
               "alternative_action": script_args.mi_alternative_action,
               "answer_type": script_args.mi_answer_type,
               "use_advantages_directly": script_args.mi_use_advantages_directly,
               "custom_advantage_position": script_args.mi_custom_advantage_position,
               "importance_sampling": script_args.mi_importance_sampling,
               "use_info_nce": script_args.mi_use_info_nce,
                "tool_turn_selection": script_args.mi_tool_turn_selection
    }


    tool_handling = {
        "max_tool_uses": script_args.max_tool_uses,
        "strict_tool_extraction": script_args.strict_tool_extraction,
        "finish_after_wrong_tool_call": script_args.finish_after_wrong_tool_call
    }


    if script_args.eps_tool_use_rate_threshold is not None or script_args.eps_exploration_threshold is not None:
        exploration_pruning_schedule = {"exploration_threshold": script_args.eps_exploration_threshold,
                                        "tool_use_rate_threshold": script_args.eps_tool_use_rate_threshold}
    else:
        exploration_pruning_schedule = None

    # Initialize the GRPO trainer
    if training_args.use_vllm:
        if script_args.aim_run_hash is None:
            os.makedirs(os.path.join(training_args.output_dir, "tool_calls", "generated_images"), exist_ok=True)

        trainer = trainer_cls(
            model=model_args.model_name_or_path,
            reward_funcs=reward_funcs,
            reward_func_weights=script_args.reward_func_weights,
            reward_func_usage=reward_func_usage,
            args=training_args,
            vlm_module=vlm_module_cls(),
            train_dataset=splits['train'].select(data_range) if data_range is not None else splits['train'],
            eval_dataset=splits.get('validation') if training_args.eval_strategy != "no" else None,
            peft_config=get_peft_config(model_args),
            freeze_vision_modules=model_args.freeze_vision_modules,
            attn_implementation=model_args.attn_implementation,
            callbacks=aim_callback,
            multi_turn=script_args.multi_turn,
            chat_template=chat_template,
            save_path=training_args.output_dir,
            processor_init_kwargs = processor_init_kwargs,
            tools = tools,
            tool_handling = tool_handling,
            use_global_buffer = script_args.global_buffer,
            vllm_address = vllm_address,
            mi_masked_vision_forward_model = script_args.mi_masked_vision_forward_model,
            mi_full_forward_model = script_args.mi_full_forward_model,
            mi_mode = mi_mode,
            scoring_batch_size_multiplier = script_args.scoring_batch_size_multiplier,
            exploration_pruning_schedule=exploration_pruning_schedule,
            iou_target_fn=iou_target_fn,
            dummy_vllm_generation=script_args.dummy_vllm_generation,
        )
    else:
        trainer = trainer_cls(
            model=model_args.model_name_or_path,
            reward_funcs=reward_funcs,
            args=training_args,
            vlm_module=vlm_module_cls(),
            train_dataset=splits['train'].select(data_range) if data_range is not None else splits['train'],
            eval_dataset=splits.get('validation') if training_args.eval_strategy != "no" else None,
            peft_config=get_peft_config(model_args),
            freeze_vision_modules=model_args.freeze_vision_modules,
            attn_implementation=model_args.attn_implementation,
            max_pixels=script_args.max_pixels,
            min_pixels=script_args.min_pixels,
            callbacks=aim_callback
        )

    if script_args.aim_run_hash is None:
        trainer.train()
    else:
        trainer.train(resume_from_checkpoint=True)

if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
