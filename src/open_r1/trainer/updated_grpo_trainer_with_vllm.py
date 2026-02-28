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
import pickle
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union, Sized
import time
import random

import torch
import torch.utils.data
import torch.distributed as dist
import transformers
import numpy as np
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from scipy.stats import spearmanr

import sys

import deepspeed
from contextlib import nullcontext

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url
from trl import GRPOTrainer
from trl.extras.vllm_client import VLLMClient

from accelerate.utils import is_peft_model, set_seed, gather_object, broadcast_object_list
import PIL.Image

import copy
from torch.utils.data import Sampler
import warnings

from trl.extras.profiling import profiling_decorator, profiling_context

from functools import partial
from torch.utils.data import DataLoader
from transformers.trainer_utils import seed_worker

from open_r1.utils.debug_utils import serialized_size_mb

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

from open_r1.utils.multi_turn_handler import Prompt, Conversations
from open_r1.utils.multi_turn_manager import MultiTurn, pad
from open_r1.utils.masker import Masker
from open_r1.utils.parser import ParsedTokenized, rescale, reduce_img_per_sample, get_processing
from open_r1.utils.tools import Tool
from open_r1.utils.buffer import Buffer

from datasets import Dataset, IterableDataset

from open_r1.vlm_modules.vlm_module import VLMBaseModule
# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, dict[str, Union[str, Callable[[list, list], list[float]]]]]
from open_r1.utils.logger import get_logger

# Get logger for this module
logger = get_logger(__name__)



class RepeatSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset in a structured manner.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        mini_repeat_count (`int`):
            Number of times to repeat each index per batch.
        batch_size (`int`, *optional*, defaults to `1`):
            Number of unique indices per batch.
        repeat_count (`int`, *optional*, defaults to `1`):
            Number of times to repeat the full sampling process.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the dataset.
        seed (`int`, *optional*):
            Random seed for reproducibility (only affects this sampler).

    Example:
    ```python
    >>> sampler = RepeatSampler(["a", "b", "c", "d", "e", "f", "g"], mini_repeat_count=2, batch_size=3, repeat_count=4)
    >>> list(sampler)
    [4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6]
    ```

    ```txt
    mini_repeat_count = 3
          -   -   -
         [0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11,      |
                                                                repeat_count = 2
          0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11, ...] |
          ---------   ---------   ---------   ---------
           ---------   ---------   ---------   ---------
            ---------   ---------   ---------   ---------
                         batch_size = 12
    ```
    """

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.shuffle = shuffle
        self.seed = seed

        if shuffle:
            self.generator = torch.Generator()  # Create a local random generator
            if seed is not None:
                self.generator.manual_seed(seed)

    def __iter__(self):
        if self.shuffle:
            # E.g., [2, 4, 3, 1, 0, 6, 5] (num_samples = 7)
            indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()
        else:
            indexes = list(range(self.num_samples))

        #    [2, 4, 3, 1, 0, 6, 5]
        # -> [[2, 4, 3], [1, 0, 6], [5]]  (batch_size = 3)
        indexes = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]

        #    [[2, 4, 3], [1, 0, 6], [5]]
        # -> [[2, 4, 3], [1, 0, 6]]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return (self.num_samples // self.batch_size) * self.batch_size * self.mini_repeat_count * self.repeat_count

class UpdatedVLMGRPOTrainerVLLM(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        reward_func_weights: list[float],
        reward_func_usage: list[str],
        args: GRPOConfig = None,
        vlm_module: VLMBaseModule = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        freeze_vision_modules: Optional[bool] = False,
        attn_implementation: str = "flash_attention_2",
        torch_dtype: str = "bfloat16",
        multi_turn: str = None,
        chat_template: dict = None,
        save_path: str = None,
        tool_handling: dict = None,
        processor_init_kwargs: dict = None,
        tools: Union[Tool, list[Tool]] = None,
        use_global_buffer: bool = False,
        vllm_address: str = None,
        mi_masked_vision_forward_model: str = None,
        mi_full_forward_model: str = None,
        mi_mode: dict = None,
        scoring_batch_size_multiplier: int = 1,
        exploration_count: int = 0,
        exploration_pruning_schedule: dict = None,
        iou_target_fn: Callable = None,
        dummy_vllm_generation: str = None,
        **kwargs,
    ):

        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")
        
        self.vlm_module = vlm_module
        self.save_path = save_path
        self.tool_handling = tool_handling
        if tools == []:
            self.tools = None
        elif isinstance(tools, Tool):
            self.tools = [tools]
        else:
            self.tools = tools

        self.masker = Masker()

        self.shuffle_dataset = args.shuffle_dataset

        self.mi_masked_vision_forward_model = mi_masked_vision_forward_model
        self.mi_full_forward_model = mi_full_forward_model
        self.mi_mode = mi_mode

        self.scoring_batch_size_multiplier = scoring_batch_size_multiplier

        self.exploration_count = exploration_count
        self.exploration_pruning_schedule = exploration_pruning_schedule

        self.iou_target_fn = iou_target_fn

        self.dummy_vllm_generation = dummy_vllm_generation

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        # FIXME
        # Remember to modify it in the invernvl
        model_init_kwargs["attn_implementation"] = attn_implementation
        if model_init_kwargs.get("torch_dtype") is None:
            model_init_kwargs["torch_dtype"] = torch_dtype
        
        assert isinstance(model, str), "model must be a string in the current implementation"
        model_id = model
        torch_dtype = model_init_kwargs.get("torch_dtype")
        if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
            pass  # torch_dtype is already a torch.dtype or "auto" or None
        elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
            torch_dtype = getattr(torch, torch_dtype)
        else:
            raise ValueError(
                "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
            )
        model_init_kwargs["use_cache"] = (
            False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
        )
            # Disable caching if gradient checkpointing is enabled (not supported)
        model_init_kwargs["use_cache"] = (
            False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
        )
        model_cls = self.vlm_module.get_model_class(model_id, model_init_kwargs)
        try:
            model = model_cls.from_pretrained(model_id, **model_init_kwargs)
        except TypeError:
            model_init_kwargs.pop("use_cache")
        model = model_cls.from_pretrained(model_id, **model_init_kwargs)

        # LoRA
        self.vision_modules_keywords = self.vlm_module.get_vision_modules_keywords()
        if peft_config is not None:
            def find_all_linear_names(model, multimodal_keywords):
                cls = torch.nn.Linear
                lora_module_names = set()
                for name, module in model.named_modules():
                    # LoRA is not applied to the vision modules
                    if any(mm_keyword in name for mm_keyword in multimodal_keywords):
                        continue
                    if isinstance(module, cls):
                        lora_module_names.add(name)
                for m in lora_module_names:  # needed for 16-bit
                    if "embed_tokens" in m:
                        lora_module_names.remove(m)
                return list(lora_module_names)
            target_modules = find_all_linear_names(model, self.vision_modules_keywords)
            peft_config.target_modules = target_modules
            model = get_peft_model(model, peft_config)

        # Freeze vision modules
        if freeze_vision_modules:
            logger.info("Freezing vision modules...")
            for n, p in model.named_parameters():
                if any(keyword in n for keyword in self.vision_modules_keywords):
                    p.requires_grad = False

        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

        # Reference model
        if is_deepspeed_zero3_enabled():
            self.ref_model = model_cls.from_pretrained(model_id, **model_init_kwargs)
        elif peft_config is None:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        # Processing class
        if processing_class is None:
            processing_cls = self.vlm_module.get_processing_class()
            #logger.info(f"processing_cls: {processing_cls}")
            processing_class = processing_cls.from_pretrained(model_id,
                                                              trust_remote_code=model_init_kwargs.get("trust_remote_code", None),
                                                              **processor_init_kwargs)

            if chat_template is not None:
                processing_class.chat_template = chat_template["chat_template"]

            #logger.info(f"processing_class: {processing_class}")
            for processing_keyword in self.vlm_module.get_custom_processing_keywords():
                if processing_keyword in kwargs:
                    setattr(processing_class, processing_keyword, kwargs[processing_keyword])
            if getattr(processing_class, "tokenizer",  None) is not None:
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
            else:
                assert isinstance(processing_class, PreTrainedTokenizerBase), "processing_class must be an instance of PreTrainedTokenizerBase if it has no tokenizer attribute"
                pad_token_id = processing_class.pad_token_id

        self.vlm_module.post_model_init(model, processing_class)
        self.vlm_module.post_model_init(self.ref_model, processing_class)

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        self.reward_funcs_per_completion = [{"reward_func": rf["func"], "is_conditional": rf["is_conditional"]} for rf in self.reward_funcs if rf["type"]=="per_completion"]
        self.reward_funcs_per_group =      [{"reward_func": rf["func"], "is_conditional": rf["is_conditional"]} for rf in self.reward_funcs if rf["type"]=="per_group"]



        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

            # Training arguments
            self.max_prompt_length = args.max_prompt_length

    # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_prompt_length = None
        if args.max_prompt_length is not None:
            warnings.warn("Setting max_prompt_length is currently not supported, it has been set to None")

        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper

        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.min_p = args.min_p
        self.repetition_penalty = args.repetition_penalty

        self.beta = args.beta
        self.epsilon = args.epsilon

        self.use_vllm = args.use_vllm

        self.multi_turn = multi_turn
        if self.multi_turn == "none" or self.multi_turn == "None":
            self.multi_turn = None

        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        # Tracks the number of iterations (forward + backward passes), including those within a gradient accumulation cycle
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates
        #self._buffered_inputs = [None] * args.gradient_accumulation_steps
        self.buffer = Buffer(max_size=args.per_device_train_batch_size * args.gradient_accumulation_steps,
                             padding_id=processing_class.pad_token_id,
                             cpu_buffer=True,
                             sample_with_replacement=True,
                             flush_after_episode=True,
                             use_global_buffer=use_global_buffer)

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        self.reward_func_weights = torch.tensor(reward_func_weights, dtype=torch.float16,
                                                device=self.accelerator.device)
        self.reward_funcs_for_reward = torch.zeros_like(self.reward_func_weights)
        self.reward_funcs_for_reward[[idx for idx, usage in enumerate(reward_func_usage) if
                                        usage in ["both", "reward"]]] = 1
        self.reward_funcs_for_sampling_weights = torch.zeros_like(self.reward_func_weights)
        self.reward_funcs_for_sampling_weights[[idx for idx, usage in enumerate(reward_func_usage) if
                                      usage in ["both", "sampling_weights"]]] = 1

        # Check if the per_device_train/eval_batch_size * num processes can be divided by the number of generations
        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * num_processes
        possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
        #if self.num_generations not in possible_values:
        #    raise ValueError(
        #        f"The global train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly "
        #        f"divisible by the number of generations per prompt ({self.num_generations}). Given the current train "
        #        f"batch size, the valid values for the number of generations are: {possible_values}."
        #    )
        if self.args.eval_strategy != "no":
            global_batch_size = args.per_device_eval_batch_size * num_processes
            possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"The global eval batch size ({num_processes} x {args.per_device_eval_batch_size}) must be evenly "
                    f"divisible by the number of generations per prompt ({self.num_generations}). Given the current "
                    f"eval batch size, the valid values for the number of generations are: {possible_values}."
                )

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)


        if self.use_vllm:
            # vLLM specific sampling arguments
            self.guided_decoding_regex = args.vllm_guided_decoding_regex

            self._last_loaded_step = -1  # tag to avoid useless loading during grad accumulation
            if self.dummy_vllm_generation != "load":
                if self.accelerator.is_main_process:
                    self.vllm_client = VLLMClient(
                        vllm_address if vllm_address is not None else args.vllm_server_host,
                        args.vllm_server_port,
                        connection_timeout=args.vllm_server_timeout
                    )
                # When using vLLM, the main process is responsible for loading the model weights. This can cause process
                # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
                # synchronize all processes after vLLM has been fully initialized.
                self.accelerator.wait_for_everyone()
        else:
            self.generation_config = GenerationConfig(
                max_new_tokens=self.max_completion_length,
                do_sample=True,
                pad_token_id=processing_class.pad_token_id,
                bos_token_id=processing_class.bos_token_id,
                eos_token_id=processing_class.eos_token_id,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                min_p=self.min_p,
                repetition_penalty=self.repetition_penalty,
                cache_implementation=args.cache_implementation,
            )

            if hasattr(self.vlm_module, "get_eos_token_id"):  # For InternVL
                self.generation_config.eos_token_id = self.vlm_module.get_eos_token_id(processing_class)
                logger.info(222, self.vlm_module.get_eos_token_id(processing_class))


        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        #for i, reward_func in enumerate(self.reward_funcs):
        #    if isinstance(reward_func, PreTrainedModel):
        #        self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if isinstance(train_dataset, Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")
        logger.info(f"In get_train_dataloader: dataloader_params bs: {self._train_batch_size * self.args.steps_per_generation}")
        dataloader_params = {
            # args.per_device_train_batch_size used to be self._train_batch_size
            "batch_size": self._train_batch_size * self.args.steps_per_generation,  # < this is the change
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = partial(
                seed_worker, num_workers=self.args.dataloader_num_workers, rank=self.args.process_index
            )

            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def _get_train_sampler(self, dataset: Optional[Dataset] = None) -> Sampler:
        # Returns a sampler that
        # 1. ensures each prompt is repeated across multiple processes. This guarantees that identical prompts are
        #    distributed to different GPUs, allowing rewards to be computed and normalized correctly within each prompt
        #    group. Using the same seed across processes ensures consistent prompt assignment, preventing discrepancies
        #    in group formation.
        # 2. repeats the batch multiple times to allow reusing generations across multiple updates. Refer to
        #    _prepare_inputs to see how the generations are stored and reused.

        # In the following figure, the values are the prompt indices. The first row shows the first sampled batch, the
        # second row shows the second sampled batch, and so on.
        #
        #                                      |   GPU 0  |   GPU 1  |
        #
        #                 global_step   step    <-───>  num_generations=2
        #                                       <-───────> per_device_train_batch_size=3
        #  grad_accum    ▲  ▲  0          0     0   0   1   1   2   2   <- Generate for the first `steps_per_generation` (prompts 0 to 11); store the completions; use the first slice to compute the loss
        #     =2         ▼  |  0          1     3   3   4   4   5   5   <- Take the stored generations and use the second slice to compute the loss
        #                   |
        #                   |  1          2     6   6   7   7   8   8   <- Take the stored generations and use the third slice to compute the loss
        #  steps_per_gen=4  ▼  1          3     9   9  10  10  11  11   <- Take the stored generations and use the fourth slice to compute the loss
        #
        #                      2          4    12  12  13  13  14  14   <- Generate for the second `steps_per_generation` (prompts 12 to 23); store the completions; use the first slice to compute the loss
        #                      2          5    15  15  16  16  17  17   <- Take the stored generations and use the second slice to compute the loss
        #                                          ...
        if dataset is None:
            dataset = self.train_dataset
        logger.info(f"in get_train_sampler: \n"
                    f"mini_repeat_count: {self.num_generations}, \n"
                    f"batch_size: {self.args.generation_batch_size // self.num_generations} \n"
                    f"repeat_count: {self.num_iterations * self.args.steps_per_generation} \n"
                    f"shuffle: {self.shuffle_dataset} \n"
                    f"seed: {self.args.seed}")
        return RepeatSampler(
            data_source=dataset,
            mini_repeat_count=self.num_generations,
            batch_size=self.args.generation_batch_size // self.num_generations,
            repeat_count=self.num_iterations * self.args.steps_per_generation,
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )

    def _prepare_inputs(
        self, generation_batch: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        # Prepares inputs for model training/evaluation by managing completion generation and batch handling.
        # During training:
        #   - Receives the local generation batch (Per-GPU batch size × steps per generation)
        #     from the modified training dataloader instead of the standard local batch
        #   - Generates completions once for the entire generation batch and splits it into batches of size
        #     `per_device_train_batch_size`
        #   - Buffers these completions and returns the appropriate slice for the current accumulation step
        #   - Optimizes by regenerating completions only periodically (every steps_per_generation * num_iterations)
        # Returns a single local batch in both cases.

        #self.monitor_gpu_usage("at the top of prepare_inputs")
        generate_every = self.args.steps_per_generation * self.num_iterations
        logger.info(f"generate_every: {generate_every}")
        logger.info(f"self._step: {self._step}")
        if self._step % generate_every == 0 or self.buffer.buffer_space_taken() == 0:
            # self._buffered_inputs=None can occur when resuming from a checkpoint
            generation_batch = self._generate_and_score_completions(generation_batch, self.model)
            logger.info(f"generation_batch after generate: {generation_batch['prompt_ids'].shape}")
            self.buffer.add(generation_batch, padding_side="right")

        inputs = self.buffer.get(batch_size=self.args.per_device_train_batch_size, deterministic=False,
                                 device=self.accelerator.device if self.buffer.cpu_buffer else None,
                                 padding_side="right")
        #inputs = self._buffered_inputs[self._step % self.args.steps_per_generation]
        self._step += 1
        #self.monitor_gpu_usage("prepare inputs: after sampled from buffer")
        return inputs

    def _enable_gradient_checkpointing(self, model: PreTrainedModel, args: GRPOConfig) -> PreTrainedModel:
        """Enables gradient checkpointing for the model."""
        # Ensure use_cache is disabled
        model.config.use_cache = False

        # Enable gradient checkpointing on the base model for PEFT
        if is_peft_model(model):
            model.base_model.gradient_checkpointing_enable()
        # Enable gradient checkpointing for non-PEFT models
        else:
            try:
                model.gradient_checkpointing_enable()
            except:
                # For InternVL; these operations are copied from the original training script of InternVL
                model.language_model.config.use_cache = False
                model.vision_model.gradient_checkpointing = True
                model.vision_model.encoder.gradient_checkpointing = True
                model.language_model._set_gradient_checkpointing()
                # This line is necessary, otherwise the `model.gradient_checkpointing_enable()` will be executed during the training process, leading to an error since InternVL does not support this operation.
                args.gradient_checkpointing = False

        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        use_reentrant = (
            "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]
        )

        if use_reentrant:
            model.enable_input_require_grads()

        return model
    
    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    def _get_per_token_logps(self, model, input_ids, attention_mask, image_grid_thw, pixel_values, num_images,
                                 batch_size, disable_dropout, return_entropies) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        # the returned entropies have the same dimensionality as the returned logps, seq_len(input_ids) - 1. The code could be extended to allow for
        # calculation of the uncertainty in generating the next token (whose true label we don't know).
        # Then the entropies had the same dimensionality as input_ids
        if disable_dropout:
            model.eval()
        max_len = input_ids.size(1)
        #logger.info(f"max_len: {max_len}")
        logp_target_len = max_len - 1
        batch_size = batch_size or input_ids.size(0)  # Chunk inputs into smaller batches to reduce memory peak
        logger.info(f"_get_per_token_logps: batch size: {batch_size}")
        logger.info(f"_get_per_token_logps: inputs_ids_size: {input_ids.size(0)}")
        all_logps = []
        all_entropies = []
        for start in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[start: start + batch_size]
            attention_mask_batch = attention_mask[start: start + batch_size]

            max_len_batch = int(attention_mask_batch.sum(dim=1).max().item())
            #logger.info(f"max_len_batch: {max_len_batch}")
            input_ids_batch = input_ids_batch[:, :max_len_batch]
            attention_mask_batch = attention_mask_batch[:, :max_len_batch]

            # Build model inputs - check if the model supports logits_to_keep (some models and VLMs don't)
            model_inputs = {"input_ids": input_ids_batch, "attention_mask": attention_mask_batch}

            if image_grid_thw is not None and pixel_values is not None:
                rows_per_image = image_grid_thw.prod(dim=-1)
                rows_per_sample = torch.split(rows_per_image, num_images)
                rows_per_sample = torch.stack([s.sum() for s in rows_per_sample])
                cum_rows = torch.cat([torch.tensor([0], device=rows_per_sample.device), rows_per_sample.cumsum(0)])
                row_start, row_end = cum_rows[start].item(), cum_rows[start + batch_size].item()
                model_inputs["pixel_values"] = pixel_values[row_start:row_end]
                cum_imgs = torch.tensor([0] + num_images).cumsum(0)
                img_start, img_end = cum_imgs[start], cum_imgs[start + batch_size]
                model_inputs["image_grid_thw"] = image_grid_thw[img_start:img_end]
            elif pixel_values is not None:
                model_inputs["pixel_values"] = pixel_values[start: start + batch_size]

            logger.info(f"_get_per_token_logps: directly before forward pass")
            logits = model(**model_inputs).logits
            logger.info(f"_get_per_token_logps: directly after forward pass")

            logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
            input_ids_batch = input_ids_batch[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
            # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
            per_token_logps = []
            per_token_entropies = []
            for logits_row, input_ids_row in zip(logits, input_ids_batch):
                log_probs = logits_row.log_softmax(dim=-1)
                token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
                per_token_logps.append(token_log_prob)

                if return_entropies:
                    per_token_entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
                    per_token_entropies.append(per_token_entropy)

            chunk_logps = torch.stack(per_token_logps)
            if return_entropies:
                chunk_entropies = torch.stack(per_token_entropies)
            #logger.info(f"chunk_logps before re-pad: {chunk_logps.size(1)}")
            if chunk_logps.size(1) < logp_target_len:
                pad_len = logp_target_len - chunk_logps.size(1)
                chunk_logps = torch.nn.functional.pad(chunk_logps, (0, pad_len), value=0.0)
                if return_entropies:
                    chunk_entropies = torch.nn.functional.pad(chunk_entropies, (0, pad_len), value=0.0)
            #logger.info(f"chunk_logps after re-pad: {chunk_logps.size(1)}")
            all_logps.append(chunk_logps)
            if return_entropies:
                all_entropies.append(chunk_entropies)


        if disable_dropout:
            model.train()

        if return_entropies:
            return torch.cat(all_logps, dim=0), torch.cat(all_entropies, dim=0)
        else:
            return torch.cat(all_logps, dim=0), None

    @profiling_decorator
    def _move_model_to_vllm(self):
        # For DeepSpeed ZeRO-3, we need to gather all parameters before operations
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        gather_if_zero3 = deepspeed.zero.GatheredParameters if zero_stage_3 else nullcontext

        if is_peft_model(self.model):
            # With PEFT and DeepSpeed ZeRO Stage 3, we must gather the full model at once before merging, as merging
            # adapters in a sharded manner is not supported.
            with gather_if_zero3(list(self.model.parameters())):
                self.model.merge_adapter()

                # Update vLLM weights while parameters are gathered
                for name, param in self.model.named_parameters():
                    # When using PEFT, we need to recover the original parameter name and discard some parameters
                    name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                    if self.model.prefix in name:
                        continue
                    # When module to save, remove its prefix and discard the original module
                    if "original_module" in name:
                        continue
                    name = name.replace("modules_to_save.default.", "")

                    if self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(name, param.data)

                # Unmerge adapters while parameters are still gathered
                self.model.unmerge_adapter()
                # Parameters will automatically be repartitioned when exiting the context
        else:
            # For non-PEFT models, simply gather and update each parameter individually.
            for name, param in self.model.named_parameters():
                with gather_if_zero3([param]):
                    if self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(name, param.data)

        # Reset cache on main process
        if self.accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()

    def _get_key_from_inputs(self, x, key):
        ele = x.get(key, None)
        assert ele is not None, f"The key {key} is not found in the input"
        if isinstance(ele, list):
            return [e for e in ele]
        else:
            return [ele]

    def _generate_and_score_completions(self, inputs: dict[str, Union[torch.Tensor, Any]], model) -> dict[str, Union[torch.Tensor, Any]]:
        # only used in image and text rethink
        format_prompt = "As before, first output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."

        logger.info(f"input keys: {inputs[0].keys()}")

        device = self.accelerator.device
        rank = dist.get_rank() if dist.is_initialized() else 0

        history = inputs.copy()

        #logger.info(f"inputs: {inputs}")

        prompts = [x["prompt"] for x in inputs]

        logger.info(f"in _generate_and_score_completions: number of prompts: {len(prompts)}")

        #added_tool_text = " Please use the tool exactly once during your reasoning."

        tool_dicts = [tool.get_tool_dict() for tool in self.tools] if self.tools is not None else None
        logger.info(f"in _generate_and_score_completions: tools: {tool_dicts}")


        # Handle both pre-loaded images and image paths
        image_paths = []
        for x in inputs:
            if "image_path" in x and x["image_path"] is not None:
                for p in self._get_key_from_inputs(x, "image_path"):
                    image_paths.append(p)
                assert len(self._get_key_from_inputs(x, "image_path")) == 1, f"Example {x} contains more than one image which is not supported atm"
            else:
                raise ValueError(f"sample {x} does not contain any image path")

        # First, have main process load weights if needed
        if self.state.global_step != self._last_loaded_step:
            t0 = time.time()
            if self.dummy_vllm_generation != "load":
                self._move_model_to_vllm()
            t1 = time.time()
            self._metrics["send_params_time"].append(t1 - t0)
            self._last_loaded_step = self.state.global_step

        # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
        all_histories = gather_object(history)
        all_image_paths = gather_object(image_paths)

        if self.dummy_vllm_generation != "load":

            if self.accelerator.is_main_process:
                t2 = time.time()
                # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and generate
                # num_generations outputs for each one. This is faster than generating outputs for each duplicate
                # prompt individually.

                ordered_set_of_image_paths = all_image_paths[:: self.num_generations]

                no_conversations = len(all_histories)

                multi_turn_manager = MultiTurn(no_conversations,
                                       processor=self.processing_class,
                                       tools=self.tools)

                multi_turn_manager.add_initial_user_prompt([h["prompt"][0] for h in all_histories], all_image_paths)
                #input_tokens = multi_turn.get_sequences(type="id")
                full_token_seq = multi_turn_manager.get_sequences(type="id", add_assistant_start=True, full_image_pad=False)[:: self.num_generations]
                input_text = multi_turn_manager.get_sequences(type="text", add_assistant_start=True, full_image_pad=False)[:: self.num_generations]
                logger.info(f"input text before generation: {input_text}")

                all_multimodal_token_inputs = [{"prompt_token_ids": full_token_seq[i],
                                                "image_path": ordered_set_of_image_paths[i]}
                       for i in range(len(ordered_set_of_image_paths))]

                logger.info(f"all_multimodal_token_inputs: {all_multimodal_token_inputs}")

                conv_round = 0
                max_conv_rounds = 5 # just for safety that we don't get stuck in endless loop. max tool calls should prevent it

                max_generation_attempts = 5

                while not all(multi_turn_manager.is_finished) and conv_round < max_conv_rounds:
                    #logger.info(f"state of mt_manager before generation {conv_round}: {multi_turn_manager.all_multi_turn}")
                    t_vllm = time.time()
                    with profiling_context(self, "vLLM.generate"):
                        vllm_generation_has_worked = False
                        attempts = 0
                        while (not vllm_generation_has_worked) and (attempts <= max_generation_attempts):
                            try:
                                completion_ids_token_based = self.vllm_client.generate_from_multimodal_token_input(
                                    prompts=all_multimodal_token_inputs,
                                    n=self.num_generations if conv_round == 0 else 1,
                                    repetition_penalty=self.repetition_penalty,
                                    temperature=self.temperature,
                                    top_p=self.top_p,
                                    top_k=-1 if self.top_k is None else self.top_k,
                                    min_p=0.0 if self.min_p is None else self.min_p,
                                    max_tokens=self.max_completion_length,
                                    guided_decoding_regex=self.guided_decoding_regex,
                                    stop_token_ids=[151643, 151658]
                                )
                                vllm_generation_has_worked = True
                            except Exception as e:
                                logger.info(f"Generation {attempts} failed with exception:", e)
                                attempts += 1
                                try:
                                    self.vllm_client.check_server(total_timeout=10.0)
                                    logger.info(f"vLLM server is up!")
                                except ConnectionError:
                                    raise ConnectionError("vLLM Server is down, aborting training.")
                    t_vllm_end = time.time()
                    self._metrics["vllm_generate_time"].append(t_vllm_end - t_vllm)
                    #logger.info(f"completion_ids: {completion_ids}")
                    #logger.info(f"completions from conv round {conv_round}: {completions}")
                    completions_token_based = self.processing_class.batch_decode(completion_ids_token_based, skip_special_tokens=True)
                    logger.info(f"completions from conv round {conv_round}: {completions_token_based}")
                    logger.info(f"completion_ids from conv round {conv_round}: {completion_ids_token_based}")

                    truncated_generations = sum(1 for comp in completion_ids_token_based if len(comp) >= self.max_completion_length)
                    logger.info(f"from {len(completion_ids_token_based)} generations, {truncated_generations} are longer than max_completion_length={self.max_completion_length}")

                    completion_idx = 0

                    multi_turn_manager.add_model_reply(completion_ids_token_based, mapping=multi_turn_manager.get_ids(is_finished=False))

                    if self.multi_turn is None:
                        #conversations.is_finished = [True] * no_conversations
                        multi_turn_manager.is_finished = [True for _ in range(no_conversations)]
                    elif self.multi_turn == "text":
                        text_rethink = "Are you sure? Think again. "
                        if conv_round == 0:
                            multi_turn_manager.add_user_message(texts=[text_rethink + format_prompt for _ in range(no_conversations)])
                        else:
                            #conversations.is_finished = [True] * no_conversations
                            multi_turn_manager.is_finished = [True for _ in range(no_conversations)]
                    elif self.multi_turn == "image":
                        image_rethink = "Are you sure? Look at the image again. "
                        if conv_round == 0:
                            multi_turn_manager.add_user_message(prompts=[{
                                                                'role': 'user',
                                                                'content': [
                                                                    {'type': 'image', 'text': None},
                                                                    {'type': 'text', 'text': image_rethink + format_prompt}
                                                                ]
                                                                } for _ in range(no_conversations)],
                                                        image_paths=multi_turn_manager.get_image_paths())
                        else:
                            #conversations.is_finished = [True] * no_conversations
                            multi_turn_manager.is_finished = [True for _ in range(no_conversations)]
                    elif self.multi_turn == "tool":

                        multi_turn_manager.handle_tool_call(save_path=os.path.join(self.save_path, "tool_calls"),
                                                       step=self.state.global_step, strict_extraction=self.tool_handling["strict_tool_extraction"],
                                                            finish_after_wrong_tool_call=self.tool_handling["finish_after_wrong_tool_call"])

                        for idx in range(no_conversations):
                            if self.tool_handling["max_tool_uses"] is not None and multi_turn_manager.get_no_tool_calls(idx) > self.tool_handling["max_tool_uses"]:
                                multi_turn_manager.is_finished[idx] = True
                                # in this case, the last tool execution is at the end of the sequence.
                                # We don't generate further and will anyway mask it for the backward pass.
                                # That's why it is more efficient to remove it from the sequence. (saves up to 5k visual tokens)
                                
                                # this if is only for idempotency reasons.
                                # by reducing the number of correct tool calls post-hoc, it should always be true
                                # because in the next round we won't go into the parent if
                                if multi_turn_manager.all_multi_turn[idx][-1].role == "user":
                                    multi_turn_manager.all_multi_turn[idx] = multi_turn_manager.all_multi_turn[idx][:-1]
                                    multi_turn_manager.all_multi_turn[idx][-1].successful_tool_call = False
                    else:
                        raise ValueError(f"Invalid value for multi_turn: {self.multi_turn}. Choose from None, 'text', 'image', or 'tool'")


                    full_token_seq = multi_turn_manager.get_sequences(type="id", add_assistant_start=True, full_image_pad=False, ignore_finished=True)
                    logger.info(f"full_token_seq: {full_token_seq}")
                    input_text = multi_turn_manager.get_sequences(type="text", add_assistant_start=True, full_image_pad=False, ignore_finished=True)
                    logger.info(f"input_text for conv_round {conv_round}: {input_text}")
                    image_paths = multi_turn_manager.get_image_paths(ignore_finished=True, flatten=False)
                    logger.info(f"image_paths: {image_paths}")

                    all_multimodal_token_inputs = [{"prompt_token_ids": full_token_seq[i],
                                                    "image_path": image_paths[i]}
                                                   for i in range(len(image_paths))]

                    conv_round += 1
                    logger.info(f"all_multimodal_token_inputs for conv_round {conv_round}: {all_multimodal_token_inputs}")
                    #logger.info(f"state of mt_manager for conv_round {conv_round}: {multi_turn_manager.all_multi_turn}")

                all_multi_turn = multi_turn_manager.all_multi_turn

                overall_tools_used = [multi_turn_manager.get_no_tool_calls(idx) for idx in range(no_conversations)]

                attempted_tool_uses = [multi_turn_manager.get_no_tool_calls(idx, type="attempt") for idx in range(no_conversations)]

                logger.info(f"overall_tools_used: {overall_tools_used}")
                tool_use_array = np.array(overall_tools_used, dtype=np.float16)
                tool_attempt_array = np.array(attempted_tool_uses, dtype=np.float16)

                tool_per_group = tool_use_array.reshape(-1, self.num_generations).mean(axis=1)
                self._metrics["tool_per_group_std"].append(float(np.std(tool_per_group)))

                self._metrics["mean_tool_use"].append(float(np.mean(tool_use_array)))

                attempts_mask = tool_attempt_array != 0

                if np.any(attempts_mask):
                    self._metrics["tool_success_rate"].append(float(np.mean(tool_use_array[attempts_mask] / tool_attempt_array[attempts_mask])))
                else:
                    self._metrics["tool_success_rate"].append(-1.0)

                t3 = time.time()
                self._metrics["generate_time"].append(t3 - t2)
            else:
                all_multi_turn = [None] * len(all_histories)

                overall_tools_used = [None] * len(all_histories)

            overall_tools_used = broadcast_object_list(overall_tools_used, from_process=0)

            def timed(msg, fn):
                t0 = time.time()
                out = fn()
                dt = time.time() - t0
                print(f"[rank {dist.get_rank()}] {msg}: {dt:.3f}s")
                return out

            logger.info(f"len all_multi_turn before split: {len(all_multi_turn)}")
            #make into list of num_processes lists such that each of them can be sent to a different process by scatter
            all_multi_turn = [all_multi_turn[i:i + len(prompts)] for i in range(0, len(all_multi_turn), len(prompts))]
            logger.info(f"len all_multi_turn after split: {len(all_multi_turn)}")

            size_mb_all = serialized_size_mb(all_multi_turn[dist.get_rank()])
            logger.info(f"broadcast payload size all: {size_mb_all} MB")

            timed("barrier-before-broadcast", lambda: dist.barrier())
            timed("broadcast_object_list", lambda: dist.scatter_object_list(scatter_object_output_list=all_multi_turn,
                                                                                             scatter_object_input_list=all_multi_turn,
                                                                                             src=0))
            timed("barrier-after-broadcast", lambda: dist.barrier())


            logger.info(f"len all_multi_turn after broadcast: {len(all_multi_turn)}")
            all_multi_turn = all_multi_turn[0]
            logger.info(f"len all_multi_turn after shorten: {len(all_multi_turn)}")

        if self.dummy_vllm_generation is not None:
            prefix = ("/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/focusreason/dummy_vllm_generation/"
                      "Qwen_2p5_7B_pr_data_cold_absolute_pixels_500_5k_image_mi_iou_Uncond_infonce_1_epoch_30_100_17p5_first_40")
            multi_turn_path = os.path.join(prefix, f'step_{self.state.global_step}_gpu_{self.accelerator.process_index}.pickle')
            tool_use_path = os.path.join(prefix, f'step_{self.state.global_step}_global_tool_use.pickle')
            if self.dummy_vllm_generation == "save":
                with open(multi_turn_path, 'wb') as handle:
                    pickle.dump(all_multi_turn, handle, protocol=pickle.HIGHEST_PROTOCOL)
                with open(tool_use_path, 'wb') as handle:
                    pickle.dump(overall_tools_used, handle, protocol=pickle.HIGHEST_PROTOCOL)

            elif self.dummy_vllm_generation == "load":
                try:
                    with open(multi_turn_path, 'rb') as handle:
                        all_multi_turn = pickle.load(handle)
                    with open(tool_use_path, 'rb') as handle:
                        overall_tools_used = pickle.load(handle)
                except FileNotFoundError:
                    logger.info(f"no vllm generations found anymore for step {self.state.global_step}, terminating")
                    sys.exit(0)
            else:
                raise ValueError(f"Invalid dummy_vllm_generation value: {self.dummy_vllm_generation}")


        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )

        #all_multi_turn = all_multi_turn[process_slice]
        multi_turn_manager = MultiTurn(batch_size=len(all_multi_turn),
                                       processor=self.processing_class,
                                       tools=self.tools)
        multi_turn_manager.all_multi_turn = all_multi_turn

        #logger.info(f"multi_turn_manager after split: {multi_turn_manager.all_multi_turn}")

        all_image_paths = multi_turn_manager.get_image_paths(flatten=False)

        images = []
        image_paths_flatten = [item for sublist in all_image_paths for item in sublist]
        images_per_sample = [len(sublist) for sublist in all_image_paths]

        #logger.info(f"after image path")

        prompt_ids_new, original_positions = multi_turn_manager.get_sequences(type="id", add_assistant_start=False, full_image_pad=True, return_positions=True)
        logger.info(f"prompt_ids_new_lens before pad: {[len(s) for s in prompt_ids_new]}")
        logger.info(f"prompt_ids_new before pad: {prompt_ids_new}")
        prompt_mask_new = [[1 for _ in seq] for seq in prompt_ids_new]
        logger.info(f"prompt_mask_new before pad: {[len(s) for s in prompt_mask_new]}")
        prompt_ids_new = pad(prompt_ids_new, padding_side='right', padding_value=self.processing_class.pad_token_id)
        logger.info(f"prompt_ids_new after pad: {[len(s) for s in prompt_ids_new]}")
        prompt_mask_new = pad(prompt_mask_new, padding_side='right', padding_value=0)
        logger.info(f"prompt_mask_new after pad: {[len(s) for s in prompt_mask_new]}")

        pixel_values_new = multi_turn_manager.get_multimodal(type="pixel_values")

        image_grid_thw_new = multi_turn_manager.get_multimodal(type="image_grid_thw")

        prompt_inputs_new = {"input_ids": torch.tensor(prompt_ids_new, dtype=torch.long, device=device),
                             "attention_mask": torch.tensor(prompt_mask_new, dtype=torch.long, device=device),
                             "image_grid_thw": torch.tensor(image_grid_thw_new,  dtype=torch.long, device=device),
                             "pixel_values": torch.tensor(pixel_values_new, dtype=torch.bfloat16, device=device)}
        prompt_inputs = super()._prepare_inputs(prompt_inputs_new)

        print(f"prompt inputs: {prompt_inputs}")

        prompt_ids = prompt_inputs["input_ids"]
        prompt_mask = prompt_inputs["attention_mask"]


        logger.info("before everything_except_model_generation mask")


        non_generation_mask_new = multi_turn_manager.get_mask(type="everything_except_model_generation")
        logger.info(f"non_generation mask: {[len(s) for s in non_generation_mask_new]}")
        non_generation_mask_new = pad(non_generation_mask_new, padding_side="right", padding_value=0)
        logger.info(f"non_generation mask after pad: {[len(s) for s in non_generation_mask_new]}")

        non_generation_mask = torch.tensor(non_generation_mask_new, device=device, dtype=torch.int8)


        if (non_generation_mask.sum(dim=1) == 0).any():
            logger.info(f"non_generation_mask contains row of zeroes!")

        logger.info("after everything_except_model_generation mask")

        if "mutual_information" in [rf["name"] for rf in self.reward_funcs]:
            use_mi_reward = True

            if self.mi_mode is None:
                raise ValueError(f"To use MI you need to specify what and how to mask!")

            if self.mi_masked_vision_forward_model == "self":
                mi_masked_vision_forward_model = model
            elif self.mi_masked_vision_forward_model == "reference":
                mi_masked_vision_forward_model = self.ref_model
            else:
                raise ValueError("mi_masked_vision_forward_model must be 'self' or 'reference'")
        else:
            use_mi_reward = False
            model_for_masked_vision_forward = None

        # Get the multimodal inputs
        multimodal_keywords = self.vlm_module.get_custom_multimodal_keywords()
        multimodal_inputs = {k: prompt_inputs[k] if k in prompt_inputs else None for k in multimodal_keywords}
        logger.info(f"for scoring: prompt ids: {prompt_ids.shape}")
        logger.info(f"for scoring: bs: {self.args.per_device_train_batch_size}")
        override_advantages = None
        t4 = time.time()
        with ((torch.no_grad())):
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip its
            # computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1:
                #logger.info("before old_per_token_logps calculation")
                logger.info(f"in old_per_token_logps: images_per_sample:  {images_per_sample}")
                old_per_token_logps, old_per_token_entropies = self._get_per_token_logps(model, prompt_ids, prompt_mask,
                                              image_grid_thw = multimodal_inputs["image_grid_thw"],
                                              pixel_values = multimodal_inputs["pixel_values"],
                                              num_images = images_per_sample,
                                              batch_size = self.args.per_device_train_batch_size * self.scoring_batch_size_multiplier,
                                              disable_dropout=True, return_entropies=True)

                if torch.isnan(old_per_token_logps).any():
                    logger.info(f"old_per_token_logps contains nan! {old_per_token_logps}")
                    logger.info(f"prompt_ids: {prompt_ids}")
                    logger.info(f"prompt_mask: {prompt_mask}")
                    logger.info(f"images: {images}")
                    sys.exit(1)
                logger.info(f"after old per token logp calculation")

                #logger.info("after old_per_token_logps calculation"
            else:
                #logger.info(f"set old_per_token_logps to None")
                old_per_token_logps = None

            if self.beta == 0.0:
                logger.info(f"set ref_per_token_logps to None")
                ref_per_token_logps = None
            elif self.ref_model is not None:
                logger.info("before ref_per_token_logps calculation with frozen model")
                #logger.info(f"rank: {rank}: directly before ref per token logps calculation")

                ref_per_token_logps, ref_per_token_entropies = self._get_per_token_logps(self.ref_model, prompt_ids, prompt_mask,
                                              image_grid_thw=multimodal_inputs["image_grid_thw"],
                                              pixel_values=multimodal_inputs["pixel_values"],
                                              num_images=images_per_sample,
                                              batch_size=self.args.per_device_train_batch_size * self.scoring_batch_size_multiplier,
                                              disable_dropout=True, return_entropies=True)


                logger.info("after ref_per_token_logps calculation")
            else:
                #logger.info("before unwrap model!")
                logger.info("before ref_per_token_logps calculation with live model")
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps, ref_per_token_entropies = self._get_per_token_logps(model, prompt_ids, prompt_mask,
                                                                        image_grid_thw=multimodal_inputs[
                                                                            "image_grid_thw"],
                                                                        pixel_values=multimodal_inputs["pixel_values"],
                                                                        num_images=images_per_sample,
                                                                        batch_size=self.args.per_device_train_batch_size * self.scoring_batch_size_multiplier,
                                                                        disable_dropout=True, return_entropies=True)
                #logger.info("after ref_per_token_logps calculation with actual model")

            contrasted_area = None
            diff = None
            contrast_diff_list = None
            negatives_list = None
            negative_bboxes = None
            positives_list = None
            stop_using_mi_rewards = False

            if use_mi_reward:
                negatives_list = [[] for _ in range(self.mi_mode["num_negatives"])]
                negative_bboxes = []
                positives_list = []
                if self.iou_target_fn is not None:
                    iou_target = self.iou_target_fn(self.state.global_step / self.state.max_steps)
                    self._metrics["iou_target"].append(iou_target)
                    if iou_target >= 1:
                        stop_using_mi_rewards = True
                if not stop_using_mi_rewards:
                    # Collect negatives
                    # negatives_list has shape (num_negatives, num_rollouts)
                    for negative_idx in range(self.mi_mode["num_negatives"]):
                        logger.info(f"before get alternative seqs")
                        input_ids_mi, image_positions_mi, reduced_images_per_sample, considered_seqs, new_bboxes = multi_turn_manager.get_alternative_sequences(
                            alternative_action=self.mi_mode["alternative_action"],
                            answer=self.mi_mode["answer_type"],
                            ground_truth=[random.choice(inp["solution"]) for inp in inputs],
                            save_path=os.path.join(self.save_path, "tool_calls"),
                            step=self.state.global_step,
                            iou_target=iou_target,
                            tool_turn_selection=self.mi_mode["tool_turn_selection"],
                            negative_bboxes=negative_bboxes,
                            same_digit_number=self.mi_mode["same_digit_number"]
                        )
                        negative_bboxes.append(new_bboxes)
                        if self.mi_mode["alternative_action"] == "alternative_tool_call":
                            alternative_mt_manager = input_ids_mi
                            #logger.info(f"alternative mt manager: {alternative_mt_manager.all_multi_turn}")

                            changed_idxs = [True if i is not False else False for i in image_positions_mi]
                            tool_turns = image_positions_mi
                            logger.info(f"changed_idxs: {changed_idxs}")
                            local_tools_used = overall_tools_used[process_slice]
                            logger.info(f"local tools used: {local_tools_used}")
                            unchanged_but_tool_used = [idx for idx in range(len(local_tools_used)) if changed_idxs[idx] is False and local_tools_used[idx] > 0]
                            logger.info(f"unchanged_but_tool_used: {unchanged_but_tool_used}")

                            pixel_values_mi = alternative_mt_manager.get_multimodal(type="pixel_values")
                            image_grid_thw_mi = alternative_mt_manager.get_multimodal(type="image_grid_thw")

                            input_ids_mi, positions = alternative_mt_manager.get_sequences(type="id", add_assistant_start=False,
                                                                 full_image_pad=True, ignore_finished=False,
                                                                 return_positions=True)

                            considered_seqs = []
                            for conv_idx in range(len(positions)):
                                if changed_idxs[conv_idx]:
                                    for turn_idx in range(len(positions[conv_idx])):
                                        if positions[conv_idx][turn_idx]["turn"] == tool_turns[conv_idx]+2:
                                            considered_seqs.append({"contrasted_area_short": [positions[conv_idx][turn_idx]["start"],
                                                                                              positions[conv_idx][turn_idx]["end"]],
                                                                    "contrasted_area":[original_positions[conv_idx][turn_idx]["start"],
                                                                                       original_positions[conv_idx][turn_idx]["end"]]})

                                else:
                                    considered_seqs.append(None)
                            logger.info(f"considered_seqs: {considered_seqs}")
                        else:
                            pixel_values_mi = multi_turn_manager.get_multimodal(type="pixel_values",
                                                                                positions=image_positions_mi)
                            image_grid_thw_mi = multi_turn_manager.get_multimodal(type="image_grid_thw",
                                                                                  positions=image_positions_mi)

                        logger.info(f"after get alternative seqs")

                        mask_mi = [[1 for _ in seq] for seq in input_ids_mi]
                        #logger.info(f"mask_mi before pad: {[len(s) for s in mask_mi]}")
                        input_ids_mi = pad(input_ids_mi, padding_side='right', padding_value=self.processing_class.pad_token_id)
                        #logger.info(f"prompt_ids_mi after pad: {[len(s) for s in input_ids_mi]}")
                        mask_mi = pad(mask_mi, padding_side='right', padding_value=0)
                        #logger.info(f"prompt_mask_new after pad: {[len(s) for s in mask_mi]}")

                        inputs_mi = {"input_ids": torch.tensor(input_ids_mi, dtype=torch.long, device=device),
                                     "attention_mask": torch.tensor(mask_mi, dtype=torch.long, device=device),
                                     "image_grid_thw": torch.tensor(image_grid_thw_mi, dtype=torch.long, device=device),
                                     "pixel_values": torch.tensor(pixel_values_mi, dtype=torch.bfloat16, device=device)}

                        if self.mi_mode["answer_type"] == "ground_truth":
                            input_ids_mi_updated_original = [v["updated_original_sequence"] for v in considered_seqs.values()]
                            mask_mi_original = [[1 for _ in seq] for seq in input_ids_mi_updated_original]
                            input_ids_mi_updated_original = pad(input_ids_mi_updated_original, padding_side='right',
                                                                padding_value=self.processing_class.pad_token_id)
                            mask_mi_updated_original = pad(mask_mi_original, padding_side='right', padding_value=0)
                            input_ids_mi_updated_original = torch.tensor(input_ids_mi_updated_original, dtype=torch.long,
                                                                         device=device)
                            mask_mi_updated_original = torch.tensor(mask_mi_updated_original, dtype=torch.long, device=device)

                        mi_batch_size = self.args.per_device_train_batch_size * self.scoring_batch_size_multiplier

                        logger.info(f"mi_batch_size: {mi_batch_size}")
                        num_images_new = reduced_images_per_sample if self.mi_mode["alternative_action"] != "alternative_tool_call" else images_per_sample
                        logger.info(f"num_images for vision fp: {num_images_new}; {len(num_images_new)}")
                        logger.info(f"image_grid_thw for vision fp: {image_grid_thw_mi}; {image_grid_thw_mi.shape}")
                        logger.info(f"pixel_values for vision fp: {pixel_values_mi.shape}")
                        vision_masked_per_token_logps, vision_masked_per_token_entropies = self._get_per_token_logps(mi_masked_vision_forward_model,
                                                                                      input_ids=inputs_mi["input_ids"],
                                                                                      attention_mask = inputs_mi["attention_mask"],
                                                  image_grid_thw=inputs_mi["image_grid_thw"],
                                                  pixel_values=inputs_mi["pixel_values"],
                                                  num_images=num_images_new,
                                                  batch_size=mi_batch_size,
                                                  disable_dropout=True, return_entropies=True)
                        logger.info(f"after vision masked logp calculation")

                        if self.mi_mode["contrasted_score"] == "entropy":
                            vision_masked_per_token_score = -vision_masked_per_token_entropies
                        elif self.mi_mode["contrasted_score"] == "log_probs":
                            vision_masked_per_token_score = vision_masked_per_token_logps
                        elif self.mi_mode["contrasted_score"] == "probs":
                            vision_masked_per_token_score = torch.exp(vision_masked_per_token_logps)
                        else:
                            raise ValueError(
                                f"contrasted_score is {self.mi_mode["contrasted_score"]}, which is unsupported. Choose from ['entropy', 'log_probs']")


                        if self.mi_mode["answer_type"] == "ground_truth":
                            if self.mi_full_forward_model == "self":
                                full_forward_model = model
                            elif self.mi_full_forward_model == "reference":
                                full_forward_model = self.ref_model
                            logger.info(f"before ground_truth logp calculation")
                            logger.info(f"input_ids_mi_updated_original: {input_ids_mi_updated_original}")
                            logger.info(f"image_grid_thw: {multimodal_inputs["image_grid_thw"]}")
                            full_forward_per_token_logps, full_forward_per_token_entropies = self._get_per_token_logps(full_forward_model,
                                                      input_ids=input_ids_mi_updated_original,
                                                      attention_mask=mask_mi_updated_original,
                                                       image_grid_thw=multimodal_inputs["image_grid_thw"],
                                                       pixel_values=multimodal_inputs["pixel_values"],
                                                       num_images=images_per_sample,
                                                       batch_size=self.args.per_device_train_batch_size * self.scoring_batch_size_multiplier,
                                                       disable_dropout=True,
                                                       return_entropies=True)
                            logger.info(f"after ground_truth logp calculation")
                            if self.mi_mode["contrasted_score"] == "entropy":
                                full_forward_score = -full_forward_per_token_entropies
                            elif self.mi_mode["contrasted_score"] == "log_probs":
                                full_forward_score = full_forward_per_token_logps
                            elif self.mi_mode["contrasted_score"] == "probs":
                                full_forward_score = torch.exp(full_forward_per_token_logps)
                            else:
                                raise ValueError(
                                    f"contrasted_score is {self.mi_mode["contrasted_score"]}, which is unsupported. "
                                    f"Choose from ['entropy', 'log_probs', 'probs']")
                        else:
                            if self.mi_full_forward_model == "self":
                                if self.mi_mode["contrasted_score"] == "entropy":
                                    full_forward_score = -old_per_token_entropies
                                elif self.mi_mode["contrasted_score"] == "log_probs":
                                    full_forward_score = old_per_token_logps
                                elif self.mi_mode["contrasted_score"] == "probs":
                                    full_forward_score = torch.exp(old_per_token_logps)
                                else:
                                    raise ValueError(f"contrasted_score is {self.mi_mode["contrasted_score"]}, which is unsupported. "
                                                     f"Choose from ['entropy', 'log_probs', 'probs']")
                            elif self.mi_full_forward_model == "reference":
                                if self.mi_mode["contrasted_score"] == "entropy":
                                    full_forward_score = -ref_per_token_entropies
                                elif self.mi_mode["contrasted_score"] == "log_probs":
                                    full_forward_score = ref_per_token_logps
                                elif self.mi_mode["contrasted_score"] == "probs":
                                    full_forward_score = torch.exp(ref_per_token_logps)
                                else:
                                    raise ValueError(
                                        f"contrasted_score is {self.mi_mode["contrasted_score"]}, which is unsupported. "
                                        f"Choose from ['entropy', 'log_probs', 'probs']")
                            else:
                                raise ValueError("mi_full_forward_model must be 'self' or 'reference'")

                        logger.info(f"masked_score: {vision_masked_per_token_score}")

                        logger.info(f"full_score: {full_forward_score}")

                        #contrast_diff_list = []
                        if self.mi_mode["use_advantages_directly"]:
                            override_advantages = torch.zeros((full_forward_score.shape[0], 3),
                                                              dtype=torch.bfloat16, device=device)

                        #contrasted_area = positions
                        # diff = full_forward_score - vision_masked_per_token_score

                        for idx in range(full_forward_score.shape[0]):
                            if (self.mi_mode["alternative_action"] == "alternative_tool_call" and changed_idxs[idx] is True) or (self.mi_mode["alternative_action"] != "alternative_tool_call" and not considered_seqs[idx]["dummy"]):

                                if not self.mi_mode["alternative_action"] == "alternative_tool_call":
                                    contrasted_area = considered_seqs[idx]["answer_position"]
                                    contrasted_area_short = considered_seqs[idx]["answer_position_short"]
                                else:
                                    contrasted_area = considered_seqs[idx]["contrasted_area"]
                                    contrasted_area_short = considered_seqs[idx]["contrasted_area_short"]

                                logger.info(f"contrasted_area: {contrasted_area}")
                                logger.info(f"contrasted_area_short: {contrasted_area_short}")

                                logger.info(
                                    f"contrasted_area in context: {prompt_ids[idx][contrasted_area[0] - 3:contrasted_area[1] + 3]}")
                                logger.info(f"contrasted_area_short in context: {inputs_mi["input_ids"][idx][contrasted_area_short[0]-3:contrasted_area_short[1]+3]}")

                                assert contrasted_area[0] > 0
                                assert contrasted_area[1] > contrasted_area[0]
                                assert contrasted_area_short[0] > 0
                                assert contrasted_area_short[1] > contrasted_area_short[0]

                                # get rid of .exp for probs, except at the very end

                                if self.mi_mode["alternative_action"] == "alternative_tool_call":
                                    original_score = full_forward_score[idx, contrasted_area[0] - 1: contrasted_area[1] - 1]
                                    alternative_score = vision_masked_per_token_score[idx, contrasted_area_short[0] - 1: contrasted_area_short[1] - 1]

                                    # TODO: unify the contrasted_score_type!
                                    if self.mi_mode["contrasted_score_type"] == "per_sequence":
                                        original_score = torch.sum(original_score, dim=0, keepdim=True)
                                        alternative_score = torch.sum(alternative_score, dim=0, keepdim=True)

                                else:
                                    answer_scores_full = full_forward_score[idx,contrasted_area[0] - 1:contrasted_area[1] - 1]

                                    logger.info(f"answer_scores_full before multiplication: {answer_scores_full}")

                                    answer_scores_short = vision_masked_per_token_score[idx,
                                    contrasted_area_short[0] - 1:
                                    contrasted_area_short[1] - 1]

                                    logger.info(f"answer_scores_short before multiplication: {answer_scores_short}")

                                    #if contrasted_area[1] - contrasted_area[0]  !=  contrasted_area_short[1] - contrasted_area_short[0]:
                                    if self.mi_mode["contrasted_score_type"] == "per_sequence":
                                        answer_scores_full = torch.sum(answer_scores_full, dim=0, keepdim=True)
                                        answer_scores_short = torch.sum(answer_scores_short, dim=0, keepdim=True)

                                    if not self.mi_mode["alternative_action"] == "alternative_tool_call":
                                        alternative_action_position = considered_seqs[idx]["alternative_action_position"]
                                        alternative_action_position_short = considered_seqs[idx][
                                            "alternative_action_position_short"]

                                    if self.mi_mode["importance_sampling"] == True:
                                        logger.info(
                                            f"alternative_action_position in context: {prompt_ids[idx][alternative_action_position[0] - 3:alternative_action_position[1] + 3]}")
                                        logger.info(
                                            f"alternative_action_position_short in context: {inputs_mi["input_ids"][idx][alternative_action_position_short[0] - 3:alternative_action_position_short[1] + 3]}")

                                        if self.mi_mode["alternative_action"] == "second_model_generation":
                                            alt_action_scores_full = full_forward_score[
                                                idx, alternative_action_position[0] - 1:
                                                     alternative_action_position[1] - 1]
                                            logger.info(
                                                f"alt_action_scores_full before multiplication: {alt_action_scores_full}")
                                            alt_action_scores_full = torch.prod(alt_action_scores_full)
                                        else:
                                            alt_action_scores_full = 1

                                        alt_action_scores_short = vision_masked_per_token_score[
                                            idx, alternative_action_position_short[0] - 1:
                                                 alternative_action_position_short[1] - 1]

                                        logger.info(
                                            f"alt_action_scores_short before multiplication: {alt_action_scores_short}")
                                        alt_action_scores_short = torch.prod(alt_action_scores_short)
                                        logger.info(f"alt_action_scores_short: {alt_action_scores_short}")
                                        logger.info(f"alt_action_scores_full: {alt_action_scores_full}")
                                        importance_sampling_ratio = alt_action_scores_short / alt_action_scores_full
                                    else:
                                        importance_sampling_ratio = 1

                                    logger.info(f"answer_scores_full: {answer_scores_full}")
                                    logger.info(f"answer_scores_short: {answer_scores_short}")
                                    logger.info(f"importance_sampling_ratio: {importance_sampling_ratio}")

                                    if self.mi_mode["alternative_action"] == "alternative_tool_call_without_execution":
                                        original_score = answer_scores_full
                                        alternative_score = answer_scores_short

                                negatives_list[negative_idx].append(alternative_score)
                                positives_list.append(original_score)

                            else:
                                #contrast_diff_list.append(None)
                                negatives_list[negative_idx].append(None)
                                positives_list.append(None)

                    # Combine negatives for final contrastive score. For this, traverse it column first, combining the rows
                    contrast_diff_list = []
                    for negative_rollout_id in range(len(negatives_list[0])):
                        invalid_count = 0
                        for negative_sample_id in range(len(negatives_list)):
                            if negatives_list[negative_sample_id][negative_rollout_id] is None:
                                invalid_count += 1

                        if self.mi_mode["multi_negative_mode"] == "strict":
                            if invalid_count > 0:
                                contrast_diff_list.append(None)
                                continue
                        else:
                            if invalid_count == len(negatives_list):
                                contrast_diff_list.append(None)
                                continue

                        if positives_list[negative_rollout_id] is None:
                            contrast_diff_list.append(None)
                            continue

                        for negative_sample_id in range(len(negatives_list)):
                            if negative_sample_id == 0:
                                if self.mi_mode["use_info_nce"]:
                                    contrast_diff_nom =  torch.logaddexp(positives_list[negative_rollout_id], positives_list[negative_rollout_id])
                                    contrast_diff_denom = torch.logaddexp(positives_list[negative_rollout_id], negatives_list[negative_sample_id][negative_rollout_id])
                                else:
                                    contrast_diff_nom = positives_list[negative_rollout_id]
                                    contrast_diff_denom = negatives_list[negative_sample_id][negative_rollout_id]
                            else:
                                contrast_diff_nom = torch.logaddexp(contrast_diff_nom, positives_list[negative_rollout_id])
                                contrast_diff_denom = torch.logaddexp(contrast_diff_denom, negatives_list[negative_sample_id][negative_rollout_id])
                        contrast_diff = contrast_diff_nom - contrast_diff_denom

                        contrast_diff_list.append(contrast_diff.detach())

                        if contrast_diff.numel() != 0:
                            # logger.info(f"diff in contrasted area: {contrast_diff}")
                            logger.info(f"contrast diff mean: {torch.mean(contrast_diff)}")
                            logger.info(f"contrast diff max: {torch.max(contrast_diff)}")
                            logger.info(f"contrast diff min: {torch.min(contrast_diff)}")
                            logger.info(f"contrast diff sum: {torch.sum(contrast_diff)}")
                            for q in [0.1, 0.3, 0.5, 0.7, 0.9]:
                                logger.info(f"q={q}: {torch.quantile(contrast_diff.to(torch.float32), q)}")

                            x_np = contrast_diff.detach().cpu().float().numpy().ravel()
                            index_for_rank_correl = np.arange(x_np.size)
                            rho, p = spearmanr(x_np, index_for_rank_correl, nan_policy='omit')
                            logger.info(f"spearman rho: {rho}, p: {p}")

                        else:
                            logger.info(f"contrast diff contains no elements!")

                        if self.mi_mode["use_advantages_directly"] == True:
                            override_advantages[negative_rollout_id, 0] = contrast_diff[0]
                            override_advantages[negative_rollout_id, 1] = 0
                            if self.mi_mode["custom_advantage_position"] == "state":
                                override_advantages[negative_rollout_id, 2] = alternative_action_position_short[0] - 1
                            else:
                                # we're currently not logging the end of the tool call...
                                raise NotImplementedError(f"custom advantage position = state_and_action not implemented!")

                    if override_advantages is not None:
                        override_advantages = override_advantages * 2.65
                        logger.info(f"override advantages: {override_advantages}")
                        mask = torch.tensor([x is not None for x in contrast_diff_list], dtype=torch.bool)
                        logger.info(f"override advantages mean: {torch.mean(override_advantages[mask], dim=0)}")


        t5 = time.time()
        self._metrics["score_time"].append(t5 - t4)

        model_generations = multi_turn_manager.get_model_generations(type="text")

        logger.info(f"model_generations for reward calculation: {model_generations}")

        # Decode the generated completions -> skip_special_tokens used to be true, but we have to set it to false, otherwise the im_start and im_end tokens that
        # distinguish the user prompt go away. Update: completions is only used to get the rewards, so im_start and im_end tokens are irrelevant for that.
        # However, the completion ids contain eos tokens which have to be ignored, because otherwise the format reward is zero.
        # completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        completions = model_generations.copy()
        logger.info(f"number of completions: {len(completions)}")
        #this works as apply_chat_template just appends before and after without realizing that there are multiple turns inside
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]
        #logger.info(f"completions after is_conv: {completions}")

        # Compute the rewards
        # No need to duplicate prompts as we're not generating multiple completions per prompt

        bbox_estimate = multi_turn_manager.get_absolute_bboxes()

        overall_tools_used = torch.tensor(overall_tools_used, dtype=torch.float16, device=device)
        completion_rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs_per_completion), device=device)
        for i, (reward_func_dict, reward_processing_class) in enumerate(
            zip(self.reward_funcs_per_completion, self.reward_processing_classes)
        ):
            reward_func = reward_func_dict['reward_func']
            is_conditional = reward_func_dict['is_conditional']
            if isinstance(reward_func, PreTrainedModel):
                raise NotImplementedError("reward function can't be a model atm")
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in reward_kwargs:
                    for example in inputs:
                        # No need to duplicate prompts as we're not generating multiple completions per prompt
                        # reward_kwargs[key].extend([example[key]] * self.num_generations)
                        reward_kwargs[key].extend([example[key]])
                #logger.info(f"reward_kwargs: {reward_kwargs}")
                output_reward_func = reward_func(prompts=prompts, completions=completions,
                                                 tool_uses = overall_tools_used,
                                                 group_size = self.num_generations,
                                                 absolute_diff = diff,
                                                 contrasted_area = contrasted_area,
                                                 contrast_diff_list = contrast_diff_list,
                                                 bbox_estimate = bbox_estimate,
                                                 **reward_kwargs)
                if output_reward_func is None:
                    completion_rewards_per_func[:, i] = 0
                else:
                    #logger.info(f"output_reward_func: {output_reward_func}")
                    completion_rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
                    if is_conditional:
                        completion_rewards_per_func[:, i] = completion_rewards_per_func[:, i] * (completion_rewards_per_func[:, 0] > 0.5)
        logger.info(f"after per-instance rewards")
        # Gather rewards across processes
        completion_rewards_per_func = self.accelerator.gather(completion_rewards_per_func)

        logger.info(f"completion_rewards_per_func: {completion_rewards_per_func}")

        # note that overall_tools_used is global, whereas prompts and completions are local, so group rewards should
        # not use prompts or completions directly
        group_rewards_per_func = torch.zeros(len(overall_tools_used), len(self.reward_funcs_per_group), device=device)
        for i, reward_func_dict in enumerate(self.reward_funcs_per_group):
            reward_func = reward_func_dict['reward_func']
            is_conditional = reward_func_dict['is_conditional']
            group_rewards_per_func[:, i] = reward_func(prompts=prompts, completions=completions,
                                                       tool_uses=overall_tools_used, group_size=self.num_generations,
                                                       **reward_kwargs)
            if is_conditional:
                # assuming that the first reward is accuracy
                group_rewards_per_func[:, i] = group_rewards_per_func[:, i] * (completion_rewards_per_func[:, 0] > 0.5)

        rewards_per_func = torch.cat((completion_rewards_per_func, group_rewards_per_func), dim=1)

        logger.info(f"after per-group rewards")
        if self.exploration_pruning_schedule is not None:

            binary_tool_use = (overall_tools_used > 0).float()  # (280)
            tool_use_rate_group_level = binary_tool_use.view(-1, self.num_generations).mean(dim=1)  # (35, 8) -> (35)

            self.exploration_count += (tool_use_rate_group_level > self.exploration_pruning_schedule["tool_use_rate_threshold"]).sum().detach().cpu().numpy().item()

            # this assumes that accuracy reward is always the first reward in the list
            no_tool_correct = (1-binary_tool_use) * (rewards_per_func[:, 0] > 0.5)
            no_tool_correct_rate_group_level = no_tool_correct.view(-1, self.num_generations).mean(dim=1)
            no_tool_correct_rate_completion_level = no_tool_correct_rate_group_level.repeat_interleave(self.num_generations, dim=0) * binary_tool_use

            if self.exploration_count < self.exploration_pruning_schedule["exploration_threshold"]:
                # TODO: Make this pretty!
                rewards = (rewards_per_func * self.reward_func_weights.unsqueeze(0)).sum(dim=1)
                extended_rewards = (rewards_per_func * self.reward_func_weights.unsqueeze(
                    0) * self.reward_funcs_for_sampling_weights.unsqueeze(0)).sum(dim=1)
                #rewards = rewards_per_func[:, 0] + 0.1
            else:
                # this assumes that accuracy reward is always the first reward in the list
                rewards = rewards_per_func[:, 0] - no_tool_correct_rate_completion_level
            self._metrics["exploration_count"].append(self.exploration_count)
            self._metrics["no_tool_correct_rate"].append(no_tool_correct_rate_group_level.detach().cpu().numpy().mean().item())
        else:
            logger.info(f"reward_funcs_for_reward: {self.reward_funcs_for_reward}")
            logger.info(f"reward_funcs_for_sampling_weights: {self.reward_funcs_for_sampling_weights}")
            # Sum the rewards from all reward functions
            rewards = (rewards_per_func * self.reward_func_weights.unsqueeze(0) * self.reward_funcs_for_reward.unsqueeze(0)).sum(dim=1)
            extended_rewards = (rewards_per_func * self.reward_func_weights.unsqueeze(0) * self.reward_funcs_for_sampling_weights.unsqueeze(0)).sum(dim=1)

        # Compute grouped-wise rewards
        # Each group consists of num_generations completions for the same prompt
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        extended_mean_grouped_rewards = extended_rewards.view(-1, self.num_generations).mean(dim=1)

        #logger.info(f"mean_grouped_rewards: {mean_grouped_rewards}")
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        extended_std_grouped_rewards = extended_rewards.view(-1, self.num_generations).std(dim=1)

        
        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        extended_mean_grouped_rewards = extended_mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        extended_std_grouped_rewards = extended_std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        sampling_weights = (extended_rewards - extended_mean_grouped_rewards) / (extended_std_grouped_rewards + 1e-4)
        sampling_weights = torch.abs(sampling_weights)

        #logger.info(f"global advantages after scoring: {advantages}")
        
        # Get only the local slice of advantages
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]
        sampling_weights = sampling_weights[process_slice]

        # Log the metrics
        overall_generation_lengths = multi_turn_manager.get_model_generation_ids(lengths=True, flatten=False)
        generation_lengths_tensor = torch.zeros((len(overall_generation_lengths), 2), dtype=torch.bfloat16, device=device)
        for idx, seq in enumerate(overall_generation_lengths):
            if len(seq) == 0:
                generation_lengths_tensor[idx][0] = float('nan')
                generation_lengths_tensor[idx][1] = float('nan')
            elif len(seq) == 1:
                generation_lengths_tensor[idx][0] = float(seq[0])
                generation_lengths_tensor[idx][1] = float('nan')
            else:
                generation_lengths_tensor[idx][0] = float(seq[0])
                generation_lengths_tensor[idx][1] = float(seq[1])

        completion_lengths = torch.nanmean(self.accelerator.gather_for_metrics(generation_lengths_tensor), dim=0)

        self._metrics["completion_length_first"].append(completion_lengths[0].item())
        self._metrics["completion_length_second"].append(completion_lengths[1].item())

        completion_length = self.accelerator.gather_for_metrics(non_generation_mask.sum(1)).float().mean().item()



        self._metrics["completion_length"].append(completion_length)

        reward_per_func_full = rewards_per_func
        reward_per_func = torch.mean(reward_per_func_full, dim=0)

        # Mean per column, but only over non-zero entries (safe for "all zeros" columns)
        nonzero_mask = reward_per_func_full.ne(0)  # bool, same shape as reward_per_func_full
        nonzero_count = nonzero_mask.sum(dim=0)  # [num_reward_funcs], int64

        # Sum only the non-zero values
        nonzero_sum = (reward_per_func_full * nonzero_mask.to(reward_per_func_full.dtype)).sum(dim=0)

        # Avoid div-by-zero: if a column has no non-zero entries, log 0.0 for that column
        reward_per_func_nonzero = torch.where(
            nonzero_count > 0,
            nonzero_sum / nonzero_count.to(reward_per_func_full.dtype),
            torch.ones_like(nonzero_sum)*-1,
        )


        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func["func"], PreTrainedModel):
                reward_func_name = reward_func["func"].config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func["name"]
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())
            self._metrics[f"rewards_non_zero/{reward_func_name}"].append(reward_per_func_nonzero[i].item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())

        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        correct_and_tool = torch.logical_and(reward_per_func_full[:, 0] > 0.5, overall_tools_used > 0)
        incorrect_and_tool = torch.logical_and(reward_per_func_full[:, 0] <= 0.5, overall_tools_used > 0)
        correct_and_no_tool = torch.logical_and(reward_per_func_full[:, 0] > 0.5, overall_tools_used == 0)
        incorrect_and_no_tool = torch.logical_and(reward_per_func_full[:, 0] <= 0.5, overall_tools_used == 0)

        self._metrics["correct,tool"].append(correct_and_tool.sum().item())
        self._metrics["incorrect,tool"].append(incorrect_and_tool.sum().item())
        self._metrics["correct,no_tool"].append(correct_and_no_tool.sum().item())
        self._metrics["incorrect,no_tool"].append(incorrect_and_no_tool.sum().item())

        logger.info(f"advantages: {advantages}")
        logger.info(f"sampling weights: {sampling_weights}")

        logger.info(f"end of updated_grpo_trainer_with_vllm")

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "non_generation_mask": non_generation_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "override_advantages": override_advantages,
            "sampling_weights": sampling_weights,
            "multimodal_inputs": multimodal_inputs,
            "images_per_sample": images_per_sample,
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        logger.info(f"in compute loss")

        # we only need this for debugging purposes
        num_images = inputs["multimodal_inputs"].pop("num_images")
        num_images = num_images.tolist()

        logger.info(f"in compute loss: num_images: {num_images}")
        

        # Get the prepared inputs
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        # the mask is applied for losses of the next token. thus we have to shift it
        """
        generated sequence	A	B	C	D	E
        mask	            o	o	x	o	o
        losses for pred of	B	C	D	E	
        loss mask	        o	x	o	o	
        """

        non_generation_mask = inputs["non_generation_mask"][:, 1:]
        #completion_ids, completion_mask, user_mask = inputs["completion_ids"], inputs["completion_mask"], inputs["user_mask"]
        multimodal_inputs = inputs["multimodal_inputs"]
        
        # Concatenate for full sequence
        #input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        #attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        #self.monitor_gpu_usage("compute loss: before live per_token_logps")
        t8 = time.time()
        # Get the current policy's log probabilities
        per_token_logps, _ = self._get_per_token_logps(model, prompt_ids, prompt_mask,
                                              image_grid_thw=multimodal_inputs["image_grid_thw"],
                                              pixel_values=multimodal_inputs["pixel_values"],
                                              num_images=num_images,
                                              batch_size=self.args.per_device_train_batch_size,
                                              disable_dropout=False,
                                              return_entropies=False)
        t9 = time.time()
        self._metrics["live_logp_time"].append(t9 - t8)

        def contains_nan(tensor: torch.Tensor, name="tensor", terminate=True, **kwargs):
            bad = ~torch.isfinite(tensor)
            if bad.any():
                # Print a small, actionable summary
                idx = bad.nonzero(as_tuple=False)
                print(f"[check_finite] {name} contains non-finite values!")
                print(
                    f"[check_finite] {name}: dtype={tensor.dtype}, device={tensor.device}, shape={tuple(tensor.shape)}")
                print(f"[check_finite] count_bad={idx.shape[0]}")
                # show a few offending entries
                max_show = min(10, idx.shape[0])
                print(f"[check_finite] first_bad_indices={idx[:max_show].tolist()}")
                print(
                    f"[check_finite] first_bad_values={tensor[tuple(idx[0].tolist())].item() if idx.shape[1] == 0 else 'see indices'}")
                for k, v in kwargs.items():
                    print(f"[check_finite] {k}: {v}")
                if terminate:
                    raise FloatingPointError(f"{name} has non-finite values")
                else:
                    return True
                
            return False
                    

        contains_nan(per_token_logps, per_token_logps=per_token_logps, prompt_ids=prompt_ids, prompt_mask=prompt_mask)

        # TODO: get rid of all user written input
        # Get rid of the prompt (-1 because of the shift done in get_per_token_logps)
        #per_token_logps = per_token_logps[:, prompt_ids.size(1) - 1:]

        # Get the advantages from inputs
        raw_advantages = inputs["advantages"]
        logger.info(f"in compute loss: raw_advantages: {raw_advantages}")
        override_advantages = inputs["override_advantages"]
        advantages = torch.zeros_like(per_token_logps)

        for i in range(advantages.size(0)):
            advantages[i, :] = raw_advantages[i]
            advantages[i, int(override_advantages[i, 1]): int(override_advantages[i, 2])] = override_advantages[i, 0]

        logger.info(f"in compute loss: advantages: {advantages}")
        contains_nan(advantages, advantages=advantages, raw_advantages=raw_advantages,
                  override_advantages=override_advantages)

        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip its computation
        # and use per_token_logps.detach() instead
        old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()

        contains_nan(old_per_token_logps, old_per_token_logps=old_per_token_logps)

        # Compute the policy ratio and clipped version
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon)

        contains_nan(coef_1, coef_1=coef_1)
        contains_nan(coef_2, coef_2=coef_2)

        per_token_loss1 = coef_1 * advantages
        per_token_loss2 = coef_2 * advantages
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        contains_nan(per_token_loss, per_token_loss=per_token_loss, per_token_loss1=per_token_loss1, per_token_loss2=per_token_loss2)

        #ignored_tokens_mask = completion_mask * user_mask

        # Add KL penalty if beta > 0
        if self.beta > 0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            contains_nan(ref_per_token_logps, ref_per_token_logps=ref_per_token_logps)
            

            per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            if contains_nan(per_token_kl, per_token_kl=per_token_kl, terminate=False):
                per_token_kl = torch.nan_to_num(
                    per_token_kl, nan=0.0, posinf=0.0, neginf=0.0
                )
            per_token_loss = per_token_loss + self.beta * per_token_kl
            #logger.info(f"per token kl of user: {(per_token_kl * ~user_mask).sum(dim=1) / (~user_mask).sum(dim=1)}")
            #logger.info(f"per token kl without user: {(per_token_kl * user_mask).sum(dim=1) / (user_mask).sum(dim=1)}")
            # Log KL divergence
            mean_kl = ((per_token_kl * non_generation_mask).sum(dim=1) / torch.clamp(non_generation_mask.sum(dim=1), min=1.0)).mean()
            self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        #logger.info(f"per token loss of user: {(per_token_loss * ~user_mask).sum(dim=1) / (~user_mask).sum(dim=1)}")
        #logger.info(f"per token loss without user: {(per_token_loss * user_mask).sum(dim=1) / (user_mask).sum(dim=1)}")

        # Compute final loss
        loss = ((per_token_loss * non_generation_mask).sum(dim=1) / torch.clamp(non_generation_mask.sum(dim=1), min=1.0)).mean()

        # Log clip ratio
        # this was the original calculation but it is a bit weird, as it only considers one-sided clipping
        # maybe some form of dr. grpo?
        #is_clipped = (per_token_loss1 < per_token_loss2).float()
        is_clipped = (coef_1 != coef_2).float()
        clip_ratio = (is_clipped * non_generation_mask).sum() / torch.clamp(non_generation_mask.sum(), min=1.0)
        self._metrics["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())

        #self.monitor_gpu_usage("compute loss: before return")

        contains_nan(loss, loss=loss, non_generation_mask_shape=non_generation_mask.shape,
                  non_generation_mask_sum=non_generation_mask.sum(dim=1))

        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def monitor_gpu_usage(self, position):
        logger.info(f"GPU usage at position: {position}")

        for i in range(torch.cuda.device_count()):
            summary = torch.cuda.memory_summary(i)
            stats = torch.cuda.memory_stats(i)
            used = torch.cuda.memory_allocated(i)
            reserved = torch.cuda.memory_reserved(i)
            total = torch.cuda.get_device_properties(i).total_memory
            #logger.info(f"GPU {i+1}: {summary}")
            #logger.info(f"GPU {i+1}: used={used / 1e9:.2f} GB, reserved={reserved / 1e9:.2f} GB, total={total / 1e9:.2f} GB")

