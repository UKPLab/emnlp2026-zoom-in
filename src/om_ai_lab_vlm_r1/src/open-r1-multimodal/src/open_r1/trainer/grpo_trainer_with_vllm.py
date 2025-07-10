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
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union, Sized
import time

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

import deepspeed
from contextlib import nullcontext

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url, pad
from trl import GRPOTrainer
from trl.extras.vllm_client import VLLMClient

from accelerate.utils import is_peft_model, set_seed, gather_object, broadcast_object_list
import PIL.Image

import copy
from torch.utils.data import Sampler
import warnings

from trl.extras.profiling import profiling_decorator, profiling_context

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

from utils.multi_turn_handler import Prompt, Conversations, get_boundaries_tokenized, TOOLS
from utils.buffer import Buffer

from vlm_modules.vlm_module import VLMBaseModule
# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, dict[str, Union[str, Callable[[list, list], list[float]]]]]


class RepeatRandomSampler(Sampler):
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
        seed (`int` or `None`, *optional*, defaults to `None`):
            Random seed for reproducibility.
    """

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.seed = seed
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()
        indexes = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count


class VLMGRPOTrainerVLLM(Trainer):
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
        **kwargs,
    ):

        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")
        
        self.vlm_module = vlm_module
        self.save_path = save_path

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
            print("Freezing vision modules...")
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
            #print(f"processing_cls: {processing_cls}")
            processing_class = processing_cls.from_pretrained(model_id, trust_remote_code=model_init_kwargs.get("trust_remote_code", None))

            if chat_template is not None:
                processing_class.chat_template = chat_template["chat_template"]

            #print(f"processing_class: {processing_class}")
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

        self.reward_funcs_per_completion = [rf["func"] for rf in self.reward_funcs if rf["type"]=="per_completion"]
        self.reward_funcs_per_group = [rf["func"] for rf in self.reward_funcs if rf["type"]=="per_group"]


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

        #self.generation_config = GenerationConfig(
        #    max_new_tokens=self.max_completion_length,
        #    do_sample=True,
        #    temperature=1,
        #    pad_token_id=pad_token_id,
        #)

        self.beta = args.beta
        self.epsilon = args.epsilon

        self.use_vllm = args.use_vllm

        self.multi_turn = multi_turn
        if self.multi_turn == "none":
            self.multi_turn = None

        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        # Tracks the number of iterations (forward + backward passes), including those within a gradient accumulation cycle
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates
        #self._buffered_inputs = [None] * args.gradient_accumulation_steps
        self.buffer = Buffer(max_size=args.per_device_train_batch_size * args.gradient_accumulation_steps,
                             padding_id=processing_class.pad_token_id,
                             cpu_buffer=False,
                             sample_with_replacement=True)

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

        # Check if the per_device_train/eval_batch_size * num processes can be divided by the number of generations
        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * num_processes
        possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
        if self.num_generations not in possible_values:
            raise ValueError(
                f"The global train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly "
                f"divisible by the number of generations per prompt ({self.num_generations}). Given the current train "
                f"batch size, the valid values for the number of generations are: {possible_values}."
            )
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
            if self.accelerator.is_main_process:
                self.vllm_client = VLLMClient(
                    args.vllm_server_host, args.vllm_server_port, connection_timeout=args.vllm_server_timeout
                )

            # vLLM specific sampling arguments
            self.guided_decoding_regex = args.vllm_guided_decoding_regex

            self._last_loaded_step = -1  # tag to avoid useless loading during grad accumulation

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
                print(222, self.vlm_module.get_eos_token_id(processing_class))


        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

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


    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, attention_mask, **custom_multimodal_inputs):
        if dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0

        logits = model(input_ids=input_ids, attention_mask=attention_mask, **custom_multimodal_inputs).logits  # (B, L, V)
        #print("after get_per_token_logps forward pass")
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            #print("get_per_token_logps: before gather")
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            #print("get_per_token_logps: after gather")
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)


    def _prepare_inputs(self, inputs):
        # Simple pass-through, just like original
        return inputs

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

        format_prompt = "As before, first output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."


        device = self.accelerator.device
        rank = dist.get_rank() if dist.is_initialized() else 0

        history = inputs.copy()

        #print(f"inputs: {inputs}")

        prompts = [x["prompt"] for x in inputs]

        #added_tool_text = " Please use the tool exactly once during your reasoning."

        #if TOOLS is not None:
        #    for prompt in prompts[0]:
        #        for content in prompt["content"]:
        #            if content["type"] == "text":
        #                content["text"] += added_tool_text


        #print(f"prompts: {prompts}")
        prompts_text = self.vlm_module.prepare_prompt(self.processing_class, inputs, tools=TOOLS)
        print(f"prompts_text: {prompts_text}")
        # Handle both pre-loaded images and image paths
        image_paths = []
        for x in inputs:
            if "image_path" in x and x["image_path"] is not None:
                for p in self._get_key_from_inputs(x, "image_path"):
                    image_paths.append(p)
                assert len(self._get_key_from_inputs(x, "image_path")) == 1, f"Example {x} contains more than one image which is not supported atm"
            else:
                raise ValueError(f"sample {x} does not contain any image path")


        # max_prompt_length is not supported yet
        # if self.max_prompt_length is not None:
        #     prompt_ids = prompt_ids[:, -self.max_prompt_length :]
        #     prompt_inputs["input_ids"] = prompt_ids
        #     prompt_mask = prompt_mask[:, -self.max_prompt_length :]
        #     prompt_inputs["attention_mask"] = prompt_mask

        ignore_user_reply_for_loss = True
        user_token_boundaries = None

        # First, have main process load weights if needed
        if self.state.global_step != self._last_loaded_step:
            self._move_model_to_vllm()
            self._last_loaded_step = self.state.global_step

        # TODO: now vllm tokenizes everything again. would be nicer to use prompt_inputs instead
        # https://docs.vllm.ai/en/latest/design/mm_processing.html#mm-processing

        # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
        all_histories = gather_object(history)
        all_prompts_text = gather_object(prompts_text)
        all_image_paths = gather_object(image_paths)

        if self.accelerator.is_main_process:
            # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and generate
            # num_generations outputs for each one. This is faster than generating outputs for each duplicate
            # prompt individually.
            ordered_set_of_histories = all_histories[:: self.num_generations]
            ordered_set_of_prompts = all_prompts_text[:: self.num_generations]
            ordered_set_of_image_paths = all_image_paths[:: self.num_generations]
            all_multimodal_inputs = [
                {"prompt": p, "image_path": i}
                for p, i in zip(ordered_set_of_prompts, ordered_set_of_image_paths)
            ]
            #print(f"all multimodal inputs: {all_multimodal_inputs}")

            no_conversations = len(all_multimodal_inputs) * self.num_generations

            conversations = Conversations(no_conversations)

            for idx in range(no_conversations):
                mod_idx = idx // self.num_generations
                conversations.add_message(
                    Prompt(pre_tokenizer_format=copy.deepcopy(ordered_set_of_histories[mod_idx]["prompt"][0]),
                           image_path=ordered_set_of_image_paths[mod_idx]), idx)

            conv_round = 0
            max_generation_attempts = 5
            max_conv_rounds = 5
            max_tool_calls = 2
            while not all(conversations.is_finished) and conv_round < max_conv_rounds:


                with profiling_context(self, "vLLM.generate"):
                    vllm_generation_has_worked = False
                    attempts = 0
                    while (not vllm_generation_has_worked) and (attempts <= max_generation_attempts):
                        try:
                            completion_ids = self.vllm_client.generate_from_multimodal_input(
                                prompts=all_multimodal_inputs,
                                n=self.num_generations if conv_round == 0 else 1,
                                repetition_penalty=self.repetition_penalty,
                                temperature=self.temperature,
                                top_p=self.top_p,
                                top_k=-1 if self.top_k is None else self.top_k,
                                min_p=0.0 if self.min_p is None else self.min_p,
                                max_tokens=self.max_completion_length,
                                guided_decoding_regex=self.guided_decoding_regex,
                            )
                            vllm_generation_has_worked = True
                        except Exception as e:
                            print(f"Generation {attempts} failed with exception:", e)
                            attempts += 1
                            try:
                                self.vllm_client.check_server(total_timeout=10.0)
                                print(f"vLLM server is up!")
                            except ConnectionError:
                                raise ConnectionError("vLLM Server is down, aborting training.")

                # TODO: maybe we should not skip special tokens
                completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
                print(f"completions: {completions}")

                completion_idx = 0
                for idx in range(no_conversations):
                    if not conversations.is_finished[idx]:
                        conversations.add_message(
                            Prompt(content=[{'text': completions[completion_idx], 'type': 'text'}], role="assistant"), idx)
                        completion_idx += 1

                if self.multi_turn is None:
                    conversations.is_finished = [True] * no_conversations
                elif self.multi_turn == "text":
                    text_rethink = "Are you sure? Think again. "
                    if conv_round == 0:
                        for idx in range(no_conversations):
                            conversations.add_message(Prompt(content=[{'text': text_rethink + format_prompt,
                                                               'type': 'text'}],
                                                     role="user"),
                                              idx)
                    else:
                        conversations.is_finished = [True] * no_conversations
                elif self.multi_turn == "image":
                    image_rethink = "Are you sure? Look at the image again. "
                    if conv_round == 0:
                        for idx in range(no_conversations):
                            conversations.add_message(Prompt(content=[{"text": None, "type": "image"},
                                                              {'text': image_rethink + format_prompt,
                                                               'type': 'text'}],
                                                     role="user",
                                                     image_path=conversations.get_image_paths()[idx][0]),
                                              idx)
                    else:
                        conversations.is_finished = [True] * no_conversations
                elif self.multi_turn == "tool":
                    conversations.handle_tool_call(save_path=os.path.join(self.save_path, "tool_calls"),
                                                   step=self.state.global_step)

                    for idx in range(no_conversations):
                        if conversations.get_no_tool_calls(idx) >= max_tool_calls:
                            conversations.is_finished[idx] = True

                full_conversations_concat = self.vlm_module.prepare_prompt(self.processing_class,
                                                                           conversations.get_full_for_hf_prep(
                                                                               ignore_finished=True), tools=TOOLS)
                # print(f"full conversations: {full_conversations_concat}")
                all_multimodal_inputs = [
                    {"prompt": p, "image_path": i}
                    for p, i in zip(full_conversations_concat, conversations.get_image_paths(ignore_finished=True))
                ]
                print(f"all_multimodal_inputs: {all_multimodal_inputs}")
                conv_round += 1

                """
                if self.multi_turn is not None:
                    for idx, completion in enumerate(completions):
                        mod_idx = idx // self.num_generations
                        if self.multi_turn == "text":


                            conversations.add_message(Prompt(content=[{'text': text_rethink + format_prompt,
                                                                       'type': 'text'}],
                                                             role="user"),
                                                      idx)

                        elif self.multi_turn == "image":
                            image_rethink = "Are you sure? Look at the image again. "
                            conversations.add_message(Prompt(content=[{"text": None, "type": "image"},
                                                                      {'text': image_rethink + format_prompt,
                                                                       'type': 'text'}],
                                                             role="user",
                                                             image_path=ordered_set_of_image_paths[mod_idx]),
                                                      idx)
                        elif self.multi_turn == "tool":
                            conversations.handle_tool_call()
                        else:
                            raise ValueError(f"multi_turn {self.multi_turn} must be either 'text' or 'image'.")
                    #print(f"get full for hf prep: {conversations.get_full_for_hf_prep()}")

                    full_conversations_concat = self.vlm_module.prepare_prompt(self.processing_class,
                                                                               conversations.get_full_for_hf_prep(
                                                                                   ignore_finished=True))
                    #print(f"full conversations: {full_conversations_concat}")
                    all_multimodal_inputs = [
                        {"prompt": p, "image_path": i}
                        for p, i in zip(full_conversations_concat, conversations.get_image_paths(ignore_finished=True))
                    ]
                    #print(f"In repeat: all_mm_inputs: {all_multimodal_inputs}\n")
                    max_generation_attempts = 5
                    with profiling_context(self, "vLLM.generate"):
                        vllm_generation_has_worked = False
                        attempts = 0
                        while (not vllm_generation_has_worked) and (attempts <= max_generation_attempts):
                            try:
                                completion_ids_new = self.vllm_client.generate_from_multimodal_input(
                                    prompts=all_multimodal_inputs,
                                    n=1,
                                    repetition_penalty=self.repetition_penalty,
                                    temperature=self.temperature,
                                    top_p=self.top_p,
                                    top_k=-1 if self.top_k is None else self.top_k,
                                    min_p=0.0 if self.min_p is None else self.min_p,
                                    max_tokens=self.max_completion_length,
                                    guided_decoding_regex=self.guided_decoding_regex,
                                )
                                vllm_generation_has_worked = True
                            except Exception as e:
                                print(f"Generation {attempts} failed with exception:", e)
                                attempts += 1
                                try:
                                    self.vllm_client.check_server(total_timeout=10.0)
                                    print(f"vLLM server is up!")
                                except ConnectionError:
                                    raise ConnectionError("vLLM Server is down, aborting training.")

                    completions_new = self.processing_class.batch_decode(completion_ids_new, skip_special_tokens=True)

                    conversations.add_messages([Prompt(role="assistant",
                                                       content=[{'text': completion,
                                                                 'type': 'text'}])
                                                for completion in completions_new])
                """

                """prompt_1_with_chat_template = self.vlm_module.prepare_prompt(self.processing_class,
                                                                             conversations.wrap_with_prompt_for_hf(conversations.get_messages_as_dicts("user", 1)),
                                                                             add_generation_prompt=None)

                # TODO: This only works with Qwen!!
                prompt_1_with_chat_template = [prompt.removeprefix("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n")
                                               for prompt in prompt_1_with_chat_template]


                full_completions = []
                user_prompt_boundaries = []
                for completions0, prompt1_with_template, completions1 in zip(conversations.get_messages("assistant", 0), prompt_1_with_chat_template, conversations.get_messages("assistant", 1)):
                    full_completions.append([completions0.get_text() + "<|im_end|>\n" + prompt1_with_template + completions1.get_text()])
                    user_prompt_boundaries.append([len(completions0.get_text() + "<|im_end|>\n"),
                                                   len(completions0.get_text() + "<|im_end|>\n") + len(prompt1_with_template)])

                #print(f"full completions: {full_completions}\n")
                if len(full_completions) > 0:
                    user_token_boundaries = []
                completion_ids = []
                for completion_idx, full_completion in enumerate(full_completions):
                    full_completion_tokenized = self.processing_class(text=full_completion, return_tensors=None,
                                                      padding=False, add_special_tokens=False,
                                                      return_offsets_mapping=True,
                                                      )
                    #print(f"rank: {rank}, tokenized without special tokens {full_completion_tokenized['input_ids'][0]}")
                    completion_ids.append(full_completion_tokenized["input_ids"][0])
                    #print(full_completion_tokenized["offset_mapping"])
                    for i, (start, end) in enumerate(full_completion_tokenized["offset_mapping"][0]):
                        if start <= user_prompt_boundaries[completion_idx][0] < end:
                            start_idx = i
                        if start < user_prompt_boundaries[completion_idx][1] <= end:
                            end_idx = i
                            break
                    user_token_boundaries.append([start_idx, end_idx])
"""
                # Testing
                #for completion, boundary in zip(completion_ids, user_token_boundaries):
                #    print(completion[boundary[0]:boundary[1]])

            all_image_paths = conversations.get_image_paths()
            full_generations = self.vlm_module.prepare_prompt(self.processing_class,
                                                              conversations.get_full_for_hf_prep(), tools=TOOLS)
            model_generations = conversations.get_model_generations()

            #overall_tools_used = np.array([conversations.get_no_tool_calls(idx) for idx in range(no_conversations)])
            overall_tools_used = [conversations.get_no_tool_calls(idx) for idx in range(no_conversations)]
            #overall_tools_used = torch.tensor([conversations.get_no_tool_calls(idx) for idx in range(no_conversations)],
            #                                  dtype=torch.float, device=self.accelerator.device)
            print(f"overall_tools_used: {overall_tools_used}")
            self._metrics["mean_tool_use"].append(float(np.mean(np.array(overall_tools_used))))
            self._metrics["std_tool_use"].append(float(np.std(np.array(overall_tools_used))))

            #print(f"full generations: {full_generations}")
            #print(f"model_generations: {model_generations}")

        else:
            full_generations = [None] * len(all_prompts_text)
            model_generations = [None] * len(all_prompts_text)
            #model_generated_boundaries = [None] * len(all_prompts_text)
            all_image_paths = [None] * len(all_prompts_text)
            overall_tools_used = [None] * len(all_prompts_text)

        #print(f"({rank}) Completion ids: {[len(lst) if lst is not None else 0 for lst in completion_ids]}")
        # Broadcast the completions from the main process to all processes, ensuring each process receives its
        # corresponding slice.

        # TODO: is it better to tokenize globally or distributed?
        full_generations = broadcast_object_list(full_generations, from_process=0)
        model_generations = broadcast_object_list(model_generations, from_process=0)
        all_image_paths = broadcast_object_list(all_image_paths, from_process=0)
        overall_tools_used = broadcast_object_list(overall_tools_used, from_process=0)

        #print(f"({rank}) Completion ids: {[len(lst) if lst is not None else 0 for lst in completion_ids]}")
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        full_generations = full_generations[process_slice]
        model_generations = model_generations[process_slice]
        all_image_paths = all_image_paths[process_slice]
        #overall_tools_used = overall_tools_used[process_slice]

        #print(f"({rank}) Completion ids after slice: {[len(lst) if lst is not None else 0 for lst in completion_ids]})")


        #print(f"lengths of completion IDs: {[len(completion_id) for completion_id in completion_ids]}")
        #for completion_id in completion_ids:
        #    for pos, idx in enumerate(completion_id):
        #        if idx == self.processing_class.eos_token_id:
                    #print(f"rank: {rank} found early eos token ({self.processing_class.eos_token_id}), "
                    #      f"pad={self.processing_class.pad_token_id} in pos {pos} with context: {completion_id[pos-5:pos+5]}")

        # Pad the completions, and concatenate them with the prompts
        #completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]

        #completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)

        #prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)

        images = []
        image_paths_flatten = [item for sublist in all_image_paths for item in sublist]
        images_per_sample = [len(sublist) for sublist in all_image_paths]

        for image_path in image_paths_flatten:
            img = PIL.Image.open(image_path)
            try:
                # Ensure minimum dimensions of 28 pixels
                w, h = img.size
                if w < 28 or h < 28:
                # Calculate new dimensions maintaining aspect ratio
                    if w < h:
                        new_w = 28
                        new_h = int(h * (28/w))
                    else:
                        new_h = 28
                        new_w = int(w * (28/h))
                img = img.resize((new_w, new_h), PIL.Image.Resampling.LANCZOS)
            except:
                pass
            images.append(img)

        hf_inputs = self.processing_class(
            text=full_generations.copy(),
            images=images,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
            return_offsets_mapping=False
        )

        # these are the token ids for "<|im_start|>assistant\n" and "<|im_end|>"
        user_token_boundaries = get_boundaries_tokenized(hf_inputs["input_ids"],
                                                         torch.tensor([151644, 77091, 198], dtype=torch.long),
                                                         torch.tensor([151645], dtype=torch.long))

        #print(f"token_boundaries: {user_token_boundaries}")

        prompt_inputs = super()._prepare_inputs(hf_inputs)

        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        #to check token boundaries
        #for idx, pi in enumerate(prompt_ids):
        #    for b in user_token_boundaries[idx]:
        #        print(f"for seq {idx}: {pi[b[0]: b[1]]}")

        #prompt_length = prompt_ids.size(1)

        non_generation_mask = torch.ones_like(prompt_mask, device=device, dtype=torch.bool)

        for i, assistant_boundaries in enumerate(user_token_boundaries):
            for boundary in assistant_boundaries:
                non_generation_mask[i, boundary[0]:boundary[1]] = False

        # Concatenate prompt_mask with completion_mask for logit computation
        #attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        # Get the multimodal inputs
        multimodal_keywords = self.vlm_module.get_custom_multimodal_keywords()
        multimodal_inputs = {k: prompt_inputs[k] if k in prompt_inputs else None for k in multimodal_keywords}
        with torch.no_grad():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip its
            # computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    model, prompt_ids, prompt_mask, **multimodal_inputs
                )
                #old_per_token_logps = old_per_token_logps[:, prompt_length - 1:]
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                #print(f"rank: {rank}: directly before ref per token logps calculation")
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_ids, prompt_mask, **multimodal_inputs
                )
            else:
                #print("before unwrap model!")
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        model, prompt_ids, prompt_mask, **multimodal_inputs
                    )

        #if self.beta != 0.0:
        #    ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:]

        #print("after ref model logps")

        # Decode the generated completions -> skip_special_tokens used to be true, but we have to set it to false, otherwise the im_start and im_end tokens that
        # distinguish the user prompt go away. Update: completions is only used to get the rewards, so im_start and im_end tokens are irrelevant for that.
        # However, the completion ids contain eos tokens which have to be ignored, because otherwise the format reward is zero.
        # completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        completions = model_generations.copy()
        #this works as apply_chat_template just appends before and after without realizing that there are multiple turns inside
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]
        #print(f"completions after is_conv: {completions}")

        # Compute the rewards
        # No need to duplicate prompts as we're not generating multiple completions per prompt
        overall_tools_used = torch.tensor(overall_tools_used, dtype=torch.float16, device=device)
        completion_rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs_per_completion), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs_per_completion, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                raise NotImplementedError("reward function can't be a model atm")
                """if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)"""
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in reward_kwargs:
                    for example in inputs:
                        # No need to duplicate prompts as we're not generating multiple completions per prompt
                        # reward_kwargs[key].extend([example[key]] * self.num_generations)
                        reward_kwargs[key].extend([example[key]])
                #print(f"input to reward func: prompts:  reward_func:={reward_func} prompts={prompts}, completions={completions}")
                #print(f"reward func: {reward_func}")
                output_reward_func = reward_func(prompts=prompts, completions=completions,
                                                 tool_uses = overall_tools_used,
                                                 group_size = self.num_generations,
                                                 **reward_kwargs)
                #print(f"output_reward_func: {output_reward_func}")
                completion_rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
        #print(f"rewards per func: {rewards_per_func}")

        #print(f"before gather rewards_per_func: {completion_rewards_per_func}")
        #print("after rewards calculation")
        # Gather rewards across processes
        completion_rewards_per_func = self.accelerator.gather(completion_rewards_per_func)
        #print(f"after gather rewards_per_func: {completion_rewards_per_func}")
        #print("after gather rewards")

        group_rewards_per_func = torch.zeros(len(overall_tools_used), len(self.reward_funcs_per_group), device=device)
        for i, reward_func in enumerate(self.reward_funcs_per_group):
            group_rewards_per_func[:, i] = reward_func(prompts=prompts, completions=completions,
                            tool_uses=overall_tools_used, group_size=self.num_generations,
                        **reward_kwargs)

        rewards_per_func = torch.cat((completion_rewards_per_func, group_rewards_per_func), dim=1)

        #print(f"after per_group rewards: {rewards_per_func}")

        #print(f"overall tools used before rewards: {overall_tools_used}")
        # Sum the rewards from all reward functions
        rewards = (rewards_per_func * self.reward_func_weights.unsqueeze(0)).sum(dim=1)
        #print(f"rewards: {rewards}")

        """use_original_pr_rewards = True
        tool_use_penalty = 1 # N in PR paper
        pixel_reasoning_threshold = 0.3 # H in PR paper
        penalty_weight = 0.05 # beta in PR paper
        curiosity_weight = 0.5 # alpha in PR paper



        if use_original_pr_rewards:
            overall_tools_used = torch.tensor(overall_tools_used, dtype=torch.float16, device=device)
            #print(f"overall_tools_used: {overall_tools_used}")
            penalty = torch.clamp(tool_use_penalty - overall_tools_used, max = 0)
            #print(f"penalty: {penalty}")
            binary_tool_use = (overall_tools_used > 0).float()
            pixel_reasoning_rate = binary_tool_use.view(-1, self.num_generations).mean(dim=1)
            #print(f"pixel_reasoning_rate: {pixel_reasoning_rate}")
            pixel_reasoning_rate = pixel_reasoning_rate.repeat_interleave(self.num_generations, dim = 0)
            #print(f"pixel_reasoning_rate after interleave: {pixel_reasoning_rate}")
            curiosity = torch.clamp(pixel_reasoning_threshold - pixel_reasoning_rate, min = 0) * binary_tool_use
            #print(f"curiosity: {curiosity}")
            new_rewards = rewards + penalty_weight * penalty + curiosity_weight * curiosity
            #print(f"new_rewards: {new_rewards}")"""

        # Compute grouped-wise rewards
        # Each group consists of num_generations completions for the same prompt
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        #print(f"mean_grouped_rewards: {mean_grouped_rewards}")
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        
        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        
        # Get only the local slice of advantages
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(non_generation_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func["func"], PreTrainedModel):
                reward_func_name = reward_func["func"].config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func["func"].__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())

        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "non_generation_mask": non_generation_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "multimodal_inputs": multimodal_inputs,
            "images_per_sample": images_per_sample,
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        # Check if we need to generate new completions or use buffered ones
        """
        if self.state.global_step % self.num_iterations == 0:
            inputs = self._generate_and_score_completions(inputs, model)
            self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = inputs
        else:
            inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
        self._step += 1
        """

        if self.state.global_step % self.num_iterations == 0:
            inputs_for_buffer = self._generate_and_score_completions(inputs, model)
            self.buffer.add(inputs_for_buffer)
            inputs = self.buffer.get(batch_size=self.args.per_device_train_batch_size, deterministic=False)
        else:
            inputs = self.buffer.get(batch_size=self.args.per_device_train_batch_size, deterministic=False)
        

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

        # Get the current policy's log probabilities
        per_token_logps = self._get_per_token_logps(model, prompt_ids, prompt_mask, **multimodal_inputs)
        # TODO: get rid of all user written input
        # Get rid of the prompt (-1 because of the shift done in get_per_token_logps)
        #per_token_logps = per_token_logps[:, prompt_ids.size(1) - 1:]

        # Get the advantages from inputs
        advantages = inputs["advantages"]

        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip its computation
        # and use per_token_logps.detach() instead
        old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()

        # Compute the policy ratio and clipped version
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        #ignored_tokens_mask = completion_mask * user_mask

        # Add KL penalty if beta > 0
        if self.beta > 0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            per_token_loss = per_token_loss + self.beta * per_token_kl
            #print(f"per token kl of user: {(per_token_kl * ~user_mask).sum(dim=1) / (~user_mask).sum(dim=1)}")
            #print(f"per token kl without user: {(per_token_kl * user_mask).sum(dim=1) / (user_mask).sum(dim=1)}")
            # Log KL divergence
            mean_kl = ((per_token_kl * non_generation_mask).sum(dim=1) / non_generation_mask.sum(dim=1)).mean()
            self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        #print(f"per token loss of user: {(per_token_loss * ~user_mask).sum(dim=1) / (~user_mask).sum(dim=1)}")
        #print(f"per token loss without user: {(per_token_loss * user_mask).sum(dim=1) / (user_mask).sum(dim=1)}")

        # Compute final loss
        loss = ((per_token_loss * non_generation_mask).sum(dim=1) / non_generation_mask.sum(dim=1)).mean()

        # Log clip ratio
        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * non_generation_mask).sum() / non_generation_mask.sum()
        self._metrics["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())

        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))

    def _get_train_sampler(self) -> Sampler:
        """Returns a sampler that ensures proper data sampling for GRPO training."""
        effective_batch_size = (
            self.args.per_device_train_batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )
        
        return RepeatRandomSampler(
            data_source=self.train_dataset,
            mini_repeat_count=self.num_generations,
            batch_size=effective_batch_size // self.num_generations,
            repeat_count=self.num_iterations,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        """Returns a sampler for evaluation."""
        return RepeatRandomSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations,
            seed=self.args.seed,
        )
