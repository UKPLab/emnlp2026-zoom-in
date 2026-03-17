import asyncio
import base64
import mimetypes
import re
import subprocess
import urllib.request
from pathlib import Path

from openai.types.chat import ChatCompletionMessage

from open_r1.utils.logger import get_logger, setup_project_logging
import torch
import argparse
from open_r1.utils.multi_turn_handler import Prompt, Conversations
from open_r1.utils.masker import get_boundaries_tokenized
from open_r1.utils.tools import Tool, Message, TOOL_CONFIGS
import copy
import os
import numpy as np
import PIL
from open_r1.utils.rewards import accuracy_reward
from trl.data_utils import is_conversational

from open_r1.utils.utils import calculate_overlap_metrics
from open_r1.vlm_modules import Qwen2VLModule
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from open_r1.preprocess_data import prepare_data
from torch.utils.data import DataLoader
import time
import json
from open_r1.utils.prompts import get_question_template
import gc
import sys

from openai import AsyncOpenAI, OpenAI

from open_r1.utils.multi_turn_manager import MultiTurn, pad

from vllm.inputs import TokensPrompt



import os



class VLLM:

    def __init__(self, model:str, tensor_parallel_size:int, gpu_memory_utilization,
                 dtype, enable_prefix_caching, max_model_len,
                 enforce_eager, limit_mm_per_prompt, mm_processor_kwargs, revision=None, disable_custom_all_reduce=False):
        self.model = model
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.dtype = dtype
        self.enable_prefix_caching = enable_prefix_caching
        self.max_model_len = max_model_len
        self.enforce_eager = enforce_eager
        self.limit_mm_per_prompt = limit_mm_per_prompt
        self.mm_processor_kwargs = mm_processor_kwargs
        self.revision = revision
        self.disable_custom_all_reduce = disable_custom_all_reduce


        self.llm = LLM(
            model=self.model,
            revision=self.revision,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            dtype=self.dtype,
            # Automatic Prefix Caching caches the KV cache of existing queries, so that a new query can
            # directly reuse the KV cache if it shares the same prefix with one of the existing queries.
            # This is particularly useful here because we generate completions from the same prompts.
            enable_prefix_caching=self.enable_prefix_caching,
            max_model_len=self.max_model_len,
            worker_extension_cls="trl.scripts.vllm_serve.WeightSyncWorkerExtension",
            disable_custom_all_reduce=self.disable_custom_all_reduce,
            enforce_eager=self.enforce_eager,
            limit_mm_per_prompt=self.limit_mm_per_prompt,
            mm_processor_kwargs=self.mm_processor_kwargs,
        )

    def generate(self, prompts, sampling_params):
        try:


            inputs_with_image = []  # will become List[Dict[str, Union[str, Dict[str, PIL.Image]]]]

            img_sizes = []


            for entry in prompts:
                if isinstance(entry["image_path"], str):
                    entry["image_path"] = [entry["image_path"]]
                images = []
                img_size = []
                for img_path in entry["image_path"]:
                    with open(img_path, 'rb') as img_file:
                        img = PIL.Image.open(img_file)
                        img.load()
                        #img = PIL.Image.open(img_path)
                    try:
                        # Ensure minimum dimensions of 28 pixels
                        w, h = img.size
                        if w < 28 or h < 28:
                            # Calculate new dimensions maintaining aspect ratio
                            if w < h:
                                new_w = 28
                                new_h = int(h * (28 / w))
                            else:
                                new_h = 28
                                new_w = int(w * (28 / h))
                            img = img.resize((new_w, new_h), PIL.Image.Resampling.LANCZOS)
                    except Exception as e:
                        logger.info(f"Warning: could not process image {img_path}: {e}")
                    img_size.append(img.size)
                    images.append(img)
                if "prompt" in entry:
                    inputs_with_image.append(
                        {"prompt": entry["prompt"],
                         "multi_modal_data": {"image": images}
                         })
                elif "prompt_token_ids" in entry:
                    inputs_with_image.append(
                        TokensPrompt(prompt_token_ids=entry["prompt_token_ids"],
                                     multi_modal_data={"image": images})
                    )
                img_sizes.append(img_size)

            logger.info(f"directly before vllm generate")
            logger.info(f"sampling_params: {sampling_params}")
            logger.info(f"inputs_with_image: {inputs_with_image}")
            all_outputs = self.llm.generate(inputs_with_image, sampling_params=sampling_params)

            logger.info(f"after vllm generate")
            #logger.info(f"{all_outputs[0].outputs[0].text}")
            # TODO: use nested loop instead of only [0]
            only_text = []
            completion_len = []
            only_tokens = []
            for output in all_outputs:
                for sample in output.outputs:
                    only_text.append(sample.text)
                    completion_len.append(len(sample.token_ids))
                    only_tokens.append(sample.token_ids)
            #only_text = [output.outputs[0].text for output in all_outputs]
            #completion_len = [len(output.outputs[0].token_ids) for output in all_outputs]
            #only_tokens = [output.outputs[0].token_ids for output in all_outputs]

            # Explicit cleanup
            del inputs_with_image
            del all_outputs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if "prompt" in prompts[0]:
                return only_text, img_sizes, completion_len
            elif "prompt_token_ids" in prompts[0]:
                return only_tokens

        finally:
            # Aggressive cleanup in finally block
            try:
                # Force garbage collection
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as cleanup_error:
                logger.warning(f"Error during cleanup: {cleanup_error}")

class VLLM_Server:
    def __init__(self, is_async: bool, max_concurrency: int = 1, port:int=8000):
        self.is_async = is_async
        self.max_concurrency = max_concurrency
        self.port = port
        if is_async:
            self.client = AsyncOpenAI(
                api_key="EMPTY",
                base_url=f"http://localhost:{self.port}/v1",
                timeout=3600
            )
        else:
            self.client = OpenAI(
                api_key="EMPTY",
                base_url=f"http://localhost:{self.port}/v1",
                timeout=3600
            )

    def process_data(self, samples: list[dict], tools: list[Tool], save_path:str, max_rounds: int):
        tool_dict = [tool.get_tool_dict() for tool in tools] if tools is not None else None
        #tool_names = [d["function"]["name"] for d in tool_dict] if tool_dict else None
        #tool_callables = {tool.get_tool_dict()["function"]["name"]:tool.callable_function for tool in tools} if tools is not None else None

        if tools is not None and len(tools) > 1:
            raise ValueError("Only one tool is supported for evaluation")

        if self.is_async:
             return asyncio.run(self.run_dataset(samples=samples, tools=tools, save_path=save_path, max_rounds=max_rounds,
                                                 max_concurrency=self.max_concurrency))
        else:
            all_replies = []
            for sample in samples:
                image_url = self.local_image_to_data_url(sample["image_path"])
                all_image_paths = [sample["image_path"]]
                all_messages = [{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            }
                        },
                        {
                            "type": "text",
                            "text": sample["text"]
                        }
                    ]
                }]

                #max_rounds = 2
                for _ in range(max_rounds):

                    response = self.client.chat.completions.create(
                        model="Qwen/Qwen3.5-9B",
                        messages=all_messages,
                        tools=tool_dict
                        #max_tokens=2048
                    )

                    print(f"Generated text: {response.choices[0].message}")

                    new_message = response.choices[0].message

                    # Final answer case
                    if len(new_message.tool_calls) == 0:
                        logger.info(f"list of tool calls empty, this sample is finished!")
                        all_replies.append({"content": response.choices[0].message.content,
                                            "reasoning": response.choices[0].message.reasoning,
                                            "tool_calls": response.choices[0].message.tool_calls})
                        break

                    # Assistant tool call message
                    assistant_message = {
                        "role": "assistant",
                        "content": new_message.content or "",
                        "tool_calls": [],
                    }

                    for tc in new_message.tool_calls:
                        assistant_message["tool_calls"].append({
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        })

                    all_messages.append(assistant_message)

                    # Execute tools and append tool results
                    for tc in new_message.tool_calls:
                        tool_name = tc.function.name
                        tool_args = json.loads(tc.function.arguments)

                        logger.info(f"Executing tool: {tool_name} with args: {tool_args}")

                        if tool_name != tool_dict[0]["function"]["name"]:
                            tool_result = {
                                "error": f"Unknown tool: {tool_name}"
                            }
                        else:
                            try:
                                tool_call_result = tools[0].call_tool({"arguments": tool_args,
                                                                       "image_paths":all_image_paths},
                                                                      save_path=os.path.join(save_path,"tool_calls"),
                                                                      base_model="qwen_3p5")
                                all_image_paths.append(tool_call_result["new_image_path"])

                                for i, part in enumerate(tool_call_result["output_message"]):
                                    if part["type"] == "image":

                                        url_format = {
                                            "type": "image_url",
                                                "image_url": {
                                                    "url": self.local_image_to_data_url(tool_call_result["new_image_path"])
                                                }
                                        },
                                        tool_call_result["output_message"][i] = url_format
                                        break
                                tool_result = tool_call_result["output_message"]
                                logger.info(f"Tool '{tool_name}' executed successfully")


                            except Exception as e:
                                tool_result = {
                                    "error": f"Tool execution failed: {str(e)}"
                                }

                        logger.info(f"Tool '{tool_name}' executed with result: {tool_result}")

                        all_messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps(tool_result),
                        })

                all_replies.append({"content": response.choices[0].message.content,
                                    "reasoning": response.choices[0].message.reasoning,
                                    "tool_calls": response.choices[0].message.tool_calls})
            return all_replies



    def local_image_to_data_url(self, path: str) -> str:
        path_obj = Path(path)
        mime_type, _ = mimetypes.guess_type(path_obj.name)
        if mime_type is None:
            mime_type = "application/octet-stream"

        with open(path_obj, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")

        return f"data:{mime_type};base64,{encoded}"

    def replace_b64_by_path(self, content, image_path: str):
        #logger.info(f"in replace_b64_by_path: {content}")
        content_copy = copy.deepcopy(content)
        for i, part in enumerate(content_copy):
            if isinstance(part, dict):
                #logger.info(part.keys())
                if part["type"] == "image_url":
                    #logger.info(f"replace image url by image path now!")
                    part["image_url"] = image_path
        return content_copy

    async def run_dataset(self, samples: list[dict], tools: list[Tool], save_path: str, max_rounds: int, max_concurrency: int = 16) -> list[dict]:
        semaphore = asyncio.Semaphore(max_concurrency)
        tasks = [self.run_one_sample(sample, tools, save_path, max_rounds, semaphore) for sample in samples]
        return await asyncio.gather(*tasks)

    async def run_one_sample(self, sample: dict, tools: list[Tool], save_path: str, max_rounds: int, semaphore: asyncio.Semaphore) -> dict:
        tool_dict = [tool.get_tool_dict() for tool in tools] if tools is not None else None
        async with (semaphore):
            image_url = self.local_image_to_data_url(sample["image_path"])

            all_messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    },
                    {
                        "type": "text",
                        "text": sample["text"]
                    }
                ]
            }]

            all_messages_with_image_path = [
                {
                    "role": "user",
                    "content": self.replace_b64_by_path(all_messages[0]["content"], sample["image_path"])
                }
            ]


            #self.replace_b64_by_path(all_messages, sample["image_path"])
            no_tool_calls = 0
            tool_calls_for_logging = []
            #max_rounds = 2
            for no_turns in range(max_rounds):
                all_image_paths = [sample["image_path"]]
                try:
                    response = await self.client.chat.completions.create(
                        model="Qwen/Qwen3.5-9B",
                        messages=all_messages,
                        tools=tool_dict
                        # max_tokens=2048
                    )
                except Exception as e:
                    logger.info(f"Error during completion: {e}")
                    #logger.info(f"all messages: {all_messages}")
                    logger.info(f"sleeping ...")
                    #time.sleep(100)
                    continue
                    #return {"status": {"generation error during": no_turns+1},
                    #       "all_messages": all_messages_with_image_path}


                logger.info(f"Generated text: {response.choices[0].message}")

                new_message = response.choices[0].message



                # Assistant tool call message
                #assistant_message = {
                #    "role": "assistant",
                #    "content": new_message.content or "",
                #    "tool_calls": [],
                #}

                #for tc in new_message.tool_calls:
                #    assistant_message["tool_calls"].append({
                #        "id": tc.id,
                #        "type": tc.type,
                #        "function": {
                #            "name": tc.function.name,
                #            "arguments": tc.function.arguments,
                #        },
                #    })

                all_messages.append(new_message)
                all_messages_with_image_path.append(new_message)

                tool_calls = new_message.tool_calls or []

                # Final answer case
                if len(tool_calls) == 0:
                    logger.info(f"list of tool calls empty, sample {sample['sample_idx']} is finished! final full multi-turn interaction: {all_messages_with_image_path}")

                    return {"status": {"success after": no_turns+1},
                            "tool_calls": tool_calls_for_logging,
                           "all_messages": all_messages_with_image_path}

                # Execute tools and append tool results
                for tc in tool_calls:
                    tool_name = tc.function.name
                    tool_args = json.loads(tc.function.arguments)

                    logger.info(f"Executing tool: {tool_name} with args: {tool_args}")

                    if tool_name != tool_dict[0]["function"]["name"]:
                        tool_result = [{
                            "type": "text",
                            "text": f"Error: Unknown tool: {tool_name}"
                        }]
                    else:
                        try:
                            tool_call_result = tools[0].call_tool({"arguments": tool_args,
                                                                   "image_paths": all_image_paths},
                                                                  save_path=os.path.join(save_path, "tool_calls"),
                                                                  base_model="qwen_3p5")
                            all_image_paths.append(tool_call_result["new_image_path"])

                            for i, part in enumerate(tool_call_result["output_message"]):
                                if part["type"] == "image":
                                    url_format = {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": self.local_image_to_data_url(tool_call_result["new_image_path"])
                                        }
                                    }
                                    tool_call_result["output_message"][i] = url_format
                                    break
                            tool_result = tool_call_result["output_message"]

                            tool_calls_for_logging.append({"bbox_wrt_target": tool_call_result["absolute_bbox_wrt_target_coords"],
                                                           "target": tool_call_result["target_image_idx"]})



                            logger.info(f"Tool '{tool_name}' executed successfully")


                        except Exception as e:
                            tool_result = [{
                                "type": "text",
                                "text": f"Error: Tool execution failed: {str(e)}"
                            }]

                    tool_result_with_image_path = self.replace_b64_by_path(tool_result, all_image_paths[-1])

                    logger.info(f"Tool '{tool_name}' executed with result: {tool_result_with_image_path}")

                    all_messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_result,
                    })
                    all_messages_with_image_path.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_result_with_image_path,
                    })
                    no_tool_calls += 1

            logger.info(f"max rounds exceeded, returning error")
            return_dict = {"status": {"error after": max_rounds},
                           "tool_calls": tool_calls_for_logging,
                           "all_messages": all_messages_with_image_path}
            return return_dict

class Evaluator:

    def __init__(self, vlm_module, processing_class, vllm_client, sampling_params:SamplingParams,
                 num_generations:int,
                 multi_turn, max_tool_uses, strict_tool_extraction, save_path, metrics):
        self.vlm_module = vlm_module
        self.processing_class = processing_class
        self.vllm_client = vllm_client
        self.sampling_params=sampling_params
        self.num_generations=num_generations
        self.multi_turn = multi_turn
        self.max_tool_uses = max_tool_uses
        self.strict_tool_extraction = strict_tool_extraction
        self.save_path = save_path
        self.metrics = {metric: [] for metric in metrics}
        #self.eval_path = None


    def evaluate(self, dataset:dict, prompt_type, batch_size, exist_ok, tools:list[Tool], qwen_3p5_eval:bool=False,
                 port:int=8000):

        # _tool_fixed_crop
        #self.eval_path = os.path.join(self.save_path, f"dataset_{dataset['dataset_name']}_prompt_{prompt_type}")

        hf_dataset = self.preprocess_dataset(dataset, prompt_type, tools)

        target_columns = ["prompt", "image_path", "solution", "accu_reward_method", "bbox"]
        available_columns =[col for col in target_columns if col in hf_dataset.column_names]

        hf_dataset = hf_dataset.select_columns(available_columns)

        if batch_size is None:
            batch_size = len(hf_dataset)
        dataloader = DataLoader(
            hf_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda x: x
        )
        self.metrics = {metric: [] for metric in self.metrics.keys()}
        logger.info(f"in evaluate: len(dataloader): {len(dataloader)}")
        batch_no = 0
        for batch in dataloader:
            logger.info(f"batch: {batch_no}/{len(dataloader)}")
            if batch_no >= 0:

                t0 = time.time()
                if qwen_3p5_eval:
                    self.vllm_server_inference(inputs=batch, tools=tools, save_path=self.save_path,
                                               max_tool_uses=self.max_tool_uses, port=port)
                else:
                    self._generate_and_score_completions(inputs=batch, tools=tools)
                t1 = time.time()
                logger.info(f"time for batch of size {batch_size}: {t1-t0}")
            #logger.info(f"metrics after batch {batch_no}: {self.metrics}")
            batch_no += 1
        logger.info(f"before dump")
        json.dump(self.metrics, open(os.path.join(self.save_path, "full_results.json"), "w"))

    def preprocess_dataset(self, dataset, prompt_type, tools):
        if dataset["dataset_name"] in ["pixel_reasoner", 'pixel_reasoner_vstar', 'pixel_reasoner_infovqa',
                                       "hr_bench_4k", "hr_bench_8k"] or dataset["dataset_name"].startswith("muffin_chihuahua") or dataset["dataset_name"].startswith("mme"):
            #logger.info(f"prompt type: {prompt_type}")
            #logger.info(f"tool name: {tool_name}")
            logger.info(f"in preprocess_dataset, tools={tools}")
            question_prompt = get_question_template(task_type=prompt_type)
            if tools is not None:
                if len(tools) == 1:
                    tool = tools[0]
                    tool_name = "crop_image" if tool.name == "crop_image_normalized" else tool.name
                    question_prompt = question_prompt.replace("{tool_name}", tool_name)
                else:
                    for idx, tool in enumerate(tools):
                        replacement_string = f"tool_name{idx+1}"
                        tool_name = "crop_image" if tool.name == "crop_image_normalized" else tool.name
                        question_prompt = question_prompt.replace(f"{{{replacement_string}}}", tool_name)
            #logger.info(f"question prompt: {question_prompt}")
            #question_prompt = question_prompt
            #logger.info(f"question prompt after tool_name insertion: {question_prompt}")
            preprocessed_dataset = prepare_data([dataset["dataset_name"]],
                                   [dataset["data_files"]],
                                   [dataset["image_folders"]],
                                   question_prompt,
                                   None)
            return preprocessed_dataset

        else:
            raise NotImplementedError(f"the requested dataset {dataset} is not implemented. Choose from 'pr_train_dataset',"
                                      f" 'pixel_reasoner_vstar', 'pixel_reasoner_infovqa'.")

    def _generate_and_score_completions(self, inputs: list[dict[str, str]], tools:list[Tool]):
        # only used in image and text rethink
        format_prompt = "As before, first output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."

        history = inputs.copy()

        logger.info(f"in _generate_and_score_completions: len(inputs): {len(inputs)}")

        #prompts = [x["prompt"] for x in inputs]

        tool_list = [tool.get_tool_dict() for tool in tools] if tools is not None else None

        # Handle both pre-loaded images and image paths
        image_paths = []
        for x in inputs:
            if "image_path" in x and x["image_path"] is not None:
                for p in x["image_path"]:
                    image_paths.append(p)
                assert len(x["image_path"]) == 1, f"Example {x} contains more than one image which is not supported atm"
            else:
                raise ValueError(f"sample {x} does not contain any image path")

        # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
        all_histories = []
        solutions = []
        prompts = []
        accu_rewards = []

        has_bboxes = "bbox" in inputs[0].keys()
        gold_bboxes = []

        for h in history:
            for _ in range(self.num_generations):
                all_histories.append(copy.deepcopy(h))
                solutions.append(h["solution"])
                prompts.append(h["prompt"])
                accu_rewards.append(h["accu_reward_method"])
                if has_bboxes:
                    gold_bboxes.append(h["bbox"])

        reward_kwargs = {"accu_reward_method": accu_rewards}

        all_image_paths = []
        for i in image_paths:
            for _ in range(self.num_generations):
                all_image_paths.append(copy.deepcopy(i))

        no_conversations = len(all_histories)

        multi_turn_manager = MultiTurn(no_conversations,
                                       processor=self.processing_class,
                                       tools=tools)
        logger.info(f"multi turn manager batch size: {multi_turn_manager.batch_size}")
        multi_turn_manager.add_initial_user_prompt([h["prompt"][0] for h in all_histories], all_image_paths)
        logger.info(f"self.num_generations before get_sequences: {self.num_generations}")
        full_token_seq = multi_turn_manager.get_sequences(type="id", add_assistant_start=True,
                                                          full_image_pad=False)#[::self.num_generations]

        input_text = multi_turn_manager.get_sequences(type="text", add_assistant_start=True, full_image_pad=False)
        self.metrics["query"] += input_text
        input_text = input_text[::self.num_generations]

        logger.info(f"input text before generation: {input_text}")

        all_multimodal_token_inputs = [{"prompt_token_ids": full_token_seq[i],
                                        "image_path": all_image_paths[i]}
                                       for i in range(0, len(all_image_paths), self.num_generations)]

        #logger.info(f"all_multimodal_token_inputs: {all_multimodal_token_inputs}")

        max_generation_attempts = 5
        conv_round = 0
        max_conv_rounds = 10  # just for safety that we don't get stuck in endless loop. max tool calls should prevent it

        while not all(multi_turn_manager.is_finished) and conv_round < max_conv_rounds:
            vllm_generation_has_worked = False
            attempts = 0
            while (not vllm_generation_has_worked) and (attempts <= max_generation_attempts):
                try:
                    logger.info(f"before num_gen change: conv round {conv_round}, sampling_params: {self.sampling_params}")
                    if conv_round == 0:
                        sampling_params = self.sampling_params
                        sampling_params.n = self.num_generations
                    else:
                        sampling_params = self.sampling_params
                        sampling_params.n = 1

                    logger.info(f"after num_gen change: sampling params: {sampling_params}")
                    completion_ids_token_based = self.vllm_client.generate(prompts=all_multimodal_token_inputs,
                                                                           sampling_params=sampling_params)
                    vllm_generation_has_worked = True
                except Exception as e:
                    logger.info(f"Generation {attempts} failed with exception:", e)
                    attempts += 1
                    try:
                        self.vllm_client.check_server(total_timeout=10.0)
                        logger.info(f"vLLM server is up!")
                    except ConnectionError:
                        raise ConnectionError("vLLM Server is down, aborting training.")
            logger.info(f"len of vllm generation: {len(completion_ids_token_based)}")
            completions_token_based = self.processing_class.batch_decode(completion_ids_token_based,skip_special_tokens=True)
            logger.info(f"completions from conv round {conv_round}: {completions_token_based}")

            multi_turn_manager.add_model_reply(completion_ids_token_based,
                                               mapping=multi_turn_manager.get_ids(is_finished=False))

            if self.multi_turn is None:
                multi_turn_manager.is_finished = [True for _ in range(no_conversations)]
            elif self.multi_turn == "text":
                text_rethink = "Are you sure? Think again. "
                if conv_round == 0:

                    multi_turn_manager.add_user_message(
                        texts=[text_rethink + format_prompt for _ in range(no_conversations)])
                else:
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
                    multi_turn_manager.is_finished = [True for _ in range(no_conversations)]
            elif self.multi_turn == "tool":
                #logger.info(f"before handle tool call.. sleep")
                #time.sleep(100)
                multi_turn_manager.handle_tool_call(save_path=os.path.join(self.save_path, "tool_calls"),
                                                    step=0, strict_extraction=self.strict_tool_extraction,
                                                    finish_after_wrong_tool_call=False)


                for idx in range(no_conversations):
                    if self.max_tool_uses is not None and multi_turn_manager.get_no_tool_calls(
                            idx) > self.max_tool_uses:
                        multi_turn_manager.is_finished[idx] = True


            full_token_seq = multi_turn_manager.get_sequences(type="id", add_assistant_start=True,
                                                              full_image_pad=False, ignore_finished=True)
            logger.info(f"full_token_seq: {full_token_seq}")
            input_text = multi_turn_manager.get_sequences(type="text", add_assistant_start=True,
                                                          full_image_pad=False, ignore_finished=True)
            logger.info(f"input_text for conv_round {conv_round}: {input_text}")
            image_paths = multi_turn_manager.get_image_paths(ignore_finished=True, flatten=False)
            logger.info(f"image_paths: {image_paths}")

            all_multimodal_token_inputs = [{"prompt_token_ids": full_token_seq[i],
                                            "image_path": image_paths[i]}
                                           for i in range(len(image_paths))]

            conv_round += 1
            logger.info(f"all_multimodal_token_inputs for conv_round {conv_round}: {all_multimodal_token_inputs}")

        logger.info(f"after multi-turn generation")
        overall_tools_used = [multi_turn_manager.get_no_tool_calls(idx) for idx in range(no_conversations)]

        attempted_tool_uses = [multi_turn_manager.get_no_tool_calls(idx, type="attempt") for idx in
                               range(no_conversations)]

        if "bbox" in inputs[0].keys():
            bboxes = multi_turn_manager.get_absolute_bboxes(flatten=False)
            logger.info(f"bboxes: {bboxes}")


            logger.info(f"gold_bboxes: {gold_bboxes}")

            overlap_metric_names = ["ious", "precision", "recall"]
            for overlap_metric_idx, overlap_metric_name in enumerate(overlap_metric_names):
                overlap_metrics = []
                for idx in range(len(gold_bboxes)):
                    if gold_bboxes[idx] is None:
                        logger.info(f"gold_bbox is None for sample {idx}, skipping")
                        continue
                    for tool_use_idx in range(len(bboxes[idx])):
                        overlap_metric = calculate_overlap_metrics(bboxes[idx][tool_use_idx], gold_bboxes[idx])[overlap_metric_idx]
                        if len(overlap_metrics) < tool_use_idx + 1:
                            overlap_metrics.append([overlap_metric])
                        else:
                            overlap_metrics[tool_use_idx].append(overlap_metric)

                logger.info(f"after {overlap_metric_name} calculation: {overlap_metrics}")

                for tool_use_idx in range(len(overlap_metrics)):
                    if len(self.metrics[overlap_metric_name]) < tool_use_idx + 1:
                        self.metrics[overlap_metric_name].append(overlap_metrics[tool_use_idx])
                    else:
                        self.metrics[overlap_metric_name][tool_use_idx] += overlap_metrics[tool_use_idx]
        else:
            logger.info(f"Skipping iou calculation for samples where bboxes are not given")

        self.metrics["tool_use"] += overall_tools_used
        self.metrics["attempted_tool_use"] += attempted_tool_uses
        self.metrics["image_sizes"] += multi_turn_manager.get_image_sizes()
        self.metrics["completion_len"] += multi_turn_manager.get_model_generation_ids(lengths=True, flatten=False)

        logger.info(f"overall_tools_used: {overall_tools_used}")

        model_generations = multi_turn_manager.get_model_generations(type="text")

        logger.info(f"model_generations for reward calculation: {model_generations}")

        completions = model_generations.copy()
        logger.info(f"number of completions: {len(completions)}")
        # this works as apply_chat_template just appends before and after without realizing that there are multiple turns inside
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]


        #solutions = [x['solution'] for x in inputs]

        accuracies = accuracy_reward(prompts=prompts, completions=completions,
                                             solution = solutions,
                                             cutoff = 1.0,
                                             **reward_kwargs)

        accuracies = np.array(accuracies)
        self.metrics["accuracy"] += accuracies.tolist()
        self.metrics["model_answer"] += completions
        self.metrics["model_answer_tokenized"] += multi_turn_manager.get_model_generations(type="id")
        self.metrics["solution"] += solutions
        self.metrics["images"] += all_image_paths



    def vllm_server_inference(self, inputs, tools, save_path, max_tool_uses, port:int=8000):

        is_async = True
        inputs = inputs

        vllm_server = VLLM_Server(is_async, max_concurrency=16, port=port)

        image_paths = []
        for x in inputs:
            if "image_path" in x and x["image_path"] is not None:
                for p in x["image_path"]:
                    image_paths.append(p)
                assert len(x["image_path"]) == 1, f"Example {x} contains more than one image which is not supported atm"
            else:
                raise ValueError(f"sample {x} does not contain any image path")

        samples = []

        for i, input in enumerate(inputs):
            only_text = ""
            for c in input["prompt"][0]["content"]:
                if c["type"] == "text":
                    only_text = c["text"]

            only_text = only_text.removeprefix("<image>\n")

            if only_text == "":
                raise ValueError(f"empty prompt in input {input}")

            logger.info(f"message {i}: {only_text}")

            samples.append({"text": only_text, "image_path": image_paths[i], "sample_idx": i})

        results = vllm_server.process_data(samples=samples, tools=tools, save_path=save_path, max_rounds=max_tool_uses)
        #{"status": {"success after": no_turns + 1},
        # "all_messages": all_messages_with_image_path}
        logger.info(f"full results: {results}")
        tool_uses = [len(r["tool_calls"]) for r in results]
        #for r in results:

        #    if "success after" in r["status"]:
        #        tool_uses.append(r["status"]["success after"]-1)
        #    elif "error after" in r["status"]:
        #        tool_uses.append(r["status"]["error after"])
        #    #elif "generation error during" in r["status"]:
        #    #    tool_uses.append(r["status"]["generation error during"]-1)
        #    else:
        #        raise ValueError(f"unknown status: {r['status']}")

        model_answers = []
        for r in results:
            model_answer = None
            for msg in r["all_messages"]:
                if isinstance(msg, ChatCompletionMessage) and msg.content is not None:
                    model_answer=msg.content
            model_answers.append(model_answer if model_answer is not None else "")

        completions = model_answers
        logger.info(f"number of completions: {len(completions)}")
        manual_bbox_change_count = 0
        new_completions = []
        # this works as apply_chat_template just appends before and after without realizing that there are multiple turns inside
        if is_conversational(inputs[0]):
            for completion in completions:
                new_completion = re.sub(r'\(([A-Za-z0-9{,3}])\)$', r'\\boxed{\1}', completion)
                if not new_completion == completion:
                    manual_bbox_change_count += 1
                new_completions.append([{"role": "assistant", "content": new_completion}])


        completions = new_completions

        # solutions = [x['solution'] for x in inputs]
        solutions = [inp["solution"] for inp in inputs]
        prompts = [inp["prompt"] for inp in inputs]
        logger.info(f"before acc reward: prompts: {prompts} \n\n completions: {completions} \n\n solutions: {solutions}")
        logger.info(f"manual_bbox_change_count: {manual_bbox_change_count}")

        accuracies = accuracy_reward(prompts=prompts, completions=completions,
                                     solution=solutions,
                                     cutoff=1.0,
                                     accu_reward_method=[inp["accu_reward_method"] for inp in inputs])

        self.metrics["accuracy"]+=accuracies
        self.metrics["tool_use"]+=tool_uses

        has_bboxes = "bbox" in inputs[0].keys()

        if has_bboxes:
            gold_bboxes = [inp["bbox"] for inp in inputs]

            global_pred_bboxes = []
            for i in range(len(inputs)):
                pred_bboxes = []
                for tc in results[i]["tool_calls"]:
                    target = tc["target"]
                    pred_bbox = tc["bbox_wrt_target"]
                    if target != 0:
                        # the coords are wrt the cropped image and have to be shifted by its upper right corner
                        new_bbox = (pred_bboxes[target - 1][0] + pred_bbox[0],
                                    pred_bboxes[target - 1][1] + pred_bbox[1],
                                    pred_bboxes[target - 1][0] + pred_bbox[2],
                                    pred_bboxes[target - 1][1] + pred_bbox[3])
                        pred_bboxes.append(new_bbox)
                    else:
                        pred_bboxes.append(pred_bbox)
                global_pred_bboxes.append(pred_bboxes)

            bboxes = global_pred_bboxes

            logger.info(f"gold bboxes: {gold_bboxes}")
            logger.info(f"bboxes: {bboxes}")

            overlap_metric_names = ["ious", "precision", "recall"]
            for overlap_metric_idx, overlap_metric_name in enumerate(overlap_metric_names):
                overlap_metrics = []
                for idx in range(len(gold_bboxes)):
                    if gold_bboxes[idx] is None:
                        logger.info(f"gold_bbox is None for sample {idx}, skipping")
                        continue
                    overlap_metrics_per_sample = []
                    for tool_use_idx in range(len(bboxes[idx])):
                        overlap_metric = calculate_overlap_metrics(bboxes[idx][tool_use_idx], gold_bboxes[idx])[overlap_metric_idx]
                        overlap_metrics_per_sample.append(overlap_metric)
                    overlap_metrics.append(overlap_metrics_per_sample)
                        #if len(overlap_metrics) < tool_use_idx + 1:
                        #    overlap_metrics.append([overlap_metric])
                        #else:
                        #    overlap_metrics[tool_use_idx].append(overlap_metric)

                logger.info(f"after {overlap_metric_name} calculation: {overlap_metrics}")

                self.metrics[overlap_metric_name] += overlap_metrics

                #for tool_use_idx in range(len(overlap_metrics)):
                #    if len(self.metrics[overlap_metric_name]) < tool_use_idx + 1:
                #        self.metrics[overlap_metric_name].append(overlap_metrics[tool_use_idx])
                #    else:
               #         self.metrics[overlap_metric_name][tool_use_idx] += overlap_metrics[tool_use_idx]






        # results is a list of dicts, where each dict has the keys 'content' (str), 'reasoning' (str) , 'tool_calls' (list)
        return results



def evaluation_process(model_path, model_class, output_path:str, tensor_parallel_size: int, image_limit: int, max_tool_uses:int,
                       strict_tool_extraction: bool,
                       dataset:dict, prompt_type: str,
                       max_tokens_per_reply: int,
                       temperature: float = 0.0,
                       num_generations: int=1,
                       batch_size:int=1000, no_vllm:bool=False,
                       enforce_eager:bool=False,
                       tool_args:list[dict]=None,
                       mm_processor_kwargs:dict=None,
                       qwen_3p5_eval:bool=False,
                       port:int=8000
                       ):
    if tool_args is None or (len(tool_args) == 1 and tool_args[0] is None) or (len(tool_args) == 1 and tool_args[0]["tool_name"] == ""):
        tools = None
    else:
        tools = []

        for tool_arg in tool_args:
            tools.append(Tool(name=tool_arg["tool_name"],
                              template_name=tool_arg["tool_template"],
                              json_customization=tool_arg["tool_json_customization"],

                              message=Message(tool_arg["tool_message"]),

                              tool_hparams=tool_arg["tool_hparams"]))

    logger.info(f"tools: {tools}")

    logger.info(f"batch size in evaluation_process: {batch_size}")

    processing_class = AutoProcessor.from_pretrained(model_class)
    processing_class.chat_template = json.load(open("qwen_chat_template_tool.json", "r"))["chat_template"]

    reuse_existing_vllm_server = True
    vllm_server_up = False

    if no_vllm:
        llm_engine = None
    elif qwen_3p5_eval:
        llm_engine = None
        try:
            urllib.request.urlopen(f"http://0.0.0.0:{port}/ping").read()
            vllm_server_up = True
            logger.info(f"server exists already!")
        except urllib.error.URLError:
            if not reuse_existing_vllm_server:
                raise RuntimeError("VLLM server already exists and reusing is disabled!")

        if not vllm_server_up:

            cmd = f"vllm serve Qwen/Qwen3.5-9B --port {port} --tensor-parallel-size {tensor_parallel_size} --max-model-len 262144 --reasoning-parser qwen3"

            if tools is not None:
                cmd += f" --enable-auto-tool-choice --tool-call-parser qwen3_coder"

            #if mm_processor_kwargs is not None:
            #    cmd += f" --mm-processor-kwargs '{mm_processor_kwargs}'"

            server = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

            for i in range(60):
                try:
                    urllib.request.urlopen(f"http://0.0.0.0:{port}/ping").read()
                    break
                except urllib.error.URLError:
                    logger.info(f"Server not ready, retrying in 10 seconds...")
                    time.sleep(10)

            logger.info(f"Server started successfully")
    else:
        llm_engine = VLLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=0.9,
            dtype=torch.bfloat16,
            max_model_len=None,
            enforce_eager=enforce_eager,
            limit_mm_per_prompt={"image": image_limit,
                                 "video": 0},
            mm_processor_kwargs=mm_processor_kwargs if mm_processor_kwargs is not None else None,
            enable_prefix_caching=False,
        )

    # Sampling parameters
    sampling_params = SamplingParams(
        n=num_generations,
        temperature=temperature,
        top_p=1.0,
        top_k=-1,
        min_p=0.0,
        max_tokens=max_tokens_per_reply,
        stop_token_ids=[151643, 151658]
    )
    logger.info(f"before initiating evaluator: num_generations: {num_generations}")
    evaluator = Evaluator(
        vlm_module=Qwen2VLModule,
        processing_class=processing_class,
        sampling_params=sampling_params,
        num_generations=num_generations,
        multi_turn="tool",
        max_tool_uses=max_tool_uses,
        strict_tool_extraction=strict_tool_extraction,
        save_path=output_path,
        vllm_client=llm_engine,
        metrics=["accuracy", "tool_use", "attempted_tool_use", "query", "solution",
                 "model_answer", "model_answer_tokenized", "images", "image_sizes", "completion_len", "ious", "precision", "recall"],
    )
    
    t0 = time.time()
    logger.info(f"Starting evaluation process for {dataset['dataset_name']}...")
    evaluator.evaluate(dataset=dataset,
                       prompt_type=prompt_type,
                       batch_size=batch_size,
                       exist_ok=False,
                       tools=tools,
                       qwen_3p5_eval=qwen_3p5_eval,
                       port=port)
    t1 = time.time()
    logger.info(f"Process took {t1 - t0} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate VLM model performance')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--model_class', type=str, required=True, help='Model class name')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output folder')
    parser.add_argument('--tensor_parallel_size', type=int, default=1, help='Tensor parallel size')
    parser.add_argument('--image_limit', type=int, default=1, help='Maximum number of images per prompt')
    parser.add_argument('--max_tool_uses', type=int, default=5, help='Maximum number of tool uses')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--data_filepath', type=str, required=True, help='Path to data files')
    parser.add_argument('--image_filepath', type=str, required=True, help='Path to image folder')
    parser.add_argument('--prompt_type', type=str, default=None, help='Type of prompt to use')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for evaluation')
    parser.add_argument('--no_vllm', action='store_true', help='Whether to build vLLM engine. For debugging')
    parser.add_argument('--enforce_eager', action='store_true', help='enforce eager in vllm. For debugging')
    parser.add_argument('--tool_configs', type=str, default="no_tool", help='tools to use, comma separated')
    parser.add_argument('--max_pixels', type=int, default=None, help='Maximum number of pixels per image')
    parser.add_argument('--min_pixels', type=int, default=None, help='Minimum number of pixels per image')
    parser.add_argument('--bbox_type', type=str, default=None, help='type of bbox coords to accept. absolute or relative')
    parser.add_argument('--strict_tool_extraction', type=bool, default=None, help='whether tool extraction is strict or not')
    parser.add_argument('--tool_padding', type=float, default=0.1, help='ratio how much the tool bbox should be increased')
    parser.add_argument('--tool_adaptive_padding_threshold', type=int, default=None, help='upper bound (in px) of tool use padding')
    parser.add_argument('--max_tokens_per_reply', type=int, default=256, help='Maximum number of tokens per reply')
    parser.add_argument('--temperature', type=float, default=0.0, help='sampling with temperature, 0.0 is greedy')
    parser.add_argument('--num_generations', type=int, default=1, help='How many generations. Value > 1 is only meaningful for temperature > 0')
    parser.add_argument('--qwen_3p5_eval', action='store_true', help='Whether Qwen 3.5 should be evaluated')
    parser.add_argument('--port', type=int, default=8000, help='port for the vLLM server, only used for q3p5 eval')

    args = parser.parse_args()

    # Get logger for this module
    logger = setup_project_logging(log_file=os.path.join(args.output_path, "evaluation.log"))

    dataset = {
        "dataset_name": args.dataset_name,
        "data_files": args.data_filepath,
        "image_folders": args.image_filepath
    }

    #tool_configs = args.tool_configs.split(",")
    #tool_config = []
    tool_configs = [TOOL_CONFIGS[tool_config] for tool_config in args.tool_configs.split(",")]

    if args.prompt_type is None:
        logger.info([tool_config["prompt_type"] for tool_config in tool_configs])
        assert len(list(set([tool_config["prompt_type"] for tool_config in tool_configs]))) == 1, "The implicit prompt_type is ambiguous. Please specify it explicitly."
        args.prompt_type = tool_configs[0]["prompt_type"]

    for tool_config in tool_configs:

        adaptive_padding_threshold = args.tool_adaptive_padding_threshold
        if adaptive_padding_threshold is None:
            adaptive_padding_threshold = 600
        if adaptive_padding_threshold < 0:
            adaptive_padding_threshold = None

        tool_config["tool_hparams"] = {"max_pixels": args.max_pixels, "min_pixels": args.min_pixels,
                                       "bbox_type": args.bbox_type,
                                       "padding": (args.tool_padding, args.tool_padding),
                                       "adaptive_padding_threshold": adaptive_padding_threshold
                                       }

    mm_processor_kwargs = {}
    if args.max_pixels is not None:
        mm_processor_kwargs["max_pixels"] = args.max_pixels
    if args.min_pixels is not None:
        mm_processor_kwargs["min_pixels"] = args.min_pixels

    if mm_processor_kwargs == {}:
        mm_processor_kwargs = None



    batch_size = args.batch_size // args.num_generations



    evaluation_process(
        args.model_path,
        args.model_class,
        args.output_path,
        args.tensor_parallel_size,
        args.image_limit,
        args.max_tool_uses,
        args.strict_tool_extraction,
        dataset,
        args.prompt_type,
        args.max_tokens_per_reply,
        args.temperature,
        args.num_generations,
        batch_size,
        args.no_vllm,
        args.enforce_eager,
        tool_configs,
        mm_processor_kwargs,
        args.qwen_3p5_eval,
        args.port
    )
