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

from open_r1.utils.utils import calculate_iou
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

from open_r1.utils.multi_turn_manager import MultiTurn, pad

from vllm.inputs import TokensPrompt



import os



class VLLM:

    def __init__(self, model:str, tensor_parallel_size:int, gpu_memory_utilization,
                 dtype, enable_prefix_caching, max_model_len,
                 enforce_eager, limit_mm_per_prompt, mm_processor_kwargs, revision=None, disable_custom_all_reduce=None):
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
            logger.info(f"inputs_with_image: {inputs_with_image}")
            all_outputs = self.llm.generate(inputs_with_image, sampling_params=sampling_params)

            logger.info(f"after vllm generate")
            #logger.info(f"{all_outputs[0].outputs[0].text}")
            only_text = [output.outputs[0].text for output in all_outputs]
            completion_len = [len(output.outputs[0].token_ids) for output in all_outputs]
            only_tokens = [output.outputs[0].token_ids for output in all_outputs]

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


class Evaluator:

    def __init__(self, vlm_module, processing_class, vllm_client, sampling_params:SamplingParams,
                 multi_turn, max_tool_uses, strict_tool_extraction, save_path, metrics):
        self.vlm_module = vlm_module
        self.processing_class = processing_class
        self.vllm_client = vllm_client
        self.sampling_params=sampling_params
        self.multi_turn = multi_turn
        self.max_tool_uses = max_tool_uses
        self.strict_tool_extraction = strict_tool_extraction
        self.save_path = save_path
        self.metrics = {metric: [] for metric in metrics}
        #self.eval_path = None


    def evaluate(self, dataset:dict, prompt_type, batch_size, exist_ok, tools:list[Tool]):

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
        batch_no = 0
        for batch in dataloader:
            logger.info(f"batch: {batch_no}/{len(dataloader)}")
            if batch_no >= 0:

                t0 = time.time()
                self._generate_and_score_completions(inputs=batch, tools=tools)
                t1 = time.time()
                logger.info(f"time for batch of size {batch_size}: {t1-t0}")
            #logger.info(f"metrics after batch {batch_no}: {self.metrics}")
            batch_no += 1
        logger.info(f"before dump")
        json.dump(self.metrics, open(os.path.join(self.save_path, "full_results.json"), "w"))

    def preprocess_dataset(self, dataset, prompt_type, tools):
        if dataset["dataset_name"] in ["pixel_reasoner", 'pixel_reasoner_vstar', 'pixel_reasoner_infovqa',
                                       "hr_bench_4k", "hr_bench_8k", "mme", "mme_lite"] or dataset["dataset_name"].startswith("muffin_chihuahua"):
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

        prompts = [x["prompt"] for x in inputs]

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
        all_histories = history
        all_image_paths = image_paths

        no_conversations = len(all_histories)

        multi_turn_manager = MultiTurn(no_conversations,
                                       processor=self.processing_class,
                                       tools=tools)

        multi_turn_manager.add_initial_user_prompt([h["prompt"][0] for h in all_histories], all_image_paths)
        full_token_seq = multi_turn_manager.get_sequences(type="id", add_assistant_start=True,
                                                          full_image_pad=False)
        input_text = multi_turn_manager.get_sequences(type="text", add_assistant_start=True, full_image_pad=False)
        self.metrics["query"] += input_text

        logger.info(f"input text before generation: {input_text}")

        all_multimodal_token_inputs = [{"prompt_token_ids": full_token_seq[i],
                                        "image_path": all_image_paths[i]}
                                       for i in range(len(all_image_paths))]

        logger.info(f"all_multimodal_token_inputs: {all_multimodal_token_inputs}")

        max_generation_attempts = 5
        conv_round = 0
        max_conv_rounds = 10  # just for safety that we don't get stuck in endless loop. max tool calls should prevent it

        while not all(multi_turn_manager.is_finished) and conv_round < max_conv_rounds:
            vllm_generation_has_worked = False
            attempts = 0
            while (not vllm_generation_has_worked) and (attempts <= max_generation_attempts):
                try:
                    completion_ids_token_based = self.vllm_client.generate(prompts=all_multimodal_token_inputs,
                                                                           sampling_params=self.sampling_params)
                    vllm_generation_has_worked = True
                except Exception as e:
                    logger.info(f"Generation {attempts} failed with exception:", e)
                    attempts += 1
                    try:
                        self.vllm_client.check_server(total_timeout=10.0)
                        logger.info(f"vLLM server is up!")
                    except ConnectionError:
                        raise ConnectionError("vLLM Server is down, aborting training.")

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

            gold_bboxes = [x['bbox'] for x in inputs]
            logger.info(f"gold_bboxes: {gold_bboxes}")
    
            ious = []
            for idx in range(len(gold_bboxes)):
                if gold_bboxes[idx] is None:
                    logger.info(f"gold_bbox is None for sample {idx}, skipping")
                    continue
                for tool_use_idx in range(len(bboxes[idx])):
                    iou = calculate_iou(gold_bboxes[idx], bboxes[idx][tool_use_idx])
                    if len(ious) < tool_use_idx + 1:
                        ious.append([iou])
                    else:
                        ious[tool_use_idx].append(iou)

            logger.info(f"after iou calculation: ious: {ious}")

            for tool_use_idx in range(len(ious)):
                if len(self.metrics["ious"]) < tool_use_idx + 1:
                    self.metrics["ious"].append(ious[tool_use_idx])
                else:
                    self.metrics["ious"][tool_use_idx] += ious[tool_use_idx]
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

        reward_kwargs = {"accu_reward_method": [x["accu_reward_method"] for x in inputs]}
        solutions = [x['solution'] for x in inputs]

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


def evaluation_process(model_path, model_class, output_path:str, tensor_parallel_size: int, image_limit: int, max_tool_uses:int,
                       strict_tool_extraction: bool,
                       dataset:dict, prompt_type: str,
                       max_tokens_per_reply: int,
                       batch_size:int=1000, no_vllm:bool=False,
                       enforce_eager:bool=False,
                       tool_args:list[dict]=None,
                       mm_processor_kwargs:dict=None):
    if tool_args is None or (len(tool_args) == 1 and tool_args[0] is None) or (len(tool_args) == 1 and tool_args[0]["tool_name"] == ""):
        tools = None
    else:
        tools = []

        for tool_arg in tool_args:
            tools.append(Tool(name=tool_arg["tool_name"],
                              template_name=tool_arg["tool_template"],
                              json_customization=tool_arg["tool_json_customization"],

                              message=Message(tool_arg["tool_message_image_pos"],
                                         tool_arg["tool_message_text_message"],
                                         tool_arg["tool_message_text_fillers"]),

                              tool_hparams=tool_arg["tool_hparams"]))

    logger.info(f"tools: {tools}")

    processing_class = AutoProcessor.from_pretrained(model_class)
    processing_class.chat_template = json.load(open("qwen_chat_template_tool.json", "r"))["chat_template"]

    if no_vllm:
        llm_engine = None
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
        n=1,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        min_p=0.0,
        max_tokens=max_tokens_per_reply,
        stop_token_ids=[151643, 151658]
    )

    evaluator = Evaluator(
        vlm_module=Qwen2VLModule,
        processing_class=processing_class,
        sampling_params=sampling_params,
        multi_turn="tool",
        max_tool_uses=max_tool_uses,
        strict_tool_extraction=strict_tool_extraction,
        save_path=output_path,
        vllm_client=llm_engine,
        metrics=["accuracy", "tool_use", "attempted_tool_use", "query", "solution",
                 "model_answer", "model_answer_tokenized", "images", "image_sizes", "completion_len", "ious"]
    )
    
    t0 = time.time()
    logger.info(f"Starting evaluation process for {dataset['dataset_name']}...")
    evaluator.evaluate(dataset=dataset,
                       prompt_type=prompt_type,
                       batch_size=batch_size,
                       exist_ok=False,
                       tools=tools)
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
        args.batch_size,
        args.no_vllm,
        args.enforce_eager,
        tool_configs,
        mm_processor_kwargs
    )
