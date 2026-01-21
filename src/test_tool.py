import os
import requests
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['VLLM_USE_V1'] = '0'

from trl.extras.vllm_client import VLLMClient

import PIL.Image

from vllm.engine.arg_utils import EngineArgs

from vllm import LLM, AsyncLLMEngine, LLMEngine, SamplingParams
from transformers import AutoProcessor

from openai import OpenAI

import json

def get_current_temperature(location: str, unit: str = "celsius"):
    """Get current temperature at a location.

    Args:
        location: The location to get the temperature for, in the format "City, State, Country".
        unit: The unit to return the temperature in. Defaults to "celsius". (choices: ["celsius", "fahrenheit"])

    Returns:
        the temperature, the location, and the unit in a dict
    """
    return {
        "temperature": 26.1,
        "location": location,
        "unit": unit,
    }


def get_temperature_date(location: str, date: str, unit: str = "celsius"):
    """Get temperature at a location and date.

    Args:
        location: The location to get the temperature for, in the format "City, State, Country".
        date: The date to get the temperature for, in the format "Year-Month-Day".
        unit: The unit to return the temperature in. Defaults to "celsius". (choices: ["celsius", "fahrenheit"])

    Returns:
        the temperature, the location, the date and the unit in a dict
    """
    return {
        "temperature": 25.9,
        "location": location,
        "date": date,
        "unit": unit,
    }

def show_image(file_path:str):
    """
    Returns the input image. It is then fed to the MLLM again so it can spot details that were missed earlier.
    :return: The input image.
    """
    return file_path



def get_function_by_name(name):
    if name == "show_image":
        return show_image

"""{
        "type": "function",
        "function": {
            "name": "get_current_temperature",
            "description": "Get current temperature at a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": 'The location to get the temperature for, in the format "City, State, Country".',
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": 'The unit to return the temperature in. Defaults to "celsius".',
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_temperature_date",
            "description": "Get temperature at a location and date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": 'The location to get the temperature for, in the format "City, State, Country".',
                    },
                    "date": {
                        "type": "string",
                        "description": 'The date to get the temperature for, in the format "Year-Month-Day".',
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": 'The unit to return the temperature in. Defaults to "celsius".',
                    },
                },
                "required": ["location", "date"],
            },
        },
    },"""


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "show_image",
            "description": "The input image gets repeated.",
            "parameters": {

            },
        },
    },
]

# "file:///pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/focusreason/chartqa_original/train_full/png/png/two_col_22791.png"

img_path_1 = "file:///pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/focusreason/chartqa_original/train_full/png/png/two_col_22791.png"
img_path_2 = "file:///pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/focusreason/chartqa_original/train_full/png/png/two_col_651.png"


MESSAGES = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user",  "content": [{
                        "type": "image_url",
                        "image_url": {"url": img_path_1},
                    },
                    {"type": "text", "text": "Describe the image in detail. If you want to look at it again, call the tool."}]},
#[{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
#    {"role": "user",  "content": [{
#                        "type": "image_url",
#                        "image_url": {"url": img_path_2},
#                    },
#                    {"type": "text", "text": "Describe the image in detail. If you want to look at it again, call the tool."}]}]
]


if __name__ == '__main__':


    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"

    try:
        response = requests.get(f"http://localhost:8000/get_tensor_parallel_size/")
        tensor_parallel_size = response.json()
    except requests.RequestException as e:
        print(f"Failed to get tensor parallel size: {e}")
        tensor_parallel_size = None

    print(f"Tensor parallel size: {tensor_parallel_size}")
    #sys.exit()

    #vllm_client = VLLMClient(
    #    "0.0.0.0", "8000", connection_timeout=10
    #)

    client = OpenAI(
        api_key=openai_api_key,
        base_url="http://0.0.0.0:8000/"#openai_api_base,
    )

    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    #model_name = "Qwen/Qwen2.5-7B-Instruct"

    tools = TOOLS
    messages = MESSAGES[:]

    #print(f"Initial messages : {messages}")

    tool_called = False
    turn = 0
    max_turns = 5
    while not tool_called and turn < max_turns:

        #completion_ids = vllm_client.generate_from_multimodal_input(
        #    prompts=messages,
        #    n=1,
        #    repetition_penalty=1.0,
        #temperature= 1.0,
        #top_p= 1.0,
        #top_k= -1,
        #min_p= 0.0,
        #max_tokens= 16,
        #guided_decoding_regex = None, tools = tools
        #)
        #print(f"completion_ids: {completion_ids}")
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.7,
            top_p=0.8,
            max_tokens=512,
            extra_body={
                "repetition_penalty": 1.05,
            },
        )

        messages.append(response.choices[0].message.model_dump())
        print(f"last message: {messages[-1]}")
        if tool_calls := messages[-1].get("tool_calls", None):
            for tool_call in tool_calls:
                call_id: str = tool_call["id"]
                if fn_call := tool_call.get("function"):
                    fn_name: str = fn_call["name"]
                    if fn_name == "show_image":
                        image_file_path = get_function_by_name(fn_name)(img_path_1)
                        fn_res = [{
                                "type": "image_url",
                                "image_url": {
                                    "url": image_file_path},
                                },
                                {"type": "text", "text": "Here is the image again."}]

                    else:
                        fn_args: dict = json.loads(fn_call["arguments"])

                        fn_res: str = json.dumps(get_function_by_name(fn_name)(**fn_args))

                    messages.append({
                        "role": "tool",
                        "content": fn_res,
                        "tool_call_id": call_id,
                    })
            tool_called = True
        turn += 1

        #print(f"messages after turn {turn} : {messages}")

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        tools=tools,
        # tool_choice="auto",
        temperature=0.7,
        top_p=0.8,
        max_tokens=512,
        extra_body={
            "repetition_penalty": 1.05,
        },
    )
    messages.append(response.choices[0].message.model_dump())
    print(f"final messages: {messages}")


    """#model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"

    llm = LLM(model=model_name,
              allowed_local_media_path=True,
              enforce_eager=True)

    processor = AutoProcessor.from_pretrained(model_name)
    prompts = []
    for prompt_text in ["What is the difference between the images?", "Beschreibe das Bild."]:

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/focusreason/chartqa_original/train_full/png/png/two_col_22791.png",
                    },
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        print(text)

        prompts.append({"prompt": text,
                "multi_modal_data":
                    {"image": PIL.Image.open(
                        "/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/focusreason/chartqa_original/train_full/png/png/two_col_22791.png")}})

    output = llm.generate(prompts=prompts)

    print(output)


    print("Done!")"""
