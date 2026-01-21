import os
os.environ['VLLM_USE_V1'] = '0'
from transformers import AutoProcessor

from vllm import LLM
import PIL.Image

model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
image_base_path: str = "/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/focusreason/chartqa_original/train_full/png/png/"

processor = AutoProcessor.from_pretrained(model_name)




def create_message(agent:str, prompt:str, image:str=None):
    if image is None:
        message = {
                    "role": agent,
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                }
    else:
        message = {
            "role": agent,
            "content": [
                {
                    "type": "image",
                    "image": os.path.join(image_base_path, image),
                },
                {"type": "text", "text": prompt},
            ],
        }
    return message

def make_hf_ready(messages: list[dict]):
    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

def make_vllm_ready(hf_ready_text: str, image_paths = list[str]):
    image_paths = [PIL.Image.open(os.path.join(image_base_path, image_path)) for image_path in image_paths]
    if len(image_paths) == 1:
        image_paths = image_paths[0]
    return {"prompt": hf_ready_text,
                    "multi_modal_data":
                        {"image": image_paths}}

if __name__ == "__main__":

    llm = LLM(model=model_name,
              allowed_local_media_path=True,
              enforce_eager=True, limit_mm_per_prompt={"image":2, "video":0},
              enable_prefix_caching=True)

    prompt_image_pairs = [
        {"prompt": "Describe the image.", "image": "two_col_22791.png", "history":[]},
        {"prompt": "Beschreibe das Bild.", "image": "two_col_22791.png", "history":[]},
    ]
    vllm_ready = []
    for prompt_image_pair in prompt_image_pairs:
        prompt = prompt_image_pair["prompt"]
        image = prompt_image_pair["image"]
        message = create_message("user", prompt=prompt, image=image)

        prompt_image_pair["history"].append(message)

        hf_ready = make_hf_ready(prompt_image_pair["history"])
        vllm_ready.append(make_vllm_ready(hf_ready, [image]))
    print(f"VLLM ready: {vllm_ready}")

    outputs = llm.generate(prompts=vllm_ready)
    vllm_ready = []
    for idx, prompt_image_pair in enumerate(prompt_image_pairs):
        output=outputs[idx]
        image = prompt_image_pair["image"]

        response = output.outputs[0].text
        print(f"Response after round 1 with id={idx}: {response}")
        message = create_message("model", prompt=response)
        prompt_image_pairs[idx]["history"].append(message)
        message = create_message("user", prompt="Are you sure? Think again.", image=image)
        prompt_image_pairs[idx]["history"].append(message)

        hf_ready = make_hf_ready(prompt_image_pair["history"])
        vllm_ready.append(make_vllm_ready(hf_ready, [image, image]))
    print(f"VLLM ready: {vllm_ready}")

    outputs = llm.generate(prompts=vllm_ready)

    for idx in range(len(prompt_image_pairs)):
        print(f"Response after round 2 with id={idx}: {outputs[idx].outputs[0].text}")

