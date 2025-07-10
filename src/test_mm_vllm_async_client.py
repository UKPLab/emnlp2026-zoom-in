import PIL.Image
from transformers import AutoProcessor
import requests

model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
processor = AutoProcessor.from_pretrained(model_name)


messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/helm/datasets/focusreason/chartqa_original/train_full/png/png/two_col_22791.png",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

prompts = [{"prompt": text,
            "image_path": "/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/helm/datasets/focusreason/chartqa_original/train_full/png/png/two_col_22791.png"
            }]



url = "http://127.0.0.1:8000/generate/"

response = requests.post(url, json={"input": prompts})
print(response.json())

print(processor.decode(response.json()["output"][0]))