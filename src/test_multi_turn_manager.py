from transformers import AutoProcessor

from open_r1.utils.multi_turn_manager import MultiTurn
from open_r1.utils.logger import setup_project_logging
from vllm import LLM
from vllm.inputs import TokensPrompt
from PIL import Image
import sys

logger = setup_project_logging(None)

if __name__ == "__main__":
    user_simple = {
        'role': 'user',
        'content': [
            {'type': 'image', 'text': None},
            {'type': 'text', 'text': "Hello World"}
        ]
    }
    user_simple_2 = {
        'role': 'user',
        'content': [
            {'type': 'image', 'text': None},
            {'type': 'text', 'text': "Hello World!!!"}
        ]
    }
    default_image = "/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/helm/datasets/pixel_reasoner/RL_data_without_video/images/a15ab079-1311-444c-8d50-1ba3544b6e06-0.jpg"

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

    mt = MultiTurn(batch_size=2, processor=processor, tools=None)

    mt.add_initial_user_prompt([user_simple, user_simple_2],
                               [default_image, default_image])

    #mt.add_model_reply([[9707,4337], [9707,220]])
    print(f"overview: {mt.all_multi_turn}")
    #print(f"all texts: {mt.get_sequences(type="text")}")
    #print(f"all token ids: {mt.get_sequences(type="id")}")

    full_token_seq = mt.get_sequences(type="id", add_assistant_start=True, full_image_pad=False)
    print(f"full token seq: {full_token_seq}")
    print(f"full token seq len: {len(full_token_seq)}; {[len(s) for s in full_token_seq]}")
    vllm_format = [TokensPrompt(prompt_token_ids = full_token_seq[i],
                                multi_modal_data={"image": [Image.open(default_image)]})
                   for i in range(len(full_token_seq))]

    llm = LLM(model="Qwen/Qwen2.5-VL-3B-Instruct")
    output = llm.generate(vllm_format)
    print(output)