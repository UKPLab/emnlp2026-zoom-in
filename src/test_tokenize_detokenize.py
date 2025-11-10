from transformers import AutoProcessor
from trl.data_utils import maybe_apply_chat_template
from PIL import Image




def get_ids(test:dict):
    if not isinstance(test, list):
        test = [test]

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    default_image = "/pfss/mlde/workspaces/mlde_wsp_KIServiceCenter/helm/datasets/pixel_reasoner/RL_data_without_video/images/a15ab079-1311-444c-8d50-1ba3544b6e06-0.jpg"

    wrapped = maybe_apply_chat_template({"prompt": test}, processor,
                              add_generation_prompt=None,
                              return_assistant_tokens_mask=False, tools=None)["prompt"]
    print(repr(wrapped))

    tokenized = processor(
        text=wrapped,
        images=Image.open(default_image),
        return_tensors=None,
        padding=False,
        add_special_tokens=False,
        return_offsets_mapping=False
    )["input_ids"]
    print(tokenized)

    return tokenized




if __name__ == "__main__":
    user_simple = {
        'role': 'user',
        'content': [
            {'type': 'image', 'text': None},
            {'type': 'text', 'text': "Hello World"}
        ]
    }
    assistant_simple = {'role': 'assistant',
                        'content': [
                            {'type': 'text', 'text': "Hello World"}
                        ]
                        }
    user_ids = get_ids(user_simple)[0]
    assistant_ids = get_ids(assistant_simple)[0]
    print(f"user_ids: {user_ids}")
    print(f"assistant_ids: {assistant_ids}")
    concat_ids = user_ids + assistant_ids
    print(len(concat_ids))

    joint_ids = get_ids([user_simple, assistant_simple])[0]
    print(len(joint_ids))


