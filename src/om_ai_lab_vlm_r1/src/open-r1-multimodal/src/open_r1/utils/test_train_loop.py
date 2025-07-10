from transformers import AutoProcessor
import copy
import time
from multi_turn_handler import Conversations, Prompt, TOOLS
from trl.data_utils import maybe_apply_chat_template


class DummyVLLM:
    def __init__(self):
        pass
    
    def generate(self, inputs, outputs):
        assert isinstance(inputs, list), "inputs must be a list"
        for input in inputs:
            assert isinstance(input, dict), "each input must be a dictionary"
            assert "prompt" in input and "image_path" in input, "input must contain 'prompt' and 'image_path' keys"
            assert isinstance(input["prompt"], str), "prompt must be a string"
            assert isinstance(input["image_path"], list), "image_path must be a list"
            assert all(isinstance(path, str) for path in input["image_path"]), "all image paths must be strings"
        return outputs


def dummy_train_loop(initial_multimodal_inputs, responses):
    vllm = DummyVLLM()
    processing_class = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    # 4 = batch size * generations
    no_conversations = 4
    conversations = Conversations(no_conversations)

    for idx in range(no_conversations):
        mod_idx = idx // 2

        conversations.add_message(
            Prompt(content=copy.deepcopy(initial_multimodal_inputs[mod_idx]["prompt"]), role="user",
                   image_path=initial_multimodal_inputs[mod_idx]["image_path"][0]), idx)

    all_multimodal_inputs = initial_multimodal_inputs

    conv_round = 0
    while not all(conversations.is_finished):


        completions = vllm.generate(all_multimodal_inputs, responses[conv_round])

        #completions = processing_class.batch_decode(outputs, skip_special_tokens=True)
        print(f"completions: {completions}")

        expanded_completions = []
        completion_idx = 0
        for idx in range(no_conversations):
            if conversations.is_finished[idx]:
                expanded_completions.append(None)
            else:
                expanded_completions.append(completions[completion_idx])
                completion_idx += 1

        print(f"expanded_completions: {expanded_completions}")

        for idx, completion in enumerate(expanded_completions):
            if completion is not None:

                conversations.add_message(Prompt(content=[{'text': completion, 'type': 'text'}], role="assistant"), idx)

        print(f"conversations: {conversations.get_image_paths()}")

        conversations.handle_tool_call()

        print(f"conversations after tool handle: {conversations.is_finished}")
        print(f"conversations after tool handle: {conversations.get_image_paths()}")

        # print(f"get full for hf prep: {conversations.get_full_for_hf_prep()}")

        full_conversations_concat = [maybe_apply_chat_template(example, processing_class, tools=TOOLS)["prompt"]
                                                for example in conversations.get_full_for_hf_prep(ignore_finished=True)]

        # print(f"full conversations: {full_conversations_concat}")
        all_multimodal_inputs = [
            {"prompt": p, "image_path": i}
            for p, i in zip(full_conversations_concat, conversations.get_image_paths(ignore_finished=True))
        ]

        print(f"new all multimodal inputs: {all_multimodal_inputs}")
        conv_round += 1
        #time.sleep(10)
    full_generations = [maybe_apply_chat_template(example, processing_class, tools=TOOLS)["prompt"]
                                                for example in conversations.get_full_for_hf_prep(ignore_finished=False)]
    print(f"full generations: {full_generations}")
    model_generations = conversations.get_model_generations()
    print(f"model generations: {model_generations}")



if __name__ == "__main__":
    prompts = ["hello", "world"]
    image_paths = [["path1"], ["path2"]]

    responses = [["finished", "<tool_call> wrong tool use", '<tool_call>{"name": "show_image", "arguments": "none"}</tool_call>', "finished"],
                 ["finished"]]


    all_multimodal_inputs = [
        {"prompt": p, "image_path": i}
        for p, i in zip(prompts, image_paths)
    ]


    dummy_train_loop(all_multimodal_inputs, responses)