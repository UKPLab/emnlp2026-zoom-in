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

    assistant_tool = {'role': 'assistant',
                      'content': [
                          {'type': 'text', 'text': "Hello <tool_call>World</tool_call>"}
                      ]
                      }

    assistant_box = {'role': 'assistant',
                     'content': [
                         {'type': 'text', 'text': "Hello World \\boxed{A}"}
                     ]
                     }
    default_image = "/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/pixel_reasoner/RL_data_without_video/images/a15ab079-1311-444c-8d50-1ba3544b6e06-0.jpg"

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

    mt = MultiTurn(batch_size=2, processor=processor, tools=None)

    mt.add_initial_user_prompt([user_simple, user_simple_2],
                               [default_image, default_image])

    #mt.add_model_reply([[9707,4337], [9707,220]])
    #print(f"overview: {mt.all_multi_turn}")
    #print(f"all texts: {mt.get_sequences(type="text")}")
    #print(f"all token ids: {mt.get_sequences(type="id")}")
    test_vllm = False
    if test_vllm:

        full_token_seq = mt.get_sequences(type="id", add_assistant_start=True, full_image_pad=False)
        print(f"full token seq: {full_token_seq}")
        print(f"full token seq len: {len(full_token_seq)}; {[len(s) for s in full_token_seq]}")
        vllm_format = [TokensPrompt(prompt_token_ids = full_token_seq[i],
                                    multi_modal_data={"image": [Image.open(default_image)]})
                       for i in range(len(full_token_seq))]

        llm = LLM(model="Qwen/Qwen2.5-VL-3B-Instruct")
        output = llm.generate(vllm_format)
        print(output)



    assistant_tool_tokenized = processor(text=[assistant_tool["content"][0]["text"]],
                                         images=None,
                                         return_tensors=None,
                                         padding=False,
                                         add_special_tokens=False,
                                         return_offsets_mapping=False)

    mt.add_model_reply(assistant_tool_tokenized["input_ids"], mapping=[1])

    mt.add_user_message(prompts = [user_simple_2], image_paths=[default_image], mapping=[1],
                        absolute_bbox_wrt_target_coordss=[(100, 100, 200, 200)], target_image_idxs=[0])
    print(f"overview after add_user_message: {mt.all_multi_turn}")
    mt.add_model_reply(assistant_tool_tokenized["input_ids"], mapping=[1])

    mt.add_user_message(prompts=[user_simple_2], image_paths=[default_image], mapping=[1],
                        absolute_bbox_wrt_target_coordss=[(10, 10, 50, 50)], target_image_idxs=[1])
    print(f"overview after add_user_message: {mt.all_multi_turn}")
    sys.exit()
    assistant_box_tokenized = processor(text=[assistant_box["content"][0]["text"]],
                                         images=None,
                                         return_tensors=None,
                                         padding=False,
                                         add_special_tokens=False,
                                         return_offsets_mapping=False)
    mt.add_model_reply(assistant_box_tokenized["input_ids"], mapping=[1])
    #print(mt.all_multi_turn)
    #input_ids, positions, images_per_sample, considered_seqs = mt.get_shortened_sequences(mode="tool_and_box")
    #print(f"input_ids: {input_ids}")
    #print(f"positions: {positions}")
    #print(f"considered_seqs: {considered_seqs}")
    out = mt.get_alternative_sequences(alternative_action="double_newline,the,answer,is", #"second_model_generation", #
                                        answer="full_generation",
                                        ground_truth=["\\boxed{42}",
                                                      "\\boxed{43}"])
    print(f"alternative result: {out}")
    short = out[3][1]["short_sequence"]
    long = out[3][1]["updated_original_sequence"]
    ans_pos = out[3][1]["answer_position"]
    ans_pos_short = out[3][1]["answer_position_short"]
    alt_action_pos = out[3][1]["alternative_action_position"]
    alt_action_pos_short = out[3][1]["alternative_action_position_short"]
    print(f"short: {short}")
    print(f"long: {long}")
    print(f"ans_pos: {long[ans_pos[0]:ans_pos[1]]}")
    print(f"ans_pos_short: {short[ans_pos_short[0]:ans_pos_short[1]]}")
    print(f"alt_action: {long[alt_action_pos[0]:alt_action_pos[1]]}")
    print(f"ans_action_short: {short[alt_action_pos_short[0]:alt_action_pos_short[1]]}")
    as_text = processor.batch_decode([short, long])
    print(f"as text: {as_text}")
    assistant_end = processor.apply_chat_template([user_simple, assistant_tool])
    #print(f"assistant_end: {assistant_end}")