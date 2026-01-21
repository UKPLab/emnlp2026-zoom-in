import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['VLLM_USE_V1'] = '0'


import PIL.Image

from vllm.engine.arg_utils import EngineArgs

from vllm import LLM, AsyncLLMEngine, LLMEngine, SamplingParams
from transformers import AutoProcessor

async def get_results(results_generator):
    for request_output in results_generator:
        output = request_output
        print(output)


if __name__ == '__main__':
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    mode = "engine"



    if mode == "offline":

        llm = LLM(model=model_name,
                  allowed_local_media_path=True,
                  enforce_eager=True)

    if mode == "online":
        engine = AsyncLLMEngine(model=model_name)

    processor = AutoProcessor.from_pretrained(model_name)
    prompts = []
    for prompt_text in ["What is the difference between the images?<|im_end|>\n<|im_start|>user\nAre you sure? Think again. As before, first output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags.<|im_end|>\n<|im_start|>assistant\n<think>\nI see that there was a mis", "Beschreibe das Bild."]:

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/focusreason/chartqa_original/train_full/png/png/two_col_22791.png"},
                    },
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        print(text)

        #print(processor.apply_chat_template(
        #    messages, tokenize=True, add_generation_prompt=True
        #))

        prompts.append({"prompt": text,
                "multi_modal_data":
                    {"image": PIL.Image.open(
                        "/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/focusreason/chartqa_original/train_full/png/png/two_col_22791.png")}})


    if mode == "engine":
        engine_args = EngineArgs(model=model_name, enforce_eager=True, num_scheduler_steps=1)
        engine = LLMEngine.from_engine_args(engine_args, )
        print(f"Scheduler config: {engine.get_scheduler_config()}")

    if mode == "engine":
        def engine_gen(prompts):
            for idx, prompt in enumerate(prompts):
                if idx == 0:
                    engine.add_request(str(idx), prompt, SamplingParams(temperature=0.0, max_tokens=10, min_tokens=2))
                else:
                    engine.add_request(str(idx), prompt, SamplingParams(temperature=0.0, max_tokens=10, min_tokens=2))

            step_no = 0
            while engine.has_unfinished_requests():
                request_outputs = engine.step()
                #print(f"after step {step_no}")
                step_no += 1
                #if step_no > 1:
                #break
                #for output in request_outputs:
                #    print(f"{step_no}: {output}")
            return request_outputs
        request_outputs = engine_gen(prompts=prompts)

        for idx, output in enumerate(request_outputs):
            print(f"Generation for seq {idx}: {output.outputs[0].text}")
            prompts[idx]["prompt"]
        #print(f"Generation for seq 0: {request_outputs[0]}")


    if mode == "offline":
        output = llm.generate(prompts=prompts)
    if mode == "online":
        results_generator = engine.generate(prompts[0], 0)

        get_results(results_generator)

    print("Done!")
