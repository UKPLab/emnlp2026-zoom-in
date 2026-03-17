from vllm import LLM, SamplingParams


if __name__ == "__main__":
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    llm = LLM(model="Qwen/Qwen3.5-9B")

    llm.generate(prompts=prompts,sampling_params=sampling_params)

