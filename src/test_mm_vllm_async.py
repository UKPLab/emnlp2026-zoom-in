import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['VLLM_USE_V1'] = '0'
import PIL.Image
import logging
import uvicorn
from fastapi import BackgroundTasks, FastAPI
from vllm import LLM, AsyncLLMEngine
from pydantic import BaseModel
from typing import List, Dict, Union

class InputRequest(BaseModel):
    input: List[Dict[str, str]]

class OutputRequest(BaseModel):
    output: List[List[int]]

def main():
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    llm = LLM(model=model_name)
    app = FastAPI()
    logger = logging.getLogger("uvicorn")

    """
    prompts = [{"prompt": text,
            "image_path":
                {"image": #PIL.Image.open(
                    "/pfss/mlde/workspaces/mlde_wsp_UKP_Multimodal/helm/datasets/focusreason/chartqa_original/train_full/png/png/two_col_22791.png"
#)
}}]
    """

    @app.post("/generate/")
    async def generate(request:InputRequest):

        inputs_with_image = [] # will become List[Dict[str, Union[str, Dict[str, PIL.Image]]]]

        for entry in request.input:
            inputs_with_image.append(
                {"prompt": entry["prompt"],
                 "multi_modal_data": {"image": PIL.Image.open(entry["image_path"])}
                 })
        #return {"inputs_with_image": inputs_with_image}
        #logger.error(f"Request: {request}")
        all_outputs = llm.generate(inputs_with_image)
        #logger.error(f"All outputs: {all_outputs}")
        completion_ids = [list(output.token_ids) for outputs in all_outputs for output in outputs.outputs]
        return {"output": completion_ids}#, "output": all_outputs}#, "completion_ids": completion_ids}
    # Start the server
    uvicorn.run(app, log_level="trace")

if __name__ == "__main__":
    main()