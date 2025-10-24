import json
import os.path
from pathlib import Path
from PIL import Image
import torch

from qwen_vl_utils import smart_resize
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from open_r1.utils.parser import ParsedTokenized, rescale


def do_pmi_analysis(full_paths:list[str], fixed_params: dict):
    for full_path in full_paths:
        results_file = os.path.join(full_path, "full_results.json")
        model_path = str(Path(full_path).parent.parent.absolute())

        get_diff(results_file, model_path, fixed_params)
        #analyze_diff(os.path.join(full_path, "diff_0.json"))

def analyze_diff(filepath:str):
    print(f"opening: {filepath}")
    #with open(filepath, "r") as f:
    #    diff = json.load(f)
    #print(diff)


def get_diff(results_file: str, model_path:str, fixed_params:dict):
    device = "cuda"
    with open(results_file, "r") as f:
        results = json.load(f)

    print(f"accuracy: {results["accuracy"]}")
    return

    processor_init_kwargs = {"min_pixels": fixed_params["min_pixels"] if "min_pixels" in fixed_params else None,
                         "max_pixels": fixed_params["max_pixels"] if "max_pixels" in fixed_params else None}

    processing_class = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct",
                                                     **processor_init_kwargs)



    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path,
                                                               attn_implementation="flash_attention_2",
                                                               torch_dtype=torch.bfloat16)
    model.to(device)

    for idx in range(193):
        if len(results["images"][idx]) != 2:
            print(f"no successful tool use in idx={idx}")
        else:
            text, images = get_sequence(results, idx=idx, fixed_params=fixed_params)
            hf_inputs = processing_class(
                text=text,
                images=images,
                return_tensors="pt",
                padding=True,
                padding_side="right",
                add_special_tokens=False,
                return_offsets_mapping=False
            )
            input_ids_list = hf_inputs["input_ids"][0].tolist()
            token_list = processing_class.tokenizer.convert_ids_to_tokens(hf_inputs["input_ids"][0], skip_special_tokens=False)
            for id, token in zip(input_ids_list, token_list):
                if token != "<|image_pad|>":
                    print(f"{id}: {token}")

            with torch.no_grad():
                per_token_logps = _get_per_token_logps_new(model, hf_inputs["input_ids"], hf_inputs["attention_mask"],
                                         hf_inputs["image_grid_thw"], hf_inputs["pixel_values"],
                                         num_images=[2], batch_size = 1 , disable_dropout=True, device=device)
    
                parser_input = hf_inputs.copy()
    
                parser = ParsedTokenized(parser_input["input_ids"],
                                         parser_input["attention_mask"],
                                         parser_input["image_grid_thw"],
                                         parser_input["pixel_values"],
                                         verbose=True)
    
                processing = [{"mode": "model_tool_call",
                               "user_turn_range": None,
                               "model_turn_range": [0, 1]},  # only consider the first model response
                              {"mode": "full_tool_user_response",
                               "user_turn_range": [1, 2],
                               # only consider the second user query as there will be the execution of the first tool call
                               "model_turn_range": [1, 2]
                               # only consider the model generation directly after the requested user generation
                               },
                              ]
    
                shorten_tokenized = parser.get_shortened_tokenized(processing, processing_class.tokenizer.pad_token_id,
                                                                   device=device, padding_side="right"
                                                                   )
    
                contrasted_area = parser.get_model_response(1)
    
                short_prompt_ids = shorten_tokenized["input_ids"]
                short_prompt_mask = shorten_tokenized["attention_mask"]
                short_multimodal_inputs = {"image_grid_thw": shorten_tokenized["image_grid_thw"],
                                           "pixel_values": shorten_tokenized["pixel_values"]}
    
                enlarged_short_prompt_ids = torch.ones_like(hf_inputs["input_ids"]) * processing_class.tokenizer.pad_token_id
                enlarged_short_prompt_ids[:, :short_prompt_ids.size(1)] = short_prompt_ids
    
                enlarged_short_attention_mask = torch.zeros_like(hf_inputs["attention_mask"])
                enlarged_short_attention_mask[:, :short_prompt_mask.size(1)] = short_prompt_mask
    
                vision_masked_per_token_logps = _get_per_token_logps_new(model,
                                                                              enlarged_short_prompt_ids,
                                                                              enlarged_short_attention_mask,
                                                                              image_grid_thw=short_multimodal_inputs[
                                                                                  "image_grid_thw"],
                                                                              pixel_values=short_multimodal_inputs[
                                                                                  "pixel_values"],
                                                                              num_images=[1],
                                                                              batch_size=1,
                                                                              disable_dropout=True, device=device)[
                    :, :short_prompt_ids.size(1) - 1]
                # shorten_tokenized
                indices = shorten_tokenized["indices"]
                indices[indices >= 0] -= 1
                logp_indices = indices[:, 1:]
    
                print(f"masked_logps: {vision_masked_per_token_logps}")
                # print(f"masked_logps: {vision_masked_per_token_logps.shape}")
    
                print(f"logp indices: {logp_indices}")
    
                rescaled_masked_logps = rescale(vision_masked_per_token_logps, logp_indices,
                                                hf_inputs["input_ids"][:, 1:], hf_inputs["attention_mask"][:, 1:],
                                                pad_token_id=-1.0875e+01)
    
                print(f"rescaled_masked_logps: {rescaled_masked_logps}")
                # print(f"rescaled_masked_logps: {rescaled_masked_logps.shape}")
    
                print(f"per_token_logps: {per_token_logps}")
                diff = per_token_logps - rescaled_masked_logps
                diff_list = diff[0].detach().cpu().tolist()
                per_token_logps_list = per_token_logps[0].detach().cpu().tolist()
                rescaled_masked_logps_list = rescaled_masked_logps[0].detach().cpu().tolist()


                diffs_as_dict = {}
                for i in range(len(token_list)):
                    if i == 0:
                        diffs_as_dict[i] = {"token": token_list[i]}
                    else:
                        diffs_as_dict[i] = {"token": token_list[i],
                                            "diff": diff_list[i-1],
                                            "per_token_logps": per_token_logps_list[i-1],
                                            "rescaled_masked_logps": rescaled_masked_logps_list[i-1]}
                diffs_as_dict[-1] = {"accuracy": results["accuracy"][idx]}
                save_path = os.path.join(str(Path(results_file).parent.absolute()), f"diff_{idx}.json")
                json.dump(diffs_as_dict, open(save_path, "w"), indent=4)



def get_sequence(results: dict, idx:int, fixed_params:dict):
    query = results["query"][idx]
    model_answer = results["model_answer"][idx][0]["content"]
    image_paths = results["images"][idx]

    model_answers = model_answer.split("</tool_call>")
    first_model_answer = model_answers[0] + "</tool_call>"
    second_model_answer = model_answers[1]
    intermediate_1 = "<|im_end|>\n<|im_start|>user\n"
    intermediate_2 = "<|im_end|>\n<|im_start|>assistant\n"
    img_path = image_paths[1]
    img_2 = Image.open(img_path, "r")
    w,h = img_2.size
    height, width = smart_resize(h,w,
                                 min_pixels=fixed_params["min_pixels"] if "min_pixels" in fixed_params.keys() else 28*28*4,
                                 max_pixels=fixed_params["max_pixels"] if "max_pixels" in fixed_params.keys() else 28*28*1024*16)

    if fixed_params["tool_config_type"] == "PR_zoom_in_old":
        tool_call = f"\nHere is the cropped image (Image Size: {width}x{height}):<|vision_start|><|image_pad|><|vision_end|>"
    else:
        raise ValueError(f"tool_config_type= {fixed_params["tool_config_type"]} is not supported")

    text = query+first_model_answer+intermediate_1+tool_call+intermediate_2+second_model_answer
    images = [Image.open(image_paths[0]), img_2]

    return text, images

def _get_per_token_logps_new(model, input_ids, attention_mask, image_grid_thw, pixel_values, num_images,
                             batch_size, disable_dropout, device):
    if disable_dropout:
        model.eval()
    max_len = input_ids.size(1)
    #print(f"max_len: {max_len}")
    logp_target_len = max_len - 1
    batch_size = batch_size or input_ids.size(0)  # Chunk inputs into smaller batches to reduce memory peak
    print(f"_get_per_token_logps_new: batch size: {batch_size}")
    print(f"_get_per_token_logps_new: inputs_ids_size: {input_ids.size(0)}")
    all_logps = []
    for start in range(0, input_ids.size(0), batch_size):
        input_ids_batch = input_ids[start: start + batch_size]
        attention_mask_batch = attention_mask[start: start + batch_size]

        max_len_batch = int(attention_mask_batch.sum(dim=1).max().item())
        #print(f"max_len_batch: {max_len_batch}")
        input_ids_batch = input_ids_batch[:, :max_len_batch].to(device)
        attention_mask_batch = attention_mask_batch[:, :max_len_batch].to(device)

        # Build model inputs - check if the model supports logits_to_keep (some models and VLMs don't)
        model_inputs = {"input_ids": input_ids_batch, "attention_mask": attention_mask_batch}

        if image_grid_thw is not None and pixel_values is not None:
            rows_per_image = image_grid_thw.prod(dim=-1)
            rows_per_sample = torch.split(rows_per_image, num_images)
            rows_per_sample = torch.stack([s.sum() for s in rows_per_sample])
            cum_rows = torch.cat([torch.tensor([0], device=rows_per_sample.device), rows_per_sample.cumsum(0)])
            row_start, row_end = cum_rows[start].item(), cum_rows[start + batch_size].item()
            model_inputs["pixel_values"] = pixel_values[row_start:row_end].to(device)
            cum_imgs = torch.tensor([0] + num_images).cumsum(0)
            img_start, img_end = cum_imgs[start], cum_imgs[start + batch_size]
            model_inputs["image_grid_thw"] = image_grid_thw[img_start:img_end].to(device)
        elif pixel_values is not None:
            model_inputs["pixel_values"] = pixel_values[start: start + batch_size].to(device)


        logits = model(**model_inputs).logits

        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids_batch = input_ids_batch[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it

        #print(f"logits: {logits}")
        #print(f"logits shape: {logits.shape}")
        #print(f"log softmax: {logits.log_softmax(dim=-1)}")
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids_batch):
            log_probs = logits_row.log_softmax(dim=-1)
            #token_log_prob = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        chunk_logps = torch.stack(per_token_logps)
        #print(f"chunk_logps before re-pad: {chunk_logps.size(1)}")
        if chunk_logps.size(1) < logp_target_len:
            pad_len = logp_target_len - chunk_logps.size(1)
            chunk_logps = torch.nn.functional.pad(chunk_logps, (0, pad_len), value=0.0)
        #print(f"chunk_logps after re-pad: {chunk_logps.size(1)}")
        all_logps.append(chunk_logps)


    if disable_dropout:
        model.train()

    return torch.cat(all_logps, dim=0)