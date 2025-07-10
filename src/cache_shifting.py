from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
import torch
from typing import List


def test():
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct", attn_implementation="flash_attention_2",
                                                 torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

    # 151643

    inputs = tokenizer(text=["My name is Qwen2",
                             "My name is Qwen2, right?"], return_tensors="pt", padding=True, padding_side="left")
    print(f"standard inputs: {inputs}")
    for seq in inputs["input_ids"]:
            print(tokenizer.convert_ids_to_tokens(list(seq)))

    additional_pads = 3

    if additional_pads > 0:
        pads = torch.ones((inputs["input_ids"].shape[0], additional_pads), dtype=torch.long)*151643
        masks = torch.zeros((inputs["input_ids"].shape[0], additional_pads), dtype=torch.long)
        inputs["input_ids"] = torch.cat((pads, inputs["input_ids"]), dim = 1)
        inputs["attention_mask"] = torch.cat((masks, inputs["attention_mask"]), dim = 1)

    print(f"padded inputs: {inputs}")
    for seq in inputs["input_ids"]:
            print(tokenizer.convert_ids_to_tokens(list(seq)))

    generation_kwargs = {
        "max_length": 20+additional_pads,  # Maximum length of the generated sequence
        "num_return_sequences": 1,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": False,
        "num_beams": 1
    }

    inputs.to("cuda")
    model.to("cuda")
    answer = model.generate(**inputs, use_cache=True, **generation_kwargs)

    print(answer)
    print(tokenizer.batch_decode(answer))

class MultiTokenCache:

    def __init__(self, batch_size, pads:torch.tensor = None, reals:torch.tensor = None, cache:torch.tensor = None):
        self.batch_size = batch_size
        self.pads = pads if pads is not None else torch.zeros(batch_size)
        self.reals = reals if reals is not None else torch.zeros(batch_size)
        self.cache = cache if cache is not None else []

        self.check_cache_validity()

    def check_cache_validity(self):
        assert 0 in self.pads, f"Validity check failed: every cache sequence contains pads"
        total = self.pads + self.reals
        print(f"total: {total}")

        assert (total[0] == total).all(), f"Validity check failed: pads + reals differ by sequence"

        if torch.sum(total) == 0:
            assert len(self.cache) == 0
        else:
            first_seq = self.cache[0]
            for seq in self.cache:
                assert seq.shape[1] == first_seq.shape[1], f"Validity check failed: cache lengths unequal"

            assert total[0] == first_seq.shape[1], f"Validity check failed: length mismatch between cache and pads+reals"

    def shift(self, generated: List[torch.tensor]):
        self.check_cache_validity()

        generated_lengths = torch.tensor([g.shape[1] for g in generated], dtype=torch.int, device=generated[0].device)
        print(f"generated lengths: {generated_lengths}")

        lengths_without_pad = self.reals+generated_lengths
        new_longest = torch.argmax(lengths_without_pad)

        print(f"new longest: {new_longest}")

        lengths_with_pad = self.pads + lengths_without_pad
        for i in range(self.batch_size):
            print(f"{i} before: {self.cache[i]}")
            print(f'{lengths_with_pad[i]} vs {lengths_without_pad[new_longest]}')
            if lengths_with_pad[i] >= lengths_without_pad[new_longest]:
                self.cache[i] = self.cache[i][:, lengths_with_pad[i]-lengths_without_pad[new_longest]:]
                self.pads[i] -= lengths_with_pad[i]-lengths_without_pad[new_longest]
                print(f"{i} truncation: {self.cache[i]}")
            if lengths_with_pad[i] < lengths_without_pad[new_longest]:
                self.cache[i] = torch.cat((
                    torch.zeros((1,lengths_without_pad[new_longest]- lengths_with_pad[i]), dtype=torch.long, device=self.cache[i].device),
                    self.cache[i]),
                    dim=1)
                self.pads[i] += lengths_without_pad[new_longest]- lengths_with_pad[i]
                print(f"{i} after padding: {self.cache[i]}")
            self.cache[i] = torch.cat((self.cache[i], generated[i]), dim=1)
            self.reals[i] += generated[i].shape[1]
            print(f"{i} after: {self.cache[i]}")

        self.check_cache_validity()
        return cache

if __name__ == "__main__":
    #test()

    cache = [
        torch.tensor([[0,0,1,2,3,4]]),
        torch.tensor([[0,1,2,3,4,5]]),
        torch.tensor([[1,2,3,4,5,6]]),
        torch.tensor([[0,0,0,0,1,2]])
    ]

    pads = [2,1,0,4]
    reals= [4,5,6,2]
    generates = [
        torch.tensor([[5001]]),
        torch.tensor([[5001, 5002]]),
        torch.tensor([[5001, 5002, 5003]]),
        torch.tensor([[5001, 5002, 5003, 5004]])
    ]

    generated_lengths = torch.tensor([g.shape[1] for g in generates], dtype=torch.int, device=generates[0].device)

    cache = MultiTokenCache(batch_size = 4, pads = torch.tensor(pads), reals= torch.tensor(reals), cache=cache)

    if not (generated_lengths[0]==generated_lengths).all():
        cache.shift(generates)
        print(cache.cache)
        print(cache.pads)
        print(cache.reals)
    else:
        print("no shifting needs to be done, use cache.update() instead")
