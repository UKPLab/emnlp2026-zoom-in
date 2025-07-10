from datasets import load_dataset

def prep_charxiv(subset):
    dataset = load_dataset("princeton-nlp/CharXiv", split=[subset])

    for sample in dataset:
        img_path = sample["figure_path"]

        question = sample["reasoning_q"]

        answer = sample["reasoning_a"]



if __name__ == '__main__':
    subset = "validation"
    prep_charxiv(subset)