# convert json to dataset

import json
import datasets

def json_to_dataset(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return datasets.Dataset.from_list(data)

if __name__ == "__main__":
    json_file = "/data/diffusion/dataset/baselines/testset/gsm8k_llama3.1_8b_instruct_testset.json"
    dataset = json_to_dataset(json_file)
    dataset.save_to_disk("/home/minhae/diffusion/dllm/examples/llada/dataset/testset/gsm8k_llama3.1_8b_instruct")