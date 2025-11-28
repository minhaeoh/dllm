# convert json to dataset

from datasets import load_from_disk
import json

if __name__ == "__main__":
    # with open("/data/diffusion/dataset/sft/gold_sft/all_predictions.jsonl", "r") as f:
    #     print(len(f.readlines()))
    #     for line in f:
    #         data = json.loads(line)
    #         print(data.keys())
    #         break
    dataset = load_from_disk("/home/minhae/diffusion/dllm/examples/llada/dataset/trainset/math_gsm8k_final")
    train = dataset['train']
    source = []
    for i in range(len(train)):
        if len(source)>6:
            break
        if train[i]['source'] not in source:
            source.append(train[i]['source'])
            print(train[i])