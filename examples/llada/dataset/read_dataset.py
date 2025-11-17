# convert json to dataset

from datasets import load_from_disk

if __name__ == "__main__":
    dataset = load_from_disk("/home/minhae/diffusion/dllm/examples/llada/dataset/testset/gsm8k_llama3.1_8b_instruct")
    print(dataset.column_names)
    # for i, row in enumerate(dataset):
    #     answer = row["gold_answer"]
    #     try:
    #         int(answer)
    #     except:
    #         print(f"Invalid answer: {answer}")
        