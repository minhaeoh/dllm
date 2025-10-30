"""
Local users
------------
python examples/llada/generate_checkpoint.py --model_name_or_path "YOUR_MODEL_PATH"

Slurm users
------------
srun -p $PARTITION --quotatype=$QUOTATYPE --gres=gpu:1 --time=03:00:000 \
    python examples/llada/generate_checkpoint.py --model_name_or_path "YOUR_MODEL_PATH"
"""

from dataclasses import dataclass

import tyro
import torch
import transformers
import accelerate

import dllm
from dllm.pipelines import llada

### config ###
data_name = "gsm8k_filter_all_1_0_1"

@dataclass
class ScriptArguments:
    model_name_or_path: str = (
        "/home/minhae/diffusion/dllm/models/LLaDA-8B-SFT/gsm8k_filter_all_1_0_1/checkpoint-final"  # "inclusionAI/LLaDA-MoE-7B-A1B-Instruct"
        # "GSAI-ML/LLaDA-8B-Instruct"
    )
    steps: int = 128
    max_new_tokens: int = 128
    block_length: int = 32
    temperature: float = 0.0
    remasking: str = "low_confidence"
    seed: int = 42

    def __post_init__(self):
        self.model_name_or_path = dllm.utils.resolve_with_base_env(
            self.model_name_or_path, "BASE_MODELS_DIR"
        )

@dataclass
class DataArguments(dllm.utils.DataArguments):
    # dataset_args: str = "allenai/tulu-3-sft-mixture[train:10000,test:1000]" 
    dataset_args: str = data_name # Use our local GSM8K dataset

data_args = tyro.cli(DataArguments)
script_args = tyro.cli(ScriptArguments)
transformers.set_seed(script_args.seed)

# # Load model & tokenizer
model = dllm.utils.get_model(model_args=script_args).eval()
tokenizer = dllm.utils.get_tokenizer(model_args=script_args, model=model)


# ----- Data loading -----
def custom_apply_chat_template(messages):
    # Don't move to device in multiprocessing context
    # Return as lists to avoid tensor serialization issues
    # for i in range(len(messages)):
    #     print(f"Message {i}: {messages[i]}")
    q_llm_input_ids = tokenizer.apply_chat_template(
        messages[:-1],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    )[0].tolist()

    q_input_ids = tokenizer.apply_chat_template(
        messages[:-2],
        add_generation_prompt=False,
        tokenize=True,
        return_tensors="pt",
    )[0].tolist()
    return {"q_llm_input_ids": q_llm_input_ids, "q_input_ids": q_input_ids, "q_len": len(q_input_ids), "gold_answer": messages[-1]["content"]}


def sft_map_fn(row) -> dict:
    return custom_apply_chat_template(row["messages"])

with accelerate.PartialState().local_main_process_first():
    dataset = dllm.data.load_sft_dataset(data_args.dataset_args)
    dataset = dataset["test"]
    print("test dataset length: ", len(dataset))
    dataset = dataset.filter(lambda x: x["source"] == "gsm8k_q_llm_cond")
    print("filtered test dataset length: ", len(dataset))
    results = dataset.map(sft_map_fn, num_proc=data_args.num_proc, remove_columns=dataset.column_names)
    # Move to device after multiprocessing is done
    input_ids_list = [torch.tensor(r["q_llm_input_ids"]).to(model.device) for r in results]
    # input_ids_list = [torch.tensor(r["q_input_ids"]).to(model.device) for r in results]
    q_len = [r["q_len"] for r in results]
    gold_answer = [r["gold_answer"].split("The answer is: ")[1] for r in results]



# --- Example 1: Batch generation ---
print("\n" + "=" * 80)
print("TEST: llada.generate()".center(80))
print("=" * 80)

# Process in batches to avoid OOM
batch_size = 2  # Adjust based on your GPU memory
all_generations = []

for batch_start in range(0, len(input_ids_list), batch_size):
    batch_end = min(batch_start + batch_size, len(input_ids_list))
    print(f"Processing batch {batch_start//batch_size + 1}/{(len(input_ids_list) + batch_size - 1)//batch_size} (samples {batch_start}-{batch_end-1})")
    
    batch_input_ids = input_ids_list[batch_start:batch_end]
    batch_q_len = q_len[batch_start:batch_end]
    
    
    out = llada.generate_two_cfg(
        model,
        tokenizer,
        batch_input_ids,
        batch_q_len,
        steps=script_args.steps,
        max_new_tokens=script_args.max_new_tokens,
        block_length=script_args.block_length,
        temperature=script_args.temperature,
        remasking=script_args.remasking,
        cfg_scale=0  # Higher scale to rely more on condition 1 (question+llm)
    )
    # out = llada.generate(
    #     model,
    #     tokenizer,
    #     batch_input_ids,        
    #     steps=script_args.steps,
    #     max_new_tokens=script_args.max_new_tokens,
    #     block_length=script_args.block_length,
    #     temperature=script_args.temperature,
    #     remasking=script_args.remasking
    # )
    
    
    for i, o in enumerate(out):
        # Extract only the generated part (after the last assistant header)
        start_index = len(batch_input_ids[i])
        stop_index = start_index + script_args.max_new_tokens
        generated_only = o[start_index:stop_index]
        total_input = tokenizer.decode(batch_input_ids[i])
        question = tokenizer.decode(batch_input_ids[i][:batch_q_len[i]])
        generated_text = tokenizer.decode(generated_only)
        print(f"Total input: {total_input}\n")
        print(f"Input question: {question}\n")
        print(f"Generated text: {generated_text}\n")
        print(f"Gold answer: {gold_answer[batch_start + i]}\n")
        print("\n" + "=" * 80)
        
    # Clear cache to free memory
    torch.cuda.empty_cache()
    break



# print(f"\nTotal generated: {len(all_generations)} samples")
# print("\n" + "=" * 80 + "\n")

