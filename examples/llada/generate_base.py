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

import wandb
from datetime import datetime
import os
import json

from datasets import load_from_disk

### config ###
data_name = "gsm8k_llama3.1_8b_instruct"
data_dir = f"/home/minhae/diffusion/dllm/examples/llada/dataset/testset/{data_name}"

model_name =  "/home/minhae/diffusion/dllm/models_nlp/LLaDA-8B-SFT/math_gsm8k_filter_all_1_0_1_ml2056/checkpoint-final"
 
task = "gsm8k"
@dataclass
class AllArguments:
    """Combined arguments for data and script"""
    # Data arguments
    dataset_args: str = data_name
    num_proc: int = 4
    
    # Script arguments
    model_name_or_path: str = (
        model_name
    )
    steps: int = 128
    max_new_tokens: int = 1024
    block_length: int = 32
    temperature: float = 0.0
    remasking: str = "low_confidence"
    seed: int = 42
    refinement: bool = False # True: refinement, False: initial
    prompt: bool = False # 0:no prompt, 1:debate prompt
    
    def __post_init__(self):
        self.model_name_or_path = dllm.utils.resolve_with_base_env(
            self.model_name_or_path, "BASE_MODELS_DIR"
        )

args = tyro.cli(AllArguments)
if args.refinement:
    config_str ="refinement"
else:
    config_str =  "initial"
transformers.set_seed(args.seed)


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")    


# Output logging setup
config_str = config_str + "_prompt"  if args.prompt else config_str+"_noprompt"
output_dir = os.path.join(os.path.dirname(__file__), "outputs",task,model_name,config_str)
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, f"{timestamp}.jsonl")

wandb.init(project="evaluation-gsm8k", name=f"{model_name}_{config_str}_{timestamp}")
wandb.config.update({
    "task": task,
    "model": model_name,
    "config": config_str,
    "timestamp": timestamp,
})


# # Load model & tokenizer
model = dllm.utils.get_model(model_args=args).eval()
tokenizer = dllm.utils.get_tokenizer(model_args=args, model=model)

import re

def extract_answer_num(text: str) -> str | None:
    cleaned = text.replace("<|endoftext|>", "").replace("<|eot_id|>", "")
    # 대소문자/공백/구두점 유연 처리
    m = re.search(r"the answer is\s*:\s*\$?\s*([-+]?\d[\d,]*(?:\.\d+)?)", cleaned, re.IGNORECASE)
    # m = re.search(r"the answer is\s*:\s*\$?\s*([-+]?\d[\d,\.]*\d)", cleaned, re.IGNORECASE)
    # m = re.search(r"the answer is\s*:\s*\$?\s*([-+]?\d[\d,\.]*)", cleaned, re.IGNORECASE)
    
    # m = re.search(r"the answer is\s*:\s*([-+]?\d[\d,\.]*)", cleaned, re.IGNORECASE)
    return m.group(1).replace(",", "") if m else None

def to_int_safe(s: str) -> int:
    return int(re.sub(r"[^\d+-]", "", s))  # 숫자/부호 외 제거 (콤마, 공백 등)

# ----- Data loading -----
def custom_apply_chat_template(row, prompt: bool):
    # Don't move to device in multiprocessing context
    # Return as lists to avoid tensor serialization issues
    messages = []
    if args.refinement:
        if prompt:
            messages.append({"role": "user", "content": "This the solutions to the problem from other agent: " + row["llm_output"]})
            messages.append({"role": "user", "content": "Using the solutions from other agents as additional information, can you provide your answer to the math problem? The original math problem is " + row["question"] + ". Your final answer should be a single numerical number, in the form 'the answer is: <answer>', at the end of your response."})
        else:
            messages.append({"role": "user", "content": row["question"]})
            messages.append({"role": "LLM generated Answer", "content": row["llm_output"]})
        
        input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        )[0].tolist()  # Keep as list, not tensor

        return {
        "input_ids": input_ids,
        "question": row["question"],
        "llm_output": row["llm_output"],
        "gold_answer": row["gold_answer"].replace(",", ""),
        "initial_correct": row["llm_correct"],
        "initial_answer": row["llm_pred_answer"],
        }
    
    else:
        if prompt:
            messages.append({"role": "user", "content": "Solve the problem step by step. Finish with The answer is: <answer>. " + row["question"]})

        else:
            messages.append({"role": "user", "content": row["question"]})

        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )[0].tolist()  # Keep as list, not tensor

        return {
            "input_ids": input_ids,
            "question": row["question"],
            "llm_output": row["llm_output"],
            "gold_answer": row["gold_answer"].replace(",", ""),
            "initial_correct": row["llm_correct"],
            "initial_answer": row["llm_pred_answer"],
        }


def sft_map_fn(row) -> dict:
    return custom_apply_chat_template(row, args.prompt)

with accelerate.PartialState().local_main_process_first():
    dataset = load_from_disk(data_dir)
    print("test dataset length: ", len(dataset))
    results = dataset.map(sft_map_fn, num_proc=args.num_proc, remove_columns=dataset.column_names)
    # Move to device after multiprocessing is done
    input_ids_list = [torch.tensor(r["input_ids"], dtype=torch.long).to(model.device) for r in results]


# --- Example 1: Batch generation ---
print("\n" + "=" * 80)
print("TEST: llada.generate()".center(80))
print("=" * 80)

# Process in batches to avoid OOM
batch_size = 8  # Adjust based on your GPU memory
all_generations = []

total_correct = 0
total_processed = 0
with open(output_file, "w", encoding="utf-8") as log_f:
    for batch_start in range(0, len(input_ids_list), batch_size):
        batch_end = min(batch_start + batch_size, len(input_ids_list))
        print(f"Processing batch {batch_start//batch_size + 1}/{(len(input_ids_list) + batch_size - 1)//batch_size} (samples {batch_start}-{batch_end-1})")
        
        batch_input_ids = input_ids_list[batch_start:batch_end]
        
        out = llada.generate(
            model,
            tokenizer,
            batch_input_ids,
            steps=args.steps,
            max_new_tokens=args.max_new_tokens,
            block_length=args.block_length,
            temperature=args.temperature,
        )
        # gold_answer = [r["gold_answer"] for r in results[batch_start:batch_end]]

        for i, o in enumerate(out):
            # Extract only the generated part (after the last assistant header)
            start_index = len(batch_input_ids[i])
            stop_index = start_index + args.max_new_tokens
            generated_only = o[start_index:stop_index]
            total_input = tokenizer.decode(batch_input_ids[i])
            question = results[batch_start + i]["question"]
            generated_text = tokenizer.decode(generated_only)
            # print(f"Total input: {total_input}\n")
            # print(f"Input question: {question}\n")
            print(f"Generated text: {generated_text}\n")
            print(f"Gold answer: {results[batch_start + i]["gold_answer"]}\n")
            # print("\n" + "=" * 80)
            generated_answer = extract_answer_num(generated_text)
            is_correct = False
            if generated_answer is not None and float(generated_answer) == float(results[batch_start + i]["gold_answer"]):
                total_correct += 1
                is_correct = True
            total_processed += 1
            wandb.log({"accuracy": total_correct/total_processed*100},step=total_processed)

            # Persist per-sample log as JSONL
            record = {
                "global_index": batch_start + i,
                "refined_correct": is_correct,
                "initial_correct": results[batch_start + i]["initial_correct"],
                "gold_answer": results[batch_start + i]["gold_answer"],                
                "refined_answer": generated_answer,
                "initial_answer": results[batch_start + i]["initial_answer"],
                "total_input": total_input,
                "input_question": question,
                "generated_text": generated_text,
                "steps": args.steps,
                "max_new_tokens": args.max_new_tokens,
                "block_length": args.block_length,
                "temperature": args.temperature,
                "remasking": args.remasking,
                "timestamp": timestamp,
                "accuracy": total_correct/total_processed*100,
            }
            log_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            log_f.flush()
        
        print(f"Total correct: {total_correct}/{total_processed}")
        print(f"Accuracy: {total_correct/total_processed*100:.2f}%")
        
        
        
        print("\n" + "=" * 80)
        # Clear cache to free memory
        torch.cuda.empty_cache()
    



# print(f"\nTotal generated: {len(all_generations)} samples")
# print("\n" + "=" * 80 + "\n")

