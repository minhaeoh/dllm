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


### config ###
data_name = "gsm8k_filter_all_1_0_1"
model_name = "gsm8k_filter_all_1_0_1_ml2056"
    

@dataclass
class ScriptArguments:
    model_name_or_path: str = (
        "/home/minhae/diffusion/dllm/models/LLaDA-8B-SFT/gsm8k_filter_all_1_0_1_ml2056/checkpoint-final"  
        # "GSAI-ML/LLaDA-8B-Instruct"
    )
    steps: int = 128
    max_new_tokens: int = 256
    block_length: int = 32
    temperature: float = 0.0
    remasking: str = "low_confidence"
    seed: int = 42
    cfg : int = 2 # 2/3
    cfg_style: str = "static" # static/dynamic
    # cfg_scale을 dataclass 필드로 직접 지정하는 것이 아니라, 아래처럼 기본값으로 받는게 올바른 방법입니다.
    cfg_scale: float = 0  # default (예: 2 CFG인 경우)
    cfg_scale1: float = 1   # 3 CFG config에 쓸 값, 사용 안하면 무시됨
    cfg_scale2: float = 3   # 3 CFG config에 쓸 값, 사용 안하면 무시됨

    prompt: bool = False

    def __post_init__(self):
        self.model_name_or_path = dllm.utils.resolve_with_base_env(
            self.model_name_or_path, "BASE_MODELS_DIR"
        )




@dataclass
class DataArguments(dllm.utils.DataArguments):
    # dataset_args: str = "allenai/tulu-3-sft-mixture[train:10000,test:1000]" 
    dataset_args: str = data_name # Use our local GSM8K dataset

@dataclass
class AllArguments:
    """Combined arguments for data and script"""
    # Data arguments
    dataset_args: str = data_name
    num_proc: int = 4
    
    # Script arguments
    model_name_or_path: str = (
        "/home/minhae/diffusion/dllm/models/LLaDA-8B-SFT/gsm8k_filter_all_1_0_1_ml2056/checkpoint-final"
    )
    steps: int = 128
    max_new_tokens: int = 128
    block_length: int = 32
    temperature: float = 0.0
    remasking: str = "low_confidence"
    seed: int = 42
    cfg: int = 2  # 2/3
    cfg_style: str = "static"  # static/dynamic
    cfg_scale: float = 0.0
    cfg_scale1: float = 1.0
    cfg_scale2: float = 3.0
    prompt: bool = False
    
    def __post_init__(self):
        self.model_name_or_path = dllm.utils.resolve_with_base_env(
            self.model_name_or_path, "BASE_MODELS_DIR"
        )

args = tyro.cli(AllArguments)
# Backwards compatibility: create separate objects
data_args = DataArguments(dataset_args=args.dataset_args, num_proc=args.num_proc)
script_args = ScriptArguments(
    model_name_or_path=args.model_name_or_path,
    steps=args.steps,
    max_new_tokens=args.max_new_tokens,
    block_length=args.block_length,
    temperature=args.temperature,
    remasking=args.remasking,
    seed=args.seed,
    cfg=args.cfg,
    cfg_style=args.cfg_style,
    cfg_scale=args.cfg_scale,
    cfg_scale1=args.cfg_scale1,
    cfg_scale2=args.cfg_scale2,
    prompt=args.prompt,
)
transformers.set_seed(script_args.seed)

# model_name = script_args.model_name_or_path.split("/")[-1]
cfg = script_args.cfg
cfg_style = script_args.cfg_style
cfg_scale = script_args.cfg_scale
cfg_scale1 = script_args.cfg_scale1
cfg_scale2 = script_args.cfg_scale2
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")    

config_str = ""
if cfg == 2:
    if cfg_style == "dynamic":
        config_str = f"cfg{cfg}_{cfg_style}"
    elif cfg_style == "static":
        config_str = f"cfg{cfg}_{cfg_style}_{cfg_scale}"
    else:
        raise ValueError(f"Invalid cfg_style: {cfg_style}")
elif cfg == 3:
    if cfg_style == "dynamic":
        config_str = f"cfg{cfg}_{cfg_style}"
    elif cfg_style == "static":
        config_str = f"cfg{cfg}_{cfg_style}_{cfg_scale1}_{cfg_scale2}"
    else:
        raise ValueError(f"Invalid cfg_style: {cfg_style}")
else:
    raise ValueError(f"Invalid cfg: {cfg}")

if script_args.prompt:
    config_str += "_prompt"

# Output logging setup
output_dir = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, f"{model_name}_{config_str}_{timestamp}.jsonl")

wandb.init(project="evaluation", name=f"{model_name}_{config_str}_{timestamp}")
wandb.config.update({
    "model": model_name,
    "config": config_str,
    "timestamp": timestamp,
})


# # Load model & tokenizer
model = dllm.utils.get_model(model_args=script_args).eval()
tokenizer = dllm.utils.get_tokenizer(model_args=script_args, model=model)

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
def custom_apply_chat_template(messages):
    # Don't move to device in multiprocessing context
    # Return as lists to avoid tensor serialization issues
    # for i in range(len(messages)):
    #     print(f"Message {i}: {messages[i]}")

    #for baseline
    if script_args.prompt:
        messages[0]["content"] = "Carefully solve the problem step by step. Finish with The answer is: <answer>. " + messages[0]["content"]
    
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
    # INSERT_YOUR_CODE
    # Ensure only unique ids: keep only the first occurrence if duplicated
    seen_ids = set()
    unique_indices = []
    for i, row in enumerate(dataset):
        row_id = row["id"]
        if row_id not in seen_ids:
            seen_ids.add(row_id)
            unique_indices.append(i)
    dataset = dataset.select(unique_indices)
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
batch_size = 8  # Adjust based on your GPU memory
all_generations = []

total_correct = 0
total_processed = 0
with open(output_file, "w", encoding="utf-8") as log_f:
    for batch_start in range(0, len(input_ids_list), batch_size):
        batch_end = min(batch_start + batch_size, len(input_ids_list))
        print(f"Processing batch {batch_start//batch_size + 1}/{(len(input_ids_list) + batch_size - 1)//batch_size} (samples {batch_start}-{batch_end-1})")
        
        batch_input_ids = input_ids_list[batch_start:batch_end]
        batch_q_len = q_len[batch_start:batch_end]
        
        if cfg == 2:
            if cfg_style == "dynamic":
                out = llada.generate_two_cfg_dynamic(
                    model,
                    tokenizer,
                    batch_input_ids,
                    batch_q_len,
                    steps=script_args.steps,
                    max_new_tokens=script_args.max_new_tokens,
                    block_length=script_args.block_length,
                    temperature=script_args.temperature,
                    remasking=script_args.remasking
                )
            elif cfg_style == "static":
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
                    cfg_scale=cfg_scale
                )
        elif cfg == 3:
            if cfg_style == "dynamic":
                out = llada.generate_three_cfg_dynamic(
                    model,
                    tokenizer,
                    batch_input_ids,
                    batch_q_len,
                    steps=script_args.steps,
                    max_new_tokens=script_args.max_new_tokens,
                    block_length=script_args.block_length,
                    temperature=script_args.temperature,
                    remasking=script_args.remasking
                )
            elif cfg_style == "static":
                out = llada.generate_three_cfg(
                    model,
                    tokenizer,
                    batch_input_ids,
                    batch_q_len,
                    steps=script_args.steps,
                    max_new_tokens=script_args.max_new_tokens,
                    block_length=script_args.block_length,
                    temperature=script_args.temperature,
                    remasking=script_args.remasking,
                    cfg_scale1=cfg_scale1,
                    cfg_scale2=cfg_scale2
                )
            else:
                raise ValueError(f"Invalid cfg_style: {cfg_style}")
            
        else:
            raise ValueError(f"Invalid cfg: {cfg}")

        
        for i, o in enumerate(out):
            # Extract only the generated part (after the last assistant header)
            start_index = len(batch_input_ids[i])
            stop_index = start_index + script_args.max_new_tokens
            generated_only = o[start_index:stop_index]
            total_input = tokenizer.decode(batch_input_ids[i])
            question = tokenizer.decode(batch_input_ids[i][:batch_q_len[i]])
            generated_text = tokenizer.decode(generated_only)
            # print(f"Total input: {total_input}\n")
            # print(f"Input question: {question}\n")
            print(f"Generated text: {generated_text}\n")
            print(f"Gold answer: {gold_answer[batch_start + i]}\n")
            # print("\n" + "=" * 80)
            generated_answer = extract_answer_num(generated_text)
            is_correct = False
            if generated_answer is not None and float(generated_answer) == float(gold_answer[batch_start + i].replace(",", "")):
                total_correct += 1
                is_correct = True
            total_processed += 1
            wandb.log({"accuracy": total_correct/total_processed*100},step=total_processed)

            # Persist per-sample log as JSONL
            record = {
                "global_index": batch_start + i,
                "total_input": total_input,
                "input_question": question,
                "generated_text": generated_text,
                "gold_answer": gold_answer[batch_start + i],
                "extracted_answer": generated_answer,
                "correct": is_correct,
                "cfg": cfg,
                "cfg_style": cfg_style,
                "cfg_scale": cfg_scale,
                "cfg_scale1": cfg_scale1,
                "cfg_scale2": cfg_scale2,
                "steps": script_args.steps,
                "max_new_tokens": script_args.max_new_tokens,
                "block_length": script_args.block_length,
                "temperature": script_args.temperature,
                "remasking": script_args.remasking,
                "timestamp": timestamp,
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

