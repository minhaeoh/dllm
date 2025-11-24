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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import load_from_disk

@dataclass
class AllArguments:
    """Combined arguments for data and script"""
    # Data arguments
    dataset_args: str = "/home/minhae/diffusion/dllm/examples/llada/dataset/testset/gsm8k_llama3.1_8b_instruct"
    num_proc: int = 8
    
    # Script arguments
    task : str = "gsm8k-filtered"
    model_name: str = "Instruct-SFT-middle"
    model_path: str = "/home/minhae/diffusion/dllm/models_nlp/LLaDA-8B-Instruct-SFT/math_gsm8k_filter_all_1_0_1_ml2056/checkpoint-335746"
    # model_name: str = "LLaDA-8B-Instruct"
    # model_path: str = "GSAI-ML/LLaDA-8B-Instruct"
    steps: int = 128
    max_new_tokens: int = 256
    block_length: int = 32
    temperature: float = 0.0
    remasking: str = "low_confidence"
    seed: int = 42
    cfg: int = 3  # 2/3/0/4
    cfg_style: str = "dynamic"  # static/dynamic
    guidance_annealing: bool = False
    guidance_type: str = "threshold" # "threshold" or "linear"
    guidance_step: float = 0.5
    epsilon: float = 0.0
    cfg_scale: float = 0.0
    cfg_scale1: float = 2.0
    cfg_scale2: float = 0.5
    alpha: float = 2.0
    beta: float = 1.0
    prompt: bool = True # False:no prompt, True:debate prompt
    log_cfg_scales: bool = False
    reverse : bool = False


args = tyro.cli(AllArguments)
transformers.set_seed(args.seed)

# Extract config values
task = args.task

cfg = args.cfg
cfg_style = args.cfg_style
cfg_scale = args.cfg_scale
cfg_scale1 = args.cfg_scale1
cfg_scale2 = args.cfg_scale2
guidance_annealing = args.guidance_annealing
guidance_type = args.guidance_type
guidance_step = args.guidance_step
epsilon = args.epsilon
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Build config string based on cfg settings
if cfg == 2:
    if cfg_style == "dynamic":
        config_str = f"cfg{cfg}_{cfg_style}"
    elif cfg_style == "static":
        config_str = f"cfg{cfg}_{cfg_style}_{cfg_scale}"
    else:
        raise ValueError(f"Invalid cfg_style: {cfg_style}")
elif cfg == 3:
    if cfg_style == "dynamic":
        config_str = f"cfg{cfg}_{cfg_style}_alpha_{args.alpha}_beta_{args.beta}_eps_{args.epsilon}"
        if args.guidance_annealing:
            config_str += "_guidance_annealing_" + str(guidance_type) 
            if guidance_type == "threshold":
                config_str += "_" + str(guidance_step)
        if args.log_cfg_scales:
            config_str += "_heatmap"
    elif cfg_style == "static":
        config_str = f"cfg{cfg}_{cfg_style}_{cfg_scale1}_{cfg_scale2}"
    else:
        raise ValueError(f"Invalid cfg_style: {cfg_style}")
elif cfg == 0:
    config_str = "base_refinement"
elif cfg == 4:
    config_str = f"cfg{cfg}_{cfg_style}_alpha_{args.alpha}_beta_{args.beta}_eps_{args.epsilon}"
    if args.guidance_annealing:
        config_str += "_guidance_annealing_" + str(guidance_type) 
        if guidance_type == "threshold":
            config_str += "_" + str(guidance_step)
    if args.log_cfg_scales:
        config_str += "_heatmap"
else:
    raise ValueError(f"Invalid cfg: {cfg}")

if args.prompt:
    config_str += "_prompt"
else:
    config_str += "_noprompt"
config_str += "_max_"+str(args.max_new_tokens)
if args.reverse:
    config_str += "_reverse"
# Output logging setup
output_dir = os.path.join(os.path.dirname(__file__), "outputs",task,args.model_name,config_str)
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, f"{timestamp}.jsonl")

# Create heatmap directory if logging cfg_scales
if args.cfg == 3 and args.log_cfg_scales:
    heatmap_dir = os.path.join(output_dir, f"heatmaps_{timestamp}")
    os.makedirs(heatmap_dir, exist_ok=True)

wandb.init(project="evaluation-gsm8k-filtered", name=f"{args.model_name}_{config_str}_{timestamp}")
wandb.config.update({
    "model": args.model_name,
    "config": config_str,
    "timestamp": timestamp,
    "task": task,
    "cfg": cfg,
    "cfg_style": cfg_style,
    "cfg_scale": cfg_scale,
    "cfg_scale1": cfg_scale1,
    "cfg_scale2": cfg_scale2,
    "guidance_type": guidance_type,
    "guidance_step": guidance_step,
    "epsilon": epsilon,
    "guidance_annealing": guidance_annealing,
    "remasking": args.remasking,
    "prompt": args.prompt,
    "reverse": args.reverse,
    "steps": args.steps,
    "max_new_tokens": args.max_new_tokens,
    "block_length": args.block_length,
    "temperature": args.temperature,
})


# # Load model & tokenizer
model = dllm.utils.get_model(model_name_or_path=args.model_path).eval()
tokenizer = dllm.utils.get_tokenizer(model_name_or_path=args.model_path, model=model)

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
    # for i in range(len(messages)):
    #     print(f"Message {i}: {messages[i]}")
    messages = []
    if args.reverse:
        if prompt :
            messages.append({"role": "user", "content": "This the solutions to the problem from other agent: " + row["llm_output"]})
            messages.append({"role": "user", "content": "Using the solutions from other agents as additional information, can you provide your answer to the math problem? The original math problem is " + row["question"] + ". Your final answer should be a single numerical number, in the form 'the answer is: <answer>', at the end of your response."})
        else :
            messages.append({"role": "LLM generated Answer", "content": row["llm_output"]})
            messages.append({"role": "user", "content": row["question"]})
            
        
    else:
        if prompt :
            messages.append({"role": "user", "content": "Solve the math problem step by step. The problem is: " + row["question"] + ". Your final answer should be a single numerical number, in the form 'the answer is: <answer>', at the end of your response."})
            messages.append({"role": "user", "content": "Using the solutions from other agents as additional information, can you provide your answer to the math problem? This the solutions to the problem from other agent: " + row["llm_output"]})
        else :
            messages.append({"role": "user", "content": row["question"]})
            messages.append({"role": "LLM generated Answer", "content": row["llm_output"]})
        
    # #for baseline
    # if script_args.prompt:
    #     messages[0]["content"] = "Carefully solve the problem step by step. Finish with The answer is: <answer>. " + messages[0]["content"]
    
    q_llm_input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    )[0].tolist()

    q_input_ids = tokenizer.apply_chat_template(
        messages[0],
        add_generation_prompt=False,
        tokenize=True,
        return_tensors="pt",
    )[0].tolist()
    return {"q_llm_input_ids": q_llm_input_ids, "q_input_ids": q_input_ids, "q_len": len(q_input_ids), "gold_answer": row["gold_answer"].replace(",", ""), 'initial_correct': row['llm_correct'], 'initial_answer': row['llm_pred_answer']}



def sft_map_fn(row) -> dict:
    return custom_apply_chat_template(row, args.prompt)

with accelerate.PartialState().local_main_process_first():
    dataset = load_from_disk(args.dataset_args)
    if task == "gsm8k-filtered":
        dataset = dataset.filter(lambda x: x['llm_correct'] == False)
        print("filtered test dataset length: ", len(dataset))
    else:
        print("total test dataset length: ", len(dataset))
    # INSERT_YOUR_CODE
    # Ensure only unique ids: keep only the first occurrence if duplicated
    
    results = dataset.map(sft_map_fn, num_proc=args.num_proc, remove_columns=dataset.column_names)
    # Move to device after multiprocessing is done
    input_ids_list = [torch.tensor(r["q_llm_input_ids"]).to(model.device) for r in results]
    # input_ids_list = [torch.tensor(r["q_input_ids"]).to(model.device) for r in results]
    q_len = [r["q_len"] for r in results]
    gold_answer = [r["gold_answer"] for r in results]



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
        
        if cfg == 0:
            out = llada.generate(
            model,
            tokenizer,
            batch_input_ids,
            steps=args.steps,
            max_new_tokens=args.max_new_tokens,
            block_length=args.block_length,
            temperature=args.temperature,
            )
            
        elif cfg == 2:
            if cfg_style == "dynamic":
                out = llada.generate_two_condition_dynamic(
                    model,
                    tokenizer,
                    batch_input_ids,
                    batch_q_len,
                    steps=args.steps,
                    max_new_tokens=args.max_new_tokens,
                    block_length=args.block_length,
                    temperature=args.temperature,
                    remasking=args.remasking
                )
            elif cfg_style == "static":
                out = llada.generate_two_condition(
                    model,
                    tokenizer,
                    batch_input_ids,
                    batch_q_len,
                    steps=args.steps,
                    max_new_tokens=args.max_new_tokens,
                    block_length=args.block_length,
                    temperature=args.temperature,
                    remasking=args.remasking,
                    cfg_scale=cfg_scale
                )
        elif cfg == 3:
            if cfg_style == "dynamic":
                # Log cfg_scales for all samples if enabled
                result = llada.generate_two_cfg_adaptive(
                    model,
                    tokenizer,
                    batch_input_ids,
                    batch_q_len,
                    steps=args.steps,
                    max_new_tokens=args.max_new_tokens,
                    block_length=args.block_length,
                    temperature=args.temperature,
                    remasking=args.remasking,
                    return_dict_in_generate=args.log_cfg_scales,
                    log_cfg_scales=args.log_cfg_scales,
                    alpha=args.alpha,
                    beta=args.beta,
                    guidance_annealing=guidance_annealing,
                    guidance_type=guidance_type,
                    guidance_step=guidance_step,
                    epsilon=epsilon
                )
                
                # Extract sequences and log cfg_scales if available
                if args.log_cfg_scales:
                    out = result["sequences"]
                    cfg_scale1_history = result.get("cfg_scale1_history", None)
                    cfg_scale2_history = result.get("cfg_scale2_history", None)
                    c1_history = result.get("c1_history", None)
                    c2_history = result.get("c2_history", None)
                    
                    # Log cfg_scale and confidence heatmaps to local files for ALL samples in batch
                    if cfg_scale1_history and cfg_scale2_history and c1_history and c2_history:
                        # Stack all steps: [num_steps, B, max_new_tokens]
                        scale1_tensor = torch.stack(cfg_scale1_history, dim=0).float()  # [num_steps, B, max_new_tokens]
                        scale2_tensor = torch.stack(cfg_scale2_history, dim=0).float()  # [num_steps, B, max_new_tokens]
                        c1_tensor = torch.stack(c1_history, dim=0).float()  # [num_steps, B, max_new_tokens]
                        c2_tensor = torch.stack(c2_history, dim=0).float()  # [num_steps, B, max_new_tokens]
                        
                        # Create heatmap for each sample in the batch
                        for sample_idx in range(scale1_tensor.shape[1]):
                            scale1_array = scale1_tensor[:, sample_idx, :].numpy()  # [num_steps, max_new_tokens]
                            scale2_array = scale2_tensor[:, sample_idx, :].numpy()  # [num_steps, max_new_tokens]
                            c1_array = c1_tensor[:, sample_idx, :].numpy()  # [num_steps, max_new_tokens]
                            c2_array = c2_tensor[:, sample_idx, :].numpy()  # [num_steps, max_new_tokens]
                            
                            # Create heatmaps (4 subplots: cfg_scale1, cfg_scale2, c1, c2)
                            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
                            
                            # CFG Scale 1 heatmap
                            sns.heatmap(scale1_array, ax=axes[0, 0], cmap='YlOrRd', cbar_kws={'label': 'cfg_scale1'})
                            axes[0, 0].set_title(f'Sample {batch_start + sample_idx}: CFG Scale 1 (cond1 confidence)')
                            axes[0, 0].set_xlabel('Token Position')
                            axes[0, 0].set_ylabel('Generation Step')
                            
                            # CFG Scale 2 heatmap
                            sns.heatmap(scale2_array, ax=axes[0, 1], cmap='YlGnBu', cbar_kws={'label': 'cfg_scale2'})
                            axes[0, 1].set_title(f'Sample {batch_start + sample_idx}: CFG Scale 2 (cond2 advantage)')
                            axes[0, 1].set_xlabel('Token Position')
                            axes[0, 1].set_ylabel('Generation Step')
                            
                            # C1 confidence heatmap
                            sns.heatmap(c1_array, ax=axes[1, 0], cmap='Greens', cbar_kws={'label': 'c1'})
                            axes[1, 0].set_title(f'Sample {batch_start + sample_idx}: C1 Confidence (cond1)')
                            axes[1, 0].set_xlabel('Token Position')
                            axes[1, 0].set_ylabel('Generation Step')
                            
                            # C2 confidence heatmap
                            sns.heatmap(c2_array, ax=axes[1, 1], cmap='Blues', cbar_kws={'label': 'c2'})
                            axes[1, 1].set_title(f'Sample {batch_start + sample_idx}: C2 Confidence (cond2)')
                            axes[1, 1].set_xlabel('Token Position')
                            axes[1, 1].set_ylabel('Generation Step')
                            
                            plt.tight_layout()
                            
                            # Save heatmap to file (local only, not to wandb)
                            heatmap_file = os.path.join(heatmap_dir, f"sample_{batch_start + sample_idx}.png")
                            fig.savefig(heatmap_file, dpi=150, bbox_inches='tight')
                            plt.close(fig)
                else:
                    out = result
                    
            elif cfg_style == "static":
                out = llada.generate_two_cfg(
                    model,
                    tokenizer,
                    batch_input_ids,
                    batch_q_len,
                    steps=args.steps,
                    max_new_tokens=args.max_new_tokens,
                    block_length=args.block_length,
                    temperature=args.temperature,
                    remasking=args.remasking,
                    cfg_scale1=cfg_scale1,
                    cfg_scale2=cfg_scale2
                )
            else:
                raise ValueError(f"Invalid cfg_style: {cfg_style}")
        elif cfg == 4:
            # Log cfg_scales for all samples if enabled
            result = llada.generate_two_cfg_adaptive_uncond(
                model,
                tokenizer,
                batch_input_ids,
                batch_q_len,
                steps=args.steps,
                max_new_tokens=args.max_new_tokens,
                block_length=args.block_length,
                temperature=args.temperature,
                remasking=args.remasking,
                return_dict_in_generate=args.log_cfg_scales,
                log_cfg_scales=args.log_cfg_scales,
                alpha=args.alpha,
                beta=args.beta,
                guidance_annealing=guidance_annealing,
                guidance_type=guidance_type,
                guidance_step=guidance_step,
                epsilon=epsilon
            )
            
            # Extract sequences and log cfg_scales if available
            if args.log_cfg_scales:
                out = result["sequences"]
                cfg_scale1_history = result.get("cfg_scale1_history", None)
                cfg_scale2_history = result.get("cfg_scale2_history", None)
                c1_history = result.get("c1_history", None)
                c2_history = result.get("c2_history", None)
                
                # Log cfg_scale and confidence heatmaps to local files for ALL samples in batch
                if cfg_scale1_history and cfg_scale2_history and c1_history and c2_history:
                    # Stack all steps: [num_steps, B, max_new_tokens]
                    scale1_tensor = torch.stack(cfg_scale1_history, dim=0).float()  # [num_steps, B, max_new_tokens]
                    scale2_tensor = torch.stack(cfg_scale2_history, dim=0).float()  # [num_steps, B, max_new_tokens]
                    c1_tensor = torch.stack(c1_history, dim=0).float()  # [num_steps, B, max_new_tokens]
                    c2_tensor = torch.stack(c2_history, dim=0).float()  # [num_steps, B, max_new_tokens]
                    
                    # Create heatmap for each sample in the batch
                    for sample_idx in range(scale1_tensor.shape[1]):
                        scale1_array = scale1_tensor[:, sample_idx, :].numpy()  # [num_steps, max_new_tokens]
                        scale2_array = scale2_tensor[:, sample_idx, :].numpy()  # [num_steps, max_new_tokens]
                        c1_array = c1_tensor[:, sample_idx, :].numpy()  # [num_steps, max_new_tokens]
                        c2_array = c2_tensor[:, sample_idx, :].numpy()  # [num_steps, max_new_tokens]
                        
                        # Create heatmaps (4 subplots: cfg_scale1, cfg_scale2, c1, c2)
                        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
                        
                        # CFG Scale 1 heatmap
                        sns.heatmap(scale1_array, ax=axes[0, 0], cmap='YlOrRd', cbar_kws={'label': 'cfg_scale1'})
                        axes[0, 0].set_title(f'Sample {batch_start + sample_idx}: CFG Scale 1 (cond1 confidence)')
                        axes[0, 0].set_xlabel('Token Position')
                        axes[0, 0].set_ylabel('Generation Step')
                        
                        # CFG Scale 2 heatmap
                        sns.heatmap(scale2_array, ax=axes[0, 1], cmap='YlGnBu', cbar_kws={'label': 'cfg_scale2'})
                        axes[0, 1].set_title(f'Sample {batch_start + sample_idx}: CFG Scale 2 (cond2 advantage)')
                        axes[0, 1].set_xlabel('Token Position')
                        axes[0, 1].set_ylabel('Generation Step')
                        
                        # C1 confidence heatmap
                        sns.heatmap(c1_array, ax=axes[1, 0], cmap='Greens', cbar_kws={'label': 'c1'})
                        axes[1, 0].set_title(f'Sample {batch_start + sample_idx}: C1 Confidence (cond1)')
                        axes[1, 0].set_xlabel('Token Position')
                        axes[1, 0].set_ylabel('Generation Step')
                        
                        # C2 confidence heatmap
                        sns.heatmap(c2_array, ax=axes[1, 1], cmap='Blues', cbar_kws={'label': 'c2'})
                        axes[1, 1].set_title(f'Sample {batch_start + sample_idx}: C2 Confidence (cond2)')
                        axes[1, 1].set_xlabel('Token Position')
                        axes[1, 1].set_ylabel('Generation Step')
                        
                        plt.tight_layout()
                        
                        # Save heatmap to file (local only, not to wandb)
                        heatmap_file = os.path.join(heatmap_dir, f"sample_{batch_start + sample_idx}.png")
                        fig.savefig(heatmap_file, dpi=150, bbox_inches='tight')
                        plt.close(fig)
            else:
                out = result
            
        else:
            raise ValueError(f"Invalid cfg: {cfg}")

        
        for i, o in enumerate(out):
            # Extract only the generated part (after the last assistant header)
            start_index = len(batch_input_ids[i])
            stop_index = start_index + args.max_new_tokens
            generated_only = o[start_index:stop_index]
            total_input = tokenizer.decode(batch_input_ids[i])
            question = tokenizer.decode(batch_input_ids[i][:batch_q_len[i]])
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
            wandb.log({"accuracy": total_correct/total_processed*100, "refined_probs": total_correct},step=total_processed)

            # Persist per-sample log as JSONL
            record = {
                "global_index": batch_start + i,
                "accuracy": total_correct/total_processed*100,
                "refined_correct": is_correct,                
                "initial_correct": results[batch_start + i]["initial_correct"],
                "alpha": args.alpha,
                "beta": args.beta,
                "extracted_answer": generated_answer,
                "initial_answer": results[batch_start + i]["initial_answer"],
                "gold_answer": results[batch_start + i]["gold_answer"],
                "total_input": total_input,
                "input_question": question,
                "generated_text": generated_text,                
                "cfg": cfg,
                "cfg_style": cfg_style,
                "cfg_scale": cfg_scale,
                "cfg_scale1": cfg_scale1,
                "cfg_scale2": cfg_scale2,
                "guidance_type": guidance_type,
                "guidance_step": guidance_step,
                "prompt": args.prompt,
                "steps": args.steps,
                "max_new_tokens": args.max_new_tokens,
                "block_length": args.block_length,
                "temperature": args.temperature,
                "remasking": args.remasking,
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

