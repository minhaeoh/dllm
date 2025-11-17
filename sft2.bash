#!/bin/sh
#SBATCH -J eval_b_r
#SBATCH -N 1            
#SBATCH -p gpu01       
#SBATCH --gres=gpu:1
#SBATCH --ntasks=12
#SBATCH -o slurm_out/%j.out  # %j는 job ID를 의미합니다


# accelerate launch --config_file scripts/accelerate_configs/deepspeed_zero2.yaml examples/llada/sft.py 
# python examples/llada/generate_checkpoint.py --prompt


# DLM refinement baseline experiment
for model_name_or_path in "/home/minhae/diffusion/dllm/models/LLaDA-8B-SFT/gsm8k_filter_all_1_0_1_ml2056/checkpoint-final" "GSAI-ML/LLaDA-8B-Instruct"; do
    for prompt in 0 1; do
        echo "Running model=$model_name_or_path, prompt=$prompt"
        python examples/llada/generate_base.py \
            --model-name-or-path "$model_name_or_path" \
            --steps 128 \
            --max-new-tokens 256 \
            --block-length 32 \
            --temperature 0.0 \
            --remasking "low_confidence" \
            --seed 42 \
            --prompt $prompt
    done
done

# CFG 2 실험: static with different scales
# for scale in 0 0.5 1.0; do
#     echo "Running cfg=2, static, scale=$scale"
#     python examples/llada/generate_checkpoint.py \
#         --cfg 2 \
#         --cfg-style static \
#         --cfg-scale $scale \
#         --prompt
# done

# # CFG 2 실험: dynamic
# echo "Running cfg=2, dynamic"
# python examples/llada/generate_checkpoint.py \
#     --cfg 2 \
#     --cfg-style dynamic \
#     --prompt

# CFG 3 실험: static with different scale combinations
# for scale1 in 0 0.5 1.0 1.5 2.0; do
#     for scale2 in 0 0.5 1.0 1.5 2.0; do
#         echo "Running cfg=3, static, scale1=$scale1, scale2=$scale2"
#         python examples/llada/generate_checkpoint.py \
#             --cfg 3 \
#             --cfg-style static \
#             --cfg-scale1 $scale1 \
#             --cfg-scale2 $scale2
#     done
# done

# CFG 3 실험: dynamic
# echo "Running cfg=3, dynamic"
# python examples/llada/generate_checkpoint.py \
#     --cfg 3 \
#     --cfg-style dynamic