# Minhae added
## env setting
```bash
conda create -n dlm python=3.12 
conda activate dlm 
conda install nvidia/label/cuda-12.8.1::cuda-toolkit 
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 —index-url https://download.pytorch.org/whl/cu128 
pip install packaging 
pip install ray 
pip install omegaconf 
pip install transformers
pip install hydra-core —upgrade
pip install datasets
MAX_JOBS=8 pip install flash-attn —no-build-isolation
MAX_JOBS=8 pip install causal-conv1d --no-build-isolation
MAX_JOBS=8 pip install mamba-ssm --no-build-isolation
pip install lightning
pip install rich
pip install timm
pip install wandb
여기에 플러스로 한두개 있었던것 같은데 이건 코드 돌리면서 깔면 됨
```
## modified or new codes
1. examples/llada/generate_checkpoint.py : sft된 모델 checkpoint를 불러와서 our cfg method을 사용해 inference하는 코드
2. examples/llada/dllm/pipelines/llada/generate_sf.py : 기존 generate.py 코드에서 generate_two_cfg 함수 추가함.
3. examples/llada/dllm/utils/data_utils.py : 우리가 만든 dataset load 할 수 있도록 고침. 현재는 데이터셋 이름에 gsm8k가 포함 되어있어야됨.


## experiment 실행방법
examples/llada 이 폴더에 dllm 넣어서 사용중. 
```bash
accelerate launch --config_file scripts/accelerate_configs/deepspeed_zero2.yaml examples/llada/sft.py
```
이때 gpu 개수만큼 deepspeed_zero2.yaml에서 num_processes: 숫자 고쳐줘야됨

## dataset 생성
1. examples/llada/dataset/gsm8k_shepherd.py : gsm8k와 shpeherd 데이터 매칭시켜서 데이터 정리하기
2. examples/llada/dataset/sort_by_q.py : 1에서 만든 데이터에서 question으로 sorting 시키는 코드. 필요할 경우에만 사용
3. examples/llada/dataset/dataloader.py : 1 or 2에서 만든 데이터를 sft.py에서 사용할 수 있도록 DatasetDict 만들기


## dataset 구조
```bash
{"splits": ["train", "test"]}
{'id': 7466, 
'messages': [{'content': 'Question: Janet pays $40/hour for 3 hours per week of clarinet lessons and $28/hour for 5 hours a week of piano lessons. How much more does she spend on piano lessons than clarinet lessons in a year?', 'role': 'user'}, {'content': ' Correct Answer: First find the total Janet spends on clarinet lessons per week: $40/hour * 3 hours/week = $<<40*3=120>>120/week\nThen find the total Janet spends on piano lessons per week: $28/hour * 5 hours/week = $<<28*5=140>>140/week\nThen subtract her weekly clarinet spending from her weekly piano spending to find the weekly difference: $140/week - $120/week = $<<140-120=20>>20/week\nThen multiply the weekly difference by the number of weeks in a year to find the annual difference: $20/week * 52 weeks/year = $<<20*52=1040>>1040/year####1040', 'role': 'assistant'}], 
'source': 'gsm8k_q_cond'}
```




# LLaDA

> **Reference**  
> 📄 Paper: [Large Language Diffusion Models](https://arxiv.org/abs/2502.09992)
> 💻 Code: [github.com/ML-GSAI/LLaDA](https://github.com/ML-GSAI/LLaDA)

This directory provides examples for finetuning open-weight LLaDA models, reproducing LLaDA by training from scratch on public data (pretraining & finetuning), and batch sampling for generation tasks.

## Table of Contents
- [Setup](#setup)
- [Files overview](#files-overview)
- [Training](#training)
    - [Finetuning LLaDA-8B-Base](#finetuning-llada-8b-base)
    - [Pretraining & Finetuning from scratch](#pretraining--finetuning-from-scratch)
- [Sampling](#sampling)

## Setup
> [!IMPORTANT]  
> **Slurm users:** Update `scripts/train.slurm.sh` and `mkdir logps`: see [(optional) Slurm setup](/README.md/#optional-slurm-setup) for details.
>
> **MoE checkpoints:** For models like [LLaDA-MoE-7B-A1B-Base](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Base), set `"model_type"` to `"lladamoe"` in the checkpoint’s `config.json`:
> ```diff
> - "model_type": "llada",
> + "model_type": "lladamoe",
> ```
>


##  Files overview
```
# tools relevant with LLaDA
dllm/pipelines/llada
├── generate.py                     # Generation utilities
├── __init__.py                     # Package initialization
├── models/
│   ├── configuration_lladamoe.py   # LLaDA-MoE model configuration
│   ├── configuration_llada.py      # LLaDA model configuration
│   ├── modeling_lladamoe.py        # LLaDA-MoE model architecture
│   └── modeling_llada.py           # LLaDA model architecture
└── trainer.py                      # Training logic (pretraining and finetuning)

# example entry points for training / sampling
examples/llada
├── generate.py                     # Generation example
├── pt.py                           # Pretraining example
├── README.md                       # Documentation (you are here)
└── sft.py                          # Supervised finetuning example
```
> [!NOTE]
>  We fixed attention mask bugs in [`modeling_lladamoe.py`](/dllm/pipelines/llada/models/modeling_lladamoe.py) and [`modeling_llada.py`](/dllm/pipelines/llada/models/modeling_llada.py). We recommend loading models with `dllm.utils.get_tokenizer`; otherwise `import dllm` before calling `AutoModel.from_pretrained` to ensure the correct models from `dllm` are used. 
> 
>  We fixed bugs in `chat_template` and standardize `mask_token` through `dllm.utils.get_tokenizer`. If you use `AutoTokenizer`, keep in mind to set `chat_template` and `mask_token` appropriately yourselves.

<!-- > [!WARNING]  
> Before loading MoE checkpoints (e.g., [inclusionAI/LLaDA-MoE-7B-A1B-Base](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Base)), first overwrite the `model_type` field from `inclusionAI/LLaDA-MoE-7B-A1B-Base/config.json`:  
> ```diff
> - "model_type": "llada",
> + "model_type": "lladamoe",
> ``` -->

## Training

> [!NOTE]
> Use `--dataset_args "allenai/tulu-3-sft-mixture[train:10000,test:1000]"` to train / eval only on a subset; 
> 
> Use `--dataset_args "allenai/tulu-3-sft-mixture | OpenCoder-LLM/opc-sft-stage2[name:educational_instruct]"` to concatenate datasets.

### Finetuning [LLaDA-8B-Base](https://huggingface.co/GSAI-ML/LLaDA-8B-Base)
We support training models with either DDP or DeepSpeed ZeRO-{1,2,3}. For example, to SFT [LLaDA-8B-Base](https://huggingface.co/GSAI-ML/LLaDA-8B-Base) for instruction following on [allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) using DeepSpeed ZeRO-2 on 8 GPUs, run:
```shell
accelerate launch \
    --config_file scripts/accelerate_configs/deepspeed_zero2.yaml \
    examples/llada/sft.py \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Base" \
    --dataset_args "allenai/tulu-3-sft-mixture" \
    --output_dir "models/LLaDA-8B-SFT/tulu-3-sft-mixture" \
    --max_length 1024 \ 
    --num_train_epochs 4 \
    --learning_rate 2e-5
```
If you are using slurm and want to train across, for example, four nodes (32 GPUs total), run:
```shell
sbatch --nodes=4 --gres=gpu:8 scripts/train.slurm.sh \
    --accelerate_config "deepspeed_zero2" \
    --script_path "examples/llada/sft.py" \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Base" \
    --dataset_args "allenai/tulu-3-sft-mixture" \
    --output_dir "models/LLaDA-8B-SFT/tulu-3-sft-mixture" \
    --max_length 1024 \ 
    --num_train_epochs 4 \
    --learning_rate 2e-5
```

### Pretraining & finetuning from scratch
> [!NOTE]
> This is an educational example demonstrating how to reproduce LLaDA pretraining and finetuning on public data. We do not guarantee performance comparable to the official LLaDA models.

Pretrain on [mlfoundations/dclm-baseline-1.0](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) using 256 GPUs (32x8) and DeepSpeed ZeRO-2:
```shell
sbatch --nodes=32 --gres=gpu:8 scripts/train.slurm.sh \
    --accelerate_config "deepspeed_zero2" \
    --script_path "examples/llada/pt.py" \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Base" \
    --dataset_args "mlfoundations/dclm-baseline-1.0" \
    --output_dir "models/LLaDA-8B-PT/dclm-baseline-1.0" \
    --max_length 1024 \ 
    --max_steps 2000 \
    --learning_rate 3e-4
```
Finetune on [allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) using 8 GPUs and DeepSpeed ZeRO-2 for better instruction following:
```shell
sbatch --nodes=4 --gres=gpu:8 scripts/train.slurm.sh \
    --accelerate_config "deepspeed_zero2" \
    --script_path "examples/llada/sft.py" \
    --model_name_or_path "models/LLaDA-8B-PT/dclm-baseline-1.0/checkpoint-final" \
    --dataset_args "allenai/tulu-3-sft-mixture" \
    --output_dir "models/LLaDA-8B-SFT/tulu-3-sft-mixture" \
    --max_length 1024 \ 
    --num_train_epochs 4 \
    --learning_rate 2e-5
```

## Sampling
We support batch sampling for standard generation and infilling generation.
See [`examples/llada/generate.py`](/examples/llada/generate.py) for a full example.
```shell
python examples/llada/generate.py --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct"
```
