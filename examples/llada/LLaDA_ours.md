<h1 align="center">dLLM</h1>

<p align="center">
Training Diffusion Large Language Models Made Simple
</p>

<p align="center">
<img
  src="assets/logo.gif"
  alt="dLLM logo">
</p>

### env setting
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
### modified or new codes
1. examples/llada/generate_checkpoint.py : sft된 모델 checkpoint를 불러와서 our cfg method을 사용해 inference하는 코드
2. examples/llada/dllm/pipelines/llada/generate_sf.py : 기존 generate.py 코드에서 generate_two_cfg 함수 추가함.
3. examples/llada/dllm/utils/data_utils.py : 우리가 만든 dataset load 할 수 있도록 고침. 현재는 데이터셋 이름에 gsm8k가 포함 되어있어야됨.


### experiment 실행방법
examples/llada 이 폴더에 dllm 넣어서 사용중. 
```bash
accelerate launch --config_file scripts/accelerate_configs/deepspeed_zero2.yaml examples/llada/sft.py
```
이때 gpu 개수만큼 deepspeed_zero2.yaml에서 num_processes: 숫자 고쳐줘야됨

### dataset 생성
1. examples/llada/dataset/gsm8k_shepherd.py : gsm8k와 shpeherd 데이터 매칭시켜서 데이터 정리하기
2. examples/llada/dataset/sort_by_q.py : 1에서 만든 데이터에서 question으로 sorting 시키는 코드. 필요할 경우에만 사용
3. examples/llada/dataset/dataloader.py : 1 or 2에서 만든 데이터를 sft.py에서 사용할 수 있도록 DatasetDict 만들기


### dataset 구조
```bash
{"splits": ["train", "test"]}
{'id': 7466, 
'messages': [{'content': 'Question: Janet pays $40/hour for 3 hours per week of clarinet lessons and $28/hour for 5 hours a week of piano lessons. How much more does she spend on piano lessons than clarinet lessons in a year?', 'role': 'user'}, {'content': ' Correct Answer: First find the total Janet spends on clarinet lessons per week: $40/hour * 3 hours/week = $<<40*3=120>>120/week\nThen find the total Janet spends on piano lessons per week: $28/hour * 5 hours/week = $<<28*5=140>>140/week\nThen subtract her weekly clarinet spending from her weekly piano spending to find the weekly difference: $140/week - $120/week = $<<140-120=20>>20/week\nThen multiply the weekly difference by the number of weeks in a year to find the annual difference: $20/week * 52 weeks/year = $<<20*52=1040>>1040/year####1040', 'role': 'assistant'}], 
'source': 'gsm8k_q_cond'}
```


