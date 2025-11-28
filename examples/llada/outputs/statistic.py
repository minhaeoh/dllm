import json
import os
import glob



def read_file(file):
    stats = {'incorrect-correct': 0, 'incorrect-incorrect': 0, 'total': 0}
            
    with open(file, "r") as f:
        for line in f:
            data = json.loads(line)
            if data['initial_correct'] == False and data['refined_correct'] == False:
                stats['incorrect-incorrect'] += 1
            else:
                stats['incorrect-correct'] += 1
            stats['total'] += 1
    return stats

def read_files(dir):
    # cfg4로 시작하는 모든 폴더 찾기
    folders = [f for f in os.listdir(dir) if os.path.isdir(os.path.join(dir, f)) and f.startswith('cfg3_')]

    print(f"Found {len(folders)} folders starting with 'cfg3_'\n")

    # 각 cfg4 폴더에 대해 처리
    for folder in sorted(folders):
        folder_path = os.path.join(dir, folder)
        
        # 폴더 안의 모든 jsonl 파일 찾기
        jsonl_files = glob.glob(os.path.join(folder_path, "*.jsonl"))
        
        if not jsonl_files:
            print(f"[{folder}] No jsonl files found\n")
            continue
        
        # 각 jsonl 파일에 대해 통계 계산
        for jsonl_file in sorted(jsonl_files):
            stats = read_file(jsonl_file)
            
            # 정확도 계산
            if stats['total'] > 0:
                refined_acc = stats['incorrect-correct'] / stats['total'] * 100
            
            
            file_name = os.path.basename(jsonl_file)
            print(f"[{folder}]")
            print(f"  {stats}")
            print(f"  Refined Accuracy: {refined_acc:.2f}%")
            print()

if __name__ == "__main__":
    dir = "/home/minhae/diffusion/dllm/examples/llada/outputs/gsm8k-filtered/Instruct-SFT-middle"
    file = "/home/minhae/diffusion/dllm/examples/llada/outputs/gsm8k-filtered/Instruct-SFT-middle/cfg3/cfg3_dynamic_alpha_2.0_beta_1.0_eps_0.25_guidance_annealing_0.2_prompt_max_256/20251122_165501.jsonl"
    # print(read_file(file))
    read_files(dir)