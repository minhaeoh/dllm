from datasets import load_from_disk
import json

data_dir = "/home/minhae/diffusion/dllm/examples/llada/dataset/before/math_gsm8k_grouped_by_index"
json_path = "/data/diffusion/dataset/sft/gold_sft/all_predictions.jsonl"


# dataset 로드
dataset = load_from_disk(data_dir)

# 1. jsonl의 index -> generated_solution 매핑 생성
index_to_generated_solution = {}
with open(json_path, "r") as f:
    for line in f:
        data = json.loads(line)
        idx = data.get("index", None)
        gen_sol = data.get("generated_solution", None)
        gen_corr = data.get("is_correct", None)
        if idx is not None:
            index_to_generated_solution[idx] = gen_sol, gen_corr

# 2. dataset에 "generated_solution" 필드 추가
#    (존재하는 index에만 추가, 없으면 None)
def add_generated_solution(example):
    idx = example.get("index", None)
    gen_sol, gen_corr = index_to_generated_solution.get(idx, None)
    example["generated_solution"] = gen_sol
    example["generated_correct"] = gen_corr
    return example

dataset_with_generated = dataset.map(add_generated_solution)
dataset_with_generated.save_to_disk("/home/minhae/diffusion/dllm/examples/llada/dataset/before/math_gsm8k_full_dataset")

# 3. 결과 확인 (예: 처음 3개 샘플)
for i in range(3):
    print(f"Sample {i}:")
    print(" Index:", dataset_with_generated[i]["index"])
    print(" Generated Solution:", dataset_with_generated[i].get("generated_solution", None))
    print("---")

# # dataset 로드
# dataset = load_from_disk(data_dir)

# # dataset의 index->question 매핑 생성
# index_to_question_dataset = {}
# for sample in dataset:
#     idx = sample["index"]
#     q = sample["question"]
#     index_to_question_dataset[idx] = q

# # jsonl의 index->question 매핑 생성
# index_to_question_json = {}
# with open(json_path, "r") as f:
#     for line in f:
#         data = json.loads(line)
#         idx = data.get("index", None)
#         q = data.get("question", None)
#         if idx is not None:
#             index_to_question_json[idx] = q

# # index 겹치는 것 체크
# shared_indices = set(index_to_question_dataset.keys()) & set(index_to_question_json.keys())
# print(f"공유되는 index 개수: {len(shared_indices)}")

# diff_questions = []
# for idx in shared_indices:
#     q1 = index_to_question_dataset[idx]
#     q2 = index_to_question_json[idx]
#     if q1 != q2:
#         diff_questions.append((idx, q1, q2))

# print(f"question이 다른 shared index 개수: {len(diff_questions)}")
# if diff_questions:
#     print("예시 (최대 5개):")
#     for idx, q1, q2 in diff_questions[:5]:
#         print(f"index {idx}:")
#         print(f"  dataset question: {q1[:100]}...")
#         print(f"  json question   : {q2[:100]}...")
# else:
#     print("모든 공유 index에서 question이 같습니다.")


# def check_invalid_samples(dataset):
#     # 이상치 조건 체크
#     invalid_samples = []
#     print(dataset.column_names)
#     print(dataset[0])
#     for idx, sample in enumerate(dataset):
#         if (sample['index'] == -1 or
#             sample['question'] == "" or
#             sample['LLM_answer'] == "" or
#             sample['gold_answer'] == None or
#             sample['gold_solution'] == None or
#             sample['match_source'] == "" or
#             sample['match_type'] == ""):
#             invalid_samples.append((idx, sample))

#     # 결과 출력
#     print(f"\nTotal samples: {len(dataset)}")
#     print(f"Invalid samples found: {len(invalid_samples)}")
    
#     if invalid_samples:
#         print("\nExample of invalid sample:")
#         idx, sample = invalid_samples[0]
#         print(f"Sample index in dataset: {idx}")
#         for key, value in sample.items():
#             if isinstance(value, str):
#                 print(f"{key}: {value[:100]}...")
#             else:
#                 print(f"{key}: {type(value)} - {value}")
        
#         # 각 필드별 "none" 또는 -1 카운트
#         field_counts = {
#             'index=-1': sum(1 for _, s in invalid_samples if s['index'] == -1),
#             'question=""': sum(1 for _, s in invalid_samples if s['question'] == ""),
#             'LLM_answer=""': sum(1 for _, s in invalid_samples if s['LLM_answer'] == ""),
#             'gold_answer=None': sum(1 for _, s in invalid_samples if s['gold_answer'] == None),
#             'gold_solution=None': sum(1 for _, s in invalid_samples if s['gold_solution'] == None),
#             'match_source=""': sum(1 for _, s in invalid_samples if s['match_source'] == ""),
#             'match_type=""': sum(1 for _, s in invalid_samples if s['match_type'] == "")
#         }
#         print("\nBreakdown of invalid fields:")
#         for field, count in field_counts.items():
#             print(f"{field}: {count} samples")

# # raw_data = load_from_disk('/home/minhae/diffusion/dllm/examples/llada/dataset/gsm8k_match_index_llm_answer2')
# # check_invalid_samples(raw_data)
# data = load_from_disk('/home/minhae/diffusion/dllm/examples/llada/dataset/gsm8k_1_0_1')
# # print(data.column_names)
# print(data['train'][0])
# # print(len(data['train']))
# # print(len(data['test']))