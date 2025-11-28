import datasets
from datasets import load_from_disk


def get_gsm8k_dataset():
  import random
  random.seed(42) # Set fixed seed for reproducibility
  """Get GSM8K dataset for finetuning."""
  raw_data_dir =  "/home/minhae/diffusion/dllm/examples/llada/dataset/before/math_gsm8k_full_dataset"
  print(f"Loading GSM8K dataset from: {raw_data_dir}")
  raw_data = load_from_disk(raw_data_dir)
  print(f"Raw data loaded: {len(raw_data)} samples")

  # Use select instead of slicing to avoid string conversion
  train_size = int(len(raw_data) * 0.9)

  splits = {
    'train': raw_data.select(range(train_size)),
    'test': raw_data.select(range(train_size, len(raw_data)))
  }
  print(f"Train split: {len(splits['train'])} samples")
  print(f"Test split: {len(splits['test'])} samples")

  def format_data_q_cond(d):
    question = "Solve the math problem step by step. The problem is: " + d['question'] + ". Your final answer should be a single numerical number, in the form 'the answer is: <answer>', at the end of your response."
    answer = ' Correct Solution: ' + d['gold_solution']+'The answer is: '+ d['gold_answer']
    messages = []
    messages.append({"role": "user", "content": question})
    messages.append({"role": "assistant", "content": answer})
    return messages, d["index"]
  
  def format_data_llm_cond(d):   
    answer = ' Correct Solution: ' + d['gold_solution']+'The answer is: ' + d['gold_answer'] 
    llm_answers = [d['llm_answer']] if isinstance(d['llm_answer'], str) else d['llm_answer']

    all_messages = []
    for llm_answer_text in llm_answers:
        messages = []
        messages.append({"role": "LLM generated Answer", "content": "Using the solutions from other agents as additional information, can you provide your answer to the math problem? This the solutions to the problem from other agent: "  + llm_answer_text})
        messages.append({"role": "assistant", "content": answer})
        all_messages.append((messages, d["index"]))
    
    return all_messages
  
  def format_data_genllm_cond(d):   
    answer = ' Correct Solution: ' + d['gold_solution']+'The answer is: ' + d['gold_answer'] 
    generated_solution = d['generated_solution']
    messages = []
    messages.append({"role": "LLM generated Answer", "content": "Using the solutions from other agents as additional information, can you provide your answer to the math problem? This the solutions to the problem from other agent: "  + generated_solution})       
    messages.append({"role": "assistant", "content": answer})
    return messages, d["index"]
  
  def format_data_q_genllm_cond(d):   
    question = "Solve the math problem step by step. The problem is: " + d['question'] + ". Your final answer should be a single numerical number, in the form 'the answer is: <answer>', at the end of your response."
    answer = ' Correct Solution: ' + d['gold_solution']+'The answer is: ' + d['gold_answer'] 
    generated_solution = d['generated_solution']
    messages = []
    messages.append({"role": "user", "content": question})
    messages.append({"role": "LLM generated Answer", "content": "Using the solutions from other agents as additional information, can you provide your answer to the math problem? This the solutions to the problem from other agent: "  + generated_solution})       
    messages.append({"role": "assistant", "content": answer})
    return messages, d["index"]

  def format_data_q_llm_cond(d):
    question = "Solve the math problem step by step. The problem is: " + d['question'] + ". Your final answer should be a single numerical number, in the form 'the answer is: <answer>', at the end of your response."
    answer = ' Correct Solution: ' + d['gold_solution']+'The answer is: ' + d['gold_answer']

    llm_answers = [d['llm_answer']] if isinstance(d['llm_answer'], str) else d['llm_answer']
    
    # 각 LLM 답변에 대해 별도의 메시지 리스트 생성
    all_messages = []
    for llm_answer_text in llm_answers:
        messages = []
        messages.append({"role": "user", "content": question})
        messages.append({"role": "LLM generated Answer", "content": "Using the solutions from other agents as additional information, can you provide your answer to the math problem? This the solutions to the problem from other agent: "  + llm_answer_text})
        messages.append({"role": "assistant", "content": answer})
        all_messages.append((messages, d["index"]))
    
    return all_messages
    


  # Process splits
  processed_splits = {}
  for split_name, split_data in splits.items():
    print(f"Processing {split_name} split...")
    processed_data = []
    for i, d in enumerate(split_data):
      if i % 1000 == 0:
        print(f"Processing sample {i}/{len(split_data)}")
      
      # Try each tokenization method and filter out None values
      try:
        task = d['source']
        formatted_q, index = format_data_q_cond(d)
        if formatted_q is not None:
          processed_data.append({"id":index,"messages": formatted_q,"source":"q_cond", "task": task})

        formatted_q_llm = format_data_q_llm_cond(d)
        for messages, index in formatted_q_llm:
          if messages is not None:
            processed_data.append({"id":index, "messages": messages, "source":"q_llm_cond", "task": task})
            
        formatted_q_genllm, index = format_data_q_genllm_cond(d)
        if formatted_q_genllm is not None:
          processed_data.append({"id":index, "messages": formatted_q_genllm, "source":"q_genllm_cond", "task": task})
            
        formatted_llm = format_data_llm_cond(d)
        for messages, index in formatted_llm:
          if messages is not None:
            processed_data.append({"id":index,"messages": messages,"source":"llm_cond", "task": task})

        formatted_genllm, index = format_data_genllm_cond(d)
        if formatted_genllm is not None:
          processed_data.append({"id":index,"messages": formatted_genllm,"source":"genllm_cond", "task": task})

      except Exception as e:
        print(f"Error processing sample {i}: {e}")
        print(f"Sample keys: {list(d.keys()) if hasattr(d, 'keys') else 'No keys'}")
        raise
    
    # Convert to HuggingFace Dataset
    processed_splits[split_name] = datasets.Dataset.from_list(processed_data)
    print(f"Processed {split_name}: {len(processed_data)} samples")
  
  # Create DatasetDict
  dataset_dict = datasets.DatasetDict(processed_splits)
  print('train samples:', len(dataset_dict['train']))
  print('test samples:', len(dataset_dict['test']))
  return dataset_dict


dataset_dict = get_gsm8k_dataset()

# Save as DatasetDict
dataset_dict.save_to_disk("/home/minhae/diffusion/dllm/examples/llada/dataset/trainset/math_gsm8k_final")



