import json
from datasets import load_dataset
from tqdm import tqdm

# ================= 設定 =================
dataset_name = "shisa-ai/shisa-v2.1-sharegpt"
input_file = "train_filtered.json"  # 既存のフィルタ済みデータ
output_file = "train_labeled_with_teacher.json"
# ========================================

def main():
    # 1. train_filtered.json を読み込み
    print(f"Loading {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        filtered_data = json.load(f)
    
    print(f"Filtered data count: {len(filtered_data)}")
    
    # instruction をキーにした set を作成（高速検索用）
    filtered_instructions = {item["instruction"] for item in filtered_data}
    
    # 2. 元データセットを読み込み
    print(f"Loading {dataset_name}...")
    ds = load_dataset(dataset_name, split="train")
    
    # 3. 元データから instruction -> output のマッピングを作成
    print("Building instruction -> output mapping...")
    instruction_to_output = {}
    
    for row in tqdm(ds):
        conversations = row['conversations']
        
        # 最初の human/gpt ペアを探す
        instruction = None
        output = None
        
        for turn in conversations:
            if turn['from'] == 'human' and instruction is None:
                instruction = turn['value']
            elif turn['from'] in ['gpt', 'assistant'] and instruction is not None and output is None:
                output = turn['value']
                break
        
        if instruction and output:
            instruction_to_output[instruction] = output
    
    print(f"Mapping created: {len(instruction_to_output)} entries")
    
    # 4. train_filtered.json の各 instruction に対して output を追加
    print("Matching instructions...")
    matched_data = []
    dropped_count = 0
    
    for item in tqdm(filtered_data):
        instruction = item["instruction"]
        
        if instruction in instruction_to_output:
            matched_data.append({
                "instruction": instruction,
                "output": instruction_to_output[instruction]
            })
        else:
            dropped_count += 1
    
    print("-" * 30)
    print(f"Input count: {len(filtered_data)}")
    print(f"Matched count: {len(matched_data)}")
    print(f"Dropped count (no match): {dropped_count}")
    print("-" * 30)
    
    # 5. 保存
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(matched_data, f, indent=4, ensure_ascii=False)
    
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    main()

