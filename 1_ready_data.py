import json
import random
from datasets import load_dataset

# 1. データセットの読み込み
dataset_name = "shisa-ai/shisa-v2.1-sharegpt"
print(f"Loading {dataset_name}...")
ds = load_dataset(dataset_name, split="train")

# 2. シャッフルして25%を抽出
# 全体の行数を取得し、その25%の数を計算
total_rows = len(ds)
sample_size = int(total_rows * 0.05)
print(f"Total rows: {total_rows}, Sampling 20%: {sample_size}")

# シャッフルされたインデックスを使用してサブセットを作成
shuffled_ds = ds.shuffle(seed=42).select(range(sample_size))

# 3. EasyDistill用にフォーマット変換
# "conversations" から最初のユーザー入力を "instruction" として抽出します
formatted_data = []

for row in shuffled_ds:
    conversations = row['conversations']
    
    # 最初の人間側の発言を探す
    instruction = ""
    for turn in conversations:
        if turn['from'] == 'human':
            instruction = turn['value']
            break
            
    if instruction:
        formatted_data.append({
            "instruction": instruction
        })

# 4. ファイルに保存 (EasyDistillの入力用)
output_file = "train_shuffled_subset.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(formatted_data, f, indent=4, ensure_ascii=False)

print(f"Saved to {output_file}")

