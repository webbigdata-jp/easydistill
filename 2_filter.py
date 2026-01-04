import json
import os
from transformers import AutoTokenizer
from tqdm import tqdm

# ================= 設定 =================
# 入力ファイル（前処理済みのJSON）
input_file = "train_shuffled_subset.json"
# 出力ファイル（フィルタリング済み）
output_file = "train_filtered.json"

# モデルのパス（ローカルにダウンロードしたもの）
model_path = "shisa-ai/shisa-v2.1-qwen3-8b"

# 最大トークン長 (Configのmax_model_lenと同じか、生成分を考慮して少し小さくする)
# 教師モデルのmax_model_lenが4096なら、入力は3072くらいに抑えないと回答生成の余地がなくなります
MAX_LENGTH = 3072
# ========================================

def main():
    print(f"Loading tokenizer from {model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    print(f"Loading data from {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Original dataset size: {len(data)}")

    filtered_data = []
    dropped_count = 0

    print("Filtering data...")
    for item in tqdm(data):
        instruction = item["instruction"]
        
        # EasyDistillのテンプレート(chat_template_kd.jinja)をシミュレートしてテキスト化
        # これで正確な入力トークン数を見積もります
        formatted_text = (

            "<|im_start|>user\n"
            f"{instruction}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )

        # トークン数を計算
        token_ids = tokenizer.encode(formatted_text, add_special_tokens=False)
        length = len(token_ids)

        if length <= MAX_LENGTH:
            filtered_data.append(item)
        else:
            dropped_count += 1
            # どんなデータが落ちたか確認したい場合はコメントアウトを外す
            # print(f"Dropped item with length {length}: {instruction[:50]}...")

    print("-" * 30)
    print(f"Original count: {len(data)}")
    print(f"Filtered count: {len(filtered_data)}")
    print(f"Dropped count : {dropped_count}")
    print("-" * 30)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, indent=4, ensure_ascii=False)
    
    print(f"Saved filtered dataset to {output_file}")

if __name__ == "__main__":
    main()

