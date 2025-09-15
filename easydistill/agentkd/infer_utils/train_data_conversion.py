import json
import re


def conversion(input_path, output_path):

    with open(input_path, "r") as f:
        data = [json.loads(line) for line in f]

    all_saves = []
    for idx, item in enumerate(data):
        # dpo_data = item["dpo_data"]
        if "\": true" not in json.dumps(item["evaluation_result"]):
            continue
        if idx >= 15000:
            break
        question = item['original_task_info']['question']
        conversations = [{
            "from": "human",
            "value": question
        }]

        conversations.append({
            "from": "gpt",
            "value":  f"<first_thought>{item['first_thought'].replace('First-thought prefix:','').replace('First-thought:', '').strip()}</first_thought>"
        })
        conversations.append({
            "from": "human",
            "value": "Observation: None"
        })

        for step_item in item["steps"]:
            if "error" in step_item.keys():
                continue
            
            search_item = step_item['code']
            conversations.append({
                "from": "gpt",
                "value":  f"<thought>{step_item['thought'].replace('Thought: ', '')}</thought><code>{search_item}</code>"
            })

            conversations.append({
                "from": "human",
                "value": f"Observation: {step_item['observation']}".strip()
            })
        

        saves = {}
        saves["conversations"] = conversations[:-1]

        all_saves.append(saves)

    # save all_saves into a json file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_saves, f, ensure_ascii=False, indent=4)