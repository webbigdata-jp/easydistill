import json
import re

def convert_jsonl_to_conversations(input_file, output_file):
    conversations = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
                
                eval_result = data.get('evaluation_result', '{}')
                if isinstance(eval_result, str):
                    eval_dict = json.loads(eval_result)
                else:
                    eval_dict = eval_result
                
                if not eval_dict.get('equivalent', False):
                    continue
                
                question = data.get('original_task_info', {}).get('question', '')
                if not question:
                    continue
                
                conversation = []
                
                conversation.append({
                    "from": "human",
                    "value": question
                })
                
                first_thought = data.get('first_thought', '')
                if first_thought:
                    conversation.append({
                        "from": "gpt",
                        "value": f"<first_thought>{first_thought}</first_thought>"
                    })
                
                steps = data.get('steps', [])
                for i, step in enumerate(steps):
                    thought = step.get('thought', '')
                    code = step.get('code', '')
                    observation = step.get('observation', '')
                    
                    if i > 0 or not first_thought:
                        conversation.append({
                            "from": "human", 
                            "value": f"Observation: {observation}"
                        })
                    
                    gpt_value = ""
                    if thought:
                        thought_content = thought.replace("Thought: ", "") if thought.startswith("Thought: ") else thought
                        gpt_value += f"<thought>{thought_content}</thought>"
                    
                    if code:
                        gpt_value += f"<code>{code}</code>"
                    
                    if gpt_value:
                        conversation.append({
                            "from": "gpt",
                            "value": gpt_value
                        })
                
                if steps:
                    last_observation = steps[-1].get('observation', '')
                    conversation.append({
                        "from": "human",
                        "value": f"Observation: {last_observation}"
                    })
                
                conversations.append({
                    "conversations": conversation
                })
                
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON line: {e}")
                continue
            except Exception as e:
                print(f"Error processing line: {e}")
                continue
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(conversations, f, indent=2, ensure_ascii=False)
    
    print(f"Converted {len(conversations)} entries to {output_file}")
    return len(conversations)

def preview_conversion(input_file, num_lines=1):
    print("Preview of conversion:")
    print("=" * 50)
    
    count = 0
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
                
                eval_result = data.get('evaluation_result', '{}')
                if isinstance(eval_result, str):
                    eval_dict = json.loads(eval_result)
                else:
                    eval_dict = eval_result
                
                if not eval_dict.get('equivalent', False):
                    continue
                
                count += 1
                if count > num_lines:
                    break
                
                print(f"Entry {count}:")
                print(f"Question: {data.get('original_task_info', {}).get('question', '')[:100]}...")
                print(f"Number of steps: {len(data.get('steps', []))}")
                print(f"Has first_thought: {'first_thought' in data}")
                print("-" * 30)
                
            except Exception as e:
                print(f"Error in preview: {e}")
                continue

# convert generated traces to train data
if __name__ == "__main__":
    input_file = "traces.jsonl" # generated traces
    output_file = "data.json" # train data
    
    print("Previewing conversion...")
    try:
        preview_conversion(input_file, 1)
    except FileNotFoundError:
        print(f"File {input_file} not found. Please update the file path.")
    except Exception as e:
        print(f"Error in preview: {e}")
    
    print("\nPerforming conversion...")
    try:
        count = convert_jsonl_to_conversations(input_file, output_file)
        print(f"Successfully converted {count} conversations!")
    except FileNotFoundError:
        print(f"File {input_file} not found. Please update the file path.")
    except Exception as e:
        print(f"Error in conversion: {e}")