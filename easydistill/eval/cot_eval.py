
# Copyright 2024 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import json, jsonlines
import argparse
import logging
import os
import re
from tqdm import tqdm
from openai import OpenAI


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def build_cot_prompts(instruction, output):
    rv_prompt_template = (
        "You are an expert judge tasked with evaluating the Reasoning Verbosity of a Chain-of-Thought (CoT) "
        "for a given problem and its answer. Reasoning Verbosity Evaluation Focus: Assess how well the CoT’s "
        "length and step complexity match the problem’s inherent difficulty. An optimal chain is neither "
        "missing essential steps nor padded with needless digressions. A simple question should be solved "
        "with a brief, direct chain; a challenging one may justifiably require a longer path with reflection "
        "and error-checking. Scoring Guidelines (0-9):\n"
        "0-1 Minimal verbosity, straightforward expression with little to no elaboration.\n"
        "2-3 Clear and concise reasoning with necessary explanations.\n"
        "4-5 Moderate verbosity with detailed explanations and thorough reasoning.\n"
        "6-7 Extensive verbosity with comprehensive justification and exploration of complex connections.\n"
        "8-9 High verbosity with deep, exhaustive exploration of reasoning; involves extensive elaboration, nested justifications, "
        "and consideration of counterarguments or alternative perspectives.\n"
        "Given Problem, Answer with hain-of-Thought, you will:\n"
        "1. Analyze the Reasoning Verbosity\n"
        "2. Determine score using the above criteria\n"
        "3. Output ONLY the integer score (0-9), place your score in <score></score>\n"
        f"Problem: {instruction}\n"
        f"Answer with Chain-of-Thought: {output}"
    )
    cd_prompt_template = (
        "You are an expert judge assessing the Cognitive Difficulty of a Chain-of-Thought (CoT) "
        "for a given problem and its answer. Cognitive Difficulty Evaluation Focus: The level of "
        "reasoning competence required for a model to follow and reproduce the chain faithfully. "
        "Judge the reasoning approach, techniques, and overall difficulty. Higher scores correspond "
        "to more advanced concepts, abstractions, or multi-layer reasoning patterns. "
        "Scoring Guidelines (0-9):\n"
        "0-1 Elementary facts or a single trivial operation.\n"
        "2-3 Multi-step arithmetic, explicit enumeration, basic rule chaining.\n"
        "4-5 Early-undergraduate logic/algebra; one non-obvious insight.\n"
        "6-7 Advanced undergraduate techniques (determinants, dynamic programming, layered code reasoning, etc).\n"
        "8-9 Graduate-level abstraction, nested proofs, intricate algorithmic analysis.\n"
        "Given Problem, Answer with hain-of-Thought, you will:\n"
        "1. Analyze the Cognitive Difficulty\n"
        "2. Determine score using the above criteria\n"
        "3. Output ONLY the integer score (0-9), place your score in <score></score>\n"
        f"Problem: {instruction}\n"
        f"Answer with Chain-of-Thought: {output}"
    )
    lc_prompt_template = (
        "You are a rigorous logical validator analyzing problem-solving components. "
        "Your task is to separately assess the validity of the reasoning process and final solution. "
        "Given Problem, Answer with hain-of-Thought, you will:\n"
        "1. Verify stepwise logical coherence and soundness\n"
        "2. Confirm all critical problem constraints are properly addressed\n"
        "3. Check for self-contradictions or unsupported leaps in logic\n"
        "4. Verify the process can actually derive the proposed solution\n"
        "5. Output ONLY the 1/0 answer (1 for true, 0 for false) for logical correctness, place your answer in <score></score>\n"
        f"Problem: {instruction}\n"
        f"Answer with Chain-of-Thought: {output}"    
    )
    return rv_prompt_template, cd_prompt_template, lc_prompt_template


def extract_score(text):
    match = re.search(r"<score>(\d+)</score>", text)
    if match:
        return int(match.group(1))
    else:
        return -1


def read_json_fields(filename):
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        logging.error("The file was not found.")
    except json.JSONDecodeError:
        logging.error("There was an error decoding the JSON file.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")


def write_data_to_json_file(data, file_path):
    try:
        with open(file_path, 'w') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
        logging.info(f"Data successfully written to {file_path}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

        
def generate_teacher_response_api(data_list, config):
    client = OpenAI(
        api_key = config["inference"]["api_key"],
        base_url = config["inference"]["base_url"]
    )
    models = client.models.list()
    model = models.data[0].id
    logging.info(model)
    outcomes = []
    for sample in tqdm(data_list, desc="Call remote model and generating responses"):
        instruction = sample["instruction"]
        output = sample["output"]
        rv_prompt_template, cd_prompt_template, lc_prompt_template = build_cot_prompts(instruction, output)
        
        def generate_score(sample, model, config):
            message = [
                {'role': 'user', 'content': sample}
            ]
            completion = client.chat.completions.create(
                messages = message,
                model = model,
                max_completion_tokens = config["inference"]["max_new_tokens"]
            )
            result = completion.choices[0].message.content
            score = extract_score(result)
            return score
    
        rv_score = generate_score(rv_prompt_template, model, config)
        cd_score = generate_score(cd_prompt_template, model, config)
        lc_score = generate_score(lc_prompt_template, model, config)
        if lc_score == 1:
            lc_score = True
        else:
            lc_score =False
        
        outcomes.append(
            {
                'instruction': instruction,
                 'output': output,
                 "reasoning_verbosity": rv_score,
                 "cognitive_difficulty": cd_score,
                 "logical_correctness": lc_score
            }
        )
    write_data_to_json_file(outcomes, config["dataset"]["output_path"])


def infer_with_teacher_model(config):
    logging.info('Generating distillation data from the teacher model!')
    data_list = read_json_fields(config["dataset"]["input_path"])
    generate_teacher_response_api(data_list, config)
       

        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to the json config file')
    args = parser.parse_args()
    config = json.load(open(args.config))
    infer_with_teacher_model(config)


if __name__ == "__main__":
    main()