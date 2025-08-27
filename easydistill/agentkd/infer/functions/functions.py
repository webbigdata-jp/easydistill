# functions/functions.py
import io
import re
import json
import multiprocessing
import types
import ast
import pickle
from contextlib import redirect_stdout
from typing import Dict, Any, Optional, Type
from openai import OpenAI
from langchain_core.pydantic_v1 import BaseModel
from langchain_openai import ChatOpenAI
from .mock_tools import MockTools
from .exceptions import LLMExecutionError, CodeExecutionError, EvaluationError
from .prompts import (
    REACT_SYSTEM_PROMPT, REACT_USER_PROMPT, 
    FIRST_THOUGHT_USER_PROMPT, FIRST_THOUGHT_SYSTEM_PROMPT,
    JUDGE_ANSWER_PROMPT
)


try:
    from ast import unparse
except ImportError:
    try:
        import astunparse
        unparse = astunparse.unparse
    except ImportError:
        
        def unparse(node):
            import ast
            return ast.dump(node)

def call_llm_api(
    user_prompt: str,
    system_prompt: str,
    api_base: Optional[str],
    api_key: Optional[str],
    model_name: str,
    max_tokens: int,
    temperature: float,
) -> BaseModel:
    try:
        client = OpenAI(api_key=api_key, base_url=api_base)
        try:
            models = client.models.list()
            dynamic_model_id = models.data[0].id if models.data else model_name
        except Exception:
            dynamic_model_id = model_name
        
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
        ]
        
        response = client.chat.completions.create(
            model=dynamic_model_id,
            messages=messages,
            temperature=temperature,
            extra_body={"max_completion_tokens": max_tokens}
        )
        response_content = response.choices[0].message.content

        return response_content
    except Exception as e:
        raise LLMExecutionError(
            f"Failed to call LLM API: {str(e)}",
            model_name=model_name
        ) from e

def one_thought_code_step(
    cfg, idx, input_query,
    first_thought="None",
    previous_context="None",
    failed_experience="None"
):
    thought_code_query = REACT_USER_PROMPT.format(
        query=input_query, idx=idx,
        first_thought=first_thought,
        failed_experience=failed_experience,
        previous_context=previous_context,
    )

    thought_code_content = call_llm_api(
        user_prompt=thought_code_query,
        system_prompt=REACT_SYSTEM_PROMPT,
        api_base=cfg.api_base,
        api_key=cfg.api_key,
        model_name=cfg.model_name,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
    )
    thought_code_content = re.sub(
        r'\<think\>\n(.*?)\</think\>\n',
        '', thought_code_content, flags=re.DOTALL
    )

    # Extract Thought and Code. The thought is everything outside/before the code block.
    code_match = re.search(r'```python(.*?)```', thought_code_content, re.DOTALL)
    if code_match:
        code_content = code_match.group(1).strip()
        # The thought is the part of the content before the code block starts.
        thought_content = thought_code_content[:code_match.start()].strip()
    else:
        # If no code block is found, there's no code, and the whole content is the thought.
        code_content = ''
        thought_content = thought_code_content.strip()

    # Clean up keywords from the extracted thought
    thought_content = thought_content.replace('Code:', '').strip()

    return thought_content, code_content


def get_first_thought(cfg, input_query):
    """Generate the first thought prefix before the Thought-Code-Observation cycle."""
    first_query = FIRST_THOUGHT_USER_PROMPT.format(query=input_query)

    initial_thought = call_llm_api(
        user_prompt=first_query,
        system_prompt=FIRST_THOUGHT_SYSTEM_PROMPT,
        api_base=cfg.api_base,
        api_key=cfg.api_key,
        model_name=cfg.model_name,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
    )
    # Remove the \<think\> tags from the response
    initial_thought = re.sub(r'\<think\>\n(.*?)\</think\>\n', '', initial_thought, flags=re.DOTALL)
    return initial_thought

def answer_evaluate_wo_repair(cfg, question, true_answer, generated_answer):
    system_prompt = """You are an expert judge evaluating whether two answers are equivalent.
        You will be given a predicted answer and a ground truth answer.
        Your task is to determine if they are semantically equivalent, ignoring minor differences in formatting, spacing, or notation.
        You must respond in JSON format with a 'equivalent' field that is either true or false."""
        
    prompt = f"""Question: {question}

    Predicted answer: {generated_answer}
        
    Ground truth answer: {true_answer}
        
    Are these answers semantically equivalent? Respond in JSON format with only the 'equivalent' field.

    Sometimes, predicted_answer may contain some irrelevant content, please ignore it, as long as predicted_answer contains the final answer, it is considered correct.
    Example: predicted_answer: {{The two sets are different because the sum of the remainders cannot equal the sum of the integers under the given constraints.}}.ground_truth: {{The two sets are different.}}. predicted_answer should be considered correct.

    Example response: {{"equivalent": true}}
    """
        
    evaluation_result_str = call_llm_api(
        user_prompt=prompt, 
        system_prompt=system_prompt, 
        api_base=cfg.api_base,
        api_key=cfg.api_key,
        model_name=cfg.model_name,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
    )
    
    return evaluation_result_str

def answer_evaluate(cfg, question, true_answer, generated_answer, thought_code_cycle):
    """Evaluate the correctness of the final answer."""
    # Create the evaluation prompt
    evaluation_prompt = JUDGE_ANSWER_PROMPT.format(
        question=question, true_answer=true_answer,
        generated_answer=generated_answer,
        thought_code_cycle=thought_code_cycle
    )
    
    # Call the LLM to evaluate the answer
    try:
        evaluation_result_str = call_llm_api(
            user_prompt=evaluation_prompt,
            system_prompt="You are a precise evaluator who analyzes reasoning processes and determines if answers are correct. Always respond with a valid JSON object and ensure all fields are properly filled.",
            api_base=cfg.api_base,
            api_key=cfg.api_key,
            model_name=cfg.model_name,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
        )
        
        # Use regex to extract fields from the response
        is_correct_match = re.search(r'"is_correct"\s*:\s*(true|false)', evaluation_result_str, re.IGNORECASE)
        error_analysis_match = re.search(r'"error_analysis"\s*:\s*"([^"]*)"', evaluation_result_str)
        correction_start_step_match = re.search(r'"correction_start_step"\s*:\s*(\d+|null)', evaluation_result_str, re.IGNORECASE)
        correction_suggestion_match = re.search(r'"correction_suggestion"\s*:\s*"([^"]*)"', evaluation_result_str)
        
        # Build a proper JSON object
        result_dict = {}
        
        # Handle is_correct field
        if is_correct_match:
            is_correct_val = is_correct_match.group(1).lower()
            result_dict["is_correct"] = True if is_correct_val == "true" else False
        else:
            result_dict["is_correct"] = False
            
        # Handle error_analysis field
        if error_analysis_match:
            # Unescape quotes and other characters
            error_analysis = error_analysis_match.group(1).replace('\\"', '"').replace('\\\\', '\\')
            result_dict["error_analysis"] = error_analysis
        else:
            result_dict["error_analysis"] = None
            
        # Handle correction_start_step field
        if correction_start_step_match:
            correction_start_step_val = correction_start_step_match.group(1).lower()
            if correction_start_step_val == "null":
                result_dict["correction_start_step"] = None
            else:
                try:
                    result_dict["correction_start_step"] = int(correction_start_step_val)
                except ValueError:
                    result_dict["correction_start_step"] = None
        else:
            result_dict["correction_start_step"] = None
            
        # Handle correction_suggestion field
        if correction_suggestion_match:
            # Unescape quotes and other characters
            correction_suggestion = correction_suggestion_match.group(1).replace('\\"', '"').replace('\\\\', '\\')
            result_dict["correction_suggestion"] = correction_suggestion
        else:
            result_dict["correction_suggestion"] = None
            
        # Serialize to proper JSON
        return json.dumps(result_dict, ensure_ascii=False)
        
    except Exception as e:
        raise EvaluationError(f"Error during evaluation: {str(e)}") from e

def _get_serializable_globals(globals_dict: dict, executed_code: str, previous_scope: dict = None) -> dict:
    """
    Filter a dictionary to only include serializable objects.
    For functions and import statements, extract their source code by parsing the original code string.
    """
    serializable_globals = {}
    
    # Start with functions from previous scope
    if previous_scope:
        for key, value in previous_scope.items():
            if isinstance(value, dict) and value.get('__type__') == 'function':
                serializable_globals[key] = value
    
    # Parse current code to extract functions and imports
    function_sources = {}
    import_statements = []
    try:
        tree = ast.parse(executed_code)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                function_sources[node.name] = unparse(node)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                import_statements.append(unparse(node))
    except (SyntaxError, AttributeError):
        pass

    # Add import statements to the serializable state
    if previous_scope and '__imports__' in previous_scope and previous_scope['__imports__'].get('__type__') == 'import_block':
        # Merge with existing imports
        existing_imports = previous_scope['__imports__']['__sources__']
        # Combine and deduplicate
        all_imports = list(dict.fromkeys(existing_imports + import_statements))
        serializable_globals['__imports__'] = {
            '__type__': 'import_block',
            '__sources__': all_imports
        }
    else:
        serializable_globals['__imports__'] = {
            '__type__': 'import_block',
            '__sources__': import_statements
        }

    # Process variables and functions in globals
    for key, value in globals_dict.items():
        # Skip built-in variables and problematic types
        if key.startswith('__') or isinstance(value, (types.ModuleType, types.CodeType)):
            continue
        
        # Handle functions by using extracted source code
        if isinstance(value, types.FunctionType):
            if key in function_sources:
                serializable_globals[key] = {
                    '__type__': 'function',
                    '__source__': function_sources[key]
                }
            continue

        # Test if object can be pickled
        try:
            pickle.dumps(value)
            serializable_globals[key] = value
        except (pickle.PicklingError, TypeError):
            pass
            
    return serializable_globals

def _execute_code_in_process(args):
    """Helper function to execute code in a separate process"""
    code, scope_dict, output_queue = args
    try:
        # Recreate the scope from the dictionary
        scope = {}
        
        # Rebuild state: imports first, then functions, then variables
        # 1. Rebuild imports
        if '__imports__' in scope_dict and scope_dict['__imports__'].get('__type__') == 'import_block':
            for import_src in scope_dict['__imports__']['__sources__']:
                try:
                    exec(import_src, scope)
                except Exception:
                    pass

        # 2. Rebuild functions from previous executions
        for key, value in scope_dict.items():
            if key == '__imports__': 
                continue  # Skip imports as they're already processed

            if isinstance(value, dict) and value.get('__type__') == 'function':
                try:
                    # Execute function definition in scope
                    exec(value['__source__'], scope)
                except Exception:
                    pass

        # 3. Load regular variables
        for key, value in scope_dict.items():
            if key == '__imports__': 
                continue
                
            # Skip functions as they're already processed
            if isinstance(value, dict) and value.get('__type__') == 'function':
                continue
                
            # Copy other serializable variables
            try:
                pickle.dumps(value)
                scope[key] = value
            except (pickle.PicklingError, TypeError):
                pass

        # Add mock tools to the scope
        scope['web_search'] = MockTools.web_search
        scope['final_answer_print'] = MockTools.final_answer_print
        
        output_stream = io.StringIO()
        with redirect_stdout(output_stream):
            exec(code, scope)
        output = output_stream.getvalue()
        if not output:
            output = "Execution successful, no output."
            
        # Filter globals to only include serializable objects
        serializable_scope = _get_serializable_globals(scope, code, scope_dict)
                
        output_queue.put({"output": output, "updated_scope": serializable_scope, "error": None})
    except Exception as e:
        output_queue.put({"output": f"Error: {e}", "updated_scope": scope_dict, "error": str(e)})

def python_interpreter(code: str, scope: Dict[str, Any], timeout: int = 10) -> Dict:
    """Execute Python code with a timeout mechanism using multiprocessing."""
    # Prepare scope for serialization
    prepared_scope = {}
    
    # Copy over the existing scope, handling special objects properly
    for key, value in scope.items():
        prepared_scope[key] = value
    
    # Create a queue for communication
    output_queue = multiprocessing.Queue()
    
    # Package arguments for the process
    args = (code, prepared_scope, output_queue)
    
    # Create and start the process
    process = multiprocessing.Process(target=_execute_code_in_process, args=(args,))
    process.start()
    
    try:
        # Wait for the result with timeout
        result = output_queue.get(timeout=timeout)
        process.join()  # Wait for process to finish
    except multiprocessing.TimeoutError:
        # Terminate the process if it times out
        process.terminate()
        process.join()
        result = {"output": f"Error: Code execution timed out after {timeout} seconds", 
                  "updated_scope": scope, 
                  "error": "timeout"}
    except Exception as e:
        # Handle other exceptions
        process.terminate()
        process.join()
        result = {"output": f"Error: {e}", 
                  "updated_scope": scope, 
                  "error": str(e)}
    
    return {"output": result["output"], "updated_scope": result["updated_scope"]}
