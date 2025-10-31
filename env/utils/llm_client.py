from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Tuple
from PIL import Image
import base64
from io import BytesIO
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor
import logging
from pathlib import Path
from tqdm import tqdm
import yaml
# from transformers import AutoTokenizer
from tqdm import tqdm
import argparse
import os
import json
import random
import sys
from dotenv import load_dotenv
load_dotenv()

project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)
from env.utils.prompts import *

logger = logging.getLogger(__name__)

def get_model_information(model_name):
    if "DeepSeek-R1-Distill-Qwen-7B" in model_name:
        base_url = "http://0.0.0.0:20000/v1"
        api_key = "EMPTY"
    elif "DeepSeek-R1-Distill-Llama-8B" in model_name:
        base_url = "http://0.0.0.0:20001/v1"
        api_key = "EMPTY"
    elif "DeepSeek-R1-Distill-Qwen-14B" in model_name:
        base_url = "http://0.0.0.0:20002/v1"
        api_key = "EMPTY"
    elif "Qwen/Qwen2.5-7B-Instruct" in model_name:
        base_url = "http://0.0.0.0:20003/v1"
        api_key = "EMPTY"
    elif "Qwen3-8B" in model_name:
        base_url = "http://0.0.0.0:20004/v1"
        api_key = "EMPTY"
    elif "Qwen3-32B" in model_name:
        base_url = "http://0.0.0.0:20005/v1"
        api_key = "EMPTY"
    elif "gpt-4o-mini" in model_name or "gpt-4o" in model_name or "gemini-2.5-flash" in model_name:
        base_url = "https://openrouter.ai/api/v1"
        api_key = os.getenv("OPENROUTER_API_KEY")
    elif "Qwen3-14B" in model_name:
        base_url = "http://0.0.0.0:40005/v1"
        api_key = "EMPTY"
    else:
        exit("Wrong model name")
    return {
        "base_url": base_url,
        "api_key": api_key
    }

class BaseGenerator(ABC):
    def __init__(self, model_name):
        """Initialize the generator with configuration."""
        self.model_name = model_name

        self.client = OpenAI(
            api_key=get_model_information(model_name)["api_key"],
            base_url=get_model_information(model_name)["base_url"],
        )
        self.batch_size = 64

    @abstractmethod
    def _prepare_prompts_for_batch(self, **kwargs) -> Any:
        """Prepare messages for the different API call."""
        pass
    
    def _generate_single(self, prompt: str, request_id: Optional[str], **kwargs ) -> Tuple[str, Optional[str]]:
        """Generate a single response with optional request ID for tracking."""

        try:
            if self.model_name == "openai/gpt-4o-mini":
                
                completion = self.client.chat.completions.create(
                    extra_body={},
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt
                                }
                            ]
                        }
                    ],  
                )
                return (completion.choices[0].message.content, request_id), completion.usage.completion_tokens
            
            else:
                
                response = self.client.completions.create(
                    model=self.model_name,
                    prompt=prompt,
                    **kwargs
                )
                # print("model paras:", kwargs)
                return (response.choices[0].text, request_id), response.usage.completion_tokens 
        
        except Exception as e:
            logger.error(f"Error generating response for request {request_id}: {str(e)}")
            raise
    
    def generate_batch(self, prompt_list: List[Tuple[str, Optional[str]]]) -> List[Tuple[str, Optional[str]]]:
        """Generate responses for a batch of messages using ThreadPoolExecutor with context manager."""
        results = []
        future_to_idx = {}
        
        with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            # Submit all tasks to the executor
            for idx, (prompt, request_id) in enumerate(prompt_list):
                future = executor.submit(self._generate_single, prompt, request_id)
                future_to_idx[future] = idx
            
            # Process futures as they complete with progress bar
            with tqdm(total=len(future_to_idx), desc="Generating responses") as pbar:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        request_id = prompt_list[idx][1]
                        logger.error(f"Error in batch generation for request {request_id}: {str(e)}")
                        results.append((str(e), request_id))
                    pbar.update(1)
        
        # Sort results to maintain original order
        if all(r[1] is not None for r in results):
            results.sort(key=lambda x: next(i for i, (_, rid) in enumerate(prompt_list) if rid == x[1]))
            
        return results
    
    @abstractmethod
    def generate(self, **kwargs) -> Any:
        """Generate content based on input."""
        pass

def load_travel_preferences_str(task, input_path: str = "env/domains/travel/travel_config.yaml") -> Dict[str, str]:
    with open(input_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    all_preferences = config["main_subtasks"][task.lower()]["dimensions"]
    return {k: ", ".join(v) for k, v in all_preferences.items()}

def process_input_file(task: str, batch: List[Dict[str, Any]], input_path: str) -> List[str]:
    # here, batches refer to the whole input data
    prompts = []

    if task == "generate_queries":
        for item in batch:
            if item["is_valid"] == 0:
                continue
            prompt = QUERY_GENERATION_INSTRUCTION.format(
                task=item["task"],
                category_candidates=load_travel_preferences_str(task=item["task"])["category"],
                tier_candidates=load_travel_preferences_str(task=item["task"])["tier"],
                style_candidates=load_travel_preferences_str(task=item["task"])["style"],
                features_candidates=load_travel_preferences_str(task=item["task"])["feature_package"]
            ) + QUERY_GENERATION_PROMPT.format(task=item["task"], category=item["preferences"]["category"], tier=item["preferences"]["tier"], style=item["preferences"]["style"], features=item["preferences"]["feature_package"])
            prompts.append(prompt)
    elif task == "validate_queries":
        for item in batch:
            # print("QUERY_VALIDATION_INSTRUCTION:", QUERY_VALIDATION_INSTRUCTION)
            prompt = QUERY_VALIDATION_INSTRUCTION + QUERY_VALIDATION_PROMPT.format(task=item["task"], category=item["preferences"]["category"], tier=item["preferences"]["tier"], style=item["preferences"]["style"], features=item["preferences"]["feature_package"])
            prompts.append(prompt)
    elif task == "count_coverage":
        for item in batch:
            prompt = COUNT_COVERAGE_INSTRUCTION + COUNT_COVERAGE_PROMPT.format(model_plan=item["model_plan"])
            prompts.append(prompt)
    else:
        raise ValueError(f"Unknown input file source: {input_path}")
    
    return prompts


class ForceBudgeGenerator(BaseGenerator):
    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", think_len: str = "more_think", temperature: float = 0.6):
        """Initialize the generator with configuration."""
        super().__init__(model_name)
        self.think_len = think_len
        self.max_thinking_len = 128
        self.temperature = temperature

    def _prepare_prompt(self, input_text: str) -> str:
        """Prepare the prompt for the model."""
        return input_text

    def _prepare_prompts_for_batch(self, input_texts: List[str]) -> List[str]:
        """Prepare messages for the different API call."""
        return [(self._prepare_prompt(text), idx) for idx, text in enumerate(input_texts)]

    def _generate_single(
        self, prompt: str, request_id: Optional[str] = None
    ) -> Tuple[str, Optional[str]]:
        if self.think_len == "no_think":
            prompt = f"{prompt}\n<think></think>"
            return super()._generate_single(prompt, request_id, max_tokens=self.max_thinking_len, temperature=self.temperature)[0]
        elif self.think_len == "regular_think":
            prompt = f"{prompt}\n<think>Okay, "
            return super()._generate_single(prompt, request_id, max_tokens=self.max_thinking_len, temperature=self.temperature)[0]
        elif self.think_len == "vanilla":
            return super()._generate_single(prompt, request_id, max_tokens=self.max_thinking_len, temperature=self.temperature)[0]
        else:
            raise ValueError(f"Unknown thinking type: {self.think_len}")
        
    def generate(self, **kwargs) -> List[Tuple[str, Optional[str]]]:
        pass
    
def batch_inference(model_name: str, input_path: str, output_path: str, task: str, think_len: str = "vanilla", begins_at: int = 0, ends_at: int = 0):

    output_model_name = model_name.split("/")[-1]
    output_path = output_path
    # if args.task == "validate_queries":
    #     output_path = os.path.join("env", "data", "runtime", "queries", "filtered_queries.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # check with the inputs
    print("[INFO] input_path: ", input_path)
    print("[INFO] model_name: ", model_name)

    if task == "validate_queries" or task == "generate_queries" or task == "count_coverage":
        input_data = json.load(open(input_path, "r", encoding='utf-8'))
        if begins_at > 0 or ends_at > 0:
            input_data = input_data[begins_at: ends_at]        
    else:
        raise NotImplementedError("Only validate_queries and generate_queries is implemented now")
    print("[INFO] the length of input data is: ", len(input_data))
    
    # turn in into the format of list of dict (including the sample nums if needed)
    prompts = process_input_file(task, input_data, input_path) 

    generator = ForceBudgeGenerator(
        model_name=model_name,
        think_len=think_len,
    )
    # pack the prompts with the request_id
    prompts = generator._prepare_prompts_for_batch(prompts)

    # get the responses
    output = generator.generate_batch(prompts)
        
    if task == "validate_queries":
        for input_data_dic in input_data:
            validation_id = input_data.index(input_data_dic)
            input_data_dic["validation_raw"] = output[validation_id][0]
        with open(output_path, 'w', encoding="utf-8") as f:
            json.dump(input_data, f, ensure_ascii=False, indent=4)
    elif task == "generate_queries":
        final_queries = []
        for item in input_data:
            if item["is_valid"] == 0:
                continue
            final_queries.append(item)
        check_length = (len(final_queries) == len(output))
        if not check_length:
            logger.warning("[CODE ERROR] The length of final_queries and output do not match.")
        for final_data_dic in final_queries:
            generation_id = final_queries.index(final_data_dic)
            final_data_dic["user_requirements"] = output[generation_id][0]
        with open(output_path, 'w', encoding="utf-8") as f:
            json.dump(final_queries, f, ensure_ascii=False, indent=4)
    elif task == "count_coverage":
        for item in input_data:
            query_id = input_data.index(item)
            item["num_paths"] = output[query_id][0]
        with open(output_path, 'w', encoding="utf-8") as f:
            json.dump(input_data, f, ensure_ascii=False, indent=4)
    else:
        raise NotImplementedError("Only validate_queries/generate_queries/count_coverage is implemented now")
    
    print("[INFO] output_path:", output_path)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument(
        "--think_len",
        type=str,
        default="no_think",
        choices=[
            "no_think",
            "regular_think",
            "vanilla"
        ],
    )
    parser.add_argument("--begins_at", type=int, default=0)
    parser.add_argument("--ends_at", type=int, default=0)
    parser.add_argument("--task", type=str, default="validate_queries", choices=["validate_queries", "generate_queries", "count_coverage"])
    # TODO: add back tool use support
    parser.add_argument("--support_tool_use", action="store_true", help="Whether to support tool use")

    args = parser.parse_args()
    batch_inference(
        model_name=args.model_name,
        input_path=args.input_path,
        output_path=args.output_path,
        think_len=args.think_len,
        begins_at=args.begins_at,
        ends_at=args.ends_at,
        task=args.task,
    )

"""
Example usage:
python env/domains/travel/filter_queries.py \
    --input_path env/data/runtime/queries/queries.json \
    --output_path env/data/runtime/queries/filtered_queries.json \
    --task validate_queries \
    --model_name gpt-4o-mini \
    --think_len vanilla \
    --begins_at 0 \
    --ends_at 0
    
python env/utils/llm_client.py \
    --input_path final_prompt/gpt-5/first_assistant_texts.json \
    --output_path final_prompt/gpt-5/first_assistant_texts.json \
    --task count_coverage \
    --model_name gpt-4o-mini \
    --think_len vanilla \
    --begins_at 0 \
    --ends_at 5
    
python env/domains/travel/filter_queries.py --input_path env/data/runtime/queries/queries.json --output_path env/data/runtime/queries/filtered_queries.json --task validate_queries --model_name gpt-4o-mini --think_len vanilla --begins_at 0 --ends_at 0
"""