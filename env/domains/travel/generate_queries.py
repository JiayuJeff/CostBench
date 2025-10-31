"""
# Generate both the training and test sets
python env/domains/travel/generate_queries.py \
    --generate_train \
    --generate_test \
    --train_search_space_path env/data/static/search_spaces/train_search_space.json \
    --test_search_space_path env/data/static/search_spaces/test_search_space.json \
    --num_queries 1000 \
    --seed 42 \
    --model_name gpt-4o \
    --query_dir env/data/runtime/queries

# Generate only the training set
python env/domains/travel/generate_queries.py \
    --generate_train \
    --train_search_space_path env/data/static/search_spaces/train_search_space.json \
    --num_queries 6000 \
    --seed 42 \
    --model_name gpt-4o \
    --query_dir env/data/runtime/queries
    
python env/domains/travel/generate_queries.py --generate_train --train_search_space_path env\data\static\search_spaces\train\search_space.json --train_config_path env/domains/travel/train_travel_config.yaml --num_queries 6000 --seed 42 --model_name gpt-4o --query_dir env\data\runtime\train_queries

# Generate only the test set
python env/domains/travel/generate_queries.py \
    --generate_test \
    --test_search_space_path env/data/static/search_spaces/test_search_space.json \
    --num_queries 1000 \
    --seed 42 \
    --model_name gpt-4o \
    --query_dir env/data/runtime/queries

Step 4: Generate query data

This script performs the following tasks:
1. Load the search_space.json file
2. Evenly allocate the number of queries for each LocationPreference
3. Randomly select preference combinations and generate query records
4. Map time information one-to-one
5. Output the complete query dataset

Data structure:
{
  "query_id": "<Query00001>",
  "TimeInfo": "<TimeInfo00001>",
  "task": "location",
  "is_location": 1,
  "preferences": {
      'category': 'city',
      'tier': 'bustling_metropolis',
      'style': 'adventure',
      'feature_package': 'history_buff'
    },
  "groundtruth": "<LocationPreference00035>"
},
{
    "query_id": "<Query00002>",
    "TimeInfo": "<TimeInfo00001>",
    "task": "transportation",
    "is_location": 0,
    "preferences": {
        'category': 'train',
        'tier': 'high_speed',
        'style': 'comfortable',
        'feature_package': 'scenic_route'
    },
    "groundtruth": "<TransportationPreference11201>"
}

In short, the core function `generate_queries(search_space_data, num_queries, seed)` first checks how many LocationPreferences exist and then determines how many queries should be generated for each LocationPreference (query_per_location = num_queries // total_locations + 1, with any excess truncated). The process is to iterate over all LocationPreferences, generate `total_locations` queries with the task type `location` (given TimeInfo and location requirements so the agent can call the tool to retrieve the LocationPreference), and then iterate over each LocationPreference to generate `(query_per_location-1) // len(remaining task types)` queries for the tasks transportation/accommodation/attraction/dining/shopping (given TimeInfo, the LocationPreference, and task-specific user requirements so the agent can call the tool to retrieve the corresponding preference). When we say "generate query," we are simply creating dictionary records like the JSON structure above without actually invoking an LLM to produce natural-language query text. Is this requirement clear?
"""

import random
import argparse
import json
import os
import sys
from sympy import im
import ast
from typing import Dict, Any, List, Tuple, Optional
project_root = os.path.join(os.path.dirname(__file__), '..', '..', '..')
sys.path.insert(0, project_root)
from env.utils.id_generator import get_global_generator
from env.utils.llm_client import batch_inference
from env.utils.parse.parse_validate import parse
from env.utils.prompts import QUERY_PROMPT
from env.core.data_types import get_all_goal_type

ORIGINAL_QUERY_STRUCTURE_NAME = "queries_raw.json"
FILTERED_QUERY_STRUCTURE_NAME = "filtered_queries.json"
FILTERED_QUERY_STRUCTURE_WITH_REQUIREMENTS_NAME = "filtered_queries_with_requirements.json"
FINAL_QUERY_NAME = "queries.json"


def parse_time_pair(time_pair_str: str) -> Tuple[str, str]:
    # Assume the format is "(<TimeInfo00427>, <TimeInfo00622>)"
    time_pair_str = time_pair_str.strip("() ")
    parts = [p.strip() for p in time_pair_str.split(",")]
    return parts[0], parts[1]


config = {}
current_config_path = None

def find_feature_dimension(subtask: str = None, requirement: str = None, config_path: str = None) -> str:
    """
    Find the feature dimension for a given subtask and requirement.
    
    Args:
        subtask: The subtask name (e.g., "location", "transportation").
        requirement: The specific requirement value (e.g., "city", "train").
        config_path: Path to the travel config file. If None, uses default.
        
    Returns:
        The dimension name (e.g., "category", "tier", "style", "features").
    """
    global config, current_config_path
    
    from generate_search_space import load_travel_config
    # Reload the configuration if the path changes
    if not config or current_config_path != config_path:
        config = load_travel_config(config_path)
        current_config_path = config_path
    if subtask is None or requirement is None:
        raise ValueError("[CODE ERROR] subtask and requirement should not be None")
    
    for dim, values in config["main_subtasks"][subtask]["dimensions"].items():
        if requirement in values:
            return dim
        
    raise ValueError(f"[CODE ERROR] Cannot find dimension for subtask {subtask} and requirement {requirement}")


def find_feature_dimension_for_tuple(subtask: str, pref_tuple: Tuple[str, str, str, str], config_path: str = None) -> Dict[str, str]:
    """
    Find the feature dimensions for all elements in a preference tuple.
    
    Args:
        subtask: The subtask name (e.g., "location", "transportation").
        pref_tuple: A tuple of preference values (e.g., ('city', 'bustling_metropolis', 'adventure', 'history_buff')).
        config_path: Path to the travel config file. If None, uses default.
        
    Returns:
        A dictionary mapping each preference value to its dimension.
    """
    
    dimension_mapping = {}
    for requirement in pref_tuple:
        dim = find_feature_dimension(subtask, requirement, config_path)
        dimension_mapping[dim] = requirement
    return dimension_mapping


def generate_query_structures(search_space_data_path: str, num_queries: int, seed: int, query_dir: str, config_path: str = None) -> List[Dict[str, Any]]:
    """
    Build a structured query dataset and distribute queries evenly across each LocationPreference.
    """
    random.seed(seed)
    generator = get_global_generator()
    
    with open(search_space_data_path, "r", encoding="utf-8") as f:
        search_space_data = json.load(f)

    # 1. Basic information
    location_mapping = search_space_data["locations"]  # {preference_tuple_str: LocationPreferenceID}
    location_information = search_space_data["location_information"]  # {LocationPreferenceID: {subtask: {pref_tuple: pref_id}}}
    time_pool = search_space_data["time_pool"]  # [("TimeInfo00001", "TimeInfo00005"), ...]
    location_pref_tuples = list(location_mapping.keys())
    location_pref_ids = list(location_mapping.values())
    total_locations = len(location_pref_ids)
    subtask_names = ["transportation", "accommodation", "attraction", "dining", "shopping"]

    # 2. Compute the allocation strategy
    queries = []
    time_pool_cycle = (tp for tp in time_pool)  # Use a generator to cycle through time slots
    
    # Determine how many queries should be generated for each task
    all_tasks = ["location"] + subtask_names  # Six tasks in total
    base_queries_per_task = num_queries // len(all_tasks)  # Base number per task
    remaining_queries = num_queries % len(all_tasks)  # Remaining queries
    
    print(f"[INFO] Total locations: {total_locations}")
    print(f"[INFO] Base queries per task: {base_queries_per_task}")
    print(f"[INFO] Remaining queries to distribute: {remaining_queries}")

    # 3. Generate queries for the location task
    location_query_count = base_queries_per_task + (1 if remaining_queries > 0 else 0)
    if remaining_queries > 0:
        remaining_queries -= 1
    
    for i in range(location_query_count):
        # Iterate through LocationPreference entries
        pref_tuple_str, location_id = list(location_mapping.items())[i % total_locations]
        
        try:
            time_pair_str = next(time_pool_cycle)
        except StopIteration:
            time_pool_cycle = (tp for tp in time_pool)
            time_pair_str = next(time_pool_cycle)
        start_time, end_time = parse_time_pair(time_pair_str)
        query = {
            "query_id": generator.generate_id("Query"),
            "TimeInfo": start_time,
            "task": "location",
            "is_location": 1,
            "goal_type": get_all_goal_type("location"),
            "preferences": find_feature_dimension_for_tuple("location", ast.literal_eval(pref_tuple_str), config_path),  # tuple
            "groundtruth": location_id
        }
        queries.append(query)

    # 4. Generate queries for the other tasks
    for task_idx, task in enumerate(subtask_names):
        task_query_count = base_queries_per_task + (1 if remaining_queries > 0 else 0)
        if remaining_queries > 0:
            remaining_queries -= 1
            
        for i in range(task_query_count):
            # Iterate through LocationPreference entries
            pref_tuple_str, location_id = list(location_mapping.items())[i % total_locations]
            location_info = location_information[location_id]
            subtask_mapping = location_info[task]
            subtask_pref_tuples = list(subtask_mapping.keys())
            
            pref_tuple = random.choice(subtask_pref_tuples)
            groundtruth = subtask_mapping[pref_tuple]
            try:
                time_pair_str = next(time_pool_cycle)
            except StopIteration:
                time_pool_cycle = (tp for tp in time_pool)
                time_pair_str = next(time_pool_cycle)
            start_time, end_time = parse_time_pair(time_pair_str)
            query = {
                "query_id": generator.generate_id("Query"),
                "TimeInfo": start_time,
                "task": task,
                "is_location": 0,
                "goal_type": get_all_goal_type(task),
                "preferences": find_feature_dimension_for_tuple(task, ast.literal_eval(pref_tuple), config_path),  # tuple
                "location_preference": location_id,
                "groundtruth": groundtruth
            }
            queries.append(query)

    # 5. Truncate extra queries
    if len(queries) > num_queries:
        queries = queries[:num_queries]
        
    # 6. Persist results
    with open(os.path.join(query_dir, ORIGINAL_QUERY_STRUCTURE_NAME), "w", encoding="utf-8") as f:
        json.dump(queries, f, ensure_ascii=False, indent=4)
        

def filter_queries(model_name: str, query_dir: str, task: str = "validate_queries") -> str:
    """
    Use an LLM to filter the generated queries and ensure they are reasonable.
    
    Args:
        queries: List of generated queries
        model_name: Name of the LLM model to use
        query_dir: Directory where query files are stored
        task: Task type ("validate_queries")
    """
    batch_inference(
        model_name=model_name,
        input_path=os.path.join(query_dir, ORIGINAL_QUERY_STRUCTURE_NAME),
        output_path=os.path.join(query_dir, FILTERED_QUERY_STRUCTURE_NAME),
        task=task,
        think_len="vanilla"
    )
    
    # parse the filtered results to mark valid/invalid queries
    parse(
        input_path=os.path.join(query_dir, FILTERED_QUERY_STRUCTURE_NAME),
        output_path=os.path.join(query_dir, FILTERED_QUERY_STRUCTURE_NAME),
        task="validate_queries"
    )

def generate_requirements(model_name: str, query_dir: str, task: str = "generate_queries"):
    """
    Generate the final `queries.json` based on the filtered queries.
    
    Args:
        filtered_queries_path: Path to the filtered query file
    """
    
    batch_inference(
        model_name=model_name,
        input_path=os.path.join(query_dir, FILTERED_QUERY_STRUCTURE_NAME),
        output_path=os.path.join(query_dir, FILTERED_QUERY_STRUCTURE_WITH_REQUIREMENTS_NAME),
        task=task,
        think_len="vanilla"
    )

    return os.path.join(query_dir, FILTERED_QUERY_STRUCTURE_WITH_REQUIREMENTS_NAME)

def combine_queries(query_dir: str):

    with open(os.path.join(query_dir, FILTERED_QUERY_STRUCTURE_WITH_REQUIREMENTS_NAME), "r", encoding="utf-8") as f:
        queries_with_requirements = json.load(f)
        
    for item in queries_with_requirements:
        item["query"] = {}
        if item["task"] == "location":
            item["query"]["input"] = QUERY_PROMPT.format(
                TimeInfo=item["TimeInfo"],
                user_requirements=item["user_requirements"],
                goal_type=item["goal_type"],
                LocationPreferenceID=""
            )
        else:
            item["query"]["input"] = QUERY_PROMPT.format(
                TimeInfo=item["TimeInfo"],
                user_requirements=item["user_requirements"],
                goal_type=item["goal_type"],
                LocationPreferenceID=" and to " + item["location_preference"]
            )
        
    with open(os.path.join(query_dir, FINAL_QUERY_NAME), "w", encoding="utf-8") as f:
        json.dump(queries_with_requirements, f, ensure_ascii=False, indent=4) 

def generate_queries(search_space_data_path: str, num_queries: int, seed: int, model_name: str, query_dir: str, config_path: str = None) -> str:
    
    # 1. Generate the initial query structures
    print("=" * 50)
    print("[INFO] Generating initial query structures...")
    if not os.path.exists(query_dir):
        os.makedirs(query_dir, exist_ok=True)
    generate_query_structures(search_space_data_path, num_queries, seed, query_dir, config_path)
    print(f"[INFO] Initial query structures saved to {os.path.join(query_dir, 'queries_raw.json')}")

    # 2. Filter out unreasonable queries
    print("=" * 50)
    print(f"[INFO] Filtering queries with {model_name}...")
    filter_queries(model_name, query_dir, task="validate_queries")
    print(f"[INFO] Filtered queries saved to {os.path.join(query_dir, 'filtered_queries.json')}")
    
    # 3. Generate the final query text
    print("=" * 50)
    print(f"[INFO] Generating user requirements with {model_name}...")
    generate_requirements(model_name, query_dir, task="generate_queries")
    print(f"[INFO] Queries with user requirements saved to {os.path.join(query_dir, 'queries_with_requirements.json')}")

    # 4. Combine everything into the final queries.json
    print("=" * 50)
    print(f"[INFO] Combining all components into final queries...")
    combine_queries(query_dir)
    print(f"[INFO] Final queries saved to {os.path.join(query_dir, 'queries.json')}")
    return os.path.join(query_dir, 'queries.json')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate queries for travel search space")
    
    # Mode selection
    parser.add_argument("--generate_train", action="store_true", help="Generate training queries")
    parser.add_argument("--generate_test", action="store_true", help="Generate testing queries")
    
    # Search space paths
    parser.add_argument("--train_search_space_path", type=str, help="Path to training search_space.json")
    parser.add_argument("--test_search_space_path", type=str, help="Path to testing search_space.json")
    
    # Configuration file paths
    parser.add_argument("--train_config_path", type=str, help="Path to training travel config file")
    parser.add_argument("--test_config_path", type=str, help="Path to testing travel config file")
    
    # Common parameters
    parser.add_argument("--num_queries", type=int, default=1000, help="Total number of queries to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--query_dir", type=str, default="env/data/runtime/queries", help="Output path for generated queries")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="LLM model name for filtering and generating queries")
    args = parser.parse_args()

    # Validate arguments
    if not args.generate_train and not args.generate_test:
        parser.error("必须指定 --generate_train 或 --generate_test 中的至少一个")
    
    # Generate the training set
    if args.generate_train:
        if not args.train_search_space_path:
            parser.error("--generate_train 需要指定 --train_search_space_path")
        if not args.train_config_path:
            parser.error("--generate_train 需要指定 --train_config_path")
        
        train_query_dir = os.path.join(args.query_dir, "train")
        print(f"\n{'=' * 60}")
        print(f"[INFO] 开始生成训练集查询...")
        print(f"[INFO] Search space: {args.train_search_space_path}")
        print(f"[INFO] Config file: {args.train_config_path}")
        print(f"[INFO] Output directory: {train_query_dir}")
        print(f"{'=' * 60}\n")
        
        generate_queries(
            search_space_data_path=args.train_search_space_path,
            num_queries=args.num_queries,
            seed=args.seed,
            model_name=args.model_name,
            query_dir=train_query_dir,
            config_path=args.train_config_path
        )
    
    # Generate the test set
    if args.generate_test:
        if not args.test_search_space_path:
            parser.error("--generate_test 需要指定 --test_search_space_path")
        if not args.test_config_path:
            parser.error("--generate_test 需要指定 --test_config_path")
        
        test_query_dir = os.path.join(args.query_dir, "test")
        print(f"\n{'=' * 60}")
        print(f"[INFO] 开始生成测试集查询...")
        print(f"[INFO] Search space: {args.test_search_space_path}")
        print(f"[INFO] Config file: {args.test_config_path}")
        print(f"[INFO] Output directory: {test_query_dir}")
        print(f"{'=' * 60}\n")
        
        generate_queries(
            search_space_data_path=args.test_search_space_path,
            num_queries=args.num_queries,
            seed=args.seed,
            model_name=args.model_name,
            query_dir=test_query_dir,
            config_path=args.test_config_path
        )

"""
Usage example:

# Generate the training set
python env/domains/travel/generate_queries.py \
    --generate_train \
    --train_search_space_path env/data/static/search_spaces/train_search_space.json \
    --num_queries 1000 \
    --seed 42 \
    --model_name gpt-4o \
    --query_dir env/data/runtime/queries

# Generate the test set
python env/domains/travel/generate_queries.py \
    --generate_test \
    --test_search_space_path env/data/static/search_spaces/test_search_space.json \
    --num_queries 1000 \
    --seed 42 \
    --model_name gpt-4o \
    --query_dir env/data/runtime/queries

# Generate both the training and test sets
python env/domains/travel/generate_queries.py \
    --generate_train \
    --generate_test \
    --train_search_space_path env/data/static/search_spaces/train_search_space.json \
    --test_search_space_path env/data/static/search_spaces/test_search_space.json \
    --num_queries 1000 \
    --seed 42 \
    --model_name gpt-4o \
    --query_dir env/data/runtime/queries
"""