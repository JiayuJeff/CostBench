"""
Step 3: Generate search space

This script performs the following tasks:
1. Build a TimeInfo pool and randomly select a target number of time pairs
2. Create a LocationPreference mapping: preference tuple -> LocationPreference ID
3. Build per-LocationPreference mappings for other subtasks: LocationPreference ID -> subtask preference tuples
4. Write the complete search space to disk

Supports both training and testing modes:
- Provide different config files to generate training or testing search spaces
- Training data is saved to {output_dir}/train/search_space.json
- Testing data is saved to {output_dir}/test/search_space.json

Data structure:
{
  "time_pool": ["(<TimeInfo00001>, <TimeInfo00005>)", "(<TimeInfo00002>, <TimeInfo00007>)", ...],
  "locations": {
    "(city,bustling_metropolis,adventure,history_buff)": "<LocationPreference00001>",
    ...
  },
  "location_information": {
    "<LocationPreference00001>": {
      "transportation": {
        "(train,economy,speed_priority,multi_stop)": "<TransportationPreference00001>",
        ...
      },
      "accommodation": {...},
      ...
    }
  }
}

Usage:
# Generate the training search space
python env/domains/travel/generate_search_space.py \
    --generate_train \
    --train_config_path env/domains/travel/train_travel_config.yaml \
    --num-times 5000 \
    --num-time-combinations 5000 \
    --seed 42
    
python env/domains/travel/generate_search_space.py --generate_train --train_config_path env/domains/travel/train_travel_config.yaml --num-times 5000 --num-time-combinations 5000 --seed 42
    
python env/domains/travel/generate_search_space.py --generate_train --train_config_path env/domains/travel/train_travel_config.yaml --num-times 5000 --num-time-combinations 5000 --seed 42

# Generate the testing search space
python env/domains/travel/generate_search_space.py \
    --generate_test \
    --test_config_path env/domains/travel/test_travel_config.yaml \
    --num-times 5000 \
    --num-time-combinations 5000 \
    --seed 42

# Generate both training and testing search spaces
python env/domains/travel/generate_search_space.py \
    --generate_train \
    --generate_test \
    --train_config_path env/domains/travel/train_travel_config.yaml \
    --test_config_path env/domains/travel/test_travel_config.yaml \
    --reset-id-history \
    --num-times 5000 \
    --num-time-combinations 5000 \
    --seed 42
"""

import sys
import os
import yaml
import json
import random
import argparse
from datetime import datetime
from pathlib import Path
from itertools import product
from typing import Dict, List, Any, Tuple

# Add the project root to sys.path
project_root = os.path.join(os.path.dirname(__file__), '..', '..', '..')
sys.path.insert(0, project_root)

from env.utils.id_generator import get_global_generator, IDGenerator


def reset_id_generation_history(mode: str = "test"):
    """
    Reset the ID generation history.
    
    This clears all ID counters so the next generated IDs start from 00001 (test) or 000001 (train).
    
    Args:
        mode: ID generation mode, "train" (6 digits) or "test" (5 digits)
    """
    print(f"\n[INFO] Resetting ID generation history (mode: {mode})...")
    
    generator = get_global_generator(mode=mode)
    
    # Show the status before resetting
    old_counts = generator.get_all_counts()
    if old_counts:
        print(f"[INFO] Current ID counts before reset:")
        total_ids = sum(old_counts.values())
        for id_type, count in old_counts.items():
            print(f"  {id_type:20s}: {count:6d} IDs")
        print(f"  {'Total':20s}: {total_ids:6d} IDs")
    else:
        print(f"[INFO] No existing ID history found")
    
    # Perform the reset
    generator.reset()
    
    # Persist the reset state
    generator.save_state()
    
    print(f"[SUCCESS] ID generation history reset complete!")
    print(f"[INFO] All ID counters are now at 0")
    next_id_format = "000001" if mode == "train" else "00001"
    print(f"[INFO] Next generated IDs will start from {next_id_format}")
    
    return True


def load_travel_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load the travel_config.yaml configuration file.
    
    Args:
        config_path: Path to the config file; use the default path when None
    
    Returns:
        Dict: Parsed configuration data
    """
    if config_path is None:
        config_path = "env/domains/travel/travel_config.yaml"
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Travel config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f"[INFO] Loaded travel config from: {config_path}")
    return config


def generate_time_pool(num_times: int, num_combinations: int, seed: int, mode: str = "test", generator=None) -> List[str]:
    """
    Build the time pool and create the requested number of time pair combinations.
    
    Args:
        num_times: Number of base TimeInfo IDs to generate
        num_combinations: Number of time pair combinations to produce
        seed: Random seed
        mode: ID generation mode, "train" (6 digits) or "test" (5 digits)
        generator: Optional ID generator instance; fall back to the global generator when None
        
    Returns:
        List[str]: List of time pair strings (e.g. "(<TimeInfo00001>, <TimeInfo00005>)")
    """
    print(f"[INFO] Generating time combinations (base times: {num_times}, combinations: {num_combinations}, seed: {seed}, mode: {mode})...")
    
    # Enforce per-mode limits
    max_limit = 999999 if mode == "train" else 99999
    if num_times > max_limit:
        raise ValueError(f"num_times cannot exceed {max_limit} in {mode} mode")
    
    # Calculate the maximum number of possible pairs
    max_combinations = num_times * (num_times - 1) // 2
    if num_combinations > max_combinations:
        raise ValueError(f"Cannot generate {num_combinations} combinations from {num_times} times. Max possible: {max_combinations}")
    
    if generator is None:
        generator = get_global_generator(mode=mode)
    
    # 1. Generate the base pool of TimeInfo IDs
    base_time_ids = []
    for i in range(num_times):
        time_id = generator.generate_id("TimeInfo")
        base_time_ids.append(time_id)
    
    print(f"[INFO] Generated {len(base_time_ids)} base TimeInfo IDs")
    
    # 2. Enumerate all possible pairs
    from itertools import combinations
    all_combinations = list(combinations(base_time_ids, 2))
    
    print(f"[INFO] Total possible combinations: {len(all_combinations)}")
    
    # 3. Randomly sample the target number of pairs
    random.seed(seed)
    selected_combinations = random.sample(all_combinations, num_combinations)
    
    # 4. Format as "(<TimeInfo00001>, <TimeInfo00005>)"
    combination_strings = []
    for combo in selected_combinations:
        combo_str = f"({combo[0]}, {combo[1]})"
        combination_strings.append(combo_str)
    
    print(f"[SUCCESS] Generated {len(combination_strings)} time combinations from {num_times} base times")
    return combination_strings


def generate_preference_combinations(subtask_config: Dict[str, List[str]]) -> List[Tuple[str, ...]]:
    """
    Generate all possible preference tuples for a single subtask.
    
    Args:
        subtask_config: Dimension configuration for the subtask
        
    Returns:
        List[Tuple]: List of all combination tuples
    """
    dimensions = subtask_config.get("dimensions", {})
    
    if not dimensions:
        print(f"[WARNING] No dimensions found for subtask")
        return []
    
    # Collect dimension names and values (using a deterministic order)
    dimension_names = sorted(dimensions.keys())  # Keep the order consistent
    dimension_values = [dimensions[dim] for dim in dimension_names]
    
    # Build the Cartesian product
    combinations = []
    for combination in product(*dimension_values):
        combinations.append(combination)
    
    print(f"[INFO] Generated {len(combinations)} combinations from {len(dimension_names)} dimensions")
    
    return combinations


def generate_location_mapping(travel_config: Dict[str, Any], mode: str = "test", generator=None) -> Dict[str, str]:
    """
    Build the mapping from location preference tuples to LocationPreference IDs.
    
    Args:
        travel_config: Travel configuration data
        mode: ID generation mode, "train" (6 digits) or "test" (5 digits)
        generator: Optional ID generator; falls back to the global generator when None
        
    Returns:
        Dict[str, str]: {preference_tuple_str: location_preference_id}
    """
    print(f"[INFO] Generating location mapping (mode: {mode})...")
    
    if "location" not in travel_config["main_subtasks"]:
        raise ValueError("Location subtask not found in config")
    
    location_config = travel_config["main_subtasks"]["location"]
    location_combinations = generate_preference_combinations(location_config)
    
    if generator is None:
        generator = get_global_generator(mode=mode)
    location_mapping = {}
    
    for combo in location_combinations:
        combo_str = str(combo)  # Convert the tuple to a string key
        location_preference_id = generator.generate_id("LocationPreference")
        location_mapping[combo_str] = location_preference_id
    
    print(f"[SUCCESS] Generated {len(location_mapping)} location mappings")
    return location_mapping


def generate_location_information(
    travel_config: Dict[str, Any], 
    location_mapping: Dict[str, str],
    mode: str = "test",
    generator=None
) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Generate preference mappings for each non-location subtask under every LocationPreference ID.
    
    Args:
        travel_config: Travel configuration data
        location_mapping: Mapping from location preference strings to LocationPreference IDs
        mode: ID generation mode, "train" (6 digits) or "test" (5 digits)
        generator: Optional ID generator; defaults to the global generator when None
        
    Returns:
        Dict: {location_preference_id: {subtask: {preference_tuple_str: preference_id}}}
    """
    print(f"[INFO] Generating location information (mode: {mode})...")
    
    # Collect every subtask except "location"
    other_subtasks = [name for name in travel_config["main_subtasks"].keys() if name != "location"]
    
    if not other_subtasks:
        print(f"[WARNING] No other subtasks found besides location")
        return {}
    
    if generator is None:
        generator = get_global_generator(mode=mode)
    
    location_information = {}
    
    for location_preference_id in location_mapping.values():
        location_info = {}
        
        for subtask_name in other_subtasks:
            subtask_config = travel_config["main_subtasks"][subtask_name]
            subtask_combinations = generate_preference_combinations(subtask_config)
            
            subtask_mapping = {}
            type_name = f"{subtask_name.capitalize()}Preference"
            
            for combo in subtask_combinations:
                combo_str = str(combo)
                preference_id = generator.generate_id(type_name)
                subtask_mapping[combo_str] = preference_id
            
            location_info[subtask_name] = subtask_mapping
        
        location_information[location_preference_id] = location_info
    
    total_preferences = sum(
        len(subtask_mapping) 
        for location_info in location_information.values() 
        for subtask_mapping in location_info.values()
    )
    
    print(f"[SUCCESS] Generated location information for {len(location_information)} LocationPreference IDs")
    print(f"           Total preference mappings: {total_preferences}")
    
    return location_information


def ensure_output_directory(output_dir: str):
    """Ensure the output directory exists."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output directory ensured: {output_path.absolute()}")


def generate_search_space(
    num_times: int,
    num_time_combinations: int,
    seed: int,
    reset_id_history: bool,
    output_dir: str,
    config_path: str = None,
    mode: str = "test"
) -> str:
    """
    Generate the search space and save it as a JSON file.

    Args:
        num_times: Number of base TimeInfo IDs to generate
        num_time_combinations: Number of time pairs to select
        seed: Random seed
        reset_id_history: Whether to reset ID history before generation
        output_dir: Target directory for outputs
        config_path: Optional config path; use the default when None
        mode: ID generation mode, "train" (6 digits) or "test" (5 digits)

    Returns:
        str: Path to the output file
    """
    # Use a separate state file for each mode
    state_file = f"env/data/runtime/id_history/id_generator_state_{mode}.json"
    
    # Create a dedicated generator for the mode (avoid clashes with the global singleton)
    generator = IDGenerator(state_file=state_file, mode=mode)
    
    # 1. Optionally reset ID history
    if reset_id_history:
        print(f"\n[INFO] Resetting ID generation history (mode: {mode})...")
        old_counts = generator.get_all_counts()
        if old_counts:
            print(f"[INFO] Current ID counts before reset:")
            total_ids = sum(old_counts.values())
            for id_type, count in old_counts.items():
                print(f"  {id_type:20s}: {count:6d} IDs")
            print(f"  {'Total':20s}: {total_ids:6d} IDs")
        else:
            print(f"[INFO] No existing ID history found")
        
        generator.reset()
        generator.save_state()
        
        print(f"[SUCCESS] ID generation history reset complete!")
        print(f"[INFO] All ID counters are now at 0")
        next_id_format = "000001" if mode == "train" else "00001"
        print(f"[INFO] Next generated IDs will start from {next_id_format}")

    # 2. Load configuration
    travel_config = load_travel_config(config_path)

    # 3. Generate the time pool (using the dedicated generator)
    time_pool = generate_time_pool(num_times, num_time_combinations, seed, mode, generator)

    # 4. Generate the location mapping
    location_mapping = generate_location_mapping(travel_config, mode, generator)

    # 5. Generate location information
    location_information = generate_location_information(travel_config, location_mapping, mode, generator)

    # 6. Gather ID usage statistics
    id_stats = generator.get_all_counts()
    id_statistics = {
        "type_breakdown": id_stats,
        "total_types": len(id_stats),
        "total_ids": sum(id_stats.values())
    }

    # 7. Assemble output data
    search_space_data = {
        "time_pool": time_pool,
        "locations": location_mapping,
        "location_information": location_information,
    }

    # 8. Save results
    output_file = os.path.join(output_dir, "search_space.json")
    metadata = generate_search_space_report(
        search_space_data, num_times, num_time_combinations, seed, reset_id_history, output_dir=output_dir
    )
    search_space_with_metadata = {
        "metadata": metadata,
        **search_space_data
    }

    ensure_output_directory(output_dir)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(search_space_with_metadata, f, indent=4, ensure_ascii=False)

    print(f"[SUCCESS] Search space saved to: {output_file}")
    return output_file


def generate_search_space_report(search_space_data: Dict[str, Any], num_times: int = 1000, num_time_combinations: int = 5000, seed: int = 42, reset_id_history: bool = True, output_dir: str = "") -> str:
    """
    Generate a statistics report for the search space.
    
    Args:
        search_space_data: Search space payload
        num_times: Number of base TimeInfo IDs generated
        output_dir: Output directory
        
    Returns:
        str: Report payload
    """
    print(f"[INFO] Generating search space report...")
    
    # Summarize time-related metrics
    time_pool_count = len(search_space_data["time_pool"])
    # Compute the theoretical maximum number of combinations
    max_possible_combinations = num_times * (num_times - 1) // 2
    
    # Summarize location counts
    locations_count = len(search_space_data["locations"])
    location_information_count = len(search_space_data["location_information"])
    
    # Count preference totals per subtask
    subtask_stats = {}
    total_preferences = 0
    
    for location_preference_id, location_info in search_space_data["location_information"].items():
        for subtask_name, subtask_mappings in location_info.items():
            if subtask_name not in subtask_stats:
                subtask_stats[subtask_name] = 0
            subtask_stats[subtask_name] += len(subtask_mappings)
            total_preferences += len(subtask_mappings)
    
    report_data = {
        "generation_summary": {
            "generated_at": datetime.now().isoformat(),
            "script_version": "2.2",
            "parameters": {
                "num_times": num_times,
                "num_time_combinations": num_time_combinations,
                "seed": seed,
                "reset_id_history": reset_id_history
            }
        },
        "time_pool": {
            "base_times_generated": num_times,
            "combinations_generated": time_pool_count,
            "max_possible_combinations": max_possible_combinations,
            "combination_coverage_rate": round(time_pool_count / max_possible_combinations * 100, 2)
        },
        "location_mapping": {
            "total_location_preferences": locations_count,
            "covered_location_preferences": location_information_count
        },
        "preference_statistics": {
            "subtask_breakdown": subtask_stats,
            "total_preferences": total_preferences,
            "avg_preferences_per_location": round(total_preferences / locations_count, 1) if locations_count > 0 else 0
        },
        "data_structure": {
            "time_pool": "List of selected TimeInfo ID pairs",
            "locations": "Dict mapping preference tuples to LocationPreference IDs",
            "location_information": "Nested dict: LocationPreference ID → Subtask → Preference tuple → Preference ID"
        },
    }
    
    return report_data


if __name__ == "__main__":
    """Entry point."""
    parser = argparse.ArgumentParser(description="Generate comprehensive search space including time and location mappings")
    
    # Mode selection
    parser.add_argument(
        '--generate_train',
        action='store_true',
        help='Generate training search space'
    )
    parser.add_argument(
        '--generate_test',
        action='store_true',
        help='Generate testing search space'
    )
    
    # Config paths
    parser.add_argument(
        '--train_config_path',
        type=str,
        help='Path to training config file'
    )
    parser.add_argument(
        '--test_config_path',
        type=str,
        help='Path to testing config file'
    )
    
    # Shared parameters
    parser.add_argument(
        '--output-dir',
        type=str,
        default='env/data/static/search_spaces',
        help='Output directory for search space files (default: env/data/static/search_spaces)'
    )
    parser.add_argument(
        '--num-times',
        type=int,
        default=1000,
        help='Total number of TimeInfo IDs to generate (max 999999 for train mode, max 99999 for test mode, default: 1000)'
    )
    parser.add_argument(
        '--num-time-combinations',
        type=int,
        default=500,
        help='Number of time combinations to select (default: 500)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for time combination selection (default: 42)'
    )
    parser.add_argument(
        '--save-id-state',
        action='store_true',
        help='Save ID generator state after generation'
    )
    parser.add_argument(
        '--reset-id-history',
        action='store_true',
        help='Reset ID generation history before starting (all counters will start from 0)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.generate_train and not args.generate_test:
        parser.error("必须指定 --generate_train 或 --generate_test 中的至少一个")
    
    # Generate the training search space
    if args.generate_train:
        if not args.train_config_path:
            parser.error("--generate_train 需要指定 --train_config_path")
        
        train_output_dir = os.path.join(args.output_dir, "train")
        print(f"\n{'=' * 60}")
        print(f"[INFO] 开始生成训练集搜索空间...")
        print(f"[INFO] Config: {args.train_config_path}")
        print(f"[INFO] Output directory: {train_output_dir}")
        print(f"{'=' * 60}\n")
        
        train_output_file = generate_search_space(
            num_times=args.num_times,
            num_time_combinations=args.num_time_combinations,
            seed=args.seed,
            reset_id_history=args.reset_id_history,
            output_dir=train_output_dir,
            config_path=args.train_config_path,
            mode="train"
        )
        print(f"[SUCCESS] 训练集搜索空间已保存到: {train_output_file}")

        # Print metadata summary
        with open(train_output_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            print("[INFO] Metadata summary:")
            print(json.dumps(data.get("metadata", {}), indent=4, ensure_ascii=False))
    
    # Generate the testing search space
    if args.generate_test:
        if not args.test_config_path:
            parser.error("--generate_test 需要指定 --test_config_path")
        
        test_output_dir = os.path.join(args.output_dir, "test")
        print(f"\n{'=' * 60}")
        print(f"[INFO] 开始生成测试集搜索空间...")
        print(f"[INFO] Config: {args.test_config_path}")
        print(f"[INFO] Output directory: {test_output_dir}")
        print(f"{'=' * 60}\n")
        
        test_output_file = generate_search_space(
            num_times=args.num_times,
            num_time_combinations=args.num_time_combinations,
            seed=args.seed,
            reset_id_history=False,  # Keep ID history for the test set and continue incrementing
            output_dir=test_output_dir,
            config_path=args.test_config_path,
            mode="test"
        )
        print(f"[SUCCESS] 测试集搜索空间已保存到: {test_output_file}")

        # Print metadata summary
        with open(test_output_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            print("[INFO] Metadata summary:")
            print(json.dumps(data.get("metadata", {}), indent=4, ensure_ascii=False))

"""
Usage examples:

# Generate the training search space
python env/domains/travel/generate_search_space.py \
    --generate_train \
    --train_config_path env/domains/travel/train_travel_config.yaml \
    --reset-id-history \
    --output-dir env/data/static/search_spaces \
    --num-times 5000 \
    --num-time-combinations 5000 \
    --seed 42 \
    --save-id-state

# Generate the testing search space
python env/domains/travel/generate_search_space.py \
    --generate_test \
    --test_config_path env/domains/travel/test_travel_config.yaml \
    --output-dir env/data/static/search_spaces \
    --num-times 5000 \
    --num-time-combinations 5000 \
    --seed 42 \
    --save-id-state

# Generate both training and testing search spaces
python env/domains/travel/generate_search_space.py \
    --generate_train \
    --generate_test \
    --train_config_path env/domains/travel/train_travel_config.yaml \
    --test_config_path env/domains/travel/test_travel_config.yaml \
    --reset-id-history \
    --output-dir env/data/static/search_spaces \
    --num-times 5000 \
    --num-time-combinations 5000 \
    --seed 42 \
    --save-id-state

# PowerShell example (training set)
python env\domains\travel\generate_search_space.py --generate_train --train_config_path env\domains\travel\train_travel_config.yaml --reset-id-history --output-dir env\data\static\search_spaces --num-times 5000 --num-time-combinations 5000 --seed 42 --save-id-state
"""