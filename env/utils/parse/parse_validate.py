import json
import os
import sys
import argparse
import re
project_root = os.path.join(os.path.dirname(__file__), '..', '..', '..')
sys.path.insert(0, project_root)
from typing import List, Dict, Any

def parse_filter_queries(validation_raw_string: str):
    """
    Parse the validation raw string to determine if there is a conflict.
    
    Returns:
        1 if no conflict,
        0 if conflict,
        None if unable to determine.
    """
    
    patterns = [
        r"\*\*(no conflict|conflict)\*\*",      # "**no conflict**" or "**conflict**"
        r"(no conflict|conflict)\n",            # "no conflict\n" or "conflict\n"
        r"(no conflict|conflict)\.",            # "no conflict." or "conflict."
        r"(no conflict|conflict)",              # "no conflict" or "conflict"
    ]

    for pattern in patterns:
        match = re.search(pattern, validation_raw_string, re.IGNORECASE)
        if match:
            matched_text = match.group(1).lower()
            if matched_text == "no conflict":
                return 1
            elif matched_text == "conflict":
                return 0
    return None

def parse(input_path, output_path, task):

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    not_parsed_count = 0
    
    for item in data:
        if task == "generate_queries":
            # Example parsing logic for generating queries
            raise NotImplementedError("Generate queries task is not implemented yet.")
        elif task == "validate_queries":
            # Example parsing logic for validating queries
            item["is_valid"] = parse_filter_queries(item["validation_raw"])
            if item["is_valid"] is None:
                not_parsed_count += 1
        else:
            raise ValueError(f"Unknown task: {task}")

    print(f"Not parsed count for task {task}: {not_parsed_count}")
    print("Writing results to", output_path)

    with open(output_path, 'w', encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse and validate JSON files.")
    parser.add_argument("--input_path", help="Path to the input JSON file.")
    parser.add_argument("--task", choices=["generate_queries", "validate_queries"], required=True, help="Task to perform: generate_queries or validate_queries.")
    parser.add_argument("--output_path", help="Path to the output JSON file.")
    args = parser.parse_args()
    
    parse(args.input_path, args.output_path, args.task)
    
"""
Example usage:
python env/utils/parse/parse_validate.py \
    --input_path env/data/runtime/queries/filtered_queries.json \
    --output_path env/data/runtime/queries/filtered_queries.json \
    --task validate_queries
"""
