"""
tool_registry.py

Central registry that manages and queries every tool (atomic and composite) in the travel domain.
Supports lookups by name, subtask, type, complexity, and more so other modules can invoke them easily.
"""

import sys
import os
import random
import math
import json
import hashlib
import numpy as np
project_root = os.path.join(os.path.dirname(__file__), '..', '..', '..')
sys.path.insert(0, project_root)

from typing import Dict, List, Any, Optional, Union
from env.domains.travel.atomic_tools import get_all_dynamic_atomic_tools
from env.domains.travel.composite_tools import get_all_dynamic_composite_tools, CompositeTool
from env.domains.travel.refinement_steps_catalog import get_max_refinement_depth
from env.core.base_types import Tool, AtomicTool

# =========================
# Tool registration and query entry points
# =========================

def get_all_tools(refinement_level: int = None) -> Dict[str, Tool]:
    """
    Retrieve all atomic and composite tools merged into a single dict.
    """
    atomic_tools = get_all_dynamic_atomic_tools(refinement_level)
    composite_tools = get_all_dynamic_composite_tools(refinement_level)
    all_tools = {}
    all_tools.update(atomic_tools)
    all_tools.update(composite_tools)
    return all_tools

def get_tool(tool_name: str, refinement_level: int = None) -> Tool:
    """
    Fetch any tool by name (atomic or composite).
    Note: the returned tool does not have the cost attribute populated.
    """
    all_tools = get_all_tools(refinement_level)
    if tool_name not in all_tools:
        raise KeyError(f"Tool '{tool_name}' not found at refinement level {refinement_level}")
    return all_tools[tool_name]

def get_tools_by_subtask(subtask: str, refinement_level: int = None) -> List[Tool]:
    """
    Return every tool (atomic and composite) associated with a specific subtask.
    """
    all_tools = get_all_tools(refinement_level)
    subtask_cap = subtask.capitalize()
    return [tool for tool in all_tools.values() if tool.name.startswith(subtask_cap)]

def get_atomic_tools(refinement_level: int = None) -> Dict[str, AtomicTool]:
    """
    Retrieve all atomic tools.
    """
    return get_all_dynamic_atomic_tools(refinement_level)

def get_composite_tools(refinement_level: int = None) -> Dict[str, CompositeTool]:
    """
    Retrieve all composite tools.
    """
    return get_all_dynamic_composite_tools(refinement_level)

def get_tools_by_input_type(input_type: str, refinement_level: int = None) -> List[Tool]:
    """
    Locate tools by their input type.
    """
    all_tools = get_all_tools(refinement_level)
    return [tool for tool in all_tools.values() if input_type in getattr(tool, "input_types", [])]

def get_tools_by_output_type(output_type: str, refinement_level: int = None) -> List[Tool]:
    """
    Locate tools by their output type.
    """
    all_tools = get_all_tools(refinement_level)
    return [tool for tool in all_tools.values() if getattr(tool, "output_type", None) == output_type]

def get_tools_by_complexity(complexity: int, refinement_level: int = None) -> List[Tool]:
    """
    Filter tools by complexity (atomic tools count as 1, composite tools use their component count).
    """
    all_tools = get_all_tools(refinement_level)
    result = []
    for tool in all_tools.values():
        if hasattr(tool, "component_tool_names"):
            if len(tool.component_tool_names) == complexity:
                result.append(tool)
        else:
            if complexity == 1:
                result.append(tool)
    return result

def export_tools_dict(output_dir: str = None, refinement_level: int = None) -> Dict[str, Any]:
    """
    Export all tools to a dict with metadata for serialization, storage, and analysis.
    """
    import datetime
    all_tools = get_all_tools(refinement_level)
    tools_dict = {name: tool.to_dict() for name, tool in all_tools.items()}

    # Compute the complexity distribution
    complexity_dist = {}
    for tool in all_tools.values():
        if hasattr(tool, "component_count"):  # Use the component_count attribute
            c = tool.component_count
        else:
            c = 1
        complexity_dist[c] = complexity_dist.get(c, 0) + 1

    metadata = {
        "generated_at": datetime.datetime.now().isoformat(),
        "refinement_level": refinement_level,
        "seed": None,
        "total_tools": len(all_tools),
        "atomic": sum(1 for t in all_tools.values() if isinstance(t, AtomicTool)),
        "composite": sum(1 for t in all_tools.values() if isinstance(t, CompositeTool)),
        "complexity_distribution": complexity_dist,
    }

    export_obj = {
        "metadata": metadata,
        "tools": tools_dict
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        tools_path = os.path.join(output_dir, f"tools_refine_{refinement_level}.json")
        import json
        with open(tools_path, "w", encoding="utf-8") as f:
            json.dump(export_obj, f, ensure_ascii=False, indent=4)
        # print(f"Exported tools dict to {tools_path}")

    return export_obj

def validate_all_tools(refinement_level: int = None) -> Dict[str, bool]:
    """
    Validate the integrity of every tool definition.
    """
    all_tools = get_all_tools(refinement_level)
    results = {}
    for name, tool in all_tools.items():
        # Composite tool validation already happens in the constructor
        if tool.get_tool_type() == "composite":
            results[name] = isinstance(tool, CompositeTool)
            if not isinstance(tool, CompositeTool):
                print(f"[ERROR] Composite tool '{name}' failed validation.")
        else:
            results[name] = isinstance(tool, AtomicTool)
            if not isinstance(tool, AtomicTool):
                print(f"[ERROR] Atomic tool '{name}' failed validation.")
    return results

def get_tool_statistics(refinement_level: int = None) -> Dict[str, Any]:
    """
    Gather statistics such as tool count and type distribution.
    """
    all_tools = get_all_tools(refinement_level)
    atomic_count = sum(1 for t in all_tools.values() if isinstance(t, AtomicTool))
    composite_count = sum(1 for t in all_tools.values() if isinstance(t, CompositeTool))
    return {
        "total": len(all_tools),
        "atomic": atomic_count,
        "composite": composite_count,
    }
    
    
def assign_costs_to_atomic_tools(output_dir: str = None, min_atomic_cost: int = 1, max_atomic_cost: int = 15, all_tools: Optional[Dict[str, Any]] = None, random_seed: int = 42, refinement_level: int = None) -> Dict[str, Any]:
    def _stable_int(s: str) -> int:
        return int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % (2**31 - 1)
    """
    Assign random costs to all atomic tools.
    """
    # Logic: traverse every tool and handle only those with type "atomic".
    # Initialize a random generator with the provided seed and assign each atomic tool
    # a random two-decimal cost between the specified min and max values.
    if all_tools is None:
        all_tools = get_all_tools(refinement_level)
        
    rng = random.Random(random_seed)
    for tool_name in all_tools["tools"]:
        tool = all_tools["tools"][tool_name]
        if tool["type"] == "atomic":
            cost = round(rng.uniform(min_atomic_cost, max_atomic_cost), 2)
            tool["cost"] = cost
            # print(f"Assigned cost {cost} to atomic tool '{tool_name}'")
            
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        tools_path = os.path.join(output_dir, f"tools_with_atomic_costs_refine_{refinement_level}.json")
        with open(tools_path, "w", encoding="utf-8") as f:
            json.dump(all_tools, f, ensure_ascii=False, indent=4)
        # print(f"Exported tools with atomic costs to {tools_path}")
        
    return all_tools

def calculate_std(min_atomic_cost: int, max_atomic_cost: int, random_seed: int) -> float:
    """
    Compute the standard deviation for a single atomic cost drawn uniformly from
    [min_atomic_cost, max_atomic_cost].

    Args:
        min_atomic_cost (int): Minimum cost.
        max_atomic_cost (int): Maximum cost.
        random_seed (int): Random seed placeholder (kept for API parity; unused in the formula).

    Returns:
        float: Standard deviation rounded to two decimals.
    """
    # Standard deviation is independent of randomness; keep the parameter for API consistency without touching global state.
    width = max_atomic_cost - min_atomic_cost
    std = width / math.sqrt(12)
    return round(std, 2)

def assign_costs_to_composite_tools(output_dir: str = None, min_atomic_cost: int = 19, max_atomic_cost: int = 21, noise_std = 0.1, all_tools: Optional[Dict[str, Any]] = None, random_seed: int = 42, refinement_level: int = None) -> Dict[str, Any]:
    """
    Assign costs to all composite tools by summing component costs and adding Gaussian noise.
    """
    if all_tools is None:
        all_tools = get_all_tools(refinement_level)
        
    # Configure a NumPy RNG without polluting global state
    def _stable_int(s: str) -> int:
        return int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % (2**31 - 1)
    
    for tool_name in all_tools["tools"]:
        tool = all_tools["tools"][tool_name]
        if tool["type"] == "composite":
            # Skip tools that already have a cost assigned
            if "cost" in tool and tool["cost"] is not None:
                print(f"[WARNING] Composite tool '{tool_name}' already has cost: {tool['cost']}")
                continue
                
            component_costs = []
            for comp_name in tool.get("component_tool_names", []):
                comp_tool = all_tools["tools"].get(comp_name)
                if comp_tool and "cost" in comp_tool and comp_tool["cost"] is not None:
                    component_costs.append(comp_tool["cost"])
                else:
                    print(f"[ERROR] Component tool '{comp_name}' not found or has no cost.")
            if component_costs:
                base_cost = sum(component_costs)
                std = noise_std
                
                # Use a stable hash of the global seed and tool name to derive a local seed
                combined_seed = _stable_int(f"{random_seed}:{tool_name}")
                rng = np.random.default_rng(combined_seed)
                noise = rng.normal(0, std * math.sqrt(len(component_costs)))
                
                total_cost = round(max(1, base_cost + noise), 2)
                tool["cost"] = total_cost
            else:
                print(f"[ERROR] Composite tool '{tool_name}' has no valid component costs.")
                
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        tools_path = os.path.join(output_dir, f"tools_with_all_costs_refine_{refinement_level}.json")
        with open(tools_path, "w", encoding="utf-8") as f:
            json.dump(all_tools, f, ensure_ascii=False, indent=4)
        # print(f"Exported tools with all costs to {tools_path}")
        
    return all_tools


def get_tool_cost_statistics(output_dir: str = None, refinement_level: int = None) -> Dict[str, Any]:
    """
    Compute and return tool cost statistics, including:
    - Average, maximum, and minimum costs for atomic tools
    - Average, maximum, and minimum costs for composite tools grouped by length k
    """
    # print("Calculating tool cost statistics...")
    with open(os.path.join(output_dir, f"tools_with_all_costs_refine_{refinement_level}.json"), "r", encoding="utf-8") as f:
        all_tools_with_costs = json.load(f)

    atomic_costs = []
    composite_costs_by_length = {}
    for tool in all_tools_with_costs["tools"].values():
        if "cost" in tool and tool["cost"] is not None:
            if tool["type"] == "atomic":
                atomic_costs.append(tool["cost"])
            elif tool["type"] == "composite":
                length = len(tool.get("component_tool_names", []))
                if length not in composite_costs_by_length:
                    composite_costs_by_length[length] = []
                composite_costs_by_length[length].append(tool["cost"])
                
    stats = {}
    if atomic_costs:
        stats["atomic_avg_cost"] = round(sum(atomic_costs) / len(atomic_costs), 2)
        stats["atomic_max_cost"] = round(max(atomic_costs), 2)
        stats["atomic_min_cost"] = round(min(atomic_costs), 2)
    for length, costs in composite_costs_by_length.items():
        if costs:
            stats[f"composite_length_{length}_avg_cost"] = round(sum(costs) / len(costs), 2)
            stats[f"composite_length_{length}_max_cost"] = round(max(costs), 2)
            stats[f"composite_length_{length}_min_cost"] = round(min(costs), 2)

    all_tools_with_costs["metadata"]["cost_statistics"] = stats
    with open(os.path.join(output_dir, f"tools_with_all_costs_refine_{refinement_level}.json"), "w", encoding="utf-8") as f:
        json.dump(all_tools_with_costs, f, ensure_ascii=False, indent=4)
    
    return stats


def assign_costs_to_tools(output_dir: str = None, refinement_level: int = None, min_atomic_cost: int = 19, max_atomic_cost: int = 21, noise_std = 0.1, random_seed: int = 42):
    """
    Assign costs to all tools: atomic tools receive random costs, composite tools sum their components plus Gaussian noise.
    """
    all_tools_without_costs = export_tools_dict(output_dir, refinement_level)
    
    # Assign random costs to atomic tools
    all_tools_with_atomic_costs = assign_costs_to_atomic_tools(output_dir, min_atomic_cost, max_atomic_cost, all_tools_without_costs, random_seed, refinement_level)

    # Assign costs to composite tools
    all_tools_with_costs = assign_costs_to_composite_tools(output_dir, min_atomic_cost, max_atomic_cost, noise_std, all_tools_with_atomic_costs, random_seed, refinement_level)

    # Generate statistics for atomic and composite tools
    stats = get_tool_cost_statistics(output_dir, refinement_level)
    
def generate_tool_interface(input_tool_path:str, output_path: str, refinement_level: int = None):
    
    with open(input_tool_path, "r", encoding="utf-8") as f:
        all_tools = json.load(f)["tools"]
    
    for tool_name, tool_dic in all_tools.items():
        if "interface" not in tool_dic:
            tool = get_tool(tool_name, refinement_level)
            tool.cost = tool_dic.get("cost", None)
            tool_dic["interface"] = tool.generate_interface()
            
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"metadata": all_tools.get("metadata", {}), "tools": all_tools}, f, ensure_ascii=False, indent=4)

def tools_ready(output_dir: str = None, refinement_level: int = None, min_atomic_cost: int = 19, max_atomic_cost: int = 21, noise_std = 0.1, random_seed: int = 42, control_tool_length: bool = False, max_tool_length: int = 8, ban_longest_tool: bool = False):
    """
    Generate and save tool definition files with costs in one step.
    """
    # If refinement_level is not provided, use the maximum depth by default
    if refinement_level is None:
        refinement_level = get_max_refinement_depth()
    
    assign_costs_to_tools(output_dir, refinement_level, min_atomic_cost, max_atomic_cost, noise_std, random_seed)
    input_tool_path = os.path.join(output_dir, f"tools_with_all_costs_refine_{refinement_level}.json")

    # Optional: filter by length at the source to keep a single entry point
    if control_tool_length or ban_longest_tool:
        with open(input_tool_path, "r", encoding="utf-8") as f:
            _tools_obj = json.load(f)
        _tools = _tools_obj.get("tools", {})

        # Compute the maximum length (composite tools only)
        longest_len = None
        if ban_longest_tool:
            for _name, _tool in _tools.items():
                if _tool.get("type") == "composite":
                    clen = _tool.get("component_count", 1)
                    if longest_len is None or clen > longest_len:
                        longest_len = clen

        erased_names = set()
        for _name, _tool in list(_tools.items()):
            if _tool.get("type") != "composite":
                continue
            clen = _tool.get("component_count", 1)
            if control_tool_length and clen > max_tool_length:
                erased_names.add(_name)
                continue
            if ban_longest_tool and longest_len is not None and clen == longest_len:
                erased_names.add(_name)

        if erased_names:
            _tools_obj["tools"] = {n: d for n, d in _tools.items() if n not in erased_names}
            with open(input_tool_path, "w", encoding="utf-8") as f:
                json.dump(_tools_obj, f, ensure_ascii=False, indent=4)
    output_tool_path = os.path.join(output_dir, f"tools_ready_refine_{refinement_level}.json")
    generate_tool_interface(input_tool_path, output_tool_path, refinement_level)
    # print(f"Tools with costs and interfaces saved to {output_tool_path}")

# =========================
# In-memory tool generation (no file IO)
# =========================
def tools_ready_in_memory(refinement_level: int = None, min_atomic_cost: int = 19, max_atomic_cost: int = 21, noise_std = 0.1, random_seed: int = 42, control_tool_length: bool = False, max_tool_length: int = 8, ban_longest_tool: bool = False) -> Dict[str, Any]:
    """
    Generate tool definitions in memory (including costs) to avoid disk IO.

    Returns an object with the same structure as the file-based version:
    {
        "metadata": {...},
        "tools": { tool_name: tool_dict, ... }
    }
    """
    # 1) Export the base definition of every tool (without writing to disk)
    all_tools_without_costs = export_tools_dict(output_dir=None, refinement_level=refinement_level)

    # 2) Assign costs to atomic tools (in memory)
    all_tools_with_atomic_costs = assign_costs_to_atomic_tools(
        output_dir=None,
        min_atomic_cost=min_atomic_cost,
        max_atomic_cost=max_atomic_cost,
        all_tools=all_tools_without_costs,
        random_seed=random_seed,
        refinement_level=refinement_level,
    )

    # 3) Assign costs to composite tools (in memory)
    all_tools_with_costs = assign_costs_to_composite_tools(
        output_dir=None,
        min_atomic_cost=min_atomic_cost,
        max_atomic_cost=max_atomic_cost,
        noise_std=noise_std,
        all_tools=all_tools_with_atomic_costs,
        random_seed=random_seed,
        refinement_level=refinement_level,
    )

    # Apply length filtering at the source (composite tools only) to mirror the file-based version
    if control_tool_length or ban_longest_tool:
        tools_dict = all_tools_with_costs.get("tools", {})
        # Compute the longest length (composite tools only)
        longest_len = None
        if ban_longest_tool:
            for _name, _tool in tools_dict.items():
                if _tool.get("type") == "composite":
                    clen = _tool.get("component_count", 1)
                    if longest_len is None or clen > longest_len:
                        longest_len = clen
        erased_names = set()
        for _name, _tool in list(tools_dict.items()):
            if _tool.get("type") != "composite":
                continue
            clen = _tool.get("component_count", 1)
            if control_tool_length and clen > max_tool_length:
                erased_names.add(_name)
                continue
            if ban_longest_tool and longest_len is not None and clen == longest_len:
                erased_names.add(_name)
        if erased_names:
            all_tools_with_costs["tools"] = {n: d for n, d in tools_dict.items() if n not in erased_names}

    # Skip computing/writing statistics to avoid IO and overhead
    return all_tools_with_costs

# =========================
# Test code
# =========================

if __name__ == "__main__":
    import argparse
    import os
    import json

    parser = argparse.ArgumentParser(description="Tool Registry Full Test")
    parser.add_argument("--refinement_level", type=int, default=4, help="Refinement level for tool generation")
    parser.add_argument("--output_dir", type=str, default="test_output", help="Directory to export tools dict and validation results")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for cost assignment")
    args = parser.parse_args()

    refinement_level = args.refinement_level
    output_dir = args.output_dir
    random_seed = args.random_seed

    # print(f"=== Tool Registry Test at Refinement Level {refinement_level} ===")
    os.makedirs(output_dir, exist_ok=True)

    tools_ready(output_dir, refinement_level, random_seed=random_seed)
    # print("Tool registry test completed.")
"""
python env/domains/travel/tool_registry.py --refinement_level 4 --output_dir env/domains/travel/test/ --random_seed 42 > env/domains/travel/logs/test.log
"""