"""
Groundtruth Solver

Graph-search-based optimal path solver for computing the minimum-cost tool invocation sequence in travel planning tasks.
The problem is modeled as a graph search:
- Nodes: state sets composed of data types
- Edges: tool invocations (atomic tools and composite tools)
- Weights: tool costs
- Goal: find the shortest path from the initial state to the goal state
"""

import heapq
from typing import Dict, List, Set, Any, Optional, Tuple, FrozenSet, Union
from dataclasses import dataclass
import sys
import os

# Add project root to sys.path
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)

from env.core.base_types import Tool, create_tool_from_dict
from env.core.data_types import get_final_type
from env.domains.travel.enums import ENUM_MAPPINGS


@dataclass
class PathResult:
    """Path solving result."""
    tools: List[str]           # Optimal tool sequence
    total_cost: float          # Total cost
    path_length: int           # Path length (number of tool invocations)
    state_sequence: List[FrozenSet[str]]  # Sequence of state transitions


class NoPathFoundError(Exception):
    """Exception raised when no path can be found."""
    pass


class GroundtruthSolver:
    """
    Optimal path solver based on Dijkstra's algorithm
    """
    
    def __init__(self, tools_with_costs: Union[Dict[str, Tool], Dict[str, Dict[str, Any]]]):
        """
        Initialize the solver.
        
        Args:
            tools_with_costs: Tool dictionary supporting two formats:
                1. Dict[str, Tool] - already converted Tool objects
                2. Dict[str, Dict[str, Any]] - raw dict format requiring conversion
        """
        if tools_with_costs and tools_with_costs.values() and isinstance(list(tools_with_costs.values())[0], dict):
            # Input is dictionary format and needs to be converted
            self.tools = self._convert_dict_to_tools(tools_with_costs)
        else:
            # Input already contains Tool objects
            self.tools = tools_with_costs
        
    def _convert_dict_to_tools(self, tools_dict: Dict[str, Dict[str, Any]]) -> Dict[str, Tool]:
        """
        Convert dictionary definitions into Tool objects.
        
        Args:
            tools_dict: Tool definitions in dictionary form
            
        Returns:
            Dict[str, Tool]: Converted tool dictionary
            
        Raises:
            ValueError: Raised when conversion fails with detailed information
        """
        converted_tools = {}
        required_fields = {"input_types", "output_type", "cost"}
        
        for tool_name, tool_data in tools_dict.items():
            try:
                # Check required fields
                missing_fields = required_fields - set(tool_data.keys())
                if missing_fields:
                    raise ValueError(f"Missing required fields: {missing_fields}")
                
                # Validate field types
                if not isinstance(tool_data["input_types"], list):
                    raise ValueError(f"input_types must be list, got {type(tool_data['input_types'])}")
                if not isinstance(tool_data["output_type"], str):
                    raise ValueError(f"output_type must be str, got {type(tool_data['output_type'])}")
                if not isinstance(tool_data["cost"], (int, float)):
                    raise ValueError(f"cost must be number, got {type(tool_data['cost'])}")
                
                # Use the existing create_tool_from_dict helper
                tool_data_copy = tool_data.copy()
                tool_data_copy["name"] = tool_name  # Ensure the name field exists
                converted_tools[tool_name] = create_tool_from_dict(tool_data_copy)
                
            except Exception as e:
                raise ValueError(f"Failed to convert tool '{tool_name}': {str(e)}")
        
        return converted_tools
        
    def _infer_initial_state(self, task: str) -> FrozenSet[str]:
        """Infer the initial state from the task name."""
        initial_types = set()
        
        # Always include TimeInfo
        initial_types.add("TimeInfo")
        
        if task == "location":
            # Location task: user preference enumerations
            for enum_class in ENUM_MAPPINGS["location"]["search"]:
                initial_types.add(enum_class.__name__)
        else:
            # Other tasks: user preference enumerations + LocationPreference
            initial_types.add("LocationPreference")
            for enum_class in ENUM_MAPPINGS[task]["search"]:
                initial_types.add(enum_class.__name__)
                
        return frozenset(initial_types)
    
    def _infer_target_types(self, task: str) -> Set[str]:
        """Infer the target types from the task name."""
        final_type = get_final_type(task)
        return {final_type}
    
    def _get_successors(self, state: FrozenSet[str]) -> List[Tuple[FrozenSet[str], str, float]]:
        """
        Get successors of the given state.
        
        Args:
            state: Current state (set of data types)
            
        Returns:
            List of (next_state, tool_name, cost)
        """
        successors = []
        
        for tool_name, tool in self.tools.items():
            # Check whether the state contains all required input types for the tool
            input_types = set(tool.input_types)
            if input_types.issubset(state):
                # Compute the new state: keep existing types and add the output type
                new_state = frozenset(state | {tool.output_type})
                cost = getattr(tool, 'cost', 0)  # Retrieve tool cost
                successors.append((new_state, tool_name, cost))
                
        return successors
    
    def _is_target_reached(self, state: FrozenSet[str], target_types: Set[str]) -> bool:
        """Check whether the target state has been reached."""
        return target_types.issubset(state)
    
    def _dijkstra_shortest_path(self, initial_state: FrozenSet[str], 
                               target_types: Set[str]) -> PathResult:
        """
        Find the shortest path using Dijkstra's algorithm.
        
        Args:
            initial_state: Starting state
            target_types: Types that must be included in the goal
            
        Returns:
            PathResult: Optimal tool sequence with cost information
            
        Raises:
            NoPathFoundError: Raised when no path can be found
        """
        # Priority queue: (cost, path_length, state, path, state_sequence)
        pq = [(0, 0, initial_state, [], [initial_state])]
        visited = set()
        
        while pq:
            current_cost, path_length, current_state, path, state_sequence = heapq.heappop(pq)
            
            # Skip already visited states
            if current_state in visited:
                continue
            visited.add(current_state)
            
            # Check whether the target has been reached
            if self._is_target_reached(current_state, target_types):
                return PathResult(
                    tools=path,
                    total_cost=current_cost,
                    path_length=path_length,
                    state_sequence=state_sequence
                )
            
            # Explore successor states
            for next_state, tool_name, tool_cost in self._get_successors(current_state):
                if next_state not in visited:
                    new_cost = current_cost + tool_cost
                    new_path_length = path_length + 1
                    new_path = path + [tool_name]
                    new_state_sequence = state_sequence + [next_state]
                    
                    # Tie-breaking: prioritize cost, then path length
                    heapq.heappush(pq, (new_cost, new_path_length, next_state, 
                                       new_path, new_state_sequence))
        
        # No path found
        raise NoPathFoundError(f"No path found from {initial_state} to target containing {target_types}")
    
    def solve(
        self, task: Optional[str] = None,
        current_state: Optional[Set[str]] = None,
        goal_types: Optional[Set[str]] = None,
        debug: bool = False
    ) -> PathResult:
        """Include a debug flag to output detailed information."""
        
        if task is not None:
            initial_state = self._infer_initial_state(task)
            target_types = self._infer_target_types(task)
            
            if debug:
                # print(f"Initial state: {initial_state}")
                # print(f"Target types: {target_types}")
                # print(f"Available tools: {len(self.tools)}")
                
                # Collect all location-related tools
                # print("\nLocation-related tools:")
                location_tools = []
                for name, tool in self.tools.items():
                    if 'Location' in name or 'location' in name.lower():
                        location_tools.append((name, tool))
                        # print(f"  {name}: {tool.input_types} -> {tool.output_type} (cost: {getattr(tool, 'cost', 'N/A')})")
                
                # Inspect specific tools that are known to be problematic
                problem_tools = ['Location_Preference_and_Search', 'Location_Planning_to_Step2', 'Location_Finalize_from_Step3']
                # print(f"\nProblem tools analysis:")
                for tool_name in problem_tools:
                    if tool_name in self.tools:
                        tool = self.tools[tool_name]
                        # print(f"  {tool_name}:")
                        # print(f"    input_types: {tool.input_types}")
                        # print(f"    output_type: {tool.output_type}")
                        # print(f"    cost: {getattr(tool, 'cost', 'N/A')}")
                        # print(f"    type: {getattr(tool, 'type', 'N/A')}")
                    else:
                        print(f"  {tool_name}: NOT FOUND")
                        
        elif current_state is not None and goal_types is not None:
            initial_state = frozenset(current_state)
            target_types = goal_types
            
            # if debug:
            #     print(f"Custom solve - Initial state: {initial_state}")
            #     print(f"Custom solve - Target types: {target_types}")
                
        else:
            raise ValueError("Must provide either 'task' or both 'current_state' and 'goal_types'")
            
        return self._dijkstra_shortest_path(initial_state, target_types)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Return solver statistics."""
        return {
            "total_tools": len(self.tools),
            "available_tool_names": list(self.tools.keys())
        }
        
def batch_update_groundtruth(
    queries: list,
    search_space_path: str
) -> list:
    """
    Batch update groundtruth information for all queries.
    
    Args:
        queries: List of queries
        search_space_path: Path to the search space file
        
    Returns:
        list: Updated queries
    """
    updated_queries = []
    
    for query in queries:
        # Update gt_id for scenario 0
        updated_query = update_query_groundtruth(query, search_space_path, scenario_index=0)
        updated_queries.append(updated_query)
    
    return updated_queries

def solve_single_query_gt(query_dict: Dict[str, Any], tools_dict: Dict[str, Tool]) -> Dict[str, Any]:
    """Compute groundtruth for a single query using in-memory tool objects.
    
    Args:
        query_dict: Query definition
        tools_dict: Dictionary of tool objects
        
    Returns:
        Dict[str, Any]: Updated query definition
    """
    solver = GroundtruthSolver(tools_with_costs=tools_dict)
    task = query_dict["task"]
    
    try:
        result = solver.solve(task=task)
        
        # Preserve the original groundtruth ID
        original_gt_id = query_dict.get("groundtruth", "")
        
        # Use the new groundtruth structure
        if "groundtruth" not in query_dict or isinstance(query_dict["groundtruth"], str):
            query_dict["groundtruth"] = {
                "original": original_gt_id,  # Retain the original groundtruth ID
                "scenarios": []
            }
        
        # Add scenario 0 (baseline without blocking)
        scenario_0 = {
            "scenario_index": 0,
            "gt_id": original_gt_id,  # Use the preserved original ID
            "tools": [
                {tool_name: getattr(solver.tools[tool_name], 'cost', None)}
                for tool_name in result.tools
            ],
            "total_cost": result.total_cost,
            "path_length": result.path_length
        }
        
        query_dict["groundtruth"]["scenarios"] = [scenario_0]
        
    except NoPathFoundError as e:
        print(f"Failed to solve query '{query_dict.get('query_id', 'unknown')}': {e}")
    
    return query_dict


def solve_from_json(query_path: str, tools_path: str, output_path: str) -> None:
    """Load queries and tools from JSON files and run the solver (kept for backward compatibility)."""
    with open(query_path, "r", encoding="utf-8") as f:
        query_data = json.load(f)
        
    with open(tools_path, "r", encoding="utf-8") as f:
        tools_data = json.load(f)["tools"]
    
    # Convert to Tool objects
    from env.core.base_types import create_tool_from_dict
    tools_dict = {tool_name: create_tool_from_dict(tool_dict) for tool_name, tool_dict in tools_data.items()}
    
    # Batch update groundtruth IDs for all queries
    from env.settings import load_config

    search_space_path = load_config().paths.search_space_path
    query_data = batch_update_groundtruth(query_data, search_space_path)
    
    # Compute groundtruth for each query
    for i, query_dict in enumerate(query_data):
        query_data[i] = solve_single_query_gt(query_dict, tools_dict)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(query_data, f, indent=4, ensure_ascii=False)
    # print(f"[INFO] Queries with initial groundtruth saved to {output_path}")

def calculate_groundtruth_id(
    task: str,
    preferences: Dict[str, str],
    search_space_path: str,
    location_preference_id: Optional[str] = None
) -> str:
    """
    Compute the groundtruth unique_id based on the task and preferences.
    
    Args:
        task: Task name, e.g., "location", "transportation"
        preferences: Preference dictionary, e.g., {"category": "city", "tier": "bustling_metropolis", ...}
        search_space_path: Path to the search space file
        location_preference_id: LocationPreference ID required for non-location tasks
        
    Returns:
        str: groundtruth unique_id, e.g., "<LocationPreference00001>"
        
    Raises:
        ValueError: If the corresponding ID cannot be found
    """
    from env.core.base_types import get_id_from_search_space
    return get_id_from_search_space(
        search_space_path=search_space_path,
        task=task,
        category=preferences.get("category"),
        tier=preferences.get("tier"),
        style=preferences.get("style"),
        feature_package=preferences.get("feature_package"),
        LocationPreferenceID=location_preference_id
    )


def update_query_groundtruth(
    query_dict: Dict[str, Any],
    search_space_path: str,
    scenario_index: int = 0
) -> Dict[str, Any]:
    """
    Update groundtruth information in the query dictionary by adding a gt_id.
    
    Args:
        query_dict: Query definition
        search_space_path: Path to the search space file
        scenario_index: Scenario index, defaults to 0
        
    Returns:
        Dict[str, Any]: Updated query definition
    """
    task = query_dict.get("task")
    preferences = query_dict.get("preferences", {})
    
    if not task:
        # print(f"[ERROR] No task found in query {query_dict.get('query_id', 'unknown')}")
        raise ValueError("[ERROR] Task is required in query_dict")
        return query_dict
    
    if not preferences:
        # print(f"[ERROR] No preferences found in query {query_dict.get('query_id', 'unknown')}")
        raise ValueError("[ERROR] Preferences are required in query_dict")
    
    try:
        # Calculate the groundtruth ID
        if task == "location":
            gt_id = calculate_groundtruth_id(
                task=task,
                preferences=preferences,
                search_space_path=search_space_path,
                location_preference_id=None
            )
        else:
            # Non-location tasks require a LocationPreference ID
            location_pref_id = query_dict.get("location_preference") 
            if not location_pref_id:
                # print(f"[ERROR] No LocationPreference ID found for {task} task in query {query_dict.get('query_id', 'unknown')}")
                raise ValueError("[ERROR] LocationPreference ID is required for non-location tasks")
                            
            gt_id = calculate_groundtruth_id(
                task=task,
                preferences=preferences,
                search_space_path=search_space_path,
                location_preference_id=location_pref_id
            )
        
        # Update the new groundtruth structure
        if "groundtruth" in query_dict and "scenarios" in query_dict["groundtruth"]:
            scenarios = query_dict["groundtruth"]["scenarios"]
            if scenario_index < len(scenarios):
                scenarios[scenario_index]["gt_id"] = gt_id
            
            # If scenario 0, also update the original field
            if scenario_index == 0:
                query_dict["groundtruth"]["original"] = gt_id
        
        # Maintain compatibility with the legacy structure
        if "gt" not in query_dict:
            query_dict["gt"] = {}
        
        gt_key = str(scenario_index)
        if gt_key not in query_dict["gt"]:
            query_dict["gt"][gt_key] = {}
        
        # Insert gt_id into the legacy structure
        query_dict["gt"][gt_key]["gt_id"] = gt_id
        
        # print(f"[INFO] Updated groundtruth ID for query {query_dict.get('query_id', 'unknown')}: {gt_id}")
        
    except Exception as e:
        print(f"[ERROR] Failed to calculate groundtruth ID for query {query_dict.get('query_id', 'unknown')}: {e}")
    
    return query_dict

import json

# Example usage
if __name__ == "__main__":
    print("=== Groundtruth Solver Example ===")
    solve_from_json(
        query_path="env/data/runtime/queries/queries.json",
        tools_path="env/data/runtime/tools/travel/tools_with_all_costs_refine_3.json",
        output_path="env/data/runtime/queries/queries_gt.json"
    )