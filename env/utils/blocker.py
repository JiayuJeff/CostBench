"""
Dynamic Blocker

Dynamic blocker – modifies the planning environment at runtime to test a
model's ability to replan.

Supports four blocking types:
1. ban_tool: disable specific tools
2. preference_change: change user requirements
3. steplen_change: adjust the visible tool-length range
4. cost_change: modify tool costs

Supports multiple trigger strategies and composite scenarios.
"""

import sys
import os
import json
import random
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import time
import uuid

project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)

from env.core.base_types import Tool
from env.utils.solver import GroundtruthSolver
from env.domains.travel.tool_registry import tools_ready, tools_ready_in_memory


class BlockType(Enum):
    """Enumeration of supported blocking types."""
    BAN_TOOL = "ban_tool"
    PREFERENCE_CHANGE = "preference_change"
    STEPLEN_CHANGE = "steplen_change"
    COST_CHANGE = "cost_change"


class BlockMode(Enum):
    """Enumeration of supported blocker modes."""
    MIXED = "mixed"
    BAN_TOOL = "ban_tool"
    PREFERENCE_CHANGE = "preference_change"
    STEPLEN_CHANGE = "steplen_change"
    COST_CHANGE = "cost_change"


@dataclass
class BlockingConfig:
    """Configuration for a single blocking event."""
    block_type: BlockType
    trigger_step: int
    parameters: Dict[str, Any]


@dataclass
class BlockingMetadata:
    """Metadata recorded for each blocking."""
    block_index: int
    block_type: str
    trigger_step: int
    parameters: Dict[str, Any]


class DynamicBlocker:
    """Main class implementing the dynamic blocker."""
    
    def __init__(
        self, 
        block_mode: str,
        # query_dict: Dict[str, Any],
        block_num: int, 
        seed_range: range,
        queries_path: str,
        refinement_level: int = 5,
        output_dir: str = "env/domains/travel/blocked_tools/"
    ):
        """
        Initialize the dynamic blocker.
        
        Args:
            block_mode: blocking mode ("mixed", "ban_tool", "preference_change", "steplen_change", "cost_change")
            block_num: planned number of blockings
            seed_range: seed range used for cost_change
            refinement_level: refinement level for tool generation
            output_dir: directory for writing tool outputs
        """
        self.block_mode = BlockMode(block_mode)
        self.block_num = block_num
        self.seed_range = seed_range
        self.refinement_level = refinement_level
        self.output_dir = output_dir
        self.queries_path = queries_path
        # self.query_dict = query_dict
        
        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Runtime state
        self.current_step = 0
        self.blocked_count = 0
        self.blocking_history = []
        self.current_state = set()  # Current data-type state (agent state)
        self.solver = None  # To be provided externally
        # Signal when a previously banned tool is reused (forces NED=1, EM=0 during eval)
        self.duplicate_banned_tool_reused = False
        
        # Independent GT-state tracking
        self.gt_state_history = []  # GT state history per scenario
        self.gt_path_history = []   # GT path history per scenario
        self.gt_current_state = None  # Current GT state
        self.gt_state_snapshots = {}  # GT state snapshot after each blocking {blocking_index: state}
        self.last_gt_step = 0  # Step at which GT state was last updated
        
        # Tool-related state
        self.original_tools = None
        self.current_tools = None
        # Maintain banned sets separately: agent sees the tool but call fails; GT solver removes it entirely
        self.agent_banned_tools = set()
        self.gt_banned_tools = set()
        self.removed_tools = set()  # Tool is completely removed (e.g. for steplen_change)
        
        # Pass-through options for filtering/cost control; default empty, configured externally via run.py
        self.filter_options = {}

        # steplen_change-related state
        self.current_tool_length_range = None
        self.next_tool_length_range = None
        self.determined_block_type = None

        # Blocking counters for the agent and GT timelines (kept in sync with self.blocked_count)
        self.agent_blocking_index = 0
        self.gt_blocking_index = 0
    
        
    def is_tool_banned(self, tool_name: str) -> bool:
        """Check whether a tool is banned (visible but call should fail)."""
        return tool_name in self.agent_banned_tools
    
    def is_tool_removed(self, tool_name: str) -> bool:
        """Check whether a tool has been removed (completely invisible)."""
        return tool_name in self.removed_tools

    def get_visible_tools(self, original_tools: List[Tool]) -> List[Tool]:
        """Return the tool list visible to the model."""
        visible_tools = []
        for tool in original_tools:
            # Only tools marked as removed become invisible
            if not self.is_tool_removed(tool.name):
                visible_tools.append(tool)
        return visible_tools
    
    def should_tool_call_fail(self, tool_name: str) -> bool:
        """Check whether a tool invocation should fail."""
        return self.is_tool_banned(tool_name)
    
    def set_solver(self, solver: GroundtruthSolver):
        """Attach the solver instance."""
        self.solver = solver
    
    def set_initial_tools(self, tools: Dict[str, Tool]):
        """Set the initial tool collection."""
        self.original_tools = {
            "tools": {name: tool.to_dict() for name, tool in tools.items()}
        }
        self.current_tools = self.original_tools.copy()
        
        # Cache all tool output types for GT state simulation
        self.cached_tool_output_types = {}
        for name, tool in tools.items():
            # AtomicTool exposes output_type, which we wrap in a single-item list
            if hasattr(tool, 'output_type'):
                self.cached_tool_output_types[name] = [tool.output_type]
            else:
                print(f"[WARNING] Tool {name} has no output_type attribute")
                self.cached_tool_output_types[name] = []
    
    def set_current_state(self, state: Set[str]):
        """Update the current agent state snapshot."""
        self.current_state = state.copy()
    
    def initialize_gt_state(self, task: str):
        """Initialise the GT state so it mirrors the agent state."""
        if self.solver and task:
            self.gt_current_state = self.solver._infer_initial_state(task)
        else:
            # print(f"[ERROR] Blocker.py line 178: initialize gt state failed.")
            self.gt_current_state = set()
        self.last_gt_step = 0
    
    def simulate_gt_tool_execution(self, tool_name: str) -> Set[str]:
        """Simulate a GT tool execution using cached tool metadata."""
        if not self.gt_current_state:
            return set()
        
        output_types = self.cached_tool_output_types.get(tool_name, [])
        # Convert frozenset to set so we can mutate it
        new_state = set(self.gt_current_state)
        new_state.update(output_types)
        
        return new_state
    
    def update_gt_state_at_blocking(self, blocking_index: int, task: str, query_dict: Dict[str, Any], block_type: BlockType):
        """Update the GT state at a blocking point.

        Steps:
        - Determine the previous scenario index `prev_scn_idx = blocking_index`
        - Compute the relative step `k`
        - Choose the baseline state from the previous blocking snapshot (or initial state)
        - Simulate tool outputs up to step `k-1`; include step `k` when the block is not ban_tool
        - Store the results in `gt_state_snapshots[blocking_index]`
        """
        # Ensure the GT initial state is ready
        if self.gt_current_state is None:
            # print(f"[ERROR] Query {query_dict.get('query_id', 'unknown')}, Blocker.py line 210 gt_current_state is None")
            self.initialize_gt_state(task)

        # Compute prev_scn_idx and relative step k
        prev_scn_idx, k = self._compute_prev_scn_and_k(query_dict, blocking_index)
        # print(f"[INFO] Query {query_dict.get('query_id', 'unknown')} get previous_scenario_index {prev_scn_idx} and relative_step {k}")
        
        # Select the baseline state
        if blocking_index - 1 in self.gt_state_snapshots:
            base_state = set(self.gt_state_snapshots[blocking_index - 1])
        elif blocking_index - 1 == -1:
            # First blocking: start from the initial state
            base_state = set(self.solver._infer_initial_state(task))
        else:
            # print(f"[ERROR] Query {query_dict.get('query_id', 'unknown')}, Blocker.py line 223 gt_current_state is None")
            base_state = set(self.gt_current_state) if self.gt_current_state is not None else set()
            # If still uninitialised, infer the initial state
            if not base_state:
                if self.solver and task:
                    base_state = set(self.solver._infer_initial_state(task))
                else:
                    base_state = set()

        # Decide whether to include step k in the simulation
        include_k = (block_type != BlockType.BAN_TOOL)
        # `upto` determines how many steps to simulate
        upto = k if include_k else (k - 1)
        # print(f"[INFO] Query {query_dict.get('query_id', 'unknown')} get upto {upto}")

        # Extract the tool sequence from scenario[prev_scn_idx] and simulate
        scenarios = query_dict.get("groundtruth", {}).get("scenarios", [])
        if prev_scn_idx >= len(scenarios) or "tools" not in scenarios[prev_scn_idx]:
            raise ValueError(f"[ERROR] Invalid scenario index {prev_scn_idx} or missing 'tools' when updating GT state at blocking {blocking_index}")
        tools_list = scenarios[prev_scn_idx]["tools"]
        
        # If the GT tool sequence is empty, mark gt_unblocked and exit safely (skip further progression)
        if not tools_list:
            try:
                query_dict["gt_unblocked"] = True
            except Exception:
                pass
            # Persist the baseline state snapshot to avoid downstream issues
            self.gt_state_snapshots[blocking_index] = set(base_state)
            self.gt_current_state = set(base_state)
            self.last_gt_step = query_dict.get("block_stats", {}).get("block_step", [])[blocking_index]
            return


        if upto >= 0:
            if upto > len(tools_list):
                # Clamp the range if it accidentally exceeds bounds (should not normally happen)
                # print(f"[ERROR] Query {query_dict.get('query_id', 'unknown')}, Blocker.py line 253 upto {upto} is out of range {len(tools_list)}")
                # print(f"[DEBUG] Blocking index {blocking_index}")
                # print(f"[DEBUG] Query {query_dict.get('query_id', 'unknown')} last scenarios tools_list {tools_list}")
                # print(f"[DEBUG] Query {query_dict.get('query_id', 'unknown')} upto {upto}")
                upto = len(tools_list) - 1
            # Simulate steps up to `upto`
            for step in range(0, upto):
                tool_dict = tools_list[step]
                tool_name = list(tool_dict.keys())[0]
                output_types = self.cached_tool_output_types.get(tool_name, [])
                base_state.update(output_types)
                # print(f"[INFO] Query {query_dict.get('query_id', 'unknown')} update base_state {output_types}")

        # Persist the snapshot and current GT state
        self.gt_state_snapshots[blocking_index] = set(base_state)
        self.gt_current_state = set(base_state)
        # last_gt_step is kept as informational metadata only
        self.last_gt_step = query_dict.get("block_stats", {}).get("block_step", [])[blocking_index]
        
        
    
    def _get_gt_tools_for_steps(self, prev_scn_idx: int, start_step: int, end_step: int, query_dict: Dict[str, Any]) -> List[str]:
        """Return GT tools for the interval [start, end) within the given scenario."""
        gt_tools = []
        scenarios = query_dict.get("groundtruth", {}).get("scenarios", [])
        if prev_scn_idx >= len(scenarios) or "tools" not in scenarios[prev_scn_idx]:
            return gt_tools
        tools_list = scenarios[prev_scn_idx]["tools"]
        for step in range(start_step, min(end_step, len(tools_list))):
            tool_dict = tools_list[step]
            tool_name = list(tool_dict.keys())[0]
            gt_tools.append(tool_name)
        return gt_tools

    def _compute_prev_scn_and_k(self, query_dict: Dict[str, Any], blocking_index: int) -> Tuple[int, int]:
        """Derive the previous scenario index and relative step k (0-based)."""
        block_steps = query_dict.get("block_stats", {}).get("block_step", [])
        if blocking_index >= len(block_steps) or block_steps[blocking_index] is None:
            raise ValueError(f"[ERROR] Invalid blocking_index {blocking_index} or missing trigger_step")
        i = blocking_index
        prev_scn_idx = i
        if i == 0:
            segment_start = 0
        else:
            if block_steps[i - 1] is None:
                raise ValueError(f"[ERROR] Missing previous trigger_step for blocking_index {i-1}")
            # segment_start = block_steps[i - 1] + 1
            segment_start = block_steps[i - 1]
        trigger_step = block_steps[i]
        k = trigger_step - segment_start
        if k < 0:
            raise ValueError(f"[ERROR] Computed negative k={k} for blocking_index {i}")
        return prev_scn_idx, k
    
    def initialize_blocking_plan(self, query_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Initialise the blocking plan (precomputed when running in static mode).
        
        Args:
            query_dict: query payload for the current task
            
        Returns:
            Dict[str, Any]: the updated query payload
        """
        # Obtain or compute the initial optimal path length
        optimal_path_length = None
        
        # Try to read it from the new groundtruth.scenarios structure
        if ("groundtruth" in query_dict and 
            "scenarios" in query_dict["groundtruth"] and 
            len(query_dict["groundtruth"]["scenarios"]) > 0 and
            "path_length" in query_dict["groundtruth"]["scenarios"][0]):
            optimal_path_length = query_dict["groundtruth"]["scenarios"][0]["path_length"]
        
        # Otherwise compute the initial groundtruth
        else:
            print(f"[WARNING] No initial groundtruth found, calculating it...")
            if not self.solver:
                raise ValueError("No initial groundtruth found and no solver available to calculate it")
            
            task = query_dict.get("task")
            initial_state = self.solver._infer_initial_state(task) if task else set()
            goal_types = {query_dict["goal_type"]}
            
            # Use all available tools to compute the initial optimal plan
            result = self.solver.solve(
                current_state=initial_state,
                goal_types=goal_types,
                debug=False
            )
            
            optimal_path_length = result.path_length
            
            # Initialise the new groundtruth structure if necessary
            if "groundtruth" not in query_dict:
                query_dict["groundtruth"] = {"original": "", "scenarios": []}
            
            # Append the initial scenario
            initial_scenario = {
                "scenario_index": 0,
                "gt_id": "",  # Calculated later
                "tools": [
                    {tool_name: getattr(self.solver.tools[tool_name], 'cost', None)}
                    for tool_name in result.tools
                ],
                "total_cost": result.total_cost,
                "path_length": result.path_length
            }
            
            query_dict["groundtruth"]["scenarios"].append(initial_scenario)
        
        if optimal_path_length is None:
            raise ValueError("Unable to determine initial path length")
        
        # Precompute blocking types and parameters
        query_dict = self._precompute_blocking_plan(query_dict)
        
        # Prepare the block_stats structure
        if "block_stats" not in query_dict:
            query_dict["block_stats"] = {}
        
        query_dict["block_stats"].update({
            "expected_block_count": self.block_num,
            "block_count": 0,
            "block_step": [None] * self.block_num,
            "block_type": [None] * self.block_num,
            "metadata": []
        })
        
        # Compute the trigger step for the first blocking
        self._calculate_next_blocking_trigger(
            query_dict=query_dict, 
            current_optimal_length=optimal_path_length,
            current_step=0,
            blocking_index=0,
        )

        return query_dict
    
    def _precompute_blocking_plan(self, query_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Precompute the type and parameter for every blocking.
        """
        # Define constants locally (avoid circular imports)
        from env.settings import load_config

        config = load_config()
        BLOCK_TYPES = list(config.blocker.block_types)
        RANDOM_SEED_INTERVAL = config.random.random_seed_interval
        MIN_TOOL_LENGTH = config.blocker.min_tool_length
        MAX_TOOLS_LENGTH = config.blocker.max_tools_length
        
        # Use tool_creation_seed as the base seed
        base_seed = query_dict.get("tool_creation_seed", 42)
        task = query_dict.get("task", "location")
        query_id = query_dict.get("query_id", "Unknown")
        
        
        # Generate the blocking-type sequence
        blocking_types = self._generate_blocking_types(base_seed, BLOCK_TYPES)
        
        # Precompute parameter payload for each blocking
        blocking_plan = []
        
        # Pre-generate ranges for steplen_change
        steplen_ranges = None
        steplen_change_count = blocking_types.count("steplen_change")
        if steplen_change_count > 0:
            if self.filter_options.get("control_tool_length", False):
                steplen_ranges = self._generate_multiple_steplen_ranges(
                    base_seed, steplen_change_count, 2, self.filter_options.get("max_tool_length", 8)
                )
            else:
                steplen_ranges = self._generate_multiple_steplen_ranges(
                    base_seed, steplen_change_count, 2, MIN_TOOL_LENGTH + self.refinement_level
                )
        
        steplen_index = 0
        for block_index, block_type in enumerate(blocking_types):
            blocking_seed = base_seed + block_index * RANDOM_SEED_INTERVAL
            
            if block_type == "cost_change":
                parameters = self._generate_cost_change_parameters(blocking_seed)
            elif block_type == "preference_change":
                parameters = self._generate_preference_change_parameters(blocking_seed, task, query_dict)
            elif block_type == "steplen_change":
                parameters = {
                    "target_range": steplen_ranges[steplen_index] if steplen_ranges else (MIN_TOOL_LENGTH + self.refinement_level)
                }
                steplen_index += 1
            elif block_type == "ban_tool":
                parameters = {"return_message_seed": blocking_seed}
            else:
                parameters = {}
            
            blocking_plan.append({
                "type": block_type,
                "parameters": parameters
            })
        
        query_dict["blocking_plan"] = blocking_plan
        return query_dict
    
    def _generate_blocking_types(self, base_seed: int, block_types: List[str]) -> List[str]:
        """
        Produce the blocking-type sequence.
        """
        TOTAL_BLOCK_TYPE = len(block_types)
        
        blocking_types = []
        
        if self.block_mode == BlockMode.MIXED:
            # Mixed mode: pick a random type for every blocking
            rng = random.Random(base_seed)
            for i in range(self.block_num):
                type_index = rng.randint(0, TOTAL_BLOCK_TYPE - 1)
                block_type = block_types[type_index]
                blocking_types.append(block_type)
        else:
            # Fixed mode: all blockings share the same type
            blocking_types = [self.block_mode.value] * self.block_num
            
        return blocking_types
    
    def _generate_multiple_steplen_ranges(
        self, 
        base_seed: int, 
        count: int, 
        min_val: int, 
        max_val: int # Upper bound for the length of the atomic tool path to the destination
    ) -> List[Tuple[int, int]]:
        """
        Generate multiple non-overlapping steplen ranges.
        """
        
        if count == 0:
            return []
            
        rng = random.Random(base_seed)
        ranges = []
        
        # Constraint: every interval must satisfy (end - start) >= (max_val // 2)
        min_required_width = max_val // 2
        total_width = max_val - min_val
        if total_width < min_required_width:
            # Relax the constraint to the widest feasible range if necessary and log a warning
            print(f"[WARNING] blocker.py line 495 Width constraint {min_required_width} infeasible for range [{min_val}, {max_val}], relaxing to {total_width}")
            min_required_width = max(0, total_width)
        
        # Compute the available integer range
        available_numbers = list(range(min_val, max_val + 1))
        
        if len(available_numbers) < count * 2:
            # If we cannot allocate enough unique ranges, allow reuse
            print(f"[WARNING] Not enough numbers to create {count} non-overlapping ranges")
            for i in range(count):
                # Always satisfy the minimum width constraint
                start = rng.randint(min_val, max_val - min_required_width)
                end = rng.randint(start + min_required_width, max_val)
                ranges.append((start, end))
            return ranges
        
        # Attempt to generate non-overlapping ranges
        used_numbers = set()
        for i in range(count):
            attempts = 0
            while attempts < 100:  # Avoid infinite loops
                # Satisfy the minimum width constraint
                start = rng.randint(min_val, max_val - min_required_width)
                end = rng.randint(start + min_required_width, max_val)
                
                # Reject ranges that overlap with a previously selected range
                range_numbers = set(range(start, end + 1))
                if not range_numbers.intersection(used_numbers):
                    ranges.append((start, end))
                    used_numbers.update(range_numbers)
                    break
                attempts += 1
            
            if attempts >= 100:
                # Allow overlap as a fallback if no disjoint range can be found
                start = rng.randint(min_val, max_val - min_required_width)
                end = rng.randint(start + min_required_width, max_val)
                ranges.append((start, end))
        
        return ranges
    
    def _generate_cost_change_parameters(self, blocking_seed: int) -> Dict[str, Any]:
        """
        Prepare parameters for a cost_change blocking.
        """
        rng = random.Random(blocking_seed)
        new_seed = rng.choice(list(self.seed_range))
        params = {"new_random_seed": new_seed}
        # Forward filter/cost control options provided via run.py
        try:
            opts = getattr(self, "filter_options", {}) or {}
            # Copy values for whitelisted keys only
            for k in [
                "control_tool_length",
                "max_tool_length",
                "ban_longest_tool",
                "min_atomic_cost",
                "max_atomic_cost",
                "noise_std",
            ]:
                if k in opts:
                    params[k] = opts[k]
        except Exception:
            pass
        return params
    
    def _generate_preference_change_parameters(
        self, 
        blocking_seed: int, 
        task: str, 
        query_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Produce parameters for a preference_change blocking.
        """
        try:
            # Read the queries file
            with open(self.queries_path, "r", encoding="utf-8") as f:
                all_queries = json.load(f)
            
            # Filter queries for the same task
            same_task_queries = [q for q in all_queries if q.get("task") == task]
            
            if not same_task_queries:
                # print(f"[ERROR] No queries found for task '{task}', using default preference")
                return self._get_default_preference(task)
            
            # Use the blocking_seed to select a different query (excluding the current one)
            rng = random.Random(blocking_seed)
            current_qid = query_dict.get("query_id")
            candidate_pool = [q for q in same_task_queries if q.get("query_id") != current_qid]
            if not candidate_pool:
                # Fallback to the full pool if excluding self empties the list
                # print(f"[ERROR] Query {query_dict.get('query_id', 'unknown')}, Blocker.py line 550 candidate_pool is empty")
                candidate_pool = same_task_queries
            selected_query = rng.choice(candidate_pool)
            
            # Extract user requirements and preferences
            if task == "location":
                return {
                    "user_requirement": selected_query.get("user_requirements", ""),
                    "preferences": selected_query.get("preferences", {}),
                    "groundtruth": selected_query.get("groundtruth", "")
                }
            else:
                return {
                    "user_requirement": selected_query.get("user_requirements", ""),
                    "preferences": selected_query.get("preferences", {}),
                    "groundtruth": selected_query.get("groundtruth", ""),
                    "location": selected_query.get("location_preference", "")
                }
            
        except Exception as e:
            # print(f"[ERROR] Failed to generate preference_change parameters: {str(e)}")
            return self._get_default_preference(task)
    
    def _calculate_next_blocking_trigger(
        self, 
        query_dict: Dict[str, Any], 
        current_optimal_length: int, 
        current_step: int,
        blocking_index: int,
    ):
        """
        Calculate the timing and type for the next blocking.
        """
        query_id = query_dict.get("query_id", "Unknown")
        
        # Look up the precomputed block type
        if "blocking_plan" not in query_dict:
            raise ValueError("Blocking plan not found in query_dict")
        
        blocking_plan = query_dict["blocking_plan"]
        
        if blocking_index >= len(blocking_plan):
            # print(f"[ERROR] Blocking index {blocking_index} exceeds plan length {len(blocking_plan)}")
            return
            
        # Read the blocking type from the plan
        block_type_str = blocking_plan[blocking_index]["type"]
        block_type = BlockType(block_type_str)
        
        # Calculate the trigger step
        if current_optimal_length == 1:
            # Single-step plan: trigger immediately
            trigger_step = current_step
        else:
            # Multi-step: trigger near the midpoint of the remaining path
            remaining_blocks = self.block_num - blocking_index
            relative_trigger = current_optimal_length // (remaining_blocks + 1)
            trigger_step = current_step + relative_trigger
        
        # Ensure we move forward at least one step
        if trigger_step < current_step + 1:
            trigger_step = current_step + 1
        
        # Persist the trigger information
        query_dict["block_stats"]["block_step"][blocking_index] = trigger_step
        query_dict["block_stats"]["block_type"][blocking_index] = block_type.value
        
        # Reuse the precomputed parameters
        parameters = blocking_plan[blocking_index]["parameters"]

        # Build metadata and append it to the log
        metadata = BlockingMetadata(
            block_index=blocking_index,
            block_type=block_type.value,
            trigger_step=trigger_step,
            parameters=parameters
        )
        
        query_dict["block_stats"]["metadata"].append(metadata.__dict__)
            
    def _generate_blocking_parameters(
        self, 
        query_dict: Dict[str, Any], 
        block_type: BlockType, 
        blocking_index: int,
    ) -> Dict[str, Any]:
        """
        Legacy helper for generating blocking parameters (deprecated and kept for compatibility).
        """
        print(f"[WARNING] _generate_blocking_parameters is deprecated, using precomputed parameters")
        return {}
    


    def _get_default_preference(self, task: str) -> Dict[str, Any]:
        """Return a default preference payload when loading from disk fails."""
        # print(f"[ERROR] Blocker.py line 646 _get_default_preference is called")
        default_preferences = {
            "location": {
                "user_requirement": "I want to change my travel preferences to explore a different type of destination...",
                "preferences": {
                    "category": "mountain",
                    "tier": "scenic_peak",
                    "style": "relaxation",
                    "feature_package": "nature_lover"
                },
                "groundtruth": "<LocationPreference00999>"
            }
        }
        
        return default_preferences.get(task, {
            "user_requirement": "I want to change my preferences...",
            "preferences": {},
            "groundtruth": ""
        })
    

    
    def should_trigger_blocking(self, current_step: int) -> bool:
        """
        Check whether a blocking should be triggered.
        
        Args:
            current_step: current execution step
            
        Returns:
            bool: True if a blocking should fire
        """
        # Abort if all planned blockings have already fired
        if self.blocked_count >= self.block_num:
            return False
        
        # Placeholder: requires callers to pass in query_dict with more information
        return False
    
    def should_trigger_blocking_with_query(
        self, 
        current_step: int, 
        query_dict: Dict[str, Any]
    ) -> bool:
        """
        Check whether a blocking should be triggered using query metadata.
        
        Args:
            current_step: current execution step
            query_dict: query payload
            
        Returns:
            bool: True if the blocking should trigger now
        """
        # Abort if all planned blockings have already fired
        if self.blocked_count >= self.block_num:
            return False
        
        # Inspect the next trigger step
        block_steps = query_dict["block_stats"]["block_step"]
        if self.blocked_count < len(block_steps):
            next_trigger_step = block_steps[self.blocked_count]
            return current_step >= next_trigger_step
        
        return False
    
    def execute_blocking(
        self,
        query_dict: Dict[str, Any],
        tool_call_request: Optional[Dict[str, Any]] = None,
        gt_only: bool = False,
    ) -> Dict[str, Any]:
        """Execute the next blocking operation."""
        
        if self.blocked_count >= self.block_num:
            return {"type": "no_blocking", "message": "No more blockings planned"}
        
        # Ensure the dual-timeline counters remain aligned
        if self.agent_blocking_index != self.gt_blocking_index:
            # raise ValueError(f"[ERROR] Mismatched blocking indices: agent={self.agent_blocking_index}, gt={self.gt_blocking_index}")
            pass
        current_index = self.agent_blocking_index

        # Retrieve metadata for the pending blocking
        if len(query_dict["block_stats"]["metadata"]) <= current_index:
            # print(f"[ERROR] Query {query_dict.get('query_id', 'unknown')}, block_stats metadata length {len(query_dict['block_stats']['metadata'])} is less than blocked_count {self.blocked_count}")
            return {"type": "no_blocking", "message": "No more blockings planned"}
        metadata = query_dict["block_stats"]["metadata"][current_index]
        block_type = BlockType(metadata["block_type"])
        parameters = metadata["parameters"]
        
        result = {"type": block_type.value, "parameters": parameters}
        
        if block_type == BlockType.BAN_TOOL:
            # GT-only catch-up: skip agent tool requests and state updates
            if gt_only:
                # Mark the GT-side banned tool only
                prev_scn_idx, k = self._compute_prev_scn_and_k(query_dict, current_index)
                scenarios = query_dict.get("groundtruth", {}).get("scenarios", [])
                if prev_scn_idx >= len(scenarios) or "tools" not in scenarios[prev_scn_idx]:
                    raise ValueError(f"[ERROR] Invalid scenario index {prev_scn_idx} for GT ban at blocking {current_index}")
                tools_list = scenarios[prev_scn_idx]["tools"]
                if k < 0 or k >= len(tools_list):
                    raise ValueError(f"[ERROR] Invalid relative step k={k} for scenario {prev_scn_idx}")
                target_index = 0 if k == 0 else (k - 1)
                gt_tool_dict = tools_list[target_index]
                gt_tool_name = list(gt_tool_dict.keys())[0]
                self.gt_banned_tools.add(gt_tool_name)
                result.update({
                    "status": "tool_banned",
                    "banned_tool_gt": gt_tool_name,
                    "message": "GT-only ban applied"
                })
            else:
                # 1) Agent ban: tool stays visible but calls fail
                if not tool_call_request or not tool_call_request.get("tool_name"):
                    # print("[ERROR] No tool call request provided for ban_tool")
                    return {"type": block_type.value, "status": "error", "message": "No tool call request provided"}
                agent_tool_name = tool_call_request.get("tool_name")
                # Record when a previously banned tool is reused (for evaluation metrics)
                try:
                    if self.block_num > 1 and agent_tool_name in self.agent_banned_tools:
                        self.duplicate_banned_tool_reused = True
                        # Persist a signal in query_dict for downstream evaluation
                        query_dict["duplicate_banned_tool_reused"] = True
                    
                except Exception:
                    pass
                self.agent_banned_tools.add(agent_tool_name)

                # 2) GT ban: read the tool used at step k in the previous scenario
                prev_scn_idx, k = self._compute_prev_scn_and_k(query_dict, current_index)
                scenarios = query_dict.get("groundtruth", {}).get("scenarios", [])
                if prev_scn_idx >= len(scenarios) or "tools" not in scenarios[prev_scn_idx]:
                    raise ValueError(f"[ERROR] Invalid scenario index {prev_scn_idx} for GT ban at blocking {current_index}")
                tools_list = scenarios[prev_scn_idx]["tools"]
                # Allow k == 0 (pre-step trigger) by using tools_list[0]
                if k < 0 or k >= len(tools_list):
                    raise ValueError(f"[ERROR] Invalid relative step k={k} for scenario {prev_scn_idx}")
                target_index = 0 if k == 0 else (k - 1)
                gt_tool_dict = tools_list[target_index]
                gt_tool_name = list(gt_tool_dict.keys())[0]
                self.gt_banned_tools.add(gt_tool_name)
                # print(f"[INFO] Query {query_dict.get('query_id', 'unknown')} gt_banned_tools {self.gt_banned_tools}")

                result.update({
                    "status": "tool_banned",
                    "banned_tool_agent": agent_tool_name,
                    "banned_tool_gt": gt_tool_name,
                    "message": "Tool temporarily unavailable"
                })
            
        elif block_type == BlockType.PREFERENCE_CHANGE:
            result.update(self._execute_preference_change(query_dict, parameters))
            
        elif block_type == BlockType.STEPLEN_CHANGE:
            result.update(self._execute_steplen_change(parameters))
            
        elif block_type == BlockType.COST_CHANGE:
            result.update(self._execute_cost_change(parameters))
        
        # Update the GT state for this blocking using the prev_scn_idx/k rule
        task = query_dict.get("task", "")
        self.update_gt_state_at_blocking(current_index, task, query_dict, block_type)

        # Update counters (dual timeline plus legacy fields)
        self.agent_blocking_index += 1
        self.gt_blocking_index += 1
        self.blocked_count = self.agent_blocking_index
        query_dict["block_stats"]["block_count"] = self.blocked_count
                
        # Recalculate groundtruth after each blocking because the state changed
        if self.solver:
            self._recalculate_groundtruth_and_next_blocking(query_dict)
        return result

    def finalize_gt(self, query_dict: Dict[str, Any]) -> None:
        """Advance GT-only blockings when the agent stops early.

        Halt immediately if a true gt_unblocked flag is encountered.
        """
        try:
            while (self.blocked_count < self.block_num) and (not bool(query_dict.get("gt_unblocked", False))):
                _ = self.execute_blocking(query_dict=query_dict, tool_call_request=None, gt_only=True)
                # The loop progressively regenerates metadata/trigger via _recalculate_groundtruth_and_next_blocking
        except Exception as _e:
            pass
            # print(f"[ERROR] finalize_gt failed: {str(_e)}")

    
    
    def _execute_ban_tool(self, tool_call_request: Dict[str, Any]) -> Dict[str, Any]:
        """Legacy placeholder; ban_tool is handled inline within execute_blocking."""
        if tool_call_request and tool_call_request.get("tool_name"):
            return {"status": "tool_banned", "banned_tool": tool_call_request.get("tool_name")}
        return {"status": "error", "message": "Invalid tool call request"}
    
    def _execute_preference_change(
        self, 
        query_dict: Dict[str, Any], 
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform a preference_change blocking."""
        # Update user requirements
        new_requirement = parameters.get("user_requirement", "")
        new_preferences = parameters.get("preferences", {})
        
        # For non-location tasks we may also override the location preference
        task = query_dict.get("task")
        result = {
            "status": "preference_changed",
            "new_user_requirement": new_requirement,
            "new_preferences": new_preferences,
            "action_required": "insert_user_message"
        }
        
        # Update location preference when provided for non-location tasks
        if task != "location" and "location" in parameters:
            new_location_preference = parameters.get("location", "")
            if new_location_preference:
                # Update location_preference within query_dict directly
                query_dict["location_preference"] = new_location_preference
                result["new_location_preference"] = new_location_preference
        
        # Reset the current state to its initial configuration because preferences changed
        if task and self.solver:
            initial_state = self.solver._infer_initial_state(task)
            self.current_state = set(initial_state)
            # Reset GT initial state as well to avoid lazy initialisation logs later
            try:
                self.initialize_gt_state(task)
            except Exception:
                pass
                # print(f"[ERROR] Query {query_dict.get('query_id', 'unknown')}, Blocker.py line 831 initialize gt state failed.")
        
        # The caller should inject the corresponding user message into the dialogue history
        return result
    
    def _execute_steplen_change(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform a steplen_change blocking."""
        # Read the new tool-length range (using the new parameter name)
        new_range = parameters.get("target_range", (2, 5))
        self.current_tool_length_range = new_range
        
        # Override strategy: recompute with the newest range, clearing prior removals
        self.removed_tools = set()
        
        # Decide which tools to remove – only composite tools, keep atomic ones
        # New rule: remove only tools whose length equals the left boundary
        if self.original_tools:
            min_len, max_len = new_range
            target_length = min_len  # Remove only tools that match the lower bound
            
            for tool_name, tool_data in self.original_tools["tools"].items():
                # Determine tool length
                if tool_data.get("type") == "atomic":
                    # Never remove atomic tools because composites depend on them
                    continue
                elif tool_data.get("type") == "composite":
                    tool_length = tool_data.get("component_count", 1)
                else:
                    tool_length = 1
                
                # Only remove composite tools whose length equals target_length
                if tool_length == target_length:
                    self.removed_tools.add(tool_name)
        
        return {
            "status": "steplen_changed",
            "new_range": new_range,
            "target_length": target_length,
            "removed_tool_count": len(self.removed_tools),
            "action_required": "update_tool_list"
        }
    
    def _execute_cost_change(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform a cost_change blocking."""
        new_seed = parameters.get("new_random_seed")
        
        try:
            # Generate tools in-memory to avoid disk IO
            new_tools = tools_ready_in_memory(
                refinement_level=self.refinement_level,
                min_atomic_cost=parameters.get("min_atomic_cost", 19),
                max_atomic_cost=parameters.get("max_atomic_cost", 21),
                noise_std=parameters.get("noise_std", 0.1),
                random_seed=new_seed,
                control_tool_length=parameters.get("control_tool_length", False),
                max_tool_length=parameters.get("max_tool_length", 8),
                ban_longest_tool=parameters.get("ban_longest_tool", False)
            )
            # Replace current_tools with the newly generated set
            self.current_tools = new_tools
            return {
                "status": "cost_changed",
                "new_seed": new_seed,
                "action_required": "update_tool_costs"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to generate new tools: {str(e)}"
            }
    
    def _filter_tools_by_length(self, length_range: Tuple[int, int]) -> Dict[str, Any]:
        """Filter tools whose length falls within the specified range."""
        if not self.original_tools:
            return {"tools": {}}
        
        min_len, max_len = length_range
        filtered_tools = {"tools": {}}
        
        for tool_name, tool_data in self.original_tools["tools"].items():
            # Determine tool length
            if tool_data.get("type") == "atomic":
                tool_length = 1
            elif tool_data.get("type") == "composite":
                tool_length = tool_data.get("component_count", 1)
            else:
                tool_length = 1
            
            # Keep tools whose length resides within the specified range
            if min_len <= tool_length <= max_len:
                filtered_tools["tools"][tool_name] = tool_data
        
        # Preserve metadata if available
        if "metadata" in self.original_tools:
            filtered_tools["metadata"] = self.original_tools["metadata"].copy()
        
        return filtered_tools
    
    def _get_available_tools_for_solver(self) -> Dict[str, Tool]:
        """
        Return the set of tools that the solver can use as Tool objects.
        
        Key logic:
        - gt_banned_tools: exclude from GT solving because they are known unavailable
        - removed_tools: fully invisible, also excluded from groundtruth calculation
        """
        # Prefer current_tools (if cost_change ran); otherwise fall back to original_tools
        tools_source = self.current_tools if self.current_tools else self.original_tools
        
        if not tools_source:
            # print("[ERROR] No tools available")
            return {}
        
        from env.core.base_types import create_tool_from_dict
        available_tools = {}
        tools_dict = tools_source.get("tools", {})
        
        for tool_name, tool_data in tools_dict.items():
            # Exclude all tools known to be unavailable when computing groundtruth (gt_banned + removed)
            if tool_name not in self.gt_banned_tools and tool_name not in self.removed_tools:
                # Convert dictionaries back into Tool objects
                tool_data_copy = tool_data.copy()
                tool_data_copy["name"] = tool_name  # Ensure the name field is present
                available_tools[tool_name] = create_tool_from_dict(tool_data_copy)
                    
        return available_tools
    
    def _recalculate_groundtruth_and_next_blocking(self, query_dict: Dict[str, Any]):
        """Recompute the groundtruth plan and schedule the next blocking."""
        def _log(level: str, message: str) -> None:
            try:
                import inspect
                caller = inspect.currentframe().f_back
                line = caller.f_lineno if caller else -1
            except Exception:
                line = -1
            qid = query_dict.get("query_id", "unknown")
            print(f"[{level}] Query {qid}, Blocker.py line {line}: {message}")

        if not self.solver:
            _log("ERROR", "No solver available for recalculating groundtruth")
            return
        
        # Skip recalculation if GT has been flagged as having no remaining steps
        if query_dict.get("gt_unblocked", False):
            # try:
            #     # _log("ERROR", f"Recalc skipped: gt_unblocked=True, blocked_count={self.blocked_count}")
            # except Exception:
            #     pass
            return
        
        try:
            last_stage = "start"
            # Step 1: build a solver that excludes banned tools
            available_tools = self._get_available_tools_for_solver()
            
            # Gather tool names for debugging
            atomic_tools = [name for name, tool in available_tools.items() if tool.get_tool_type() == 'atomic']
            composite_tools = [name for name, tool in available_tools.items() if tool.get_tool_type() == 'composite']
            last_stage = "after_get_available_tools"
            try:
                # _log("DEBUG", f"Available tools -> atomic: {len(atomic_tools)}, composite: {len(composite_tools)}")
                if not available_tools:
                    _log("ERROR", f"No available tools for temp_solver at blocked_count={self.blocked_count}")
            except Exception:
                pass
            
            # Instantiate a temporary solver for recalculation
            temp_solver = GroundtruthSolver(tools_with_costs=available_tools)
            last_stage = "after_temp_solver_init"
            
            # Recompute the optimal path from the current state
            goal_types = {query_dict["goal_type"]}
            task = query_dict.get("task")
            last_stage = "after_read_goal_task"
            
            # Create a new scenario; start by determining scenario_index
            scenario_index = self.blocked_count
            try:
                # _log("DEBUG", f"Recalc begin -> scenario_index={scenario_index}, blocked_count={self.blocked_count}")
                pass
            except Exception:
                pass
            
            # Choose the starting state based on the most recent blocking type
            if (scenario_index > 0 and 
                "block_stats" in query_dict and 
                "metadata" in query_dict["block_stats"] and
                len(query_dict["block_stats"]["metadata"]) >= scenario_index):
                
                # Retrieve metadata for the most recently executed blocking (scenario_index - 1)
                current_metadata = query_dict["block_stats"]["metadata"][scenario_index - 1]
                current_block_type = current_metadata.get("block_type")
                
                if current_block_type == "preference_change":
                    # preference_change: recompute from the initial state because preferences changed
                    solve_state = self.solver._infer_initial_state(task) if self.solver else set()
                    last_stage = "solve_state_from_initial_due_to_preference_change"
                    
                else:
                    # ban_tool / steplen_change / cost_change: resume from the GT snapshot captured earlier
                    solve_state = self._get_gt_state_at_blocking(scenario_index - 1, task, query_dict)
                    last_stage = "solve_state_from_gt_snapshot"
                    
                    
            else:
                # Scenario 0: always start from the initial state
                solve_state = self.solver._infer_initial_state(task) if self.solver else set()
                last_stage = "solve_state_from_initial"
            
            try:
                # _log("DEBUG", f"Solve state prepared -> size={len(solve_state)}; goal_types={goal_types}")
                pass
            except Exception:
                pass
            last_stage = "before_solve"
            result = temp_solver.solve(
                current_state=solve_state,
                goal_types=goal_types,
                debug=True  # Enable solver debug output
            )
            last_stage = "after_solve"
            try:
                # _log("DEBUG", f"Solve result -> path_length={result.path_length}, total_cost={result.total_cost}, tools_count={len(result.tools) if hasattr(result, 'tools') else 'NA'}")
                pass
            except Exception:
                pass
            new_scenario = {
                "scenario_index": scenario_index,
                "gt_id": "",  # Calculated later
                "tools": [
                    {tool_name: getattr(temp_solver.tools[tool_name], 'cost', None)}
                    for tool_name in result.tools
                ],
                "total_cost": result.total_cost,
                "path_length": result.path_length
            }
            last_stage = "build_new_scenario"
            try:
                # _log("DEBUG", f"New scenario built -> index={scenario_index}, tools={len(new_scenario['tools'])}")
                pass
            except Exception:
                pass
            
            # Attach block_info when available
            if scenario_index > 0:
                metadata = query_dict["block_stats"]["metadata"][scenario_index - 1]
                new_scenario["block_info"] = {
                    "type": metadata["block_type"],
                    "trigger_step": metadata["trigger_step"],
                    "parameters": metadata["parameters"]
                }
            last_stage = "attach_block_info_if_any"
            
            # Insert the rebuilt scenario into the scenarios array
            if "groundtruth" not in query_dict:
                query_dict["groundtruth"] = {"original": "", "scenarios": []}
            
            scenarios = query_dict["groundtruth"]["scenarios"]
            
            # Ensure the scenarios array is large enough
            while len(scenarios) <= scenario_index:
                scenarios.append({})
                
            scenarios[scenario_index] = new_scenario
            last_stage = "append_scenario"
            
            # Compute and assign a new gt_id
            # Blocked scenarios require the latest preferences when calculating gt_id
            from env.utils.solver import update_query_groundtruth, calculate_groundtruth_id
            search_space_path = "env/data/static/search_spaces/test_search_space.json"
            
            # If the latest blocking was a preference_change, use the updated preferences/location
            current_preferences = query_dict.get("preferences", {})
            current_location_pref_id = query_dict.get("location_preference")
            last_stage = "prepare_gt_id_inputs"
            
            if (scenario_index > 0 and 
                "block_stats" in query_dict and 
                "metadata" in query_dict["block_stats"] and
                len(query_dict["block_stats"]["metadata"]) >= scenario_index):
                
                # Inspect the metadata for the most recent blocking
                executed_blocking_metadata = query_dict["block_stats"]["metadata"][scenario_index - 1]
                if executed_blocking_metadata.get("block_type") == "preference_change":
                    # Update preferences from the blocking parameters
                    if "preferences" in executed_blocking_metadata.get("parameters", {}):
                        current_preferences = executed_blocking_metadata["parameters"]["preferences"]
                    # For non-location tasks, update the location_preference_id as well
                    if task != "location" and "location" in executed_blocking_metadata.get("parameters", {}):
                        new_location_pref = executed_blocking_metadata["parameters"]["location"]
                        if new_location_pref:
                            current_location_pref_id = new_location_pref
            
            # Calculate gt_id
            task = query_dict.get("task")
            if task == "location":
                gt_id = calculate_groundtruth_id(
                    task=task,
                    preferences=current_preferences,
                    search_space_path=search_space_path,
                    location_preference_id=None
                )
            else:
                gt_id = calculate_groundtruth_id(
                    task=task,
                    preferences=current_preferences,
                    search_space_path=search_space_path,
                    location_preference_id=current_location_pref_id
                )
            last_stage = "after_calculate_gt_id"
            try:
                # _log("DEBUG", f"GT ID calculated -> {gt_id if isinstance(gt_id, str) else type(gt_id)}")
                pass
            except Exception:
                pass
                
            
            # Record the computed gt_id in the scenario
            scenarios[scenario_index]["gt_id"] = gt_id
            last_stage = "assign_gt_id"
            
            # Schedule the next blocking trigger
            if self.blocked_count < self.block_num:
                try:
                    # _log("DEBUG", f"Calculating next trigger -> current_optimal_length={result.path_length}, current_step={self.current_step}, next_index={self.blocked_count}")
                    pass
                except Exception:
                    pass
                self._calculate_next_blocking_trigger(
                    query_dict=query_dict,
                    current_optimal_length=result.path_length,
                    current_step=self.current_step,
                    blocking_index=self.blocked_count,
                )
                        
        except Exception as e:
            try:
                _log("ERROR", f"Failed to recalculate groundtruth at stage={last_stage}: {str(e)}")
            except Exception:
                _log("ERROR", f"Failed to recalculate groundtruth (stage unknown): {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _get_gt_state_at_blocking(self, blocking_index: int, task: str, query_dict: Dict[str, Any]) -> set:
        """
        Retrieve the GT state snapshot at the specified blocking index.
        
        Args:
            blocking_index: zero-based blocking index
            task: task identifier
            query_dict: query payload containing groundtruth scenarios
            
        Returns:
            set: the GT state expected at that blocking point
        """
        # Return the cached snapshot when available
        if blocking_index in self.gt_state_snapshots:
            return set(self.gt_state_snapshots[blocking_index])
        elif blocking_index == 0:
            # First blocking: fallback to the initial state
            return set(self.solver._infer_initial_state(task))
        
        # Fallback: rebuild from metadata (should rarely happen)
        # print(f"[ERROR] Query {query_dict.get('query_id', 'unknown')}, Blocker.py line 1096 gt_state_snapshots {self.gt_state_snapshots} is not in {blocking_index}")
        if self.solver and task:
            base_state = set(self.solver._infer_initial_state(task))
        else:
            base_state = set()

        try:
            prev_scn_idx, k = self._compute_prev_scn_and_k(query_dict, blocking_index)
            # Use metadata to decide whether to include step k
            block_type_str = query_dict.get("block_stats", {}).get("metadata", [])[blocking_index].get("block_type")
            include_k = (block_type_str != BlockType.BAN_TOOL.value)
            upto = k if include_k else (k - 1)
            scenarios = query_dict.get("groundtruth", {}).get("scenarios", [])
            if prev_scn_idx < len(scenarios) and "tools" in scenarios[prev_scn_idx] and upto >= 0:
                tools_list = scenarios[prev_scn_idx]["tools"]
                upto = min(upto, len(tools_list) - 1)
                for step in range(0, upto + 1):
                    tool_dict = tools_list[step]
                    tool_name = list(tool_dict.keys())[0]
                    base_state.update(self.cached_tool_output_types.get(tool_name, []))
        except Exception:
            pass

        return set(base_state)
    
    def _get_blocking_step(self, blocking_index: int, query_dict: Dict[str, Any]) -> int:
        """Return the trigger step for the given blocking index."""
        if ("block_stats" in query_dict and 
            "metadata" in query_dict["block_stats"] and 
            blocking_index < len(query_dict["block_stats"]["metadata"])):
            
            return query_dict["block_stats"]["metadata"][blocking_index].get("trigger_step", 0)
        return 0
    
    def update_current_step(self, step: int):
        """Update the cached current step."""
        self.current_step = step
    
    def get_current_tools(self) -> Dict[str, Any]:
        """Return the currently active tool bundle."""
        return self.current_tools if self.current_tools else {}
    
    def get_blocking_status(self) -> Dict[str, Any]:
        """Return a summary of the current blocking status."""
        # Compute the count of visible tools correctly
        visible_tool_count = 0
        if self.original_tools:
            original_tool_names = list(self.original_tools.get("tools", {}).keys())
            visible_tool_count = len([name for name in original_tool_names 
                                    if not self.is_tool_removed(name)])
        
        from env.core.base_types import get_all_tasks
        return {
            "blocked_count": self.blocked_count,
            "expected_count": self.block_num,
            "current_step": self.current_step,
            "banned_tools_count": len(self.agent_banned_tools),  # Visible but failing for the agent
            "removed_tools_count": len(self.removed_tools) / len(get_all_tasks()),  # Fully invisible
            "visible_tool_count": visible_tool_count / len(get_all_tasks()) # Normalised at runtime (six tasks)
        }


# Convenience factory
def create_blocker(
    block_mode: str,
    block_num: int,
    queries_path: str,
    seed_range: range = range(1, 100),
    refinement_level: int = 4,
    output_dir: str = "env/domains/travel/blocked_tools/"
) -> DynamicBlocker:
    """Construct a DynamicBlocker instance."""
    return DynamicBlocker(
        block_mode=block_mode,
        block_num=block_num,
        seed_range=seed_range,
        refinement_level=refinement_level,
        output_dir=output_dir,
        queries_path=queries_path
    )