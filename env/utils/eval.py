"""
Evaluation utilities for CostBench
Calculates evaluation metrics including accuracy, cost, and tool-path-related metrics.

Example commands:
python env/utils/eval.py \
  --input final_prompt/gpt-5/results_gpt-5_unblocked_refinement_3.json \
  --output final_prompt/gpt-5/results_gpt-5_unblocked_refinement_3_latest.json
  
python env/utils/eval.py \
  --input final_prompt/gemini-2.5-pro/results_gemini-2.5-pro_unblocked_refinement_5.json \
  --output final_prompt/gemini-2.5-pro/results_gemini-2.5-pro_unblocked_refinement_5_latest.json
"""
import re
import json
import argparse
from typing import Dict, List, Any, Optional, Union
from difflib import SequenceMatcher
from pathlib import Path
import sys
import os

# Add the project root to sys.path so atomic_tools can be imported
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)
from env.domains.travel.atomic_tools import get_all_dynamic_atomic_tools, get_all_subtasks
from env.settings import load_config

BAN_TOOL_RETURN_SENTENCES = tuple(load_config().messages.ban_tool_return_sentences)


def is_invalid_tool_call(tool_content: str) -> bool:
    """
    Determine whether a tool call is invalid (i.e., not a banned tool).

    Args:
        tool_content: Content returned by the tool.

    Returns:
        True if the tool call is invalid (starts with [ERROR] but is not in BAN_TOOL_RETURN_SENTENCES).
        False if the tool call is valid or corresponds to a banned tool.
    """
    if not tool_content or not tool_content.strip().startswith("[ERROR]"):
        return False
    
    # Check whether the sentence is listed in BAN_TOOL_RETURN_SENTENCES
    for ban_sentence in BAN_TOOL_RETURN_SENTENCES:
        if ban_sentence in tool_content:
            return False  # This is a banned tool, so it is not treated as an invalid call
    
    return True  # Starts with [ERROR] but is not in the banned list, so it is invalid


def get_atomic_tool_names(refinement_level: int = 5) -> set:
    """
    Retrieve the names of all atomic tools.

    Args:
        refinement_level: Refinement level, defaults to 5.

    Returns:
        set: Set containing every atomic tool name.
    """
    try:
        atomic_tools = get_all_dynamic_atomic_tools(refinement_level)
        return set(atomic_tools.keys())
    except Exception:
        # If retrieval fails, fall back to the basic atomic tool naming patterns
        subtasks = get_all_subtasks()
        atomic_names = set()
        
        for subtask in subtasks:
            subtask_cap = subtask.capitalize()
            # Add the base tools
            atomic_names.add(f"Decide_{subtask_cap}_Preference")
            atomic_names.add(f"Search_{subtask_cap}_Candidates")
            # Add the refinement steps
            for step in range(1, refinement_level + 1):
                atomic_names.add(f"{subtask_cap}_Refinement_Step{step}")
            # Add the select tool
            atomic_names.add(f"Select_Final_{subtask_cap}")
        
        return atomic_names


def extract_candidate_id(answer_text: str) -> Optional[str]:
    """
    Extract the candidate ID from the <answer>...</answer> tags (with relaxed matching).

    Args:
        answer_text: Text that contains the answer content.

    Returns:
        The candidate ID if one is found; otherwise None.
    """
    # 0) For qwen-series models, strip the reasoning chain first
    if "<think>" in answer_text and "</think>" in answer_text:
        answer_text = answer_text.split("</think>")[1]
    
    # 1) First try to grab the content inside <answer> ... </answer>
    answer_scope_match = re.search(r'<answer>([\s\S]*?)</answer>', answer_text, re.IGNORECASE)

    # If no answer tags are found, return None immediately
    if not answer_scope_match:
        return None

    # Only search within the answer tag content
    answer_content = answer_scope_match.group(1)

    # 2) Try matching tokens with angle brackets and curly braces first
    #    Supports:
    #    - "<AccommodationCandidate{Candidate_ID:24499}>"
    #    - "<AccommodationCandidate{Candidate_ID: 27959}>"
    #    - "<DiningCandidate{42198}>"
    curly_with_key_pattern = re.compile(
        r'<\s*(Transportation|Location|Accommodation|Attraction|Dining|Shopping)Candidate\s*\{\s*Candidate_ID\s*:\s*(\d{5})\s*\}\s*>',
        re.IGNORECASE,
    )
    curly_simple_pattern = re.compile(
        r'<\s*(Transportation|Location|Accommodation|Attraction|Dining|Shopping)Candidate\s*\{\s*(\d{5})\s*\}\s*>',
        re.IGNORECASE,
    )

    m = curly_with_key_pattern.search(answer_content)
    if m:
        return m.group(2)
    m = curly_simple_pattern.search(answer_content)
    if m:
        return m.group(2)

    # 3) Perform strict matching within the narrowed scope: require angle brackets
    #    Examples:
    #    - "<LocationCandidate00004>"
    #    - "< TransportationCandidate 00123 >"
    strict_pattern = re.compile(r'<\s*(Transportation|Location|Accommodation|Attraction|Dining|Shopping)Candidate\s*(\d{5})\s*>', re.IGNORECASE)

    # 4) Relaxed matching: allow content without angle brackets
    #    Examples:
    #    - "LocationCandidate00004"
    #    - "TransportationCandidate 00123"
    loose_pattern = re.compile(r'(Transportation|Location|Accommodation|Attraction|Dining|Shopping)Candidate\s*(\d{5})', re.IGNORECASE)

    # Strict match (with angle brackets)
    m = strict_pattern.search(answer_content)
    if m:
        return m.group(2)
    
    # Loose match (without angle brackets)
    m = loose_pattern.search(answer_content)
    if m:
        return m.group(2)

    return None


# ----------------------
# Helper utility functions (shared with visualization/external scripts)
# ----------------------
def match_model_name(file_model_name: Optional[str], target_model_name: Optional[str]) -> bool:
    """Determine whether the model name recorded in the results matches the target model name.

    Supports both the full name (e.g., "Qwen/Qwen2.5-7B-Instruct") and the short name (e.g., "Qwen2.5-7B-Instruct").
    """
    if not file_model_name or not target_model_name:
        return False
    file_short = file_model_name.split('/')[-1]
    target_short = target_model_name.split('/')[-1]
    return (
        file_model_name == target_model_name
        or file_short == target_model_name
        or file_model_name == target_short
        or file_short == target_short
    )


def parse_timestamp_from_filename(file_path: Union[str, Path]) -> Optional[int]:
    """Parse the timestamp from a results filename formatted as:
    results_{model_short}_{block_info}_{timestamp}.json
    """
    try:
        p = Path(file_path)
        stem = p.stem
        ts_str = stem.split('_')[-1]
        return int(ts_str)
    except Exception:
        return None


def extract_gt_id_number(gt_id: str) -> Optional[str]:
    """
    Extract the numeric portion from a ground-truth ID.

    Args:
        gt_id: For example "<TransportationPreference01589>"

    Returns:
        The numeric portion, e.g., "01589".
    """
    pattern = r'<(?:Transportation|Location|Accommodation|Attraction|Dining|Shopping)(?:Preference|Candidate)(\d+)>'
    match = re.search(pattern, gt_id, re.IGNORECASE)
    
    if match:
        return match.group(1)
    
    return None


def get_agent_final_answer(conversation_history: List[Dict[str, Any]]) -> Optional[str]:
    """
    Retrieve the agent's final answer from the conversation history.

    Args:
        conversation_history: The sequence of conversation messages.

    Returns:
        The candidate ID of the final answer, or None if absent.
    """
    # Traverse backwards to find the last assistant response
    for message in reversed(conversation_history):
        if message.get("role") == "assistant":
            content = message.get("content", "")
            if content is None:
                return None
            if "<answer>" in content or "Candidate" in content:
                return extract_candidate_id(content)
    
    return None


def get_agent_tool_path_and_cost(conversation_history: List[Dict[str, Any]], block_steps: List[int]) -> tuple:
    """
    Extract the agent's tool path and total cost after the last blocking event.

    Args:
        conversation_history: The conversation history.
        block_steps: List of steps where blocking occurred.

    Returns:
        (tool_path: List[str], total_cost: float)
    """
    tool_path = []
    total_cost = 0.0
    
    # Determine the starting point after the most recent blocking step (tolerate None values)
    try:
        cleaned_block_steps = [b for b in (block_steps or []) if isinstance(b, int)]
        last_block_step = max(cleaned_block_steps) if cleaned_block_steps else 0
    except Exception:
        last_block_step = 0
    current_step = 0
    
    for i, message in enumerate(conversation_history):
        # Skip system and user messages
        if message.get("role") in ["system", "user"]:
            continue
            
        if message.get("role") == "assistant":
            current_step += 1
            
            # Only consider tool calls that happen after the last blocking event
            if current_step > last_block_step:
                tool_calls = message.get("tool_calls", [])
                for tool_call in tool_calls:
                    tool_name = tool_call.get("name")
                    if tool_name:
                        tool_path.append(tool_name)
        
        elif message.get("role") == "tool":
            # Only account for cost after the last blocking event
            if current_step > last_block_step:
                content = message.get("content", "")
                # Extract the cost value from the tool response
                cost_match = re.search(r'Cost:\s*([\d.]+)', content, re.IGNORECASE)
                if cost_match:
                    value_str = cost_match.group(1).rstrip(' .。;,)]')
                    try:
                        total_cost += float(value_str)
                    except Exception:
                        pass
    
    return tool_path, total_cost


def extract_full_agent_steps(conversation_history: List[Dict[str, Any]]) -> tuple:
    """
    Extract the full sequence of agent tool calls (counted per assistant turn) and parse each step's cost.
    Invalid tool calls are filtered out (start with [ERROR] but are not in BAN_TOOL_RETURN_SENTENCES).

    Returns:
        tuple: (steps: List[Dict], total_tool_calls: int, valid_tool_calls: int, atomic_tool_calls: int)
        Each step looks like {"step": int, "tool": str, "cost": float, "is_valid": bool, "is_atomic": bool}.
    """
    steps = []
    current_step = 0
    i = 0
    n = len(conversation_history)
    total_tool_calls = 0
    valid_tool_calls = 0
    atomic_tool_calls = 0
    
    # Cache the set of atomic tool names
    atomic_tool_names = get_atomic_tool_names()
    
    while i < n:
        msg = conversation_history[i]
        role = msg.get("role")
        if role == "assistant":
            current_step += 1
            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                # Only take the first tool call (consistent with runtime behavior)
                try:
                    tool_name = tool_calls[0].get("name") if isinstance(tool_calls[0], dict) else tool_calls[0].function.name
                except Exception:
                    tool_name = None
                
                # Find the first tool response after this assistant message
                cost = 0.0
                is_valid = True
                j = i + 1
                while j < n:
                    nxt = conversation_history[j]
                    if nxt.get("role") == "tool":
                        content = nxt.get("content", "") or ""
                        
                        # Determine whether the tool call is invalid
                        if is_invalid_tool_call(content):
                            is_valid = False
                        
                        # Extract the cost value
                        m = re.search(r"Cost:\s*([\d.]+)", content, re.IGNORECASE)
                        if m:
                            try:
                                value_str = m.group(1).rstrip(' .。;,)]')
                                cost = float(value_str)
                            except Exception:
                                cost = 0.0
                        break
                    # If the next message is another assistant reply, break because no tool response exists for this step
                    if nxt.get("role") == "assistant":
                        break
                    j += 1
                
                if tool_name:
                    total_tool_calls += 1
                    is_atomic = tool_name in atomic_tool_names
                    
                    if is_valid:
                        valid_tool_calls += 1
                    
                    if is_atomic:
                        atomic_tool_calls += 1
                    
                    steps.append({
                        "step": current_step, 
                        "tool": tool_name, 
                        "cost": cost,
                        "is_valid": is_valid,
                        "is_atomic": is_atomic
                    })
        i += 1
    
    return steps, total_tool_calls, valid_tool_calls, atomic_tool_calls


def build_segmented_gt_path(scenarios: List[Dict[str, Any]], block_steps: List[int]) -> tuple:
    """
    Construct a segmented ground-truth path based on the blocking schedule.

    Correct logic:
    1. Each scenario contains the full ground-truth path recomputed after its corresponding blocking.
    2. block_steps stores the absolute step number where each blocking is triggered.
    3. The ground-truth path takes steps from each scenario up to the trigger of the next blocking.

    Args:
        scenarios: A list of scenarios, each with tools and path_length.
        block_steps: List of absolute step numbers where blocking is triggered.

    Returns:
        tuple: (gt_path, gt_cost)
    """
    gt_path: List[str] = []
    gt_cost: float = 0.0

    if not scenarios:
        return gt_path, gt_cost

    # Convert each scenario's tools into a list of (name, cost) tuples
    def scenario_tools_tuple_list(scn: Dict[str, Any]) -> List[tuple]:
        tools = scn.get("tools", [])
        result = []
        for t in tools:
            if isinstance(t, dict) and t:
                name = list(t.keys())[0]
                cost = float(list(t.values())[0])
                result.append((name, cost))
        return result

    scenario_steps = [scenario_tools_tuple_list(s) for s in scenarios]

    # If there is no blocking, return the full path from scenario[0]
    if not block_steps:
        for name, cost in scenario_steps[0]:
            gt_path.append(name)
            gt_cost += cost
        return gt_path, gt_cost

    # With blocking: compute how many steps to take in each segment based on absolute trigger steps (allow None values)
    cleaned_blocks = [b for b in (block_steps or []) if isinstance(b, int)]
    sorted_blocks = sorted(cleaned_blocks)
    current_step = 0  # Total number of GT steps accumulated so far
    
    # Iterate over each scenario segment
    for scenario_idx in range(len(scenarios)):
        scenario_tools = scenario_steps[scenario_idx] if scenario_idx < len(scenario_steps) else []
        
        if scenario_idx < len(sorted_blocks):
            # Intermediate segments: take steps only up to the next blocking trigger
            next_trigger_step = sorted_blocks[scenario_idx]
            steps_to_take = next_trigger_step - current_step
            
            # Ensure we do not exceed the scenario length and avoid negative counts
            steps_to_take = max(0, min(steps_to_take, len(scenario_tools)))
            
            # Append the GT steps for this segment
            for i in range(steps_to_take):
                if i < len(scenario_tools):
                    name, cost = scenario_tools[i]
                    gt_path.append(name)
                    gt_cost += cost

            current_step += steps_to_take
            
        else:
            # Last segment: take all remaining steps in this scenario
            for name, cost in scenario_tools:
                gt_path.append(name)
                gt_cost += cost
            break

    return gt_path, gt_cost


def get_groundtruth_info(query_dict: Dict[str, Any]) -> tuple:
    """
    Retrieve the ground-truth tool path and cost.

    Args:
        query_dict: Query payload to inspect.

    Returns:
        (gt_tool_path: List[str], gt_cost: float)

    Raises:
        ValueError: If the scenario_index does not match the recorded block count.
    """
    scenarios = query_dict.get("groundtruth", {}).get("scenarios", [])
    if not scenarios:
        return [], 0.0
    
    block_count = query_dict.get("block_stats", {}).get("block_count", 0)
    last_scenario = scenarios[-1]
    
    # Validate the scenario_index
    if last_scenario.get("scenario_index") != block_count:
        raise ValueError(f"Scenario index mismatch: expected {block_count}, got {last_scenario.get('scenario_index')}")
    
    # Adapt to the new tools structure: list[dict{tool_name: cost}] -> extract an ordered list of tool names
    tools_list = last_scenario.get("tools", [])
    gt_tool_names: List[str] = []
    try:
        for item in tools_list:
            if isinstance(item, dict) and len(item) == 1:
                tool_name = next(iter(item.keys()))
                gt_tool_names.append(tool_name)
            else:
                # Fall back to the legacy format (list[str]) or unexpected entries
                if isinstance(item, str):
                    gt_tool_names.append(item)
    except Exception:
        # Fallback: if parsing fails, return an empty list
        gt_tool_names = []
    gt_cost = last_scenario.get("total_cost", 0.0)
    
    return gt_tool_names, gt_cost


def calculate_edit_distance(seq1: List[str], seq2: List[str]) -> int:
    """
    Compute the minimum edit distance between two sequences.

    Args:
        seq1: Sequence one.
        seq2: Sequence two.

    Returns:
        The minimum edit distance.
    """
    if not seq1 and not seq2:
        return 0
    if not seq1:
        return len(seq2)
    if not seq2:
        return len(seq1)
    
    # Use dynamic programming to compute the edit distance
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialization
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # Populate the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[m][n]


def check_answer_accuracy(agent_answer: Optional[str], scenarios: List[Dict[str, Any]], actual_block_count: int = 0) -> bool:
    """
    Check whether the agent's answer matches the gt_id of the final scenario.

    Args:
        agent_answer: Candidate ID returned by the agent.
        scenarios: List of ground-truth scenarios.
        actual_block_count: Number of blocking steps observed, used to locate the final scenario.

    Returns:
        True if the answer matches the final scenario's gt_id; otherwise False.
    """
    if not agent_answer or not scenarios:
        return False
    
    # Determine the final scenario index based on the actual number of blocking events
    target_scenario_index = actual_block_count
    
    if target_scenario_index < len(scenarios):
        target_scenario = scenarios[target_scenario_index]
        gt_id = target_scenario.get("gt_id", "")
        gt_number = extract_gt_id_number(gt_id)
        return gt_number and agent_answer == gt_number
    
    return False


def check_previous_gt_answer_accuracy(agent_answer: Optional[str], scenarios: List[Dict[str, Any]]) -> bool:
    """
    Check whether the agent's answer matches any scenario's gt_id (historical GT accuracy).

    Args:
        agent_answer: Candidate ID returned by the agent.
        scenarios: List of ground-truth scenarios.

    Returns:
        True if the answer matches any scenario's gt_id; otherwise False.
    """
    if not agent_answer:
        return False
    
    for scenario in scenarios:
        gt_id = scenario.get("gt_id", "")
        gt_number = extract_gt_id_number(gt_id)
        if gt_number and agent_answer == gt_number:
            return True
    
    return False


def evaluate_single_query(query_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate every metric for a single query.

    Args:
        query_dict: Query payload to evaluate.

    Returns:
        A dictionary containing the evaluation results.
    """
    # Basic metadata and flags
    block_stats = query_dict.get("block_stats", {})
    expected_block_count = block_stats.get("expected_block_count", 0)
    # Favor the agent-side blocking count; if missing, fall back to the final GT count
    actual_block_count_agent = block_stats.get("block_count_agent", block_stats.get("block_count", 0))
    # New rule: only include cases where the agent's blocking count exactly matches the expectation
    is_block_valid = (expected_block_count == actual_block_count_agent)

    conversation_history = query_dict.get("conversation_history", [])
    conversation_length = len(conversation_history)
    scenarios = query_dict.get("groundtruth", {}).get("scenarios", [])
    block_steps = block_stats.get("block_step", [])
    is_goal_state = query_dict.get("is_goal_state", None)

    # Flag: if the final scenario has no tools, treat the GT as producing no further steps
    gt_unblocked = False
    try:
        if scenarios:
            last_scn_tools = scenarios[-1].get("tools", [])
            if not last_scn_tools:
                gt_unblocked = True
    except Exception:
        gt_unblocked = False

    # Additional flag: determine whether GT is incomplete (either gt_unblocked or blocking still contains None)
    # Note: has_null_block_step is computed below; this variable is a placeholder until gt_incomplete is finalized
    gt_incomplete = False

    # 1) Answer
    agent_final_answer = get_agent_final_answer(conversation_history)
    has_answer = agent_final_answer is not None

    # Validity definition: either blocking occurred, or the task was unblocked (expected_block_count == 0)
    no_goal_state = (is_goal_state is False)
    is_valid = (is_block_valid or expected_block_count == 0)

    # Indicator: if any block_step is None, the GT is considered incomplete
    try:
        has_null_block_step = any(x is None for x in (block_steps or []))
    except Exception:
        has_null_block_step = False

    # Unified definition of GT incompleteness
    gt_incomplete = bool(gt_unblocked or has_null_block_step)

    if not is_valid:
        return {
            "is_valid": False,
            "query_id": query_dict.get("query_id", ""),
            "expected_block_count": expected_block_count,
            "actual_block_count": actual_block_count_agent,
            "is_block_valid": is_block_valid,
            "has_answer": has_answer,
            "no_goal_state": no_goal_state,
            "agent_final_answer": agent_final_answer,
            "has_null_block_step": has_null_block_step,
            "gt_unblocked": gt_unblocked,
            "gt_incomplete": gt_incomplete,
            "conversation_length": conversation_length
        }

    # 2) Accuracy
    final_answer_correct = check_answer_accuracy(agent_final_answer, scenarios, actual_block_count_agent)
    previous_gt_answer_correct = check_previous_gt_answer_accuracy(agent_final_answer, scenarios)

    # 3) Full agent path and cost; construct segmented GT path and cost
    try:
        agent_steps, total_tool_calls, valid_tool_calls, atomic_tool_calls = extract_full_agent_steps(conversation_history)
        # Build the path using only valid tool calls
        agent_tool_path = [s["tool"] for s in agent_steps if s["is_valid"]]
        # agent_tool_path = [s["tool"] for s in agent_steps]

        # Relax the scenario_index check: do not abort the evaluation if it mismatches; use it only for diagnostics
        scenarios_nonempty = scenarios or []
        if scenarios_nonempty:
            _ = scenarios_nonempty[-1].get("scenario_index")

        # Build the segmented GT path first for path-based metrics
        gt_tool_path, gt_cost_full = build_segmented_gt_path(scenarios, block_steps)

        # Unified cost strategy: compute the entire trajectory regardless of blocking
        # Agent cost: sum only valid tool calls
        agent_cost = sum(s["cost"] for s in agent_steps if s["is_valid"])
        
        # GT cost: use the full cost from build_segmented_gt_path
        gt_cost = gt_cost_full

        # 4) Edit distance and exact match (agent path vs. segmented GT path)
        edit_distance = calculate_edit_distance(agent_tool_path, gt_tool_path)
        tool_path_exact_match = (agent_tool_path == gt_tool_path)
        
        # Compute the per-query normalized edit distance
        max_path_length = max(len(agent_tool_path), len(gt_tool_path))
        normalized_edit_distance_single = edit_distance / max_path_length if max_path_length > 0 else 0.0
        
        # Compute the cost ratio: (greedy_cost - agent_cost)/(greedy_cost - gt_cost)
        cost_ratio = None
        try:
            stimulation = query_dict.get("stimulation", {})
            greedy_cost = stimulation.get("avg_cost", 0.0)
            if greedy_cost > 0 and gt_cost > 0:
                cost_ratio = (greedy_cost - agent_cost) / (greedy_cost - gt_cost)
        except Exception:
            cost_ratio = None
        
        # Compute the valid search ratio and atomic tool ratio
        valid_search_ratio = valid_tool_calls / total_tool_calls if total_tool_calls > 0 else 0.0
        atomic_tool_ratio = atomic_tool_calls / total_tool_calls if total_tool_calls > 0 else 0.0

        # If the agent failed to reach the goal state, force NED to 1 and mark as not an exact match
        # Override rule 1: when the goal state is not reached, force NED=1 and EM=False
        if no_goal_state:
            normalized_edit_distance_single = 1.0
            # Set the raw edit distance to the maximum possible value for consistency
            edit_distance = max_path_length
            tool_path_exact_match = False

        # Override rule 2: if a previously banned tool is reused without triggering a new ban (in repeated ban_tool scenarios)
        # force NED=1 and EM=False
        try:
            if bool(query_dict.get("duplicate_banned_tool_reused", False)):
                normalized_edit_distance_single = 1.0
                edit_distance = max_path_length
                tool_path_exact_match = False
        except Exception:
            pass

    except (ValueError, KeyError) as e:
        return {
            "is_valid": False,
            "query_id": query_dict.get("query_id", ""),
            "error": str(e),
            "conversation_length": conversation_length
        }

    return {
        "is_valid": True,
        "query_id": query_dict.get("query_id", ""),
        "final_answer_correct": final_answer_correct,
        "previous_gt_answer_correct": previous_gt_answer_correct,
        "agent_final_answer": agent_final_answer,
        "agent_tool_path": agent_tool_path,
        "agent_cost": agent_cost,
        "gt_tool_path": gt_tool_path,
        "gt_cost": gt_cost,
        "edit_distance": edit_distance,
        "normalized_edit_distance_single": normalized_edit_distance_single,
        "tool_path_exact_match": tool_path_exact_match,
        "expected_block_count": expected_block_count,
        "actual_block_count": actual_block_count_agent,
        "is_block_valid": is_block_valid,
        "has_answer": has_answer,
        "no_goal_state": no_goal_state,
        "gt_unblocked": gt_unblocked,
        "has_null_block_step": has_null_block_step,
        "gt_incomplete": gt_incomplete,
        # Additional metrics
        "cost_ratio": cost_ratio,
        "valid_search_ratio": valid_search_ratio,
        "atomic_tool_ratio": atomic_tool_ratio,
        "total_tool_calls": total_tool_calls,
        "valid_tool_calls": valid_tool_calls,
        "atomic_tool_calls": atomic_tool_calls,
        "conversation_length": conversation_length
    }


def eval(cleaned_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluate every query and compute aggregate metrics.

    Args:
        cleaned_results: List of cleaned query results.

    Returns:
        Dictionary containing all aggregated metrics.
    """
    total_queries = len(cleaned_results)
    valid_queries = []
    invalid_queries = []
    all_results = []
    
    # Track the counts of invalid categories in detail
    no_answer_count = 0
    block_incomplete_count = 0
    no_goal_state_count = 0
    
    # Evaluate each query
    for query_dict in cleaned_results:
        result = evaluate_single_query(query_dict)
        all_results.append(result)

        # Count the detailed categories, excluding samples where the GT is unblocked
        if not result.get("gt_unblocked", False):
            if result.get("no_goal_state"):
                no_goal_state_count += 1
            if not result.get("has_answer", True):
                no_answer_count += 1
            if result.get("expected_block_count", 0) > 0 and not result.get("is_block_valid", False):
                block_incomplete_count += 1

            if result.get("is_valid"):
                valid_queries.append(result)
            else:
                invalid_queries.append(result)
    
    # Samples participating in aggregate metrics:
    # - Exclude cases where the GT is incomplete (including None or empty tool segments)
    # - Only include valid samples (those with the necessary fields for accuracy)
    metrics_queries = [
        r for r in all_results
        if (
            not r.get("gt_incomplete", False)
            and r.get("is_valid", False)
        )
    ]


    # Precompute path metrics:
    # - Exclude samples where the GT is incomplete
    # - If the agent did not complete blocking as expected or reach the goal, count as worst case (NED=1, EM=0, ED=max(len(agent), len(gt)))
    try:
        path_base_queries = [
            r for r in all_results
            if (
                not r.get("gt_incomplete", False)
                and r.get("is_block_valid", False)
                and (r.get("conversation_length", 0) > 3)
            )
        ]
        path_n = len(path_base_queries)
        if path_n == 0:
            avg_edit_distance_all = 0.0
            avg_normalized_edit_distance_all = 0.0
            tool_path_exact_match_count_all = 0
            tool_path_exact_match_ratio_all = 0.0
        else:
            edits = []
            neds = []
            ems = []
            for q in path_base_queries:
                agent_path = q.get("agent_tool_path", []) or []
                gt_path = q.get("gt_tool_path", []) or []
                agent_ok = bool(q.get("is_block_valid", False) and not q.get("no_goal_state", False))
                if agent_ok:
                    ned_val = float(q.get("normalized_edit_distance_single", 0.0))
                    em_val = bool(q.get("tool_path_exact_match", False))
                    try:
                        ed_val = float(q.get("edit_distance", 0.0))
                    except Exception:
                        ed_val = 0.0
                else:
                    # Worst-case scenario
                    max_len = max(len(agent_path), len(gt_path))
                    ed_val = float(max_len)
                    ned_val = 1.0 if max_len > 0 else 0.0
                    em_val = False
                neds.append(ned_val)
                ems.append(em_val)
                edits.append(ed_val)
            avg_edit_distance_all = (sum(edits) / path_n) if path_n > 0 else 0.0
            avg_normalized_edit_distance_all = (sum(neds) / path_n) if path_n > 0 else 0.0
            tool_path_exact_match_count_all = sum(1 for x in ems if x)
            tool_path_exact_match_ratio_all = (tool_path_exact_match_count_all / path_n) if path_n > 0 else 0.0
    except Exception:
        avg_edit_distance_all = 0.0
        avg_normalized_edit_distance_all = 0.0
        tool_path_exact_match_count_all = 0
        tool_path_exact_match_ratio_all = 0.0


    # Compute overall metrics
    block_valid_num = len(metrics_queries)
    block_invalid_num = block_incomplete_count  # Legacy compatibility: counts invalid blocking cases
    no_answer_num = no_answer_count
    
    if block_valid_num == 0:
        return {
            "total_queries": total_queries,
            "valid_num": 0,
            "block_invalid_num": block_invalid_num,
            "no_answer_num": no_answer_num,
            "no_goal_state_num": no_goal_state_count,
            "accuracy_metrics": {
                "final_answer_accuracy": 0.0,
                "previous_gt_answer_accuracy": 0.0  # This metric may need further refinement based on requirements
            },
            "cost_metrics": {
                "avg_agent_cost": 0.0,
                "avg_gt_cost": 0.0,
                "cost_ratio": 0.0
            },
            "tool_path_metrics": {
                "avg_edit_distance": avg_edit_distance_all,
                "avg_normalized_edit_distance": avg_normalized_edit_distance_all,
                "tool_path_exact_match_count": tool_path_exact_match_count_all,
                "tool_path_exact_match_ratio": tool_path_exact_match_ratio_all
            },
            "anomaly_metrics": {
                "agent_cost_below_gt_count": 0,
                "agent_cost_below_gt_ratio": 0.0,
                "query_ids": []
            }
        }
    
    # Calculate accuracy metrics (only consider valid samples with conversation length > 3)
    accuracy_queries = [
        q for q in metrics_queries
        if q.get("conversation_length", 0) > 3
    ]
    accuracy_num = len(accuracy_queries)

    if accuracy_num == 0:
        final_answer_correct_count = 0
        final_answer_accuracy = 0.0
        previous_gt_answer_correct_count = 0
        previous_gt_answer_accuracy = 0.0
    else:
        final_answer_correct_count = sum(1 for q in accuracy_queries if bool(q.get("final_answer_correct", False)))
        final_answer_accuracy = final_answer_correct_count / accuracy_num

        previous_gt_answer_correct_count = sum(1 for q in accuracy_queries if bool(q.get("previous_gt_answer_correct", False)))
        previous_gt_answer_accuracy = previous_gt_answer_correct_count / accuracy_num
    
    # Cost metrics consider only samples where GT is complete, the agent satisfied blocking, and the goal state was reached
    cost_queries = [
        q for q in metrics_queries
        if (
            (not q.get("no_goal_state", False))
            and q.get("is_block_valid", False)
            and (q.get("conversation_length", 0) > 3)
        )
    ]
    cost_valid_num = len(cost_queries)
    
    if cost_valid_num == 0:
        total_agent_cost = 0.0
        total_gt_cost = 0.0
        avg_agent_cost = 0.0
        avg_gt_cost = 0.0
        cost_ratio = 0.0
    else:
        total_agent_cost = sum(q["agent_cost"] for q in cost_queries)
        total_gt_cost = sum(q["gt_cost"] for q in cost_queries)
        avg_agent_cost = total_agent_cost / cost_valid_num
        avg_gt_cost = total_gt_cost / cost_valid_num
        cost_ratio = avg_agent_cost / avg_gt_cost if avg_gt_cost > 0 else float('inf')

    # Approach A: normalize by the "atomic unit gap" (using E_atomic = expected atomic cost)
    # Default to the empirical mean of 20 unless callers supply configuration overrides
    try:
        # Replace with provided values if cleaned_results[0]["run_args"] includes them later
        min_atomic = cleaned_results[0].get("run_args", {}).get("min_atomic_cost")
        max_atomic = cleaned_results[0].get("run_args", {}).get("max_atomic_cost")
        if min_atomic is not None and max_atomic is not None:
            expected_atomic_cost = (float(min_atomic) + float(max_atomic)) / 2.0
        else:
            expected_atomic_cost = 20.0
    except Exception:
        expected_atomic_cost = 20.0

    per_query_delta_atomic = []
    per_query_ratio = []
    for q in cost_queries:
        agent_c = q["agent_cost"]
        gt_c = q["gt_cost"]
        # Delta in atomic units (avoid division by zero; fall back to raw difference if expected_atomic_cost <= 0)
        if expected_atomic_cost and expected_atomic_cost > 0:
            per_query_delta_atomic.append((agent_c - gt_c) / expected_atomic_cost)
        else:
            per_query_delta_atomic.append(agent_c - gt_c)
        # Per-query ratio (avoid gt_c == 0)
        per_query_ratio.append((agent_c / gt_c) if gt_c > 0 else float('inf'))

    if cost_valid_num == 0:
        avg_delta_atomic = 0.0
        mean_per_query_ratio = 0.0
    else:
        avg_delta_atomic = sum(per_query_delta_atomic) / cost_valid_num
        # Approach B: average the per-query ratios
        mean_per_query_ratio = sum(per_query_ratio) / cost_valid_num
    
    anomaly_query_ids = []
    for q in cost_queries:
        if q["agent_cost"] < q["gt_cost"]:
            anomaly_query_ids.append(q["query_id"])
    anomaly_count = len(anomaly_query_ids)
    anomaly_ratio = (anomaly_count / cost_valid_num) if cost_valid_num > 0 else 0.0
    
    # Compute the average cost ratio (only for valid values and samples that reached the goal state)
    cost_ratios = [q.get("cost_ratio") for q in cost_queries if q.get("cost_ratio") is not None]
    avg_cost_ratio = sum(cost_ratios) / len(cost_ratios) if cost_ratios else None
    
    # Aggregate the valid search and atomic tool ratios
    total_tool_calls_all = sum(q.get("total_tool_calls", 0) for q in metrics_queries)
    valid_tool_calls_all = sum(q.get("valid_tool_calls", 0) for q in metrics_queries)
    atomic_tool_calls_all = sum(q.get("atomic_tool_calls", 0) for q in metrics_queries)
    
    overall_valid_search_ratio = valid_tool_calls_all / total_tool_calls_all if total_tool_calls_all > 0 else 0.0
    overall_atomic_tool_ratio = atomic_tool_calls_all / total_tool_calls_all if total_tool_calls_all > 0 else 0.0
    
    # Path metrics: reuse the precomputed values (GT incomplete removed; unfinished agents counted as worst case)
    avg_edit_distance = avg_edit_distance_all
    avg_normalized_edit_distance = avg_normalized_edit_distance_all
    tool_path_exact_match_count = tool_path_exact_match_count_all
    tool_path_exact_match_ratio = tool_path_exact_match_ratio_all

    # ------------------------------
    # Stimulation (random/greedy) aggregation (include all unless errors are reported)
    # ------------------------------
    try:
        stim_entries = []
        # Agent vs. stimulation: aggregate within each query, then average across queries
        per_query_agent_vs_stim_avg_ed: List[float] = []
        per_query_agent_vs_stim_avg_ned: List[float] = []
        # all_results and cleaned_results share the same order, so indexes align to retrieve gt_cost
        for idx, q in enumerate(cleaned_results):
            try:
                stim = q.get("stimulation")
                # Skip entries without stimulation data or with explicit errors
                if not isinstance(stim, dict):
                    continue
                if isinstance(stim.get("error"), str) and stim.get("error"):
                    continue

                # Parse the per-query average metrics from stimulation
                stim_avg_cost = float(stim.get("avg_cost", 0.0) or 0.0)
                stim_avg_edit = float(stim.get("avg_edit", 0.0) or 0.0)
                stim_avg_ned = float(stim.get("avg_ned", 0.0) or 0.0)
                # has_exact_match: record 1 if any stimulation run exactly matches the GT path; otherwise 0
                stim_em_flag = 1.0 if (int(stim.get("has_exact_match", 0) or 0) > 0) else 0.0

                # Retrieve the GT cost for this query (same definition as the agent: segmented GT path cost)
                # Treat missing or out-of-range indexes as zero
                try:
                    gt_cost_this = float(all_results[idx].get("gt_cost", 0.0) or 0.0)
                except Exception:
                    gt_cost_this = 0.0

                stim_entries.append({
                    "stim_avg_cost": stim_avg_cost,
                    "stim_avg_edit": stim_avg_edit,
                    "stim_avg_ned": stim_avg_ned,
                    "stim_em_flag": stim_em_flag,
                    "gt_cost": gt_cost_this,
                })

                # Compute the edit-distance gaps between the agent path and each stimulation path, then average per query
                try:
                    agent_path_for_this = (all_results[idx].get("agent_tool_path") or [])
                    stim_paths_for_this = stim.get("paths") or []
                    if isinstance(stim_paths_for_this, list) and len(stim_paths_for_this) > 0:
                        ed_vals: List[float] = []
                        ned_vals: List[float] = []
                        for sp in stim_paths_for_this:
                            sp_list = sp or []
                            ed_val = float(calculate_edit_distance(agent_path_for_this, sp_list))
                            max_len = max(len(agent_path_for_this), len(sp_list))
                            ned_val = (ed_val / max_len) if max_len > 0 else 0.0
                            ed_vals.append(ed_val)
                            ned_vals.append(ned_val)
                        per_query_agent_vs_stim_avg_ed.append(sum(ed_vals) / len(ed_vals))
                        per_query_agent_vs_stim_avg_ned.append(sum(ned_vals) / len(ned_vals))
                except Exception:
                    pass
            except Exception:
                # Skip problematic entries
                continue

        stim_valid_num = len(stim_entries)
        if stim_valid_num == 0:
            stim_cost_metrics = {
                "total_agent_cost": 0.0,
                "total_gt_cost": 0.0,
                "avg_agent_cost": 0.0,
                "avg_gt_cost": 0.0,
                "cost_ratio": 0.0,
                "avg_delta_atomic": 0.0,
                "mean_per_query_ratio": 0.0,
                "agent_cost_below_gt_count": 0,
                "agent_cost_below_gt_ratio": 0.0,
            }
            stim_path_metrics = {
                "avg_edit_distance": 0.0,
                "avg_normalized_edit_distance": 0.0,
                "tool_path_exact_match_count": 0,
                "tool_path_exact_match_ratio": 0.0,
            }
        else:
            # Cost: reuse the agent aggregation method but with stimulation per-query averages
            total_stim_cost = sum(e["stim_avg_cost"] for e in stim_entries)
            total_gt_cost_stim = sum(e["gt_cost"] for e in stim_entries)
            avg_stim_cost = total_stim_cost / stim_valid_num
            avg_gt_cost_stim = total_gt_cost_stim / stim_valid_num if stim_valid_num > 0 else 0.0
            cost_ratio_stim = (avg_stim_cost / avg_gt_cost_stim) if avg_gt_cost_stim > 0 else float('inf')

            # Atomic-unit deviation
            per_query_delta_atomic_stim: List[float] = []
            per_query_ratio_stim: List[float] = []
            for e in stim_entries:
                agent_c = e["stim_avg_cost"]
                gt_c = e["gt_cost"]
                if expected_atomic_cost and expected_atomic_cost > 0:
                    per_query_delta_atomic_stim.append((agent_c - gt_c) / expected_atomic_cost)
                else:
                    per_query_delta_atomic_stim.append(agent_c - gt_c)
                per_query_ratio_stim.append((agent_c / gt_c) if gt_c > 0 else float('inf'))

            avg_delta_atomic_stim = sum(per_query_delta_atomic_stim) / stim_valid_num
            mean_per_query_ratio_stim = sum(per_query_ratio_stim) / stim_valid_num

            agent_cost_below_gt_count_stim = sum(1 for e in stim_entries if e["stim_avg_cost"] < e["gt_cost"])
            agent_cost_below_gt_ratio_stim = agent_cost_below_gt_count_stim / stim_valid_num

            stim_cost_metrics = {
                "total_agent_cost": total_stim_cost,
                "total_gt_cost": total_gt_cost_stim,
                "avg_agent_cost": avg_stim_cost,
                "avg_gt_cost": avg_gt_cost_stim,
                "cost_ratio": cost_ratio_stim,
                "avg_delta_atomic": avg_delta_atomic_stim,
                "mean_per_query_ratio": mean_per_query_ratio_stim,
                "agent_cost_below_gt_count": agent_cost_below_gt_count_stim,
                "agent_cost_below_gt_ratio": agent_cost_below_gt_ratio_stim,
            }

            # Paths: use stimulation per-query average NED/ED; EM uses has_exact_match as a binary flag
            avg_edit_distance_stim = sum(e["stim_avg_edit"] for e in stim_entries) / stim_valid_num
            avg_ned_stim = sum(e["stim_avg_ned"] for e in stim_entries) / stim_valid_num
            em_count_stim = int(sum(1 for e in stim_entries if e["stim_em_flag"] > 0))
            em_ratio_stim = em_count_stim / stim_valid_num

            stim_path_metrics = {
                "avg_edit_distance": float(avg_edit_distance_stim),
                "avg_normalized_edit_distance": float(avg_ned_stim),
                "tool_path_exact_match_count": int(em_count_stim),
                "tool_path_exact_match_ratio": float(em_ratio_stim),
            }

        # Agent vs. stimulation averaged across queries
        try:
            if per_query_agent_vs_stim_avg_ed and per_query_agent_vs_stim_avg_ned:
                agent_vs_stim_avg_ed_overall = sum(per_query_agent_vs_stim_avg_ed) / len(per_query_agent_vs_stim_avg_ed)
                agent_vs_stim_avg_ned_overall = sum(per_query_agent_vs_stim_avg_ned) / len(per_query_agent_vs_stim_avg_ned)
            else:
                agent_vs_stim_avg_ed_overall = 0.0
                agent_vs_stim_avg_ned_overall = 0.0
        except Exception:
            agent_vs_stim_avg_ed_overall = 0.0
            agent_vs_stim_avg_ned_overall = 0.0
    except Exception:
        # If any error occurs in the aggregation, return empty stimulation metrics
        stim_valid_num = 0
        stim_cost_metrics = {
            "total_agent_cost": 0.0,
            "total_gt_cost": 0.0,
            "avg_agent_cost": 0.0,
            "avg_gt_cost": 0.0,
            "cost_ratio": 0.0,
            "avg_delta_atomic": 0.0,
            "mean_per_query_ratio": 0.0,
            "agent_cost_below_gt_count": 0,
            "agent_cost_below_gt_ratio": 0.0,
        }
        stim_path_metrics = {
            "avg_edit_distance": 0.0,
            "avg_normalized_edit_distance": 0.0,
            "tool_path_exact_match_count": 0,
            "tool_path_exact_match_ratio": 0.0,
        }
        agent_vs_stim_avg_ed_overall = 0.0
        agent_vs_stim_avg_ned_overall = 0.0
    
    return {
        "total_queries": total_queries,
        "valid_num": block_valid_num,
        "block_invalid_num": block_invalid_num,
        "no_answer_num": no_answer_num,
        "no_goal_state_num": no_goal_state_count,
        "accuracy_metrics": {
            "final_answer_accuracy": final_answer_accuracy,
            "final_answer_correct_count": final_answer_correct_count,
            "previous_gt_answer_accuracy": previous_gt_answer_accuracy,
            "previous_gt_answer_correct_count": previous_gt_answer_correct_count
        },
        "cost_metrics": {
            "total_agent_cost": total_agent_cost,
            "total_gt_cost": total_gt_cost,
            "avg_agent_cost": avg_agent_cost,
            "avg_gt_cost": avg_gt_cost,
            "cost_ratio": cost_ratio,  # Original agent-to-GT ratio
            "stimulation_cost_ratio": avg_cost_ratio,  # Cost ratio derived from stimulation results
            "avg_delta_atomic": avg_delta_atomic,
            "mean_per_query_ratio": mean_per_query_ratio,
            "agent_cost_below_gt_count": anomaly_count,
            "agent_cost_below_gt_ratio": anomaly_ratio,
        },
        "tool_path_metrics": {
            "avg_edit_distance": avg_edit_distance,
            "avg_normalized_edit_distance": avg_normalized_edit_distance,
            "tool_path_exact_match_count": tool_path_exact_match_count,
            "tool_path_exact_match_ratio": tool_path_exact_match_ratio
        },
        "tool_usage_metrics": {
            "total_tool_calls": total_tool_calls_all,
            "valid_tool_calls": valid_tool_calls_all,
            "atomic_tool_calls": atomic_tool_calls_all,
            "valid_search_ratio": overall_valid_search_ratio,
            "atomic_tool_ratio": overall_atomic_tool_ratio
        },
        "stimulation_metrics": {
            "participating_num": stim_valid_num,
            "cost_metrics": stim_cost_metrics,
            "tool_path_metrics": stim_path_metrics,
            "agent_vs_stimulation_path_metrics": {
                "avg_edit_distance": float(agent_vs_stim_avg_ed_overall),
                "avg_normalized_edit_distance": float(agent_vs_stim_avg_ned_overall),
            },
        },
        # "detailed_results": valid_queries,
        # "invalid_queries": invalid_queries
    }


# Retain the original function name for compatibility
def evaluate_single_run(query_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Backward-compatible single-query evaluation function"""
    return evaluate_single_query(query_dict)


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-evaluate a CostBench results JSON file using env.utils.eval"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to an existing results_*.json file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path. Defaults to <input_stem>_reeval.json alongside the input file."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the input file in-place instead of writing to a new file."
    )
    return parser.parse_args()


def main_cli() -> None:
    args = parse_cli_args()

    input_path = args.input.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    cleaned_results = payload.get("results")
    if not isinstance(cleaned_results, list):
        raise ValueError("Input JSON missing 'results' list")

    scores = eval(cleaned_results)

    payload.setdefault("stats", {})["scores"] = scores

    if args.overwrite:
        output_path = input_path
    else:
        if args.output is not None:
            output_path = args.output
        else:
            output_path = input_path.with_name(f"{input_path.stem}_reeval.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=4)

    print(f"[INFO] Re-evaluated metrics saved to {output_path}")


if __name__ == "__main__":
    main_cli()