"""
CostBench Runtime - VLLM OpenAI Compatible Server Mode

Run concurrent inference against an OpenAI-compatible server launched with
`vllm serve`.
"""

import random
import hashlib
import sys
import os
from tqdm import tqdm
import json
import requests
import asyncio
import argparse
import time
import uuid
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
from concurrent.futures import ThreadPoolExecutor
import threading
from tqdm import tqdm
# import dns.resolver
import httpx

# Add the project root directory to the import path
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, project_root)

# OpenAI client
from openai import AsyncOpenAI

# Project modules
from env.utils.blocker import create_blocker
from env.utils.solver import GroundtruthSolver
from env.utils.id_generator import get_global_generator
from env.domains.travel.tool_registry import tools_ready, tools_ready_in_memory
from env.core.base_types import Tool, create_tool_from_dict
from env.core.data_types import get_final_type
from env.utils.solver import solve_from_json, solve_single_query_gt
from env.utils.blocker import DynamicBlocker
from env.utils.eval import evaluate_single_run
from env.utils.eval import build_segmented_gt_path, calculate_edit_distance
from env.utils.example_generator import example_for_refinement_level
from env.domains.travel.refinement_steps_catalog import (
    get_refinement_dimensions,
    get_atomic_sequence,
)
from env.settings import load_config, resolve_model_credentials

CONFIG = load_config()
SEED_RANGE = CONFIG.random.seed_range

# Stable hash to keep results consistent across processes/environments
def _stable_int(s: str) -> int:
    return int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % (2**31 - 1)


def load_tools_from_json(args, tool_path) -> Dict[str, Tool]:
    # Load the JSON and construct Tool objects via create_tool_from_dict
    tool_file_path = os.path.join(tool_path, f"tools_ready_refine_{args.refinement_level}.json")
    with open(tool_file_path, "r", encoding="utf-8") as f:
        tool_dicts = json.load(f).get("tools", [])
    if not tool_dicts:
        raise ValueError(f"No tools found in {tool_file_path}")

    all_tools = {tool_name: create_tool_from_dict(tool_dict) for tool_name, tool_dict in tool_dicts.items()}
    return all_tools


def get_model_info(args) -> Dict[str, Any]:
    model_name = getattr(args, "model_name", None)
    if model_name is None:
        raise ValueError("model_name must be provided in args")

    return resolve_model_credentials(model_name)


def get_env_ready(args) -> Dict[str, Any]:
    
    # Initialize the id generator
    id_generator = get_global_generator()
    
    # load queries
    all_queries = []
    with open(args.query_path, "r", encoding="utf-8") as f:
        all_queries = json.load(f)
    if not all_queries:
        raise ValueError(f"[ERROR] No queries found in {args.query_path}")

    # load model
    client = AsyncOpenAI(
        base_url=get_model_info(args)["base_url"],
        api_key=get_model_info(args)["api_key"],
        timeout=1800, 
    )
    
    return {
        "id_generator": id_generator,
        "client": client,
        "queries": all_queries,
    }
    
    
def get_refinement_dimensions_str(args) -> str:
    
    dimensions = get_refinement_dimensions(args.refinement_level)
    return ", ".join([dimensions[i] for i in range(1, args.refinement_level + 1) if i in dimensions])

def get_tool_types_str(args) -> str:
    
    return "Deciding Preference, Searching Candidates, Refining Options, Final Recommendation" if args.refinement_level != 0 else "Deciding Preference, Searching Candidates, Final Recommendation"

def get_tool_num_str(args) -> str:
    return NUM_TOOL_TYPES if args.refinement_level != 0 else NUM_TOOL_TYPES - 1

def get_atomic_tool_sequence_str(args, task: str) -> str:
    return get_atomic_sequence(args.refinement_level, task)

async def run_single_query(
    client: AsyncOpenAI,
    query_dict: Dict[str, Any], 
    args,
    query_index: int,
    tool_creation_seed: int
) -> Dict[str, Any]:
    """
    Full execution flow for handling a single query.
    
    Args:
        client (AsyncOpenAI): OpenAI client instance
        query_dict (Dict[str, Any]): Dictionary containing the query payload
        args: Command-line arguments
        query_index (int): Index of the query within the list
        tool_creation_seed (int): Random seed used for tool generation
        
    Returns:
        Dict[str, Any]: Query dictionary enriched with execution results
    """
    task = query_dict.get("task")
    query_id = query_dict["query_id"]
    query_dict["tool_creation_seed"] = tool_creation_seed

    
    # 0. Initialize the solver, blocker, and tools (supports in-memory and file modes)
    #    and prepare tool snapshots used by stimulation (initial + cost_change updates)
    if getattr(args, "tool_mode", "memory") == "memory":
        # In-memory mode: build the tool dictionary and instantiate directly
        tools_dict_in_memory = tools_ready_in_memory(
            refinement_level=args.refinement_level,
            min_atomic_cost=args.min_atomic_cost,
            max_atomic_cost=args.max_atomic_cost,
            noise_std=args.noise_std,
            random_seed=tool_creation_seed,
            control_tool_length=getattr(args, "control_tool_length", False),
            max_tool_length=getattr(args, "max_tool_length", CONFIG.tool_defaults.max_tool_length),
            ban_longest_tool=getattr(args, "ban_longest_tool", False)
        )
        init_tools = {
            tool_name: create_tool_from_dict({**tool_dict, "name": tool_name})
            for tool_name, tool_dict in tools_dict_in_memory.get("tools", {}).items()
        }
        # Initial tool snapshot (dict form) for stimulation
        tool_snapshots_for_simulation: List[Dict[str, Any]] = [tools_dict_in_memory]
    else:
        # File mode: generate to disk and read back
        unique_tool_dir = os.path.join(
            args.tool_output_dir,
            str(uuid.uuid4()),
            f"query_{query_id.replace('<','').replace('>','')}_{tool_creation_seed}"
        )
        os.makedirs(unique_tool_dir, exist_ok=True)
        tools_ready(
            output_dir=unique_tool_dir,
            refinement_level=args.refinement_level,
            min_atomic_cost=args.min_atomic_cost,
            max_atomic_cost=args.max_atomic_cost,
            noise_std=args.noise_std,
            random_seed=tool_creation_seed,
            control_tool_length=getattr(args, "control_tool_length", False),
            max_tool_length=getattr(args, "max_tool_length", CONFIG.tool_defaults.max_tool_length),
            ban_longest_tool=getattr(args, "ban_longest_tool", False)
        )
        init_tools = load_tools_from_json(args, unique_tool_dir)
        # In file mode, convert Tool objects back to dict snapshots
        init_tools_snapshot_dict = {
            "tools": {name: tool.to_dict() for name, tool in init_tools.items()}
        }
        tool_snapshots_for_simulation: List[Dict[str, Any]] = [init_tools_snapshot_dict]
    
    # Source-level filtering is already handled in tools_ready_*, so no extra length filters here

    # Solve the initial ground truth using the in-memory tool objects
    from env.utils.solver import solve_single_query_gt
    query_dict = solve_single_query_gt(query_dict, init_tools)

    solver = GroundtruthSolver(
        tools_with_costs=init_tools,
    )

    # Initialize the blocker
    if args.use_blocker:
        blocker = create_blocker(
            block_mode=args.block_mode,
            block_num=args.block_num,
            seed_range=SEED_RANGE,
            refinement_level=args.refinement_level,
            output_dir=args.changed_tool_output_dir,
            queries_path=args.query_path,
            # query_dict=query_dict
        )
        blocker.set_solver(solver)
        # Pass through filtering and cost-control options for reuse during cost_change
        try:
            blocker.filter_options = {
                "control_tool_length": getattr(args, "control_tool_length", CONFIG.tool_defaults.control_tool_length),
                "max_tool_length": getattr(args, "max_tool_length", CONFIG.tool_defaults.max_tool_length),
                "ban_longest_tool": getattr(args, "ban_longest_tool", CONFIG.tool_defaults.ban_longest_tool),
                "min_atomic_cost": getattr(args, "min_atomic_cost", CONFIG.tool_defaults.min_atomic_cost),
                "max_atomic_cost": getattr(args, "max_atomic_cost", CONFIG.tool_defaults.max_atomic_cost),
                "noise_std": getattr(args, "noise_std", CONFIG.tool_defaults.noise_std),
            }
        except Exception:
            pass
    else:
        blocker = None
    
    # 1. Initialize the query state
    result_query = query_dict.copy()
    conversation_history = []
    current_step = 0
    max_steps = args.max_tool_steps
    
    # 2. Initialize blocker-specific state if enabled
    if args.use_blocker:
        # Build the initial blocking plan
        result_query = blocker.initialize_blocking_plan(result_query)
        blocker.set_initial_tools(init_tools)
        blocker.set_current_state(solver._infer_initial_state(task))   
        # Explicitly initialize the GT state to avoid lazy initialization logs
        try:
            blocker.initialize_gt_state(task)
        except Exception:
            pass
    
    # require_goal_state: goal type and the no-blocker state set
    final_type = get_final_type(task) if task else None
    # Maintain a separate non-blocker state set because blocker owns the state when enabled
    current_state_no_blocker = None
    if not args.use_blocker:
        try:
            current_state_no_blocker = set(solver._infer_initial_state(task)) if task else set()
        except Exception:
            current_state_no_blocker = set()
    
    # 3. Build the initial system prompt and user message
    EXAMPLE_CONTENT = example_for_refinement_level(args.refinement_level)
    # if args.provide_atomic_tool_sequence:
    #     example_content_str = EXAMPLE_CONTENT.format(
    #             example_atomic_tool_sequence=EXAMPLE_ATOMIC_TOOL_SEQUENCE
    #         )
    # else:
    #     example_content_str = EXAMPLE_CONTENT.format(example_atomic_tool_sequence="")
    example_content = "" if not args.use_example else EXAMPLE_CONTENT
    task_atomic_tool_sequence_str = TASK_ATOMIC_TOOL_SEQUENCE_TEMPLATE.format(
        atomic_tool_sequence=get_atomic_tool_sequence_str(args=args, task=task.capitalize())
    ) if args.provide_atomic_tool_sequence else ""
    if args.refinement_level != 0:
        refinement_dimensions_str=get_refinement_dimensions_str(args)
        refinement_content = REFINEMENT_CONTENT_TEMPLATE.format(
            task=task.capitalize(),
            num_dimensions=args.refinement_level,
            refinement_dimensions_str=refinement_dimensions_str,
        )
        system_prompt = QUERY_INSTRUCTION.format(
            task=task.capitalize(),
            tool_num_str=get_tool_num_str(args),
            tool_types_str=get_tool_types_str(args),
            refinement_content=refinement_content,
            example_content=example_content,
            composite_concept_content=COMPOSITE_CONCEPT_CONTENT if getattr(args, "provide_composite_concept", False) else "",
            task_atomic_tool_sequence_str=task_atomic_tool_sequence_str,
            need_loc_pref="LocationPreference, " if task.lower() != "location" else ""
        )
    else:
        system_prompt = QUERY_INSTRUCTION.format(
            task=task.capitalize(),
            tool_num_str=get_tool_num_str(args),
            tool_types_str=get_tool_types_str(args),
            refinement_content="",
            example_content=example_content,
            composite_concept_content=COMPOSITE_CONCEPT_CONTENT if getattr(args, "provide_composite_concept", False) else "",
            task_atomic_tool_sequence_str=task_atomic_tool_sequence_str,
            need_loc_pref="LocationPreference, " if task.lower() != "location" else ""
        )
    user_message = query_dict["query"]["input"]
    
    conversation_history.append({"role": "system", "content": system_prompt})
    conversation_history.append({"role": "user", "content": user_message})
    
    # 4. Run the dialogue loop
    execution_log = []
    agent_path: List[str] = []
    
    while current_step < max_steps:
        step_start_time = time.time()
        
        # Check whether a pre-step ban_tool blocking should trigger
        pre_step_blocking_result = None
        if args.use_blocker:
            blocker.update_current_step(current_step + 1)  # Pre-check the upcoming step
            if blocker.should_trigger_blocking_with_query(current_step + 1, result_query):
                # Check whether the next blocking type is ban_tool
                if len(result_query["block_stats"]["metadata"]) <= blocker.blocked_count:
                    print(f"[WARNING] Query {query_dict.get('query_id', 'unknown')}, block_stats metadata length {len(result_query['block_stats']['metadata'])} is less than blocked_count {blocker.blocked_count}")
                    break
                next_block_type = result_query["block_stats"]["metadata"][blocker.blocked_count]["block_type"]
                if next_block_type == "ban_tool":
                    # Mark it for execution during the tool call
                    pre_step_blocking_result = "ban_tool_pending"
                
        try:
            # Exit early when require_goal_state is enabled and the goal type is already available
            if args.require_goal_state and final_type is not None:
                goal_in_state_now = False
                if args.use_blocker and blocker is not None:
                    goal_in_state_now = final_type in blocker.current_state
                elif current_state_no_blocker is not None:
                    goal_in_state_now = final_type in current_state_no_blocker
                if goal_in_state_now:
                    break

            # Send the request to the model
            tool_interface = [tool.generate_interface(provide_composite_concept=getattr(args, "provide_composite_concept", False)) for tool in _get_available_tools(args, init_tools, blocker, task)] # List[Dict[str, Any]]
            
            # Print the model-visible context when print_tool_interface is enabled
            if args.print_tool_interface:
                print(f"\n[DEBUG] ===== Model-side context (Query: {query_dict.get('query_id', 'Unknown')}, Step: {current_step + 1}) =====")
                print(f"[DEBUG] System Prompt (Instruction):")
                print(f"{conversation_history[0]['content']}")
                print(f"\n[DEBUG] User Message:")
                print(f"{conversation_history[1]['content']}")
                print(f"\n[DEBUG] Tool Interface ({len(tool_interface)} tools):")
                for i, tool in enumerate(tool_interface):
                    print(f"  Tool {i+1}: {tool.get('function', {}).get('name', 'Unknown')}")
                    print(f"    Cost: {tool["function"]["description"].split("This tool has a cost of ")[1].split(" units. ")[0]}")
                    print(f"    Description: {tool.get('function', {}).get('description', 'No description')}")
                    print(f"    Parameters: {json.dumps(tool.get('function', {}).get('parameters', {}), indent=6, ensure_ascii=False)}")
                print(f"[DEBUG] ===== End of model-side context =====\n")
            
            # Choose tool_choice based on whether the goal state has been achieved (force "required" only when needed)
            tool_choice_value = "auto"
            need_continue_tools = False
            if args.require_goal_state:
                if final_type is not None:
                    if args.use_blocker and blocker is not None:
                        need_continue_tools = final_type not in blocker.current_state
                    elif current_state_no_blocker is not None:
                        need_continue_tools = final_type not in current_state_no_blocker
            tool_choice_value = "required" if need_continue_tools else "auto"

            response = await client.chat.completions.create(
                model=args.model_name,
                messages=conversation_history,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                tools=tool_interface,
                tool_choice=tool_choice_value,
                parallel_tool_calls=False
            )
            # print(f"[DEBUG] response type: {type(response)}")
            # print(f"[DEBUG] response: {response}")
            
            assistant_message = response.choices[0].message
            conversation_history.append({
                "role": "assistant", 
                "content": assistant_message.content,
                "tool_calls": assistant_message.tool_calls
            })

            # Step executed successfully; update the counter
            current_step += 1
            
            # Prefer an answer-first strategy: stop if <answer> appears alongside tool calls
            should_terminate = False
            if assistant_message.content and "<answer>" in assistant_message.content and "</answer>" in assistant_message.content:
                answer = assistant_message.content.split("<answer>")[1].split("</answer>")[0]
                num_digits = len([ch for ch in answer if ch.isdigit()])
                if num_digits == ID_LENGTH:
                    should_terminate = True
            if should_terminate:
                execution_log.append({
                    "step": current_step,
                    "tool_calls": [tc.function.name for tc in (assistant_message.tool_calls or [])],
                    "blocking_result": None,
                    "success": True,
                    "execution_time": time.time() - step_start_time,
                    "note": "Final answer present with tool_calls; prioritized answer and terminated"
                })
                break

            # Handle tool invocations
            if assistant_message.tool_calls:
                tool_results = []
                blocking_result = None

                tool_calls_list = assistant_message.tool_calls or []
                if len(tool_calls_list) > 1:
                    print(f"[WARNING] Multiple tool calls detected at step {current_step}, only the first will be executed; others will receive error tool messages to satisfy API constraints.")

                # Respond to each tool call; execute the first and return placeholders for the rest
                tool_call = tool_calls_list[0]
                tool_name = tool_call.function.name

                # Visualization: print visible tools and GT alignment at the start of each step (prefixed with [AGENT])
                if getattr(args, "vis_agent", False):
                    try:
                        available_tools_for_vis = _get_available_tools(args, init_tools, blocker, task)
                        # Filter out tools currently removed by the blocker
                        visible_tools_map = {t.name: t for t in available_tools_for_vis}
                        visible_str = ", ".join([
                            f"{name}(c={getattr(t, 'cost', 0)})" for name, t in sorted(visible_tools_map.items(), key=lambda kv: kv[0])
                        ]) if visible_tools_map else "<none>"
                        # Compute the corresponding GT step name
                        try:
                            scenarios_vis = result_query.get("groundtruth", {}).get("scenarios", []) or []
                            block_steps_vis = result_query.get("block_stats", {}).get("block_step", []) or []
                            gt_tool_path_vis, _ = build_segmented_gt_path(scenarios_vis, block_steps_vis)
                            gt_step_name = gt_tool_path_vis[current_step - 1] if (current_step - 1) < len(gt_tool_path_vis) else "<none>"
                        except Exception:
                            gt_step_name = "<none>"
                        print(f"[AGENT] Query {query_id} step {current_step} \n visible: {visible_str} | visible_count: {len(visible_tools_map)} | gt: {gt_step_name}")
                        print("\n\n")
                    except Exception:
                        pass
                
                # Check whether ban_tool blocking should trigger
                if (pre_step_blocking_result == "ban_tool_pending" and 
                    args.use_blocker and 
                    blocker.should_trigger_blocking_with_query(current_step, result_query)):
                    
                    # Execute the ban_tool blocking branch
                    blocking_result = blocker.execute_blocking(
                        query_dict=result_query, 
                        tool_call_request={"tool_name": tool_name}
                    )
                    # Use a local RNG to pick a response sentence to keep behavior reproducible per query/step
                    _msg_rng_seed = _stable_int(f"{tool_creation_seed}:{current_step}:ban_tool_message")
                    _msg_rng = random.Random(_msg_rng_seed)
                    try: 
                        tool_obj = _find_tool_by_name(init_tools, tool_name)
                    except Exception:
                        raise Exception(f"Failed to find tool '{tool_name}'")
                    tool_result = {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": tool_name,
                        "content": _msg_rng.choice(BAN_TOOL_RETURN_SENTENCES) + f"{(' Cost: ' + str(tool_obj.cost)) if tool_obj is not None and tool_obj.has_cost() else ''}"
                    }
                                        
                # Check whether the tool call is already blocked (from an earlier ban_tool)
                elif args.use_blocker and blocker.should_tool_call_fail(tool_name):
                    # Retrieve the return_message_seed associated with the current blocking
                    return_message_seed = tool_creation_seed  # Default value
                    if "blocking_plan" in result_query:
                        # Locate the corresponding ban_tool blocking entry
                        for bp in result_query["blocking_plan"]:
                            if bp["type"] == "ban_tool" and "return_message_seed" in bp["parameters"]:
                                return_message_seed = bp["parameters"]["return_message_seed"]
                                break
                    
                    _msg_rng = random.Random(return_message_seed)
                    try: 
                        tool_obj = _find_tool_by_name(init_tools, tool_name)
                    except Exception:
                        raise Exception(f"Failed to find tool '{tool_name}'")
                    tool_result = {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": tool_name,
                        "content": _msg_rng.choice(BAN_TOOL_RETURN_SENTENCES) + f" Cost: {tool_obj.cost}." if tool_obj is not None and tool_obj.has_cost() else ""
                    }
                                        
                else:
                    # Execute the tool normally
                    try:
                        available_tools = _get_available_tools(args, init_tools, blocker, task) # List[Tool]
                        available_tools_dict = {tool.name: tool for tool in available_tools} # Dict[str, Tool]
                        tool_obj = _find_tool_by_name(available_tools_dict, tool_name)
                        if tool_obj:
                            # Pre-check for reachability: ensure all required input types are present
                            precheck_failed = False
                            required_types = set(tool_obj.input_types)
                            current_types = set()
                            if args.use_blocker and blocker is not None:
                                current_types = set(blocker.current_state)
                            else:
                                if current_state_no_blocker is None:
                                    current_types = set()
                                else:
                                    current_types = set(current_state_no_blocker)
                            if not required_types.issubset(current_types):
                                precheck_failed = True

                            if precheck_failed:
                                tool_result = {
                                    "tool_call_id": tool_call.id,
                                    "role": "tool",
                                    "name": tool_name,
                                    "content": (
                                        f"[ERROR] Missing required input types for tool '{tool_name}'. "
                                        f"Required: {sorted(list(required_types))}, Current: {sorted(list(current_types))}"
                                        f" Cost: {tool_obj.cost}." if tool_obj is not None and tool_obj.has_cost() else ""
                                    )
                                }
                            else:
                                # Simulate tool execution
                                tool_output = _execute_tool_simulation(
                                    tool_obj=tool_obj,
                                    arguments_str=tool_call.function.arguments, 
                                    refinement_level=args.refinement_level,
                                    available_tools=available_tools_dict  # Provide a dict instead of a list
                                )

                                # Update state only when execution succeeds
                                is_success = True
                                if isinstance(tool_output, str):
                                    if tool_output.startswith("[ERROR]") or tool_output.startswith("Tool execution failed") or tool_output.startswith("Error:"):
                                        is_success = False

                                if is_success:
                                    if args.use_blocker and blocker is not None:
                                        new_state = blocker.current_state | {tool_obj.output_type}
                                        blocker.set_current_state(new_state)
                                    else:
                                        if current_state_no_blocker is None:
                                            current_state_no_blocker = set()
                                        current_state_no_blocker = current_state_no_blocker | {tool_obj.output_type}

                                tool_result = {
                                    "tool_call_id": tool_call.id,
                                    "role": "tool", 
                                    "name": tool_name,
                                    "content": tool_output if ("Cost:" in (tool_output or "")) else (tool_output + f" Cost: {tool_obj.cost}")
                                }
                        else:
                            tool_result = {
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": tool_name, 
                                "content": f"[ERROR] You have requested an invalid tool '{tool_name}'. Note that you should call the exact name of the tool as specified in the tool list."
                            }
                            
                    except Exception as e:
                        tool_result = {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": tool_name,
                            "content": f"[ERROR] Error executing tool: {str(e)}"
                        }
                
                tool_results.append(tool_result)
                # Record the agent path
                try:
                    agent_path.append(tool_name)
                    if getattr(args, "vis_agent", False):
                        step_cost_dbg = None
                        try:
                            step_cost_dbg = getattr(tool_obj, "cost", None)
                        except Exception:
                            step_cost_dbg = None
                        if pre_step_blocking_result == "ban_tool_pending":
                            print(f"[AGENT] Query {query_id} step {current_step} choose: {tool_name} (cost={step_cost_dbg}) [ban_tool]")
                        else:
                            print(f"[AGENT] Query {query_id} step {current_step} choose: {tool_name} (cost={step_cost_dbg})")
                        print("\n\n")
                except Exception:
                    pass

                # Add placeholder error messages for additional tool calls
                if len(tool_calls_list) > 1:
                    for extra_call in tool_calls_list[1:]:
                        tool_results.append({
                            "tool_call_id": extra_call.id,
                            "role": "tool",
                            "name": extra_call.function.name,
                            "content": "[ERROR] Multiple tool calls in one step are not allowed. Only the first tool call was executed. Please call one tool at a time."
                        })

                # Append all tool results to the conversation history
                conversation_history.extend(tool_results)
                
                # Record execution metadata
                execution_log.append({
                    "step": current_step,
                    "tool_calls": [tc.function.name for tc in assistant_message.tool_calls],
                    "blocking_result": blocking_result,
                    "success": blocking_result is None,
                    "execution_time": time.time() - step_start_time
                })
                
                # Check whether any additional blocking should trigger after this step
                post_step_blocking_result = None
                conversation_will_continue = True  # Assume the dialogue continues
                
                if args.use_blocker and blocking_result is None:  # Only check when no blocking has occurred in this step
                    # Determine whether a non-ban_tool blocking needs to trigger
                    if blocker.should_trigger_blocking_with_query(current_step, result_query):
                        # Inspect the blocking type
                        if len(result_query["block_stats"]["metadata"]) <= blocker.blocked_count:
                            print(f"[WARNING] Query {query_dict.get('query_id', 'unknown')}, block_stats metadata length {len(result_query['block_stats']['metadata'])} is less than blocked_count {blocker.blocked_count}")
                            break
                        next_block_metadata = result_query["block_stats"]["metadata"][blocker.blocked_count]
                        next_block_type = next_block_metadata["block_type"]
                        
                        # Only non-ban_tool blocking types trigger after the step completes
                        if next_block_type != "ban_tool":
                            # Check whether the conversation will continue (by inspecting for a final answer)
                            if assistant_message.content and "<answer>" in assistant_message.content and "</answer>" in assistant_message.content:
                                answer = assistant_message.content.split("<answer>")[1].split("</answer>")[0]
                                num_digits = len([ch for ch in answer if ch.isdigit()])
                                if num_digits == ID_LENGTH:
                                    conversation_will_continue = False
                            
                            if conversation_will_continue:
                                post_step_blocking_result = blocker.execute_blocking(
                                    query_dict=result_query,
                                    tool_call_request=None
                                )
                                
                                # Update the execution log
                                execution_log[-1]["post_blocking_result"] = post_step_blocking_result
                            else:
                                # The conversation has ended; this blocking is ineffective and should not increase blocked_count
                                # Create a placeholder result to record the skipped blocking without incrementing the counter
                                post_step_blocking_result = {
                                    "type": next_block_type, 
                                    "status": "skipped_conversation_ended",
                                    "message": f"{next_block_type} blocking skipped because conversation ended with final answer"
                                }
                                execution_log[-1]["post_blocking_result"] = post_step_blocking_result
                                # Important: do not increment blocked_count because the blocking never took effect
                                # blocker.blocked_count and result_query["block_stats"]["block_count"] remain unchanged
                
                # If a cost_change occurred, refresh init_tools with the updated costs
                final_blocking_result = blocking_result or post_step_blocking_result
                if (final_blocking_result and 
                    final_blocking_result.get("type") == "cost_change" and
                    final_blocking_result.get("status") == "cost_changed"):
                    
                    # Pull the updated tools from the blocker
                    updated_tools_dict = blocker.get_current_tools().get("tools", {})
                    if updated_tools_dict:
                        # Rebuild Tool objects and refresh init_tools (reusing create_tool_from_dict)
                        init_tools = {
                            tool_name: create_tool_from_dict({**tool_dict, "name": tool_name})
                            for tool_name, tool_dict in updated_tools_dict.items()
                        }
                        # Source-level filtering already happened in tools_ready_*, so skip redundant filtering here
                # Append a tool snapshot for stimulation using the blocker's current tool state
                try:
                    tool_snapshots_for_simulation.append(blocker.get_current_tools())
                except Exception:
                    pass
                
                # If a preference change occurred, append a new user message
                if final_blocking_result and final_blocking_result.get("status") == "preference_changed":
                    pref_loc = final_blocking_result.get("new_location_preference") if task.lower() != "location" else ""
                        
                    location_pref_str = (f"This time I want to go to {pref_loc}.") if (task and task.lower() != "location" and pref_loc) else ""
                    new_user_message = f"{location_pref_str}" + final_blocking_result.get("new_user_requirement", "") + f" The previously provided TimeInfo information is still valid."
                    if new_user_message:
                        _msg_rng_seed_pref = _stable_int(f"{tool_creation_seed}:{current_step}:preference_change_message")
                        _msg_rng_pref = random.Random(_msg_rng_seed_pref)
                        new_user_message_template = _msg_rng_pref.choice(PREFERENCE_CHANGE_USER_MESSAGE_TEMPLATES)
                        conversation_history.append({
                            "role": "user",
                            "content": new_user_message_template.format(new_user_message=new_user_message)
                        })
                    else:
                        print(f"[WARNING] Query {query_dict.get('query_id', 'unknown')}, new_user_message is empty")
                    # After a preference change, reset to the initial state when no blocker is used to avoid stale reachability state
                    if not args.use_blocker:
                        try:
                            current_state_no_blocker = set(solver._infer_initial_state(task)) if task else set()
                        except Exception:
                            current_state_no_blocker = set()

                # In require_goal_state mode, if the goal is achieved and no preference change occurred,
                # append one final text-only assistant request in the same round (tool_choice='none') and then stop
                if args.require_goal_state and final_type is not None:
                    goal_in_state_after = False
                    if args.use_blocker and blocker is not None:
                        goal_in_state_after = final_type in blocker.current_state
                    elif current_state_no_blocker is not None:
                        goal_in_state_after = final_type in current_state_no_blocker
                    if goal_in_state_after and not (final_blocking_result and final_blocking_result.get("status") == "preference_changed"):
                        try:
                            tool_interface_final = [tool.generate_interface(provide_composite_concept=getattr(args, "provide_composite_concept", False)) for tool in _get_available_tools(args, init_tools, blocker, task)]
                            
                            # Print the model-visible context for the final response when print_tool_interface is enabled
                            if args.print_tool_interface:
                                print(f"\n[DEBUG] ===== Final response model-side context (Query: {query_dict.get('query_id', 'Unknown')}, Final Step) =====")
                                print(f"[DEBUG] System Prompt (Instruction):")
                                print(f"{conversation_history[0]['content']}")
                                print(f"\n[DEBUG] Current Conversation History ({len(conversation_history)} messages):")
                                for idx, msg in enumerate(conversation_history):
                                    role = msg.get('role', 'unknown')
                                    content = msg.get('content', '')[:200] + '...' if len(msg.get('content', '')) > 200 else msg.get('content', '')
                                    print(f"  Message {idx+1} ({role}): {content}")
                                print(f"\n[DEBUG] Tool Interface ({len(tool_interface_final)} tools):")
                                for i, tool in enumerate(tool_interface_final):
                                    print(f"  Tool {i+1}: {tool.get('function', {}).get('name', 'Unknown')}")
                                    print(f"    Description: {tool.get('function', {}).get('description', 'No description')}")
                                print(f"[DEBUG] Tool Choice: none (final response)")
                                print(f"[DEBUG] ===== End of final response model-side context =====\n")
                            
                            final_response = await client.chat.completions.create(
                                model=args.model_name,
                                messages=conversation_history,
                                temperature=args.temperature,
                                max_tokens=args.max_tokens,
                                tools=tool_interface_final,
                                tool_choice="none",
                                parallel_tool_calls=False
                            )
                            final_assistant_message = final_response.choices[0].message
                            conversation_history.append({
                                "role": "assistant",
                                "content": final_assistant_message.content,
                                "tool_calls": final_assistant_message.tool_calls
                            })
                            execution_log.append({
                                "step": current_step,
                                # Record tool calls (if any) from the final text-only response
                                "tool_calls": [tc.function.name for tc in (final_assistant_message.tool_calls or [])],
                                "blocking_result": final_blocking_result,
                                "success": True,
                                "execution_time": time.time() - step_start_time,
                                "note": "Goal reached; issued final text-only assistant response and terminated"
                            })
                        except Exception as _e:
                            # Terminate even if the final text-only response fails to send
                            execution_log.append({
                                "step": current_step,
                                "tool_calls": [tc.function.name for tc in assistant_message.tool_calls],
                                "blocking_result": final_blocking_result,
                                "success": False,
                                "execution_time": time.time() - step_start_time,
                                "error": str(_e),
                                "note": "Goal reached; attempted final text-only response but failed, terminating anyway"
                            })
                        break
                
            else:
                if assistant_message.content and "<answer>" in assistant_message.content:
                    # The model provided a final answer; end the dialogue
                    execution_log.append({
                        "step": current_step,
                        "tool_calls": [],
                        "blocking_result": None,
                        "success": True,
                        "execution_time": time.time() - step_start_time,
                        "note": "No tool calls - final answer"
                    })
                    break
                else:
                    # The model neither called a tool nor provided an answer
                    if args.require_goal_state:
                        # In require_goal_state mode, continue attempting (enforced by tool_choice="required")
                        continue
                    else:
                        # Preserve the original behavior: end the loop to avoid infinite retries
                        break
                
        except Exception as e:
            execution_log.append({
                "step": current_step,
                "error": str(e),
                "success": False,
                "execution_time": time.time() - step_start_time
            })
            break
    
    # 5. Assemble the results and evaluation metadata
    # Visualization: print the agent path at the end
    if getattr(args, "vis_agent", False):
        try:
            print(f"[AGENT] Query {query_id} path: " + (" -> ".join(agent_path) if agent_path else "<empty>"))
            print("\n\n")
        except Exception:
            pass
    # Record whether the goal state was reached (regardless of require_goal_state)
    is_goal_state = False
    try:
        if final_type is not None:
            if args.use_blocker and blocker is not None:
                is_goal_state = final_type in blocker.current_state
            else:
                if current_state_no_blocker is not None:
                    is_goal_state = final_type in current_state_no_blocker
    except Exception:
        is_goal_state = False
    result_query.update({
        # "execution_log": execution_log,
        "conversation_history": conversation_history,
        "total_steps": current_step,
        "status": "completed" if current_step < max_steps else "max_steps_reached",
        "is_goal_state": is_goal_state
    })
    
    # Visualization: print the GT path (same style as stimulation, prefixed with [GT])
    if getattr(args, "vis_gt", False):
        try:
            scenarios_vis = result_query.get("groundtruth", {}).get("scenarios", []) or []
            block_steps_vis = result_query.get("block_stats", {}).get("block_step", []) or []
            gt_tool_path_vis, _ = build_segmented_gt_path(scenarios_vis, block_steps_vis)
            print(f"[GT] Query {query_id} path: " + (" -> ".join(gt_tool_path_vis) if gt_tool_path_vis else "<empty>"))
            print("\n\n")
        except Exception:
            pass
    
    # 6. Optional random-strategy stimulation (runs after Agent and GT finish)
    if getattr(args, "use_stimulation", False):
        try:
            stimulation_summary = _run_random_stimulation(
                query_dict=result_query,
                task=task,
                args=args,
                initial_tool_snapshots=tool_snapshots_for_simulation
            )
            result_query["stimulation"] = stimulation_summary
        except Exception as _stim_e:
            result_query["stimulation"] = {"error": str(_stim_e)}

    if args.use_blocker:
        # Post-processing: adjust ineffective blockings
        # If steplen_change/cost_change/preference_change triggers but the dialogue ends immediately after,
        # treat them as ineffective and subtract them from blocked_count
        effective_blocked_count = 0
        for i in range(blocker.blocked_count):
            if i < len(result_query["block_stats"]["metadata"]):
                block_metadata = result_query["block_stats"]["metadata"][i]
                block_type = block_metadata["block_type"]
                trigger_step = block_metadata["trigger_step"]
                
                # For steplen_change/cost_change/preference_change blockings, if the dialogue ends quickly afterwards,
                # treat them as ineffective
                if block_type in ["steplen_change", "cost_change", "preference_change"]:
                    # If the trigger step + 1 >= total steps, the blocking had no effect
                    if trigger_step + 1 >= current_step:
                        continue  # Skip this ineffective blocking
                
                # Count effective blockings
                effective_blocked_count += 1
        
        # Update blocked_count with the effective number
        blocker.blocked_count = effective_blocked_count
        # Keep the dual timeline counters in sync
        try:
            blocker.agent_blocking_index = effective_blocked_count
            blocker.gt_blocking_index = effective_blocked_count
        except Exception:
            pass
        
        result_query["blocking_status"] = blocker.get_blocking_status()

        # Record agent and GT blocking counts
        try:
            result_query["block_stats"]["block_count_agent"] = blocker.blocked_count
        except Exception:
            pass

        # If blocker is enabled and GT has remaining placeholders (None) without gt_unblocked, finalize GT blocking slots
        try:
            bs = result_query.get("block_stats", {}) or {}
            steps = bs.get("block_step", []) or []
            types = bs.get("block_type", []) or []
            md = bs.get("metadata", []) or []
            expected = int(bs.get("expected_block_count", getattr(args, "block_num", 0) or 0))
            has_none = any(x is None for x in steps) or any(x is None for x in types) or (len(md) < expected)
            gt_not_done = blocker.blocked_count < expected
            if (not bool(result_query.get("gt_unblocked", False))) and gt_not_done and has_none:
                blocker.finalize_gt(result_query)
        except Exception as _fin_e:
            pass

        # Set block_count to the final GT count and synchronize block_count_gt
        try:
            result_query["block_stats"]["block_count_gt"] = blocker.blocked_count
            result_query["block_stats"]["block_count"] = blocker.blocked_count
        except Exception:
            pass
        
        # Populate the evaluation field
        # The evaluation phase aligns with the agent blocking count when available, otherwise falls back to the GT count
        actual_block_count = result_query.get("block_stats", {}).get("block_count_agent", blocker.blocked_count)
        expected_block_count = blocker.block_num
        
        # Determine the evaluation target
        if actual_block_count == 0:
            # No blocking occurred; use the original scenario
            target_scenario_index = 0
        else:
            # Blocking occurred; use the last scenario after the applied blocking
            target_scenario_index = actual_block_count
        
        # Determine the expected gt_id
        expected_gt = ""
        if "groundtruth" in result_query and "scenarios" in result_query["groundtruth"]:
            scenarios = result_query["groundtruth"]["scenarios"]
            if target_scenario_index < len(scenarios):
                expected_gt = scenarios[target_scenario_index].get("gt_id", "")
        
        result_query["evaluation"] = {
            "expected_block_count": expected_block_count,
            "actual_block_count": actual_block_count,
            "target_scenario_index": target_scenario_index,
            "expected_gt": expected_gt,
            "actual_result": "",  # Filled during evaluation
            "success": False  # Computed during evaluation
        }
    
    return result_query


def _get_available_tools(args: Any, init_tools: Dict[str, Tool], blocker: DynamicBlocker, task: Optional[str] = None) -> List[Tool]:
    """Return the list of tools currently visible to the model."""
    if not blocker:
        available_tools = list(init_tools.values())
    else:
        # Ask the blocker for the visible tools
        available_tools = blocker.get_visible_tools(list(init_tools.values()))
    # available_tools: List[Tool]
    
    # If a task is provided, filter to tools related to that task
    if task:
        task_filtered_tools = []
        for tool in available_tools:
            # Check whether the tool name contains the capitalized task name
            task_cap = task.capitalize()
            if task_cap in tool.name:
                task_filtered_tools.append(tool)
        return task_filtered_tools
    else:
        print("[WARNING] No task specified, returning all available tools without filtering.")
    
    return available_tools


def _find_tool_by_name(tools: Dict[str, Tool], tool_name: str) -> Optional[Tool]:
    """Look up a tool by name."""
    return tools.get(tool_name)  


def _execute_tool_simulation(
    tool_obj: Tool,
    arguments_str: str,
    refinement_level: int,
    available_tools: Dict[str, Tool]
) -> str:
    """Simulate executing a tool."""
    if tool_obj.name not in available_tools:
        raise ValueError(f"Tool '{tool_obj.name}' is not available for execution.")
    
    try:
        # Process the inputs
        tool_inputs = tool_obj.process_input(arguments_str)

        # Simulate execution
        output = tool_obj.execute(
            input_data=tool_inputs, 
            available_tools=available_tools
        )

        # Process the output
        formatted_output = tool_obj.process_output(
            tool_output=output, 
            refinement_level=refinement_level, 
            available_tools=available_tools
        )
        
        # Append cost information
        formatted_output += (" Cost: " + str(tool_obj.cost)) if (tool_obj is not None and tool_obj.has_cost()) else ""
        if "Cost: " not in formatted_output:
            raise ValueError(f"Tool '{tool_obj.name}' returned invalid output (No cost): {formatted_output}")
            
        return formatted_output
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Tool execution failed: {str(e)}"


def _build_tool_objects_from_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Tool]:
    """Convert a tool snapshot (dict) back into Tool objects."""
    from env.core.base_types import create_tool_from_dict
    tools_dict = snapshot.get("tools", {}) or {}
    tool_objs: Dict[str, Tool] = {}
    for name, td in tools_dict.items():
        td_copy = dict(td)
        td_copy["name"] = name
        tool_objs[name] = create_tool_from_dict(td_copy)
    return tool_objs


def _filter_task_tools(tool_objs: Dict[str, Tool], task: Optional[str]) -> Dict[str, Tool]:
    if not task:
        return tool_objs
    task_cap = task.capitalize()
    return {name: t for name, t in tool_objs.items() if task_cap in name}


def _is_composite_tool(tool_dict: Dict[str, Any]) -> bool:
    return (tool_dict or {}).get("type") == "composite"


def _component_count(tool_dict: Dict[str, Any]) -> int:
    if (tool_dict or {}).get("type") == "composite":
        try:
            return int(tool_dict.get("component_count", 1))
        except Exception:
            return 1
    return 1


def _run_random_stimulation(
    query_dict: Dict[str, Any],
    task: Optional[str],
    args,
    initial_tool_snapshots: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Run random-strategy simulations for a single query and aggregate metrics."""
    # Basic metadata
    scenarios = query_dict.get("groundtruth", {}).get("scenarios", []) or []
    block_stats = query_dict.get("block_stats", {}) or {}
    block_steps = block_stats.get("block_step", []) or []
    metadata_list = block_stats.get("metadata", []) or []
    expected_block_count = int(block_stats.get("expected_block_count", getattr(args, "block_num", 0)))
    query_id = query_dict.get("query_id", "unknown")

    # Build the segmented GT path
    gt_tool_path, _gt_cost = build_segmented_gt_path(scenarios, block_steps)

    # Debugging: optionally print segmented GT path information when --vis_stimulation is enabled
    try:
        if getattr(args, "vis_stimulation", False):
            scenario_lengths = []
            for _sc in scenarios:
                tools_list = _sc.get("tools", []) or []
                scenario_lengths.append(len(tools_list))
            print(
                f"[DEBUG] Query {query_id} build_segmented_gt_path -> "
                f"block_steps={block_steps}, "
                f"scenario_lengths={scenario_lengths}, "
                f"gt_len={len(gt_tool_path)}, gt_path={gt_tool_path}"
            )
    except Exception:
        pass

    # Prepare the tool snapshots used for stimulation: initial snapshot plus each cost_change update
    snapshots: List[Dict[str, Any]] = []
    if initial_tool_snapshots and isinstance(initial_tool_snapshots, list):
        snapshots = list(initial_tool_snapshots)
    else:
        snapshots = []
    # Generate missing snapshots for cost_change entries in metadata as a fallback
    try:
        from env.domains.travel.tool_registry import tools_ready_in_memory
        for md in metadata_list:
            if md.get("block_type") == "cost_change":
                params = md.get("parameters", {})
                new_seed = params.get("new_random_seed")
                if new_seed is None:
                    continue
                # Skip generation if enough snapshots already exist; otherwise, append a new one
                # Duplicates are allowed and will be consumed sequentially when triggered
                new_tools = tools_ready_in_memory(
                    refinement_level=getattr(args, "refinement_level", None),
                    min_atomic_cost=params.get("min_atomic_cost", getattr(args, "min_atomic_cost", 19)),
                    max_atomic_cost=params.get("max_atomic_cost", getattr(args, "max_atomic_cost", 21)),
                    noise_std=params.get("noise_std", getattr(args, "noise_std", 0.1)),
                    random_seed=new_seed,
                    control_tool_length=params.get("control_tool_length", getattr(args, "control_tool_length", False)),
                    max_tool_length=params.get("max_tool_length", getattr(args, "max_tool_length", 8)),
                    ban_longest_tool=params.get("ban_longest_tool", getattr(args, "ban_longest_tool", False))
                )
                snapshots.append(new_tools)
    except Exception:
        pass

    # Initial state and goal type
    final_type = get_final_type(task) if task else None
    try:
        # Infer the initial state via GroundtruthSolver
        solver_tmp = GroundtruthSolver(tools_with_costs=_build_tool_objects_from_snapshot(snapshots[0])) if snapshots else None
        initial_state = set(solver_tmp._infer_initial_state(task)) if (solver_tmp and task) else set()
    except Exception:
        initial_state = set()

    # Aggregation containers
    total_costs: List[float] = []
    total_steps_list: List[int] = []
    success_flags: List[bool] = []
    blocked_hits: List[bool] = []  # Whether the blocking count equals args.block_num
    neds: List[float] = []
    edits: List[int] = []
    exact_match_any = False
    paths_record: List[List[str]] = []  # Record each stimulation path

    stim_runs = max(int(getattr(args, "stimulation_num", 0) or 0), 0)
    if stim_runs <= 0:
        return {
            "avg_tool_calls": 0.0,
            "has_exact_match": 0,
            "success_ratio": 0.0,
            "blocked_ratio": 0.0,
            "min_ned": 0.0,
            "avg_ned": 0.0,
            "min_cost": 0.0,
            "avg_cost": 0.0
        }

    # Stable randomness seeded by query + tool_creation_seed + run_index
    def _rng_for(i: int) -> random.Random:
        return random.Random(_stable_int(f"{query_id}:{getattr(args, 'tool_creation_seed', 0)}:stim:{i}"))

    for i in range(stim_runs):
        rng = _rng_for(i)

        # Initialize state
        state = set(initial_state)
        last_output = None
        just_blocked_cooldown = 0  # >0 relaxes the linear constraint for this step
        step = 0
        current_snapshot_idx = 0
        removed_tools: set = set()  # Tools hidden due to steplen or ban events
        sim_blocked_count = 0
        path: List[str] = []
        total_cost = 0.0

        # Pointer to the next blocking event
        next_block_idx = 0

        # Main loop
        while step < getattr(args, "max_tool_steps", 5):
            step += 1

            # Check whether a pre-step ban_tool blocking should trigger
            pre_step_ban = False
            if next_block_idx < len(metadata_list):
                md = metadata_list[next_block_idx]
                if md.get("block_type") == "ban_tool" and int(md.get("trigger_step", 0)) == step:
                    pre_step_ban = True

            # Build candidate tools (from the current snapshot, minus removed tools, filtered by task and reachability)
            if current_snapshot_idx >= len(snapshots):
                break
            tool_objs_all = _build_tool_objects_from_snapshot(snapshots[current_snapshot_idx])
            tool_objs_all = _filter_task_tools(tool_objs_all, task)

            # Filter out removed tools
            tool_objs_visible = {n: t for n, t in tool_objs_all.items() if n not in removed_tools}

            # Visualization: print visible tools and GT step at the start of each iteration
            if getattr(args, "vis_stimulation", False):
                try:
                    visible_str = ", ".join([
                        f"{name}(c={getattr(t, 'cost', 0)})" for name, t in sorted(tool_objs_visible.items(), key=lambda kv: kv[0])
                    ]) if tool_objs_visible else "<none>"
                    gt_step_name = gt_tool_path[step - 1] if (step - 1) < len(gt_tool_path) else "<none>"
                    print(f"[STIM] Query {query_id} run {i+1}/{stim_runs} step {step} \n visible: {visible_str} | visible_count: {len(tool_objs_visible)} | gt: {gt_step_name}")
                    print("\n\n")
                except Exception:
                    pass

            # Reachability + linear constraint
            candidates: List[Tool] = []
            for t in tool_objs_visible.values():
                req_types = set(getattr(t, "input_types", []) or [])
                if not req_types.issubset(state):
                    continue
                if last_output is not None and just_blocked_cooldown == 0:
                    # Linear constraint: must include the previous step's output
                    if last_output not in req_types:
                        continue
                candidates.append(t)

            if not candidates:
                break  # No available tools; exit early

            # Choose a candidate tool: random by default; --greedy selects the best cost efficiency (atomic: cost, composite: cost/component_count)
            if getattr(args, "greedy", False):
                def _tool_score(t: Tool) -> float:
                    try:
                        cost_val = float(getattr(t, "cost", 0.0) or 0.0)
                    except Exception:
                        cost_val = 0.0
                    tool_type = ""
                    try:
                        tool_type = str(t.get_tool_type())
                    except Exception:
                        tool_type = "atomic"
                    if tool_type == "composite":
                        comp_cnt = 1
                        try:
                            comp_cnt = int(getattr(t, "component_count", 1) or 1)
                        except Exception:
                            comp_cnt = 1
                        comp_cnt = max(1, comp_cnt)
                        return cost_val / float(comp_cnt)
                    else:
                        return cost_val
                # Greedy mode: pick the tool with the lowest score (cost or unit component cost)
                selected_tool: Tool = min(candidates, key=_tool_score)
            else:
                selected_tool: Tool = rng.choice(candidates)

            # Handle pre-step ban cases
            if pre_step_ban:
                # Count one blocking event
                sim_blocked_count += 1
                next_block_idx += 1
                # Failed call: pay the cost, keep the state unchanged, and mark the tool as removed
                step_cost = float(getattr(selected_tool, "cost", 0.0) or 0.0)
                total_cost += step_cost
                path.append(selected_tool.name)
                removed_tools.add(selected_tool.name)
                # Flag just_blocked (relax the linear constraint next step)
                just_blocked_cooldown = 1
                if getattr(args, "vis_stimulation", False):
                    try:
                        print(f"[STIM] Query {query_id} run {i+1}/{stim_runs} step {step} choose: {selected_tool.name} (cost={step_cost}, cum={total_cost}) [ban_tool]")
                        print("\n\n")
                    except Exception:
                        pass
            else:
                # Execute normally
                step_cost = float(getattr(selected_tool, "cost", 0.0) or 0.0)
                total_cost += step_cost
                path.append(selected_tool.name)
                try:
                    out_type = getattr(selected_tool, "output_type", None)
                    if out_type:
                        state.add(out_type)
                        last_output = out_type
                except Exception:
                    pass
                if getattr(args, "vis_stimulation", False):
                    try:
                        print(f"[STIM] Query {query_id} run {i+1}/{stim_runs} step {step} choose: {selected_tool.name} (cost={step_cost}, cum={total_cost})")
                        print("\n\n")
                    except Exception:
                        pass

            # Check whether the goal is reached
            goal_reached_now = bool(final_type and final_type in state)

            # Handle post-step blocking (steplen/cost_change/preference_change)
            if next_block_idx < len(metadata_list):
                md2 = metadata_list[next_block_idx]
                btype = md2.get("block_type")
                if btype != "ban_tool" and int(md2.get("trigger_step", 0)) == step:
                    if goal_reached_now:
                        # Dialogue already ended; skip this blocking without counting it
                        pass
                    else:
                        # Apply the blocking and increment the count
                        sim_blocked_count += 1
                        if btype == "steplen_change":
                            tr = md2.get("parameters", {}).get("target_range")
                            if isinstance(tr, (list, tuple)) and len(tr) == 2:
                                min_len, max_len = int(tr[0]), int(tr[1])
                                snap_dict = snapshots[current_snapshot_idx].get("tools", {}) if current_snapshot_idx < len(snapshots) else {}
                                for name, td in snap_dict.items():
                                    if _is_composite_tool(td):
                                        cc = _component_count(td)
                                        if not (min_len <= cc <= max_len):
                                            removed_tools.add(name)
                                if getattr(args, "vis_stimulation", False):
                                    try:
                                        print(f"[STIM] Query {query_id} run {i+1}/{stim_runs} step {step} block: steplen_change -> keep [{min_len},{max_len}] components; now_removed={len(removed_tools)})")
                                        print("\n\n")
                                    except Exception:
                                        pass
                        elif btype == "cost_change":
                            # Switch to the next snapshot
                            if current_snapshot_idx + 1 < len(snapshots):
                                current_snapshot_idx += 1
                                if getattr(args, "vis_stimulation", False):
                                    try:
                                        print(f"[STIM] Query {query_id} run {i+1}/{stim_runs} step {step} block: cost_change -> snapshot {current_snapshot_idx}")
                                        print("\n\n")
                                    except Exception:
                                        pass
                        elif btype == "preference_change":
                            # Reset state
                            state = set(initial_state)
                            last_output = None
                            if getattr(args, "vis_stimulation", False):
                                try:
                                    print(f"[STIM] Query {query_id} run {i+1}/{stim_runs} step {step} block: preference_change -> state reset")
                                    print("\n\n")
                                except Exception:
                                    pass
                        # Mark just_blocked
                        just_blocked_cooldown = 1
                    # Consume this blocking entry
                    next_block_idx += 1

            # Cool down just_blocked after the step
            if just_blocked_cooldown > 0:
                just_blocked_cooldown -= 1

            if goal_reached_now:
                break

        # Per-run summary
        if getattr(args, "vis_stimulation", False):
            try:
                print(f"[STIM] Query {query_id} run {i+1}/{stim_runs} path: " + (" -> ".join(path) if path else "<empty>"))
                print("\n\n")
            except Exception:
                pass
        total_costs.append(total_cost)
        total_steps_list.append(len(path))
        success_flags.append(bool(final_type and final_type in state))
        blocked_hits.append(sim_blocked_count == int(getattr(args, "block_num", expected_block_count) or 0))

        # NED/EM metrics (aligned with eval.py; NED defaults to 1 when the goal is not reached)
        try:
            edit = calculate_edit_distance(path, gt_tool_path)
            max_len = max(len(path), len(gt_tool_path))
            ned = (edit / max_len) if max_len > 0 else 0.0
            if not success_flags[-1]:
                ned = 1.0
            neds.append(float(ned))
            edits.append(int(edit))
            if path == gt_tool_path:
                exact_match_any = True
        except Exception:
            neds.append(1.0 if not success_flags[-1] else 0.0)
            try:
                edit_fallback = max(len(path), len(gt_tool_path)) if not success_flags[-1] else 0
            except Exception:
                edit_fallback = 0
            edits.append(int(edit_fallback))

        # Record this run's path
        try:
            paths_record.append(list(path))
        except Exception:
            pass

    # Aggregate results
    def _safe_avg(vals: List[float]) -> float:
        return (sum(vals) / len(vals)) if vals else 0.0

    return {
        "avg_tool_calls": _safe_avg([float(s) for s in total_steps_list]),
        "has_exact_match": 1 if exact_match_any else 0,
        "success_ratio": _safe_avg([1.0 if x else 0.0 for x in success_flags]),
        "blocked_ratio": _safe_avg([1.0 if x else 0.0 for x in blocked_hits]),
        "min_ned": (min(neds) if neds else 0.0),
        "avg_ned": _safe_avg(neds),
        "min_edit": (min(edits) if edits else 0),
        "avg_edit": _safe_avg([float(e) for e in edits]),
        "min_cost": (min(total_costs) if total_costs else 0.0),
        "avg_cost": _safe_avg(total_costs),
        "paths": paths_record,
        "runs": int(stim_runs)
    }


async def run_all_query(args, queries, id_generator, client):
    """
    Run all queries concurrently with a tqdm progress bar, limited by args.num_threads.
    """
    # Generate stable per-query seeds from the global seed + query_id to avoid ordering effects
    def per_query_seed(global_seed: int, qid: str) -> int:
        return _stable_int(f"{global_seed}:{qid}")

    # Create a semaphore to throttle concurrency
    semaphore = asyncio.Semaphore(args.num_threads)
    
    async def run_single_query_with_semaphore(client, query_dict, args, query_index, tool_creation_seed):
        """Wrapper that applies the semaphore for concurrency control."""
        async with semaphore:
            return await run_single_query(
                client=client,
                query_dict=query_dict,
                args=args,
                query_index=query_index,
                tool_creation_seed=tool_creation_seed
            )
    
    # Build the task list
    tasks = [
        run_single_query_with_semaphore(
            client=client,
            query_dict=query,
            args=args,
            query_index=i,
            tool_creation_seed=per_query_seed(args.tool_creation_seed, query.get("query_id", str(i)))
        )
        for i, query in enumerate(queries[args.start_index : args.end_index if args.end_index != -1 else None])
    ]
    
    results = []
    queries_with_results = []
    progress = tqdm(total=len(tasks), desc=f"Running queries (max {args.num_threads} concurrent): ")

    for coro in asyncio.as_completed(tasks):
        try:
            result = await coro
            results.append(result)
            queries_with_results.append(result)
        except Exception as e:
            idx = len(results)
            queries_with_results.append({
                "query_id": queries[idx].get("query_id", f"query_{idx}"),
                "error": str(e),
                "status": "failed"
            })
        finally:
            progress.update(1)

    progress.close()
    return queries_with_results
    
def save_show_results(args, queries_with_results, output_path):
    """
    Save results to disk and optionally display summary information.
    """

    # Clean up non-serializable objects
    cleaned_results = []
    for result in queries_with_results:
        cleaned_result = result.copy()
        
        # Remove non-serializable items inside conversation_history
        if "conversation_history" in cleaned_result:
            cleaned_history = []
            for msg in cleaned_result["conversation_history"]:
                cleaned_msg = {
                    "role": msg["role"],
                    "content": msg.get("content")
                }
                # Convert tool_calls into a simple structure when present
                if "tool_calls" in msg and msg["tool_calls"]:
                    if hasattr(msg["tool_calls"][0], 'function'):  # OpenAI response object
                        cleaned_msg["tool_calls"] = [
                            {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                            for tc in msg["tool_calls"]
                        ]
                    else:  # Already in dict form
                        cleaned_msg["tool_calls"] = msg["tool_calls"]
                cleaned_history.append(cleaned_msg)
            cleaned_result["conversation_history"] = cleaned_history
        
        cleaned_results.append(cleaned_result)
        
    # Evaluation
    from env.utils.eval import eval as eval_costbench, get_agent_final_answer, check_answer_accuracy
    # Attach key runtime cost parameters to assist evaluation (optional normalization)
    # To avoid large data copies, store a brief run_args summary per result
    run_args_summary = {
        "min_atomic_cost": args.min_atomic_cost,
        "max_atomic_cost": args.max_atomic_cost,
        "noise_std": args.noise_std,
        "refinement_level": args.refinement_level,
        "use_blocker": args.use_blocker,
    }
    for _item in cleaned_results:
        try:
            _item["run_args"] = run_args_summary
        except Exception:
            pass

    scores = eval_costbench(cleaned_results)

    # Populate evaluation.actual_result and evaluation.success for each query
    for item in cleaned_results:
        try:
            agent_answer = get_agent_final_answer(item.get("conversation_history", []))
        except Exception:
            agent_answer = None
        # Ensure the evaluation field exists
        eval_field = item.get("evaluation", {})
        eval_field["actual_result"] = agent_answer or ""
        try:
            # Compute success using the agent's blocking count
            actual_block_count = item.get("block_stats", {}).get("block_count_agent", item.get("block_stats", {}).get("block_count", 0))
            success_flag = check_answer_accuracy(agent_answer, item.get("groundtruth", {}).get("scenarios", []), actual_block_count)
        except Exception:
            success_flag = False
        eval_field["success"] = bool(success_flag)
        item["evaluation"] = eval_field

        # Record gt_unblocked consistent with eval.py for downstream analysis
        try:
            per_eval = evaluate_single_run(item)
            item["gt_unblocked"] = bool(per_eval.get("gt_unblocked", False))
        except Exception:
            pass
        
    final_results = {
        "stats": {
            "run_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "model_name": args.model_name,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "num_threads": args.num_threads,
            "query_path": args.query_path,
            "total_queries": len(queries_with_results),
            "use_blocker": args.use_blocker,
            "block_mode": args.block_mode if args.use_blocker else None,
            "block_num": args.block_num if args.use_blocker else 0,
            "refinement_level": args.refinement_level,
            "max_tool_steps": args.max_tool_steps,
            "tool_creation_seed": args.tool_creation_seed,
            "tool_output_dir": args.tool_output_dir,
            "changed_tool_output_dir": args.changed_tool_output_dir,
            "min_atomic_cost": args.min_atomic_cost,
            "max_atomic_cost": args.max_atomic_cost,
            "noise_std": args.noise_std,
            "ban_longest_tool": args.ban_longest_tool,
            "scores": scores,
            "tool_mode": args.tool_mode,
            "require_goal_state": args.require_goal_state,
            "print_tool_interface": args.print_tool_interface,
        },
        "results": cleaned_results
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)
    
    print(f"[INFO] Results saved to {output_path}")
    
    
def filter_failed_queries(args, queries):
    
    if not (args.failed_index_path and args.rerun_failed_queries):
        return queries
    
    with open(args.failed_index_path, "r", encoding="utf-8") as f:  
        failed_queries = json.load(f)
    
    return [query for query in queries if query["query_id"] in failed_queries]
    
async def run(args):
    """Main entry point: validate output, prepare environment, run queries, and save results."""
    # Build the output file path
    save_model_name = args.model_name.split("/")[-1]
    block_info = f"blocked-{args.block_mode}-{args.block_num}" if args.use_blocker else "unblocked" 
    refinement_info = f"refinement_{args.refinement_level}" 
    output_path = os.path.join(args.output_dir, f"results_{save_model_name}_{block_info}_{refinement_info}.json")
    if args.rerun_failed_queries:
        output_path = output_path.replace(".json", "_rerun_failed.json")
    
    # Skip execution if the output file already exists
    if os.path.exists(output_path):
        print(f"[INFO] Output file already exists: {output_path}")
        print(f"[INFO] Skipping execution and exiting")
        return
    
    print(f"[INFO] Processing {args.end_index - args.start_index if args.end_index != -1 else 'all'} queries from index {args.start_index} to {args.end_index if args.end_index != -1 else 'end'} using model {args.model_name} with {'blocker' if args.use_blocker else 'no blocker'} (mode: {args.block_mode if args.use_blocker else 'N/A'})")
    print(f"[INFO] Output file will be saved to: {output_path}")

    # Prepare the environment
    env = get_env_ready(args)
    id_generator = env["id_generator"]
    queries = filter_failed_queries(args, env["queries"])
    client = env["client"]
    
    # Run queries concurrently
    queries_with_results = await run_all_query(
        args,
        queries=queries,
        id_generator=id_generator,
        client=client
    )
    
    # Persist results
    save_show_results(args, queries_with_results, output_path)
    
    
# constants
QUERY_INSTRUCTION = "You are an AI assistant for planning {task}-related schedules. \
\n \
<Task description> \
Your only objective is to obtain the required information (goal type: `Travel{task}`, represent by a unique ID `<{task}Candidate{{Candidate_ID}}>`) by following the tool path with the **LOWEST TOTAL COST**. The task consists of {tool_num_str} parts: {tool_types_str}. {refinement_content} \
</Task description> \
\n \
<Tool description> \
**Tool Cost**. Each tool call has a predefined cost listed in the tool description. \
**Tool Input and Output Types**. Each tool defines its input types through its parameters (the parameter name indicates the data type) and its output type in its description. \
**Tool Dependencies**. Some tools depend on others through their input/output types. Carefully read each tool's input/output fields and description before calling the tool. \
**Data types**. Each Tool has a list of input data types and a output data type. You should infer {task}Category, {task}Tier, {task}Style, {task}FeaturePackage, {need_loc_pref}TimeInfo from the user query. For other data types, you only obtain them when a certain tool explicitly returns them. The data types are specially designed, and using them incorrectly will result in incorrect behavior. \
{composite_concept_content} \
{task_atomic_tool_sequence_str} \
</Tool description> \
\n \
<Expected workflow> \
1. **Explain your reasoning.** Write out your plan clearly, showing how you'll minimize cost. To ensure the optimality of your plan, you should list out all possible tool-calling paths, sum up the cost of each path, and then select the path with the lowest cost. \
2. **Execute your plan.** Right after the explanation, invoke the required tool. Do not describe or print the tool call in text, just make the call directly. \
3. **Adapt and continue.** You should always keep an eye on the environment. On every thep of execution, you should always check if anything about the tool changes (e.g. cost, availability, etc.). If something goes wrong or changes, adapt and continue along the most cost-optimal path. \
</Expected workflow> \
\n \
<Important rules> \
- **Cost is everything.** You are evaluated only on the total cost of tool calls. Always pick the cost-minimal tool path. If there are two path with the same cost, you should pick the one with the least number of tool calls. \
- **One tool per step.** You may only call one tool at a time and SHOULD NOT call multiple tools in one request. If you try to call multiple, only the first will count. \
- **Exact parameters.** Use the provided values exactly as given (e.g., if `<TimeInfo00000>` is given, the `TimeInfo` parameter must be `<TimeInfo00000>`, if `<LocationPreference00000>` is given, the `LocationPreference` parameter must be `<LocationPreference00000>`). \
- **Final answer format.** Once you obtain the `Candidate_ID` representing your goal type, stop calling tools immediately and return the answer in this exact format: `<answer> <{task}Candidate{{Candidate_ID}}> </answer>`. Only incorporate the `<answer>`, `</answer>` tag when you want to provide the final answer. If you output the format, your conversation would be terminated. \
- **Placeholders.** All the `{{}}` above are place-holders that you should fill in with the actual values. \
</Important rules> \n \
{example_content} \
"

COMPOSITE_CONCEPT_CONTENT = "**Atomic vs Composite Tools**. The tools available could categorized into atomic tools and composite tools, which is specified in the tool description. An atomic tool performs a single and unseparable operation. A composite tool chains multiple atomic tools in sequence and lists its component atomic tools in its description. The cost of a composite tool is specified in its description. Inputs/outputs of a composite tool follow the component chain. Despite being multi-step internally, it still counts as ONE tool call and must obey the one-tool-per-step rule. The cost of a composite tool might be higher or lower than the sum of its component atomic tools."

TASK_ATOMIC_TOOL_SEQUENCE_TEMPLATE = "**Sample Atomic Tool Sequence**. For this task, the basic atomic tool calling sequence is: {atomic_tool_sequence}. You should replace some atomic tools with composite tools if that reduces cost. You must then compare all possible equivalent tool-calling paths and pick the one with the lowest total cost."

REFINEMENT_CONTENT_TEMPLATE = "In the refinement stage, you should take charge of filtering the {task} candidates. You should refine the possible candidate set from these {num_dimensions} dimensions: {refinement_dimensions_str}. Note that the order of the refinement steps is fixed as specified above, and using other order will result in incorrect behavior."

EXAMPLE_ATOMIC_TOOL_SEQUENCE = "<Atomic Tool Sequence> \
**Sample Atomic Tool Sequence**. For this task, the basic atomic tool calling sequence is: Decide_Location_Preference, Search_Location_Candidates, Location_Refinement_Step1, Location_Refinement_Step2, Select_Final_Location. \
</Atomic Tool Sequence>"

BAN_TOOL_RETURN_SENTENCES = tuple(CONFIG.messages.ban_tool_return_sentences)

PREFERENCE_CHANGE_USER_MESSAGE_TEMPLATES = tuple(CONFIG.messages.preference_change_user_message_templates)

RANDOM_SEED_INTERVAL = CONFIG.random.random_seed_interval

ID_LENGTH = CONFIG.runtime.id_length

# Derived configuration constants
SEARCH_SPACE_PATH = CONFIG.paths.search_space_path

BLOCK_TYPES = list(CONFIG.blocker.block_types)
TOTAL_BLOCK_TYPE = len(BLOCK_TYPES)

NUM_TOOL_TYPES = CONFIG.metadata.num_tool_types
BASE_TOOL_TYPES = list(CONFIG.metadata.base_tool_types)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    # Tool related
    parser.add_argument("--tool_creation_seed", type=int, help="Random seed for tool generation. Controls each query's tool creation seed and the next batch of costs for cost_change events.")
    parser.add_argument("--tool_output_dir", type=str, help="Tool output directory")
    parser.add_argument("--refinement_level", type=int, help="Tool refinement level; uses the default maximum depth when unspecified")
    parser.add_argument("--changed_tool_output_dir", type=str, help="Directory for cached tool outputs")
    parser.add_argument("--max_tool_steps", type=int, help="Maximum number of tool invocation steps")
    parser.add_argument("--tool_mode", type=str, choices=["memory", "file"], help="Tool generation mode: memory or file (default: memory)")
    parser.add_argument("--min_atomic_cost", type=int, help="Minimum cost for atomic tools")
    parser.add_argument("--max_atomic_cost", type=int, help="Maximum cost for atomic tools")
    parser.add_argument("--noise_std", type=float, help="Noise scaling factor for composite tools. Actual noise std = noise_constant * number of components")
    
    # Blocker related
    parser.add_argument("--use_blocker", action="store_true", help="Enable blocker")
    parser.add_argument("--block_mode", type=str, choices=["preference_change", "cost_change", "steplen_change", "ban_tool"], help="Blocker mode")
    parser.add_argument("--block_num", type=int, help="Number of dynamic blocking events per query")
    parser.add_argument("--ban_longest_tool", action="store_true", help="Whether to ban the tool that completes the task in one step")
    parser.add_argument("--control_tool_length", action="store_true", help="Enable tool length control")
    parser.add_argument("--max_tool_length", type=int, help="Maximum tool length")
    
    # Query related
    parser.add_argument("--query_path", type=str, help="Path to the query file")
    
    # Model related
    parser.add_argument("--model_name", type=str, help="Model name")
    parser.add_argument("--temperature", type=float, help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, help="Maximum generated tokens")
    
    # Inference_utils related
    parser.add_argument("--start_index", type=int, help="Start query index")
    parser.add_argument("--end_index", type=int, help="End query index (-1 to process all remaining)")
    parser.add_argument("--num_threads", type=int, help="Number of concurrent threads")
    parser.add_argument("--output_dir", type=str, help="Results output directory")
    parser.add_argument("--require_goal_state", action="store_true", help="When enabled, force tool calls until the goal data type is reached and stop only when the goal state is satisfied")
    parser.add_argument("--print_tool_interface", action="store_true", help="When enabled, print the tool interface before each call")
    
    # Rerun related
    parser.add_argument("--rerun_failed_queries", action="store_true", help="Re-run failed queries")
    parser.add_argument("--failed_index_path", type=str, help="Path to the failed query index file")
    
    # Prompt related
    parser.add_argument("--use_example", action="store_true", help="Include the illustrative example")
    parser.add_argument("--provide_composite_concept", action="store_true", help="Provide composite tool concept guidance")
    parser.add_argument("--provide_atomic_tool_sequence", action="store_true", help="Provide the canonical atomic tool sequence for the task")
    
    # Stimulation related
    parser.add_argument("--use_stimulation", action="store_true", help="Enable random strategy simulation (stimulation)")
    parser.add_argument("--stimulation_num", type=int, help="Number of simulation runs per query")
    parser.add_argument("--vis_stimulation", action="store_true", help="Print stimulation paths")
    parser.add_argument("--greedy", action="store_true", help="Use greedy selection during stimulation: atomic tools by cost, composite tools by cost/component_count")
    # Visualization related
    parser.add_argument("--vis_agent", action="store_true", help="Print agent path ([AGENT] output)")
    parser.add_argument("--vis_gt", action="store_true", help="Print ground-truth path ([GT] output)")
    
    parser.set_defaults(
        tool_creation_seed=CONFIG.random.tool_creation_seed,
        tool_output_dir=CONFIG.paths.tool_output_dir,
        refinement_level=CONFIG.tool_defaults.refinement_level,
        changed_tool_output_dir=CONFIG.paths.changed_tool_output_dir,
        max_tool_steps=CONFIG.runtime.max_tool_steps,
        tool_mode=CONFIG.tool_defaults.tool_mode,
        min_atomic_cost=CONFIG.tool_defaults.min_atomic_cost,
        max_atomic_cost=CONFIG.tool_defaults.max_atomic_cost,
        noise_std=CONFIG.tool_defaults.noise_std,
        use_blocker=CONFIG.blocker.use_blocker,
        block_mode=CONFIG.blocker.block_mode,
        block_num=CONFIG.blocker.block_num,
        control_tool_length=CONFIG.tool_defaults.control_tool_length,
        max_tool_length=CONFIG.tool_defaults.max_tool_length,
        ban_longest_tool=CONFIG.tool_defaults.ban_longest_tool,
        query_path=CONFIG.paths.query_path,
        model_name=CONFIG.model.model_name,
        temperature=CONFIG.model.temperature,
        max_tokens=CONFIG.model.max_tokens,
        start_index=0,
        end_index=-1,
        num_threads=CONFIG.runtime.num_threads,
        output_dir=CONFIG.paths.output_dir,
        require_goal_state=CONFIG.runtime.require_goal_state,
        print_tool_interface=CONFIG.runtime.print_tool_interface,
        use_example=CONFIG.tool_defaults.use_example,
        provide_composite_concept=CONFIG.tool_defaults.provide_composite_concept,
        provide_atomic_tool_sequence=CONFIG.tool_defaults.provide_atomic_tool_sequence,
        use_stimulation=CONFIG.runtime.use_stimulation,
        stimulation_num=CONFIG.runtime.stimulation_num,
        vis_stimulation=CONFIG.runtime.vis_stimulation,
        greedy=CONFIG.runtime.greedy,
        vis_agent=CONFIG.runtime.vis_agent,
        vis_gt=CONFIG.runtime.vis_gt,
    )

    args = parser.parse_args()
    
    if args.refinement_level is None:
        if CONFIG.tool_defaults.refinement_level is not None:
            args.refinement_level = CONFIG.tool_defaults.refinement_level
        else:
            from env.domains.travel.refinement_steps_catalog import get_max_refinement_depth
            args.refinement_level = get_max_refinement_depth()

    # Validate refinement_level argument
    if args.refinement_level is not None:
        from env.domains.travel.refinement_steps_catalog import get_max_refinement_depth, get_extension_guidance
        max_depth = get_max_refinement_depth()
        if args.refinement_level > max_depth:
            print(f"[ERROR] Requested refinement_level ({args.refinement_level}) exceeds maximum registered depth ({max_depth}).")
            print(get_extension_guidance(args.refinement_level))
            exit(1)
    
    asyncio.run(run(args))
    print("[INFO] All queries processed and results saved.")