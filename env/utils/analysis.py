#!/usr/bin/env python3

import argparse
import json
import os
import re
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure the project root is importable before loading helper modules.
project_root = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.abspath(project_root))

from env.settings import load_config

# You can tweak these constants to control figure typography globally.
FONT_SIZE_TITLE = 22
FONT_SIZE_LABEL = 20
FONT_SIZE_TICK = 16
FONT_SIZE_LEGEND = 16

# Reuse the GT trajectory construction logic from eval.py.
try:
    from env.utils.eval import build_segmented_gt_path as build_gt_path_segmented
except Exception:
    build_gt_path_segmented = None  # Fallback if the import fails.


# BAN tool return sentences kept consistent with reeval.py (these errors count as "valid tool calls").
BAN_TOOL_RETURN_SENTENCES = tuple(load_config().messages.ban_tool_return_sentences)


def is_invalid_tool_call(tool_content: str) -> bool:
    """Determine whether a tool call is invalid.

    Definition: content that starts with "[ERROR]" and is not in BAN_TOOL_RETURN_SENTENCES is invalid;
    everything else is treated as valid.
    """
    if not tool_content or not str(tool_content).strip().startswith("[ERROR]"):
        return False
    for ban_sentence in BAN_TOOL_RETURN_SENTENCES:
        if ban_sentence in tool_content:
            return False
    return True


def extract_first_tool_name(tool_calls: Any) -> Optional[str]:
    """Extract the first tool name from the tool_calls field (supports dict or object forms)."""
    if not tool_calls:
        return None
    try:
        first = tool_calls[0]
        if isinstance(first, dict):
            return first.get("name")
        # Support object form: .function.name
        return getattr(getattr(first, "function", None), "name", None)
    except Exception:
        return None


def extract_cost_from_tool_message(content: str) -> float:
    """Extract "Cost: <float>" from the content text of a tool message. Returns 0.0 if not found."""
    try:
        m = re.search(r"Cost:\s*([\d.]+)", content or "", re.IGNORECASE)
        if m:
            value_str = m.group(1).rstrip(' .。;,)]')
            return float(value_str)
    except Exception:
        pass
    return 0.0


def extract_full_agent_steps(conversation_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract the full sequence of agent steps (only the first tool_call counts per step) with cost and validity.

    Returns: List[{"step": int, "tool": str, "cost": float, "is_valid": bool}]
    """
    steps: List[Dict[str, Any]] = []
    current_step = 0
    i = 0
    n = len(conversation_history or [])

    while i < n:
        msg = conversation_history[i]
        role = msg.get("role")
        if role == "assistant":
            current_step += 1
            tool_name = extract_first_tool_name(msg.get("tool_calls") or [])

            # Look for the first subsequent tool response to parse cost and validity
            cost = 0.0
            is_valid = True
            j = i + 1
            while j < n:
                nxt = conversation_history[j]
                if nxt.get("role") == "tool":
                    content = nxt.get("content") or ""
                    if is_invalid_tool_call(content):
                        is_valid = False
                    cost = extract_cost_from_tool_message(content)
                    break
                if nxt.get("role") == "assistant":
                    break
                j += 1

            if tool_name:
                steps.append({
                    "step": current_step,
                    "tool": tool_name,
                    "cost": float(cost),
                    "is_valid": bool(is_valid),
                })
        i += 1

    return steps


def build_valid_agent_path(steps: List[Dict[str, Any]]) -> Tuple[List[str], List[Tuple[str, float, bool]]]:
    """Build two views from the step list:
    - Valid path (only keep tool names where is_valid=True)
    - Full visualization trajectory (triples of tool, cost, is_valid)
    """
    path: List[str] = []
    vis: List[Tuple[str, float, bool]] = []
    for s in steps:
        tool = s.get("tool", "")
        cost = float(s.get("cost", 0.0))
        is_valid = bool(s.get("is_valid", True))
        vis.append((tool, cost, is_valid))
        if is_valid:
            path.append(tool)
    return path, vis


def calculate_edit_distance(seq1: List[str], seq2: List[str]) -> int:
    """Minimum edit distance (Levenshtein distance with unit cost for insert/delete/replace)."""
    if not seq1 and not seq2:
        return 0
    if not seq1:
        return len(seq2)
    if not seq2:
        return len(seq1)

    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


def format_trajectory_display(trajectory: List[Tuple], trajectory_type: str) -> str:
    """Format output similar to the display used in reeval.py."""
    if not trajectory:
        return f"    {trajectory_type}: (Empty)\n"

    lines = [f"    {trajectory_type}:"]
    total_cost = 0.0
    for i, item in enumerate(trajectory, 1):
        if len(item) == 3:
            tool_name, cost, is_valid = item  # type: ignore[misc]
            valid_mark = "✓" if is_valid else "✗"
            lines.append(f"      {i:2d}. {tool_name:<40} Cost: {cost:>8.2f} [{valid_mark}]")
            if is_valid:
                total_cost += cost
        else:
            tool_name, cost = item  # type: ignore[misc]
            lines.append(f"      {i:2d}. {tool_name:<40} Cost: {cost:>8.2f}")
            total_cost += cost
    lines.append(f"      {'Total Cost:':<43} {total_cost:>8.2f}")
    lines.append("")
    return "\n".join(lines)


def load_results(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("results", [])


def normalize_assistant_content(content: Any) -> Optional[str]:
    """Normalize assistant message content into a plain text string. Return None for empty text."""
    if content is None:
        return None
    try:
        if isinstance(content, list):
            texts: List[str] = []
            for part in content:
                if isinstance(part, dict):
                    if "text" in part:
                        texts.append(str(part.get("text", "")))
                    elif "type" in part and part.get("type") == "text" and "value" in part:
                        texts.append(str(part.get("value", "")))
                    else:
                        texts.append(str(part))
                else:
                    texts.append(str(part))
            content_str = "\n".join(t for t in texts if t is not None)
        else:
            content_str = str(content)
    except Exception:
        content_str = str(content)

    if content_str is None:
        return None
    if str(content_str).strip():
        return str(content_str)
    return None


def extract_first_assistant_text(conv: List[Dict[str, Any]]) -> Optional[str]:
    """Extract the first assistant plain-text reply in a conversation (content field, excluding tool_calls)."""
    for msg in conv or []:
        if msg.get("role") != "assistant":
            continue
        content_str = normalize_assistant_content(msg.get("content"))
        if content_str is not None:
            return content_str
    return None


def extract_assistant_text_messages(conv: List[Dict[str, Any]]) -> List[str]:
    """Collect every assistant message in the conversation that contains text."""
    texts: List[str] = []
    for msg in conv or []:
        if msg.get("role") != "assistant":
            continue
        content_str = normalize_assistant_content(msg.get("content"))
        if content_str is None:
            continue
        texts.append(content_str)
    return texts


def run_unblocked_coverage_to_json(models_dir: Path, refinement_level: int, output_path: Path) -> None:
    """Scan unblocked results, extract each query's first assistant text reply, and write them to JSON.

    - Locate: results_{MODEL}_unblocked_refinement_{refinement_level}.json
    - Extract: the content of the first conversation_history entry where role == "assistant"
    - Output: write a List[Dict] where every element is {"query_id": str, "model_plan": str|None}
    """
    unblocked = find_unblocked_files(models_dir, refinement_level)
    if not unblocked:
        print("No unblocked result files found (check --model / --refinement_level).")
        return

    # Deduplicate by query_id, allowing later entries to overwrite earlier ones.
    collected_map: Dict[str, Optional[str]] = {}

    for _model_short, path in sorted(unblocked.items()):
        try:
            results = load_results(path)
        except Exception as e:
            print(f"Failed to load results: {e} -> {path}")
            continue

        for item in results or []:
            qid_raw = item.get("query_id")
            if qid_raw is None:
                continue
            query_id = str(qid_raw)
            conv_history = item.get("conversation_history", [])
            if not isinstance(conv_history, list):
                conv_history = []
            if len(conv_history) <= 3:
                continue
            first_text = extract_first_assistant_text(conv_history)
            # If multiple files share the same query_id, later entries overwrite earlier ones.
            collected_map[query_id] = first_text if first_text is not None else None

    # Ensure the output directory exists.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Convert to the required list-of-dicts structure.
    collected_list: List[Dict[str, Optional[str]]] = []
    for qid, txt in collected_map.items():
        collected_list.append({
            "query_id": qid,
            "model_plan": txt
        })
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(collected_list, f, ensure_ascii=False, indent=2)
    print(f"Wrote JSON: {output_path}  (total {len(collected_list)} entries)")


def model_short_from_filename(p: Path) -> Optional[str]:
    """Extract the short MODEL name from a filename: results_{MODEL}_...json -> MODEL."""
    m = re.match(r"results_(.+?)_", p.name)
    return m.group(1) if m else None


def find_result_files(models_dir: Path, refinement_level: int, block_num: int) -> Dict[str, Dict[str, Path]]:
    """Recursively find unblocked and cost-change result files for the given refinement and block_num.

    Returns: { model_short: {"unblocked": Path, "cost_change": Path} }
    """
    pairs: Dict[str, Dict[str, Path]] = {}

    # Look for unblocked files
    unblocked_files = list(models_dir.rglob(f"results_*_unblocked_refinement_{refinement_level}.json"))

    # Look for cost-change files (exact match for block_num and refinement_level)
    cost_change_files = list(models_dir.rglob(
        f"results_*_blocked-cost_change-{block_num}_refinement_{refinement_level}.json"
    ))

    # Build index
    unblocked_by_model: Dict[str, Path] = {}
    for p in unblocked_files:
        m = model_short_from_filename(p)
        if m and m not in unblocked_by_model:
            unblocked_by_model[m] = p

    cost_change_by_model: Dict[str, Path] = {}
    for p in cost_change_files:
        m = model_short_from_filename(p)
        if m and m not in cost_change_by_model:
            cost_change_by_model[m] = p

    # Intersect to produce pairs
    for m in sorted(set(unblocked_by_model.keys()) & set(cost_change_by_model.keys())):
        pairs[m] = {"unblocked": unblocked_by_model[m], "cost_change": cost_change_by_model[m]}

    return pairs


def find_result_files_for_block_mode(models_dir: Path, refinement_level: int, block_num: int, mode: str) -> Dict[str, Dict[str, Path]]:
    """Recursively find result files for a given block mode alongside unblocked files.

    Returns: { model_short: {"unblocked": Path, "blocked": Path} }
    File naming: results_{MODEL}_blocked-{mode}-{block_num}_refinement_{refinement_level}.json
    """
    pairs: Dict[str, Dict[str, Path]] = {}

    # Look for unblocked files
    unblocked_files = list(models_dir.rglob(f"results_*_unblocked_refinement_{refinement_level}.json"))

    # Look for blocked files of the specified mode
    blocked_files = list(models_dir.rglob(
        f"results_*_blocked-{mode}-{block_num}_refinement_{refinement_level}.json"
    ))

    # Build index
    unblocked_by_model: Dict[str, Path] = {}
    for p in unblocked_files:
        m = model_short_from_filename(p)
        if m and m not in unblocked_by_model:
            unblocked_by_model[m] = p

    blocked_by_model: Dict[str, Path] = {}
    for p in blocked_files:
        m = model_short_from_filename(p)
        if m and m not in blocked_by_model:
            blocked_by_model[m] = p

    # Intersect to produce pairs
    for m in sorted(set(unblocked_by_model.keys()) & set(blocked_by_model.keys())):
        pairs[m] = {"unblocked": unblocked_by_model[m], "blocked": blocked_by_model[m]}

    return pairs


def find_unblocked_files(models_dir: Path, refinement_level: int) -> Dict[str, Path]:
    """Recursively find unblocked result files for the given refinement level and index them by model.

    Returns:
        Dict[str, Path]: Mapping from model_short to file path.
    """
    files = list(models_dir.rglob(f"results_*_unblocked_refinement_{refinement_level}.json"))
    by_model: Dict[str, Path] = {}
    for p in files:
        m = model_short_from_filename(p)
        if m and m not in by_model:
            by_model[m] = p
    return by_model


def index_results_by_query_id(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(item.get("query_id")): item for item in (results or []) if item.get("query_id") is not None}


def cost_change_blocked_successfully(item: Dict[str, Any]) -> bool:
    """Check whether the cost_change run actually triggered blocking for this query."""
    try:
        bs = item.get("block_stats", {}) or {}
        count_agent = int(bs.get("block_count_agent", bs.get("block_count", 0)) or 0)
        types = list(bs.get("block_type", []) or [])
        return (count_agent >= 1) and ("cost_change" in [str(t) for t in types])
    except Exception:
        return False


def blocked_successfully(item: Dict[str, Any], mode: str) -> bool:
    """Generic blocked-success check: block_count_agent>=1 and block_type includes the specified mode."""
    try:
        bs = item.get("block_stats", {}) or {}
        count_agent = int(bs.get("block_count_agent", bs.get("block_count", 0)) or 0)
        types = list(bs.get("block_type", []) or [])
        return (count_agent >= 1) and (mode in [str(t) for t in types])
    except Exception:
        return False


def both_reached_goal(unblocked_item: Dict[str, Any], cost_item: Dict[str, Any]) -> bool:
    return bool(unblocked_item.get("is_goal_state") is True and cost_item.get("is_goal_state") is True)


def compute_and_print_similarity(query_id: str,
                                 unblocked_item: Dict[str, Any],
                                 cost_item: Dict[str, Any],
                                 vis_path: bool,
                                 mode_display: str = "Cost-Change") -> Tuple[int, float, bool, int, float, bool]:
    # Extract step lists and valid paths for both runs (Agent)
    steps_unblocked = extract_full_agent_steps(unblocked_item.get("conversation_history", []))
    steps_cost = extract_full_agent_steps(cost_item.get("conversation_history", []))

    agent_path_unblocked, vis_unblocked = build_valid_agent_path(steps_unblocked)
    agent_path_cost, vis_cost = build_valid_agent_path(steps_cost)

    # Build both GT paths (following the segmented logic in eval.py)
    def build_gt_path(item: Dict[str, Any]) -> List[str]:
        scenarios = (item.get("groundtruth", {}) or {}).get("scenarios", [])
        block_steps = (item.get("block_stats", {}) or {}).get("block_step", [])
        if build_gt_path_segmented is not None:
            try:
                gt_path, _ = build_gt_path_segmented(scenarios, block_steps)
                return gt_path
            except Exception:
                pass
        # Fallback: take the tool sequence from the last scenario directly
        gt_path: List[str] = []
        try:
            last_scn = scenarios[-1] if scenarios else {}
            tools_list = last_scn.get("tools", []) or []
            for t in tools_list:
                if isinstance(t, dict) and t:
                    gt_path.append(list(t.keys())[0])
                elif isinstance(t, str):
                    gt_path.append(t)
        except Exception:
            gt_path = []
        return gt_path

    gt_path_unblocked = build_gt_path(unblocked_item)
    gt_path_cost = build_gt_path(cost_item)

    # Compute Agent vs Agent metrics
    agent_ed = calculate_edit_distance(agent_path_unblocked, agent_path_cost)
    agent_max_len = max(len(agent_path_unblocked), len(agent_path_cost))
    agent_ned = (agent_ed / agent_max_len) if agent_max_len > 0 else 0.0
    agent_em = (agent_path_unblocked == agent_path_cost)

    # Compute GT vs GT metrics
    gt_ed = calculate_edit_distance(gt_path_unblocked, gt_path_cost)
    gt_max_len = max(len(gt_path_unblocked), len(gt_path_cost))
    gt_ned = (gt_ed / gt_max_len) if gt_max_len > 0 else 0.0
    gt_em = (gt_path_unblocked == gt_path_cost)

    # Print the four paths
    print(f"Query: {query_id}")
    print(f"  Unblocked GT Path:   {gt_path_unblocked}")
    print(f"  Unblocked Agent Path:{agent_path_unblocked}")
    print(f"  {mode_display} GT Path: {gt_path_cost}")
    print(f"  {mode_display} Agent Path: {agent_path_cost}")

    # Print both sets of comparative metrics
    print(f"  [Agent vs Agent] (Unblocked Agent vs Cost-Change Agent)  ED={agent_ed}  NED={agent_ned:.3f}  EM={agent_em}")
    print(f"  [GT vs GT]       (Unblocked GT vs Cost-Change GT)        ED={gt_ed}  NED={gt_ned:.3f}  EM={gt_em}")

    if vis_path:
        print("  Paths (Agent Trajectories):")
        print("  - Unblocked")
        print(format_trajectory_display(vis_unblocked, "Agent Cost-Counted"))
        print("  - Baseline")
        print(format_trajectory_display(vis_cost, "Agent Cost-Counted"))
    print("-" * 100)
    return agent_ed, agent_ned, agent_em, gt_ed, gt_ned, gt_em


def compute_and_print_agent_vs_gt_metrics(label: str,
                                          agent_path: List[str],
                                          item: Dict[str, Any]) -> Tuple[int, float, bool]:
    """Compute and print Agent vs GT metrics (ED/NED/EM) using the GT trajectory logic from eval.py."""
    try:
        scenarios = (item.get("groundtruth", {}) or {}).get("scenarios", [])
        block_steps = (item.get("block_stats", {}) or {}).get("block_step", [])
        if build_gt_path_segmented is not None:
            gt_path, _gt_cost = build_gt_path_segmented(scenarios, block_steps)
        else:
            # Fallback: concatenate the tool names from the final scenario when import fails
            gt_path = []
            try:
                last_scn = scenarios[-1] if scenarios else {}
                tools_list = last_scn.get("tools", []) or []
                for t in tools_list:
                    if isinstance(t, dict) and t:
                        gt_path.append(list(t.keys())[0])
                    elif isinstance(t, str):
                        gt_path.append(t)
            except Exception:
                gt_path = []

        ed = calculate_edit_distance(agent_path, gt_path)
        max_len = max(len(agent_path), len(gt_path))
        ned = (ed / max_len) if max_len > 0 else 0.0
        em = (agent_path == gt_path)

        print(f"  [GT vs {label}] ED: {ed}  NED: {ned:.3f}  EM: {em}")
        return ed, ned, em
    except Exception:
        print(f"  [GT vs {label}] ED: 0  NED: 0.000  EM: False  (GT construction failed)")
        return 0, 0.0, False


def build_gt_path_for_item(item: Dict[str, Any]) -> List[str]:
    """Build the GT tool path for a single result (using eval.py's segmented logic). Return [] on failure."""
    try:
        scenarios = (item.get("groundtruth", {}) or {}).get("scenarios", [])
        block_steps = (item.get("block_stats", {}) or {}).get("block_step", [])
        if build_gt_path_segmented is not None:
            try:
                gt_path, _ = build_gt_path_segmented(scenarios, block_steps)
                return gt_path
            except Exception:
                pass
        # Fallback: take the tool sequence from the final scenario directly
        gt_path: List[str] = []
        try:
            last_scn = scenarios[-1] if scenarios else {}
            tools_list = last_scn.get("tools", []) or []
            for t in tools_list:
                if isinstance(t, dict) and t:
                    gt_path.append(list(t.keys())[0])
                elif isinstance(t, str):
                    gt_path.append(t)
        except Exception:
            gt_path = []
        return gt_path
    except Exception:
        return []


def _ensure_entries(raw: Any) -> List[Dict[str, Any]]:
    """Extract the result list from the loaded object, supporting both list and dict formats."""
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if isinstance(raw, dict):
        results = raw.get("results")
        if isinstance(results, list):
            return [item for item in results if isinstance(item, dict)]
    return []


def run_visualize_failed_datapoints(result_path: Path, show_vis: bool) -> None:
    """Print every sample in the result file that failed to reach the goal state, matching cost_change_similarity styling."""
    if not result_path.exists() or not result_path.is_file():
        print(f"Error: result file not found or unusable: {result_path}")
        sys.exit(1)

    try:
        with result_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        print(f"Failed to load results: {e} -> {result_path}")
        sys.exit(1)

    entries = _ensure_entries(raw)
    total_entries = len(entries)

    failed_entries: List[Dict[str, Any]] = []
    for item in entries:
        is_goal = item.get("is_goal_state")
        if bool(is_goal is True):
            continue
        failed_entries.append(item)

    failed_count = len(failed_entries)

    print("=" * 100)
    print("FAILED QUERIES - Goal State Not Reached")
    print("=" * 100)
    print(f"Source file: {result_path}")
    print(f"Failed queries: {failed_count} / {total_entries}")
    print("-")

    if not failed_entries:
        print("No entries found with is_goal_state != True.")
        print("=" * 100)
        return

    for item in failed_entries:
        query_id = item.get("query_id")
        query_str = str(query_id) if query_id is not None else "<unknown>"
        task = item.get("task") or "<unknown>"
        goal_type = item.get("goal_type") or item.get("goalType") or "<unknown>"
        status = item.get("status")
        evaluation = item.get("evaluation") or {}
        success_flag = evaluation.get("success")
        actual_result = evaluation.get("actual_result") or evaluation.get("actualResult")

        print(f"Query: {query_str}")
        print(f"  Task: {task}    Goal Type: {goal_type}")
        print(f"  is_goal_state: {item.get('is_goal_state')}    status: {status}")
        if success_flag is not None or actual_result:
            print(f"  eval.success: {success_flag}    eval.actual_result: {actual_result}")

        steps = extract_full_agent_steps(item.get("conversation_history", []))
        agent_path, vis_agent = build_valid_agent_path(steps)
        gt_path = build_gt_path_for_item(item)

        print(f"  Agent Path: {agent_path}")
        print(f"  GT Path:    {gt_path}")
        compute_and_print_agent_vs_gt_metrics("Agent", agent_path, item)

        assistant_texts = extract_assistant_text_messages(item.get("conversation_history", []))
        if assistant_texts:
            print("  Assistant Text Messages:")
            for idx, text in enumerate(assistant_texts, 1):
                print(f"    #{idx}:")
                clean_text = text.rstrip()
                if not clean_text:
                    continue
                formatted_text = textwrap.indent(clean_text, "      ")
                print(formatted_text)

        if show_vis:
            print("  Agent Trajectory (step / cost / valid):")
            formatted = format_trajectory_display(vis_agent, "Agent Cost-Counted")
            # format_trajectory_display already includes indentation; add two spaces for alignment
            formatted_lines = ["    " + line if idx == 0 else line for idx, line in enumerate(formatted.splitlines())]
            print("\n".join(formatted_lines))

        stimulation = item.get("stimulation")
        if stimulation:
            stim_paths = stimulation.get("paths") or []
            if stim_paths:
                pretty_paths = [list(path) for path in stim_paths if isinstance(path, (list, tuple))]
                print(f"  Stimulation paths (raw): {pretty_paths}")

        print("-" * 100)

    print("=" * 100)


def run_all_coverage(models: List[str], refinement_levels: List[int], output_dir: Path, figure_path: Optional[Path], colors: Optional[List[str]] = None) -> None:
    """Read first_assistant_texts_refine_{level}.json for each model/level, compute average num_paths,
    derive coverage rate = avg / (2^(2+level)), print metrics, and create a bar chart."""
    def parse_num_paths(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            s = str(value)
            m = re.search(r"^\s*([0-9]+(?:\.[0-9]+)?)", s)
            return float(m.group(1)) if m else None
        except Exception:
            return None

    # Collect metrics: {model: {level: (avg_paths, coverage_rate, count)}}
    metrics: Dict[str, Dict[int, Tuple[float, float, int]]] = {}
    for model in models:
        metrics[model] = {}
        for level in refinement_levels:
            json_path = output_dir / model / f"first_assistant_texts_refine_{level}.json"
            if not json_path.exists():
                print(f"[WARN] Missing file: {json_path}")
                continue
            try:
                with json_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"[WARN] Failed to load {json_path}: {e}")
                continue

            if not isinstance(data, list):
                print(f"[WARN] Unexpected JSON format (expect list): {json_path}")
                continue

            values: List[float] = []
            for item in data:
                try:
                    num_raw = item.get("num_paths")
                except Exception:
                    num_raw = None
                v = parse_num_paths(num_raw)
                if v is not None:
                    values.append(v)

            avg = (sum(values) / len(values)) if values else 0.0
            denom = float(2 ** (2 + int(level)))
            coverage = (avg / denom) if denom > 0 else 0.0
            metrics[model][int(level)] = (avg, coverage, len(values))
            print(f"{model} (task sequence={level}): avg_paths={avg:.3f}, coverage_rate={coverage:.4f} ({len(values)} queries)")

    # Plotting: retain the provided order (no additional sorting)
    models_sorted = models

    if not models_sorted:
        print("[WARN] No data available to plot.")
        return

    # Prepare grouped bar chart
    import numpy as np  # lazy import to avoid hard dep if not plotting
    import matplotlib.pyplot as plt  # lazy import for plotting only
    try:
        import matplotlib
        # Apply global font size settings
        matplotlib.rcParams['axes.titlesize'] = FONT_SIZE_TITLE
        matplotlib.rcParams['figure.titlesize'] = FONT_SIZE_TITLE
        matplotlib.rcParams['axes.labelsize'] = FONT_SIZE_LABEL
        matplotlib.rcParams['xtick.labelsize'] = FONT_SIZE_TICK
        matplotlib.rcParams['ytick.labelsize'] = FONT_SIZE_TICK
        matplotlib.rcParams['legend.fontsize'] = FONT_SIZE_LEGEND
    except Exception:
        pass
    levels_sorted = sorted(set(int(l) for l in refinement_levels))

    x = np.arange(len(models_sorted), dtype=float)
    width = 0.8 / max(1, len(levels_sorted))

    # Use custom colors if provided; otherwise fall back to matplotlib defaults
    if colors and len(colors) >= len(levels_sorted):
        bar_colors = colors[:len(levels_sorted)]
    else:
        bar_colors = None  # matplotlib will use its default color cycle

    fig, ax = plt.subplots(figsize=(max(8, len(models_sorted) * 0.8), 5))
    for idx, level in enumerate(levels_sorted):
        heights = []
        for m in models_sorted:
            heights.append(metrics.get(m, {}).get(level, (0.0, 0.0, 0))[1])
        bar_kwargs = {"width": width, "label": f"task sequence={level}"}
        if bar_colors:
            bar_kwargs["color"] = bar_colors[idx]
        ax.bar(x + (idx - (len(levels_sorted) - 1) / 2) * width, heights, **bar_kwargs)

    ax.set_ylabel("coverage rate")
    ax.set_xticks(x)
    # Capitalize first letter of model short names in x tick labels
    # Special handling: convert "gpt" to "GPT" and make all labels italic
    pretty_labels = []
    for m in models_sorted:
        try:
            short = m.split('/')[-1]
        except Exception:
            short = m
        # Capitalize first letter
        label = (short[:1].upper() + short[1:]) if short else short
        # Replace "Gpt" with "GPT" (case-insensitive replacement)
        if label.lower().startswith('gpt'):
            label = 'GPT' + label[3:]
        # Make label italic using matplotlib's mathtext italic formatting
        label = r'$\mathit{' + label.replace('-', r'\text{-}') + r'}$'
        pretty_labels.append(label)
    ax.set_xticklabels(pretty_labels, rotation=30, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout()
    out_path = figure_path if figure_path else (output_dir / "coverage_rate_bar.png")
    try:
        fig.savefig(out_path, dpi=200)
        print(f"Saved figure: {out_path}")
    except Exception as e:
        print(f"[WARN] Failed to save figure {out_path}: {e}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Analysis utilities: similarity / coverage")
    parser.add_argument("--type", required=True, choices=[
        "cost_change_similarity",
        "steplen_change_similarity",
        "ban_tool_similarity",
        "preference_change_similarity",
        "unblocked_coverage",
        "all_coverage",
        "all_print_similarity",
        "all_unblocked_avg_steps",
        "visualize_failed"
    ], help="Analysis type")
    parser.add_argument("--model", required=False, help="Directory containing result files (recursively scans results_*.json)")
    parser.add_argument("--refinement_level", type=int, required=False, help="refinement level")
    parser.add_argument("--block_num", type=int, help="Block index for blocked modes (required for similarity modes)")
    parser.add_argument("--vis_path", action="store_true", help="Print detailed visualization output")
    parser.add_argument("--file", required=False, help="Path to the result file for visualize_failed mode")
    parser.add_argument("--output_path_path", help="Output JSON path for unblocked_coverage (defaults to <model>/first_assistant_texts_refine_{refinement_level}.json)")
    parser.add_argument("--models", nargs="+", help="Model names (subdirectories) for all_coverage; comma-separated supported")
    parser.add_argument("--output_dir", help="Root directory containing model outputs for all_coverage")
    parser.add_argument("--refinement_levels", nargs="+", help="Refinement levels for all_coverage; comma-separated supported")
    parser.add_argument("--figure_path", required=False, help="Path to save the coverage figure (default <output_dir>/coverage_rate_bar.png)")
    parser.add_argument("--colors", required=False, help="Custom bar colors for all_coverage (comma or space separated, e.g., #1f77b4,#ff7f0e,#2ca02c)")
    parser.add_argument("--block_types", required=False, help="Block types to compare in all_print_similarity (comma or space separated)")
    parser.add_argument("--model_names", required=False, help="Model names to summarize in all_print_similarity (comma or space separated)")
    args = parser.parse_args()

    if args.type == "unblocked_coverage":
        if not args.model:
            print("Error: --model is required (only for --type unblocked_coverage).")
            sys.exit(1)
        if not args.refinement_level:
            print("Error: --refinement_level is required (only for --type unblocked_coverage).")
            sys.exit(1)
        models_dir = Path(args.model)
        if not models_dir.exists() or not models_dir.is_dir():
            print(f"Error: directory not found or unavailable: {models_dir}")
            sys.exit(1)
        # Default output path when not provided: <model>/first_assistant_texts_refine_{refinement_level}.json
        if args.output_path_path:
            output_path = Path(args.output_path_path)
        else:
            output_path = models_dir / f"first_assistant_texts_refine_{int(args.refinement_level)}.json"
        run_unblocked_coverage_to_json(models_dir, args.refinement_level, output_path)
        return

    if args.type == "all_coverage":
        if not args.models or not args.output_dir or not args.refinement_levels:
            print("Error: --models, --output_dir, and --refinement_levels are required (only for --type all_coverage).")
            sys.exit(1)
        output_dir = Path(args.output_dir)
        # Support model names separated by spaces or commas
        models_arg = args.models
        models: List[str] = []
        if isinstance(models_arg, list):
            if len(models_arg) == 1:
                raw = str(models_arg[0])
                for part in re.split(r"[,\s]+", raw):
                    if part.strip():
                        models.append(part.strip())
            else:
                for token in models_arg:
                    if token is None:
                        continue
                    for part in str(token).split(","):
                        if part.strip():
                            models.append(part.strip())
        else:
            for part in re.split(r"[,\s]+", str(models_arg)):
                if part.strip():
                    models.append(part.strip())

        if not models:
            print("Error: no valid model names parsed (check --models).")
            sys.exit(1)

        # Parse refinement levels (comma/space supported)
        levels_arg = args.refinement_levels
        levels: List[int] = []
        def parse_level_token(tok: str) -> Optional[int]:
            tok = tok.strip()
            if not tok:
                return None
            try:
                return int(tok)
            except Exception:
                return None
        if isinstance(levels_arg, list):
            if len(levels_arg) == 1:
                for part in re.split(r"[,\s]+", str(levels_arg[0])):
                    lv = parse_level_token(part)
                    if lv is not None:
                        levels.append(lv)
            else:
                for token in levels_arg:
                    for part in str(token).split(","):
                        lv = parse_level_token(part)
                        if lv is not None:
                            levels.append(lv)
        else:
            for part in re.split(r"[,\s]+", str(levels_arg)):
                lv = parse_level_token(part)
                if lv is not None:
                    levels.append(lv)

        if not levels:
            print("Error: no valid refinement levels parsed (check --refinement_levels).")
            sys.exit(1)

        # Parse color arguments
        colors: Optional[List[str]] = None
        if args.colors:
            colors = []
            for part in re.split(r"[,\s]+", str(args.colors)):
                c = part.strip()
                if c:
                    colors.append(c)

        fig_path = Path(args.figure_path) if args.figure_path else None
        run_all_coverage(models, levels, output_dir, fig_path, colors)
        return

    if args.type == "visualize_failed":
        if not args.file:
            print("Error: --file is required (only for --type visualize_failed).")
            sys.exit(1)
        run_visualize_failed_datapoints(Path(args.file), args.vis_path)
        return

    if args.type == "all_print_similarity":
        # Parse model names and block types
        if not args.model_names:
            print("Error: --model_names is required (only for --type all_print_similarity).")
            sys.exit(1)
        if not args.block_types:
            print("Error: --block_types is required (only for --type all_print_similarity).")
            sys.exit(1)
        if not args.refinement_level:
            print("Error: --refinement_level is required (only for --type all_print_similarity).")
            sys.exit(1)

        def parse_list_arg(raw: str) -> List[str]:
            vals: List[str] = []
            for part in re.split(r"[\s,]+", str(raw)):
                tok = part.strip()
                if tok:
                    vals.append(tok)
            return vals

        model_names = parse_list_arg(args.model_names)
        block_types = [bt.lower() for bt in parse_list_arg(args.block_types)]
        models_dir = Path(args.model) if args.model else Path(".")
        if not models_dir.exists() or not models_dir.is_dir():
            print(f"Error: directory not found or unavailable: {models_dir}")
            sys.exit(1)
        block_num = int(args.block_num) if args.block_num is not None else 1

        # Print header information
        print("=" * 100)
        print("ALL PRINT SIMILARITY - Mean Agent NED & GT NED (Blocked vs Unblocked)")
        print("=" * 100)
        print(f"Model dir: {models_dir}")
        print(f"Refinement level: {args.refinement_level}   Block num: {block_num}")
        print(f"Models: {', '.join(model_names)}")
        print(f"Block types: {', '.join(block_types)}")
        print("Notes:")
        print("  - mean_Agent_NED: Average normalized edit distance between unblocked and blocked agent paths")
        print("  - mean_GT_NED: Average normalized edit distance between unblocked and blocked GT paths")
        print("-")

        # Pre-build an unblocked index for fallback lookups
        unblocked_by_model = find_unblocked_files(models_dir, int(args.refinement_level))

        def summarize_for_model_and_mode(model_key_in: str, mode_key_in: str) -> Tuple[int, float, float]:
            """Return (compared_count, mean_agent_ned, mean_gt_ned). If unavailable, return (0, 0.0, 0.0)."""
            mode_key = mode_key_in.lower()
            model_key_norm = model_key_in.strip()

            # Locate paired files
            if mode_key == "cost_change":
                pairs = find_result_files(models_dir, int(args.refinement_level), block_num)
                files = pairs.get(model_key_norm)
                if not files:
                    # Attempt case-insensitive match as a fallback
                    for k in pairs.keys():
                        if k.lower() == model_key_norm.lower():
                            files = pairs[k]
                            break
                if not files:
                    return 0, 0.0, 0.0
                unblocked_path = files.get("unblocked")
                blocked_path = files.get("cost_change")
                blocked_ok = cost_change_blocked_successfully
            else:
                pairs = find_result_files_for_block_mode(models_dir, int(args.refinement_level), block_num, mode_key)
                files = pairs.get(model_key_norm)
                if not files:
                    for k in pairs.keys():
                        if k.lower() == model_key_norm.lower():
                            files = pairs[k]
                            break
                if not files:
                    return 0, 0.0, 0.0
                unblocked_path = files.get("unblocked")
                blocked_path = files.get("blocked")
                blocked_ok = lambda item: blocked_successfully(item, mode_key)

            if not (unblocked_path and blocked_path):
                return 0, 0.0, 0.0

            try:
                unblocked_results = load_results(unblocked_path)
                blocked_results = load_results(blocked_path)
            except Exception:
                return 0, 0.0, 0.0

            idx_unblocked = index_results_by_query_id(unblocked_results)
            idx_blocked = index_results_by_query_id(blocked_results)
            common_ids = sorted(set(idx_unblocked.keys()) & set(idx_blocked.keys()))

            agent_neds: List[float] = []
            gt_neds: List[float] = []
            for qid in common_ids:
                ub = idx_unblocked[qid]
                bk = idx_blocked[qid]
                if not blocked_ok(bk):
                    continue
                if not both_reached_goal(ub, bk):
                    continue

                # Compute Agent trajectory NED
                steps_unblocked = extract_full_agent_steps(ub.get("conversation_history", []))
                steps_blocked = extract_full_agent_steps(bk.get("conversation_history", []))
                agent_path_unblocked, _ = build_valid_agent_path(steps_unblocked)
                agent_path_blocked, _ = build_valid_agent_path(steps_blocked)

                ed = calculate_edit_distance(agent_path_unblocked, agent_path_blocked)
                max_len = max(len(agent_path_unblocked), len(agent_path_blocked))
                ned = (ed / max_len) if max_len > 0 else 0.0
                agent_neds.append(ned)

                # Compute GT trajectory NED
                gt_path_unblocked = build_gt_path_for_item(ub)
                gt_path_blocked = build_gt_path_for_item(bk)
                gt_ed = calculate_edit_distance(gt_path_unblocked, gt_path_blocked)
                gt_max_len = max(len(gt_path_unblocked), len(gt_path_blocked))
                gt_ned = (gt_ed / gt_max_len) if gt_max_len > 0 else 0.0
                gt_neds.append(gt_ned)

            if not agent_neds:
                return 0, 0.0, 0.0
            mean_agent_ned = sum(agent_neds) / len(agent_neds)
            mean_gt_ned = sum(gt_neds) / len(gt_neds) if gt_neds else 0.0
            return len(agent_neds), mean_agent_ned, mean_gt_ned

        # Main loop output
        for model_name in model_names:
            print(f"Model: {model_name}")
            for bt in block_types:
                compared, mean_agent_ned, mean_gt_ned = summarize_for_model_and_mode(model_name, bt)
                print(f"  {bt}: compared={compared}  mean_Agent_NED={mean_agent_ned:.3f}  mean_GT_NED={mean_gt_ned:.3f}")
            print("-" * 100)
        print("=" * 100)
        return

    if args.type == "all_unblocked_avg_steps":
        # Compute average valid agent steps for specified models at the given refinement level
        if not args.model_names:
            print("Error: --model_names is required (only for --type all_unblocked_avg_steps).")
            sys.exit(1)
        if not args.refinement_level:
            print("Error: --refinement_level is required (only for --type all_unblocked_avg_steps).")
            sys.exit(1)

        def parse_list_arg2(raw: str) -> List[str]:
            vals: List[str] = []
            for part in re.split(r"[\s,]+", str(raw)):
                tok = part.strip()
                if tok:
                    vals.append(tok)
            return vals

        model_names = parse_list_arg2(args.model_names)
        models_dir = Path(args.model) if args.model else Path(".")
        if not models_dir.exists() or not models_dir.is_dir():
            print(f"Error: directory not found or unavailable: {models_dir}")
            sys.exit(1)

        # Gather unblocked files
        refinement_level = int(args.refinement_level)
        unblocked_map = find_unblocked_files(models_dir, refinement_level)

        # Normalize keys to tolerate date/timestamp suffixes in filenames, e.g.
        # results_claude-sonnet-4-20250514_unblocked_refinement_2.json -> allow "claude-sonnet-4"
        norm_map: Dict[str, Path] = {}
        for k, v in unblocked_map.items():
            norm_map[k] = v
            # Remove trailing -YYYYMMDD or -YYYYMMDDHHMM timestamp suffixes
            m = re.match(r"^(.+?)-(\d{8})(?:\d{4})?$", k)
            if m:
                base = m.group(1)
                if base not in norm_map:
                    norm_map[base] = v

        print("=" * 100)
        print("ALL UNBLOCKED AVG STEPS - Average Valid Agent Steps (Goal Only)")
        print("=" * 100)
        print(f"Model dir: {models_dir}")
        print(f"Refinement level: {refinement_level}")
        print(f"Models: {', '.join(model_names)}")
        print("-")

        global_counts: List[int] = []

        def get_unblocked_path_for_model(model_key_in: str) -> Optional[Path]:
            p = norm_map.get(model_key_in)
            if p:
                return p
            # Attempt case-insensitive match
            for k, v in norm_map.items():
                if k.lower() == model_key_in.lower():
                    return v
            return None

        # Compute averages for agent valid steps and GT steps
        for model_name in model_names:
            path = get_unblocked_path_for_model(model_name)
            if not path:
                print(f"Model: {model_name}")
                print("  No unblocked result file found")
                print("-" * 100)
                continue

            try:
                results = load_results(path)
            except Exception as e:
                print(f"Model: {model_name}")
                print(f"  Failed to load: {e}")
                print("-" * 100)
                continue

            step_counts: List[int] = []
            gt_step_counts: List[int] = []
            greedy_counts: List[float] = []  # Per-query average steps for stimulation paths (if multiple, average first)
            for item in results or []:
                if not bool(item.get("is_goal_state") is True):
                    continue
                steps = extract_full_agent_steps(item.get("conversation_history", []))
                agent_path, _ = build_valid_agent_path(steps)
                step_counts.append(len(agent_path))
                gt_path = build_gt_path_for_item(item)
                gt_step_counts.append(len(gt_path))

                # Stimulation path counts: if stimulation.paths exist, measure their lengths
                try:
                    stim = item.get("stimulation") or {}
                    stim_paths = stim.get("paths") or []
                    per_query_path_lengths: List[int] = []
                    if isinstance(stim_paths, list):
                        for pth in stim_paths:
                            try:
                                p_list = list(pth or [])
                            except Exception:
                                p_list = []
                            if p_list:
                                per_query_path_lengths.append(len(p_list))
                    if per_query_path_lengths:
                        greedy_counts.append(sum(per_query_path_lengths) / len(per_query_path_lengths))
                except Exception:
                    pass

            if step_counts:
                avg_steps = sum(step_counts) / len(step_counts)
                avg_gt_steps = (sum(gt_step_counts) / len(gt_step_counts)) if gt_step_counts else 0.0
                avg_greedy_steps = (sum(greedy_counts) / len(greedy_counts)) if greedy_counts else 0.0
                global_counts.extend(step_counts)
                print(f"Model: {model_name}")
                print(f"  Qualified queries: {len(step_counts)}")
                print(f"  Average valid agent steps: {avg_steps:.3f}")
                print(f"  Average GT steps: {avg_gt_steps:.3f}")
                print(f"  Greedy avg steps (stimulation): {avg_greedy_steps:.3f}  (queries with stimulation: {len(greedy_counts)})")
                print("-" * 100)
            else:
                print(f"Model: {model_name}")
                print("  No qualified (goal-state) queries.")
                print("-" * 100)

        if global_counts:
            print("Across Models (Goal-State Only)")
            print(f"  Total qualified queries: {len(global_counts)}")
            print(f"  Average valid agent steps: {(sum(global_counts)/len(global_counts)):.3f}")
        print("=" * 100)
        return

    def run_block_mode_similarity(mode_key: str, mode_display: str) -> None:
        if not args.model:
            print(f"Error: --model is required (only for --type {mode_key}_similarity).")
            sys.exit(1)
        if not args.refinement_level:
            print(f"Error: --refinement_level is required (only for --type {mode_key}_similarity).")
            sys.exit(1)
        if args.block_num is None:
            print(f"Error: --block_num is required (only for --type {mode_key}_similarity).")
            sys.exit(1)

        models_dir = Path(args.model)
        if not models_dir.exists() or not models_dir.is_dir():
            print(f"Error: directory not found or unavailable: {models_dir}")
            sys.exit(1)

        # cost_change supports the legacy implementation (different dictionary keys)
        if mode_key == "cost_change":
            pairs = find_result_files(models_dir, args.refinement_level, args.block_num)
        else:
            pairs = find_result_files_for_block_mode(models_dir, args.refinement_level, args.block_num, mode_key)

        if not pairs:
            print("No comparable file pairs found (check --model / --refinement_level / --block_num).")
            sys.exit(1)

        print("=" * 100)
        print(f"{mode_display.upper()} vs UNBLOCKED - Agent Path Similarity (ED/NED/EM)")
        print("=" * 100)
        print(f"Model dir: {models_dir}")
        print(f"Refinement level: {args.refinement_level}   Block num: {args.block_num}")
        print()
        print("Terminology and comparison definitions:")
        print("  - ED (Edit Distance): Minimum number of insert/delete/replace operations between tool sequences")
        print("  - NED (Normalized ED): ED / max(len(seqA), len(seqB))")
        print("  - EM (Exact Match): Whether the sequences are identical (True/False)")
        print(f"  - [Agent vs Agent]: Compare the unblocked agent path with the {mode_display} agent path")
        print(f"  - [GT vs GT]:      Compare the unblocked GT path with the {mode_display} GT path")
        print("-")

        total_pairs = 0
        total_compared = 0
        # Global Agent vs Agent metrics
        global_sum_ed = 0
        global_sum_ned = 0.0
        global_em_count = 0
        # Global GT vs GT metrics
        global_gt_sum_ed = 0
        global_gt_sum_ned = 0.0
        global_gt_em_count = 0

        for model_short, files in pairs.items():
            unblocked_path = files.get("unblocked")
            blocked_path = files.get("cost_change") if mode_key == "cost_change" else files.get("blocked")
            if not (unblocked_path and blocked_path):
                continue

            print(f"Model: {model_short}")
            print(f"  Unblocked: {unblocked_path}")
            print(f"  {mode_display}: {blocked_path}")

            try:
                unblocked_results = load_results(unblocked_path)
                blocked_results = load_results(blocked_path)
            except Exception as e:
                print(f"  Failed to load results: {e}")
                print("-" * 100)
                continue

            idx_unblocked = index_results_by_query_id(unblocked_results)
            idx_blocked = index_results_by_query_id(blocked_results)

            common_ids = sorted(set(idx_unblocked.keys()) & set(idx_blocked.keys()))
            print(f"  Paired queries: {len(common_ids)}")
            print(f"  Conditions: {mode_key} must have triggered blocking and both sides require is_goal_state=True")
            print("-" * 100)

            total_pairs += len(common_ids)
            model_sum_ed = 0
            model_sum_ned = 0.0
            model_em_count = 0
            model_compared = 0

            for qid in common_ids:
                item_unblocked = idx_unblocked[qid]
                item_blocked = idx_blocked[qid]

                # Filtering criteria
                if mode_key == "cost_change":
                    if not cost_change_blocked_successfully(item_blocked):
                        continue
                else:
                    if not blocked_successfully(item_blocked, mode_key):
                        continue
                if not both_reached_goal(item_unblocked, item_blocked):
                    continue

                agent_ed, agent_ned, agent_em, gt_ed, gt_ned, gt_em = compute_and_print_similarity(
                    qid, item_unblocked, item_blocked, args.vis_path, mode_display=mode_display
                )
                total_compared += 1
                model_compared += 1
                # Per-model (Agent vs Agent)
                model_sum_ed += agent_ed
                model_sum_ned += agent_ned
                if agent_em:
                    model_em_count += 1
                # Global (Agent vs Agent)
                global_sum_ed += agent_ed
                global_sum_ned += agent_ned
                if agent_em:
                    global_em_count += 1
                # Global (GT vs GT)
                global_gt_sum_ed += gt_ed
                global_gt_sum_ned += gt_ned
                if gt_em:
                    global_gt_em_count += 1

            print()

            # Per-model summary
            if model_compared > 0:
                avg_ed = model_sum_ed / model_compared
                avg_ned = model_sum_ned / model_compared
                em_rate = model_em_count / model_compared
                print(f"[Model Summary] {model_short}")
                print(f"  Compared: {model_compared}")
                print(f"  Mean ED:  {avg_ed:.3f}")
                print(f"  Mean NED: {avg_ned:.3f}")
                print(f"  EM Rate:  {em_rate:.3f}")
                print("-" * 100)
            else:
                print(f"[Model Summary] {model_short}")
                print("  No entries after filtering.")
                print("-" * 100)

        print("=" * 100)
        print(f"Total paired entries (query_id aligned): {total_pairs}")
        print(f"Entries passing filters and compared: {total_compared}")
        if total_compared > 0:
            global_mean_ed = global_sum_ed / total_compared
            global_mean_ned = global_sum_ned / total_compared
            global_em_rate = global_em_count / total_compared
            print("-")
            print("Global summary (Across all models)")
            print(f"  Compared: {total_compared}")
            print(f"  Mean ED:  {global_mean_ed:.3f}")
            print(f"  Mean NED: {global_mean_ned:.3f}")
            print(f"  EM Rate:  {global_em_rate:.3f}")
            # Global GT comparison summary
            print("-")
            print("Global GT comparison summary (GT vs GT across queries)")
            print(f"  Mean ED:  {(global_gt_sum_ed/total_compared):.3f}")
            print(f"  Mean NED: {(global_gt_sum_ned/total_compared):.3f}")
            print(f"  EM Rate:  {(global_gt_em_count/total_compared):.3f}")
        print("=" * 100)

    if args.type == "cost_change_similarity":
        run_block_mode_similarity("cost_change", "Cost-Change")
        return
    if args.type == "steplen_change_similarity":
        run_block_mode_similarity("steplen_change", "Steplen-Change")
        return
    if args.type == "ban_tool_similarity":
        run_block_mode_similarity("ban_tool", "Ban-Tool")
        return
    if args.type == "preference_change_similarity":
        run_block_mode_similarity("preference_change", "Preference-Change")
        return


if __name__ == "__main__":
    main()


"""
Analysis utilities:
1) Compare path similarity (ED/NED/EM) between unblocked and cost_change agent runs.
2) Extract the "first assistant text reply (content)" from each conversation in unblocked results.

Usage examples:
python env/utils/analysis.py \
    --type visualize_failed \
    --file final_prompt/gpt-5/first_assistant_texts_refine_5.json \
    --vis_path > final_prompt/gpt-5/failure_vis/gpt-5_failed_queries.log

# Similarity analysis
python env/utils/analysis.py \
    --type cost_change_similarity \
    --model final_prompt/gpt-5 \
    --refinement_level 2 \
    --block_num 1 \
    --vis_path > final_prompt/gpt-5/examine_raw_data.log
    
python env/utils/analysis.py \
    --type cost_change_similarity \
    --model final_prompt/Qwen3-32B \
    --refinement_level 2 \
    --block_num 1 \
    --vis_path > final_prompt/Qwen3-32B/examine_raw_data.log

# Unblocked coverage (export each query's first assistant text reply as JSON)
python env/utils/analysis.py \
    --type unblocked_coverage \
    --model final_prompt/gpt-5 \
    --refinement_level 2 \
    --output_path_path final_prompt/gpt-5/first_assistant_texts_refine_{refinement_level}.json  

python env/utils/analysis.py --type unblocked_coverage \
  --model final_prompt/gemini-2.5-pro \
  --refinement_level 2 \
  --output_path_path final_prompt/gemini-2.5-pro/first_assistant_texts_refine_2.json

python env/utils/analysis.py \
  --type all_coverage \
  --models gpt-5,gemini-2.5-pro,Qwen3-32B,Qwen3-14B,Qwen3-8B \
  --refinement_levels 2,3,4,5 \
  --output_dir final_prompt \
  --figure_path env/vis/plots/coverage_rate_bar.pdf \
  --colors "#1f77b4,#ff7f0e,#2ca02c,#d62728"
  
python env/utils/analysis.py --type all_coverage --models gpt-5,gemini-2.5-pro,Qwen3-32B,Qwen3-14B,Qwen3-8B --refinement_levels 2,3,4,5 --output_dir final_prompt --figure_path env/vis/plots/coverage_rate_bar.pdf --colors "#4a2377,#f55f74,#8cc5e3,#0d7d87"
  
python env/utils/analysis.py \
  --type all_print_similarity \
  --model /data/yumeng/CostBench/final_prompt \
  --refinement_level 2 \
  --block_types cost_change,ban_tool,preference_change,steplen_change \
  --model_names gemini-2.5-pro,Qwen3-8B,Qwen3-14B,Qwen3-32B,gpt-5 > final_prompt/block_analysis.log

python env/utils/analysis.py \
  --type all_unblocked_avg_steps \
  --model /data/yumeng/CostBench/final_prompt \
  --refinement_level 2 \
  --model_names gemini-2.5-pro,Qwen3-8B,Qwen3-14B,Qwen3-32B,gpt-5,claude-sonnet-4,deepseek-chat-v3.1,Llama-3.1-8B-Instruct,gpt-4o > final_prompt/unblocked_avg_steps.log

python env/utils/analysis.py \
    --type visualize_failed \
    --file final_check/gemini-2.5-pro/results_gemini-2.5-pro_unblocked_refinement_2.json \
    --vis_path > final_check/gemini-2.5-pro/failed_queries.log

Notes:
- File discovery:
  - Recursively search for two categories of result files under --model (cross-subdirectories allowed):
    1) Unblocked:   results_{MODEL}_unblocked_refinement_{refinement_level}.json
    2) Cost-change: results_{MODEL}_blocked-cost_change-{block_num}_refinement_{refinement_level}.json
  - Pair files by {MODEL} (one pair per model).

- Pairing and filtering:
  - Only compare entries with the same query_id.
  - Filter criteria:
    1) The cost_change entry must have an actual block (block_stats.block_count_agent >= 1 and includes cost_change).
    2) Both entries must satisfy is_goal_state == True.
  - Skip entries that do not meet the criteria (no worst-case fallback).

- Path extraction:
  - Follow the logic in eval.py/reeval.py:
    - For each assistant message, only take the name of the first tool_call as the tool for that step.
    - The step cost is taken from the first subsequent tool message containing "Cost: <float>".
    - "Invalid tool calls" (content starts with [ERROR] and is not in BAN_TOOL_RETURN_SENTENCES) are excluded;
      [ERROR] responses contained in BAN_TOOL_RETURN_SENTENCES are treated as valid and included.

- Output:
  - For each qualifying query, print ED, NED, and EM.
  - If --vis_path is provided, also print a visualization of both paths (aligned with the format used in reeval.py).
"""