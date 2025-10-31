"""
python env/vis/vis.py \
  --type error-analysis \
  --model_names Qwen3-8B,Qwen3-14B,Qwen3-32B,gpt-5,gemini-2.5-pro,Llama-3.1-8B-Instruct,claude-sonnet-4,deepseek-chat-v3.1,gpt-4o \
  --refinement_level 2 \
  --experiment_result_dir final_prompt/ \
  --error_json env/errors/error_dump.json
"""

"""
python env/vis/vis.py \
  --type all_block_comparison \
  --model_names Qwen/Qwen3-14B,claude-sonnet-4,gemini-2.5-pro,gpt-5 \
  --refinement_level 2 \
  --experiment_result_dir final_prompt/ \
  --output_dir env/vis/plots/all_block_comparison \
  --colors "cost_change:#e78ac3,ban_tool:#fc8d62,preference_change:#8da0cb,steplen_change:#a6d854,unblocked:#66c2a5"
  
python env/vis/vis.py --type all_block_comparison --model_names Qwen/Qwen3-14B,claude-sonnet-4,gemini-2.5-pro,gpt-5 --refinement_level 2 --experiment_result_dir final_prompt/ --output_dir env/vis/plots/all_block_comparison --colors "cost_change:#e78ac3,ban_tool:#fc8d62,preference_change:#a6d854,steplen_change:#8da0cb,unblocked:#66c2a5"
"""

"""
python env/vis/vis.py \
    --type all_noise_vis \
    --model_names Qwen/Qwen3-8B,Qwen/Qwen3-14B,Qwen/Qwen3-32B,gemini-2.5-pro \
    --refinement_level 2 \
    --experiment_result_dir noise/ \
    --noise_values 0.1,0.5,1.0,5.0,10.0 \
    --output_dir env/vis/plots/all_noise_vis
    
python env/vis/vis.py --type all_noise_vis --model_names Qwen/Qwen3-8B,Qwen/Qwen3-14B,Qwen/Qwen3-32B,gemini-2.5-pro --refinement_level 2 --experiment_result_dir noise/ --noise_values 0.1,0.5,1.0,5.0,10.0 --output_dir env/vis/plots/all_noise_vis
"""

"""
python env/vis/vis.py \
  --type all_unblock_scale \
  --model_names Qwen/Qwen3-8B,Qwen/Qwen3-14B,Qwen/Qwen3-32B,gemini-2.5-pro,gpt-5 \
  --experiment_result_dir final_prompt/ \
  --output_dir env/vis/plots/all_unblock_scale
  
python env/vis/vis.py --type all_unblock_scale --model_names gpt-5,gemini-2.5-pro,Qwen/Qwen3-32B,Qwen/Qwen3-14B,Qwen/Qwen3-8B --experiment_result_dir final_prompt/ --output_dir env/vis/plots/all_unblock_scale --colors "#eddca5,#c99b38,#8fd7d7,#00b0be"
"""

"""
python env/vis/vis.py \
    --type all_block_scale \
    --model_names Qwen/Qwen3-8B,Qwen/Qwen3-14B,Qwen/Qwen3-32B,gemini-2.5-pro \
    --experiment_result_dir final_prompt/ \
    --refinement_level 4 \
    --block_nums 0,1,2,3 \
    --output_dir env/vis/plots/all_block_scale
    
python env/vis/vis.py \
  --type all_block_scale \
  --model_names Qwen3-14B \
  --experiment_result_dir final_prompt/Qwen3-14B\
  --refinement_level 4 \
  --block_nums 0,1,2,3 \
  --block_types ban_tool,cost_change,preference_change,steplen_change \
  --output_dir env/vis/plots/all_block_scale
  
python env/vis/vis.py --type all_block_scale --model_names Qwen3-14B --experiment_result_dir final_prompt/Qwen3-14B --refinement_level 4 --block_nums 0,1,2,3 --block_types ban_tool,cost_change,preference_change,steplen_change --output_dir env/vis/plots/all_block_scale --colors ban_tool:#3594cc,cost_change:#ea801c,preference_change:#8cc5e3,steplen_change:#f0b077

python env/vis/vis.py --type all_block_scale --model_names gemini-2.5-pro --experiment_result_dir final_prompt/gemini-2.5-pro --refinement_level 4 --block_nums 0,1,2,3 --block_types ban_tool,cost_change,preference_change,steplen_change --output_dir env/vis/plots/all_block_scale --colors ban_tool:#3594cc,cost_change:#ea801c,preference_change:#8cc5e3,steplen_change:#f0b077
"""

"""
python env/vis/vis.py \
  --model_name Qwen/Qwen3-8B \
  --type blocked_comparison \
  --refinement_level 2 \
  --block_num 1 \
  --experiment_result_dir final_prompt/Qwen3-8B \
  --output_dir env/vis/plots/Qwen3-8B/blocked_comparison/refinement_level_2

python env/vis/vis.py \
  --model_name Qwen/Qwen3-14B \
  --type blocked_comparison \
  --refinement_level 2 \
  --block_num 1 \
  --experiment_result_dir final_prompt/Qwen3-14B \
  --output_dir env/vis/plots/Qwen3-14B/blocked_comparison/refinement_level_2
  
python env/vis/vis.py \
  --model_name Llama-3.1-8B-Instruct \
  --type blocked_comparison \
  --refinement_level 2 \
  --block_num 1 \
  --experiment_result_dir final_prompt/Llama-3.1-8B-Instruct \
  --output_dir env/vis/plots/Llama-3.1-8B-Instruct/blocked_comparison/refinement_level_2
  
python env/vis/vis.py \
  --model_name Qwen/Qwen3-32B \
  --type blocked_comparison \
  --refinement_level 2 \
  --block_num 1 \
  --experiment_result_dir final_prompt/Qwen3-32B \
  --output_dir env/vis/plots/Qwen3-32B/blocked_comparison/refinement_level_2
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from pathlib import Path
import argparse
from typing import List, Dict, Any, Optional, Tuple
import math
import re

# =============================
# Global font size configuration
# =============================
# You can tweak these constants to control figure typography globally.
FONT_SIZE_TITLE = 23
FONT_SIZE_LABEL = 22
FONT_SIZE_TICK = 22
FONT_SIZE_LEGEND = 14

# Apply to matplotlib rcParams so they take effect unless explicitly overridden
matplotlib.rcParams['axes.titlesize'] = FONT_SIZE_TITLE
matplotlib.rcParams['figure.titlesize'] = FONT_SIZE_TITLE
matplotlib.rcParams['axes.labelsize'] = FONT_SIZE_LABEL
matplotlib.rcParams['xtick.labelsize'] = FONT_SIZE_TICK
matplotlib.rcParams['ytick.labelsize'] = FONT_SIZE_TICK
matplotlib.rcParams['legend.fontsize'] = FONT_SIZE_LEGEND
# matplotlib.rcParams['font.family'] = 'sans-serif'
# matplotlib.rcParams['font.sans-serif'] = ['Roboto']
# matplotlib.rcParams['font.serif'] = ['Times', 'DejaVu Serif', 'Nimbus Roman No9 L']


def _parse_timestamp_from_filename(path: Path) -> int:
    """Extract the integer timestamp from saved results filename.

    Expected pattern: results_{model}_{block_info}_{timestamp}.json
    """
    try:
        stem = path.stem  # results_{...}_{ts}
        # timestamp is the last underscore-separated token
        ts_str = stem.split('_')[-1]
        return int(ts_str)
    except Exception:
        return -1


def _strip_date_suffix(name: str) -> str:
    """Strip trailing date/datetime suffix like -20250514 or -20250514T123456.

    Keeps the rest unchanged. If no suffix, returns the original name.
    """
    if not isinstance(name, str):
        return name
    # Common patterns: -YYYYMMDD or -YYYYMMDDHHMMSS
    return re.sub(r"-\d{8}(?:\d{6})?$", "", name)


def _match_model_name(file_model_name: str, user_model_name: str) -> bool:
    """Return True if a saved run's model matches the requested model.

    Accepts either full name (e.g., "Qwen/Qwen2.5-7B-Instruct") or short name
    (e.g., "Qwen2.5-7B-Instruct").
    """
    if not file_model_name or not user_model_name:
        return False
    # Allow optional date suffix in file name (e.g., claude-sonnet-4-20250514)
    file_model_name_stripped = _strip_date_suffix(file_model_name)
    user_model_name_stripped = _strip_date_suffix(user_model_name)
    # Exact match or short-name match (with suffix stripped)
    short_in_file = file_model_name_stripped.split('/')[-1]
    short_in_user = user_model_name_stripped.split('/')[-1]
    return (
        file_model_name_stripped == user_model_name_stripped
        or short_in_file == user_model_name_stripped
        or file_model_name_stripped == short_in_user
        or short_in_file == short_in_user
    )


def _ensure_gpt_upper(s: str) -> str:
    """Ensure occurrences like gpt-xxx or Gpt-xxx are rendered as GPT-xxx."""
    try:
        return re.sub(r'(?i)gpt-', 'GPT-', s) if isinstance(s, str) else s
    except Exception:
        return s


def _pretty_baseline_name(baseline: str) -> str:
    """Convert baseline internal name to display name.
    
    Special handling:
    - steplen_change -> Remove Tools
    - Others: replace _ with space and title case
    """
    if not isinstance(baseline, str):
        return str(baseline)
    if baseline == 'steplen_change':
        return 'Remove Tools'
    return baseline.replace('_', ' ').title()


def load_data(model_name: str, experiment_result_dir: str, blocked_refinement_level: Optional[int] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Discover and load unblocked and blocked runs for a given model.

    Returns:
        (unblocked_data, blocked_data)
        - unblocked_data: list of stats dicts (one per refinement_level, latest by timestamp)
        - blocked_data: list of stats dicts for baseline comparison across block modes
    """
    result_dir = Path(experiment_result_dir)
    if not result_dir.exists():
        raise FileNotFoundError(f"Experiment result directory not found: {experiment_result_dir}")

    # Scan result files
    candidate_files = sorted(result_dir.glob('results_*.json'))
    runs: List[Tuple[Dict[str, Any], int, Path]] = []
    for fp in candidate_files:
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                data = json.load(f)
            stats = data.get('stats', {})
            file_model = stats.get('model_name')
            if not _match_model_name(file_model, model_name):
                continue
            ts = _parse_timestamp_from_filename(fp)
            runs.append((stats, ts, fp))
        except Exception:
            continue

    if not runs:
        return [], []

    # Build latest unblocked per refinement_level
    latest_unblocked_by_ref: Dict[int, Tuple[Dict[str, Any], int]] = {}
    for stats, ts, _ in runs:
        if stats.get('use_blocker'):
            continue
        rlv = int(stats.get('refinement_level', -1))
        if rlv not in latest_unblocked_by_ref or ts > latest_unblocked_by_ref[rlv][1]:
            latest_unblocked_by_ref[rlv] = (stats, ts)

    unblocked_data: List[Dict[str, Any]] = [v[0] for k, v in sorted(latest_unblocked_by_ref.items(), key=lambda x: x[0])]

    # Prepare blocked comparison at a chosen refinement level
    blocked_runs = [(s, ts) for (s, ts, _) in runs if s.get('use_blocker')]
    available_ref_levels = sorted({int(s.get('refinement_level', -1)) for (s, _) in blocked_runs})
    if blocked_refinement_level is None:
        blocked_ref_level = available_ref_levels[-1] if available_ref_levels else (unblocked_data[-1]['refinement_level'] if unblocked_data else -1)
    else:
        blocked_ref_level = blocked_refinement_level

    modes = ['ban_tool', 'cost_change', 'preference_change', 'steplen_change']
    latest_blocked_by_mode: Dict[str, Tuple[Dict[str, Any], int]] = {}
    for stats, ts in blocked_runs:
        if int(stats.get('refinement_level', -1)) != int(blocked_ref_level):
            continue
        mode = stats.get('block_mode')
        if mode not in modes:
            continue
        if mode not in latest_blocked_by_mode or ts > latest_blocked_by_mode[mode][1]:
            latest_blocked_by_mode[mode] = (stats, ts)

    blocked_data: List[Dict[str, Any]] = [v[0] for k, v in sorted(latest_blocked_by_mode.items(), key=lambda x: modes.index(x[0]))]

    # Add one unblocked run for baseline comparison (same refinement level if available)
    if blocked_data:
        # pick unblocked at the same ref level
        baseline_unblocked = None
        for stats, _ in latest_unblocked_by_ref.items():
            pass  # placeholder to keep structure
        if blocked_ref_level in latest_unblocked_by_ref:
            baseline_unblocked = latest_unblocked_by_ref[blocked_ref_level][0]
        elif unblocked_data:
            baseline_unblocked = unblocked_data[-1]
        if baseline_unblocked is not None:
            blocked_data.append(baseline_unblocked)
    
    return unblocked_data, blocked_data


def _candidate_result_dirs(base_dir: Path, model_name: str) -> List[Path]:
    """Return plausible subdirectories under base_dir where this model's results may live.

    Includes:
      - base_dir itself
      - base_dir / short_model_name
      - base_dir / model_name with '/' replaced by '-'
    """
    candidates: List[Path] = []
    if base_dir.exists():
        candidates.append(base_dir)
    short_name = model_name.split('/')[-1]
    dash_full = model_name.replace('/', '-')
    for sub in [short_name, dash_full]:
        p = base_dir / sub
        if p.exists() and p.is_dir():
            candidates.append(p)
    # Deduplicate while preserving order
    seen = set()
    unique: List[Path] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def _discover_runs(model_name: str, experiment_result_dir: str) -> List[Tuple[Dict[str, Any], int, Path, Dict[str, Any]]]:
    """Scan a directory for results_*.json and return matching runs for a model.

    Returns list of tuples: (stats, timestamp, path, full_json_data)
    """
    result_dir = Path(experiment_result_dir)
    if not result_dir.exists():
        return []
    # Build candidate directories to search
    dirs_to_scan = _candidate_result_dirs(result_dir, model_name)
    candidate_files: List[Path] = []
    for d in dirs_to_scan:
        candidate_files.extend(sorted(d.glob('results_*.json')))
    runs: List[Tuple[Dict[str, Any], int, Path, Dict[str, Any]]] = []
    for fp in candidate_files:
        try:
            if fp.stat().st_size == 0:
                print(f"[WARN] Empty result file skipped: {fp}")
                continue
            with open(fp, 'r', encoding="utf-8") as f:
                data = json.load(f)
            stats = data.get('stats', {})
            file_model = stats.get('model_name')
            if not _match_model_name(file_model, model_name):
                continue
            ts = _parse_timestamp_from_filename(fp)
            if ts < 0:
                # Fallback: use file modification time if no timestamp suffix found
                try:
                    ts = int(fp.stat().st_mtime)
                except Exception:
                    ts = 0
            if not isinstance(stats, dict) or not stats:
                print(f"[WARN] Missing or empty stats, skipped: {fp.name}")
                continue
            runs.append((stats, ts, fp, data))
        except json.JSONDecodeError:
            print(f"[ERROR] JSON parse failed, skipped: {fp}")
        except Exception as e:
            print(f"[ERROR] Failed to read {fp}: {e}")
    return runs


def prepare_blocked_comparison_entries(
    model_name: str,
    experiment_result_dir: str,
    refinement_level: int,
    block_num: Optional[int]
) -> Dict[str, Dict[str, Any]]:
    """Pick latest entries per block mode for a given refinement_level and block_num.

    Returns mapping: mode -> full_json_data. Also includes key 'unblocked' if available.
    """
    runs = _discover_runs(model_name, experiment_result_dir)
    if not runs:
        return {}

    modes = ['ban_tool', 'cost_change', 'preference_change', 'steplen_change']
    latest_by_mode: Dict[str, Tuple[int, Dict[str, Any]]] = {}
    latest_unblocked: Tuple[int, Dict[str, Any]] = None  # type: ignore

    for stats, ts, _fp, full in runs:
        try:
            use_blocker = bool(stats.get('use_blocker'))
            rlv = int(stats.get('refinement_level', -1))
            if rlv != int(refinement_level):
                continue
            if use_blocker:
                if block_num is not None and int(stats.get('block_num', -1)) != int(block_num):
                    continue
                mode = stats.get('block_mode')
                if mode in modes:
                    prev = latest_by_mode.get(mode)
                    if (prev is None) or (ts > prev[0]):
                        latest_by_mode[mode] = (ts, full)
            else:
                # unblocked baseline (ignore block_num)
                prev_u = latest_unblocked
                if (prev_u is None) or (ts > prev_u[0]):
                    latest_unblocked = (ts, full)
        except Exception:
            continue

    result: Dict[str, Dict[str, Any]] = {}
    for m in modes:
        if m in latest_by_mode:
            result[m] = latest_by_mode[m][1]
    if latest_unblocked is not None:
        result['unblocked'] = latest_unblocked[1]
    return result


def prepare_block_scaling_entries(
    model_name: str,
    experiment_result_dir: str,
    refinement_level: int,
    block_mode: str
) -> Dict[int, Dict[str, Any]]:
    """Pick latest entries per block_num for a given refinement_level and a fixed block_mode.

    Returns mapping: block_num -> full_json_data
    """
    runs = _discover_runs(model_name, experiment_result_dir)
    if not runs:
        return {}

    latest_by_bn: Dict[int, Tuple[int, Dict[str, Any]]] = {}
    for stats, ts, _fp, full in runs:
        try:
            if not stats.get('use_blocker'):
                continue
            rlv = int(stats.get('refinement_level', -1))
            if rlv != int(refinement_level):
                continue
            mode = stats.get('block_mode')
            if mode != block_mode:
                continue
            bn = int(stats.get('block_num', -1))
            prev = latest_by_bn.get(bn)
            if (prev is None) or (ts > prev[0]):
                latest_by_bn[bn] = (ts, full)
        except Exception:
            continue

    result: Dict[int, Dict[str, Any]] = {}
    for bn, tpl in sorted(latest_by_bn.items(), key=lambda x: x[0]):
        result[bn] = tpl[1]
    return result


def _extract_agent_perf(stats: Dict[str, Any]) -> Dict[str, float]:
    scores = stats.get('scores', {}) if stats else {}
    am = scores.get('accuracy_metrics', {}) or {}
    tpm = scores.get('tool_path_metrics', {}) or {}
    vals = {
        'final_answer_accuracy': am.get('final_answer_accuracy', 0.0),
        'avg_normalized_edit_distance': tpm.get('avg_normalized_edit_distance', 0.0),
        'avg_edit_distance': tpm.get('avg_edit_distance', 0.0),
        'tool_path_exact_match_ratio': tpm.get('tool_path_exact_match_ratio', 0.0),
    }
    out: Dict[str, float] = {}
    for k, v in vals.items():
        try:
            f = float(v if v is not None else 0.0)
            if math.isnan(f) or math.isinf(f):
                print(f"[WARN] Invalid numeric value for {k}: {v}. Coerced to 0.0")
                f = 0.0
            out[k] = f
        except Exception:
            print(f"[WARN] Non-numeric value for {k}: {v}. Coerced to 0.0")
            out[k] = 0.0
    return out


def _extract_costs(stats: Dict[str, Any]) -> Tuple[float, float, float]:
    scores = stats.get('scores', {}) if stats else {}
    cm = scores.get('cost_metrics', {}) or {}
    stim = scores.get('stimulation_metrics', {}) or {}
    stim_cm = (stim.get('cost_metrics', {}) or {})
    keys = [('agent_cost', cm.get('avg_agent_cost', 0.0)), ('gt_cost', cm.get('avg_gt_cost', 0.0)), ('stim_cost', stim_cm.get('avg_agent_cost', 0.0))]
    outs: List[float] = []
    for name, v in keys:
        try:
            f = float(v if v is not None else 0.0)
            if math.isnan(f) or math.isinf(f):
                print(f"[WARN] Invalid numeric value for {name}: {v}. Coerced to 0.0")
                f = 0.0
            outs.append(f)
        except Exception:
            print(f"[WARN] Non-numeric value for {name}: {v}. Coerced to 0.0")
            outs.append(0.0)
    return outs[0], outs[1], outs[2]


def _extract_stim_path(stats: Dict[str, Any]) -> Dict[str, float]:
    scores = stats.get('scores', {}) if stats else {}
    stim = scores.get('stimulation_metrics', {}) or {}
    tpm = (stim.get('tool_path_metrics', {}) or {})
    vals = {
        'avg_normalized_edit_distance': tpm.get('avg_normalized_edit_distance', 0.0),
        'avg_edit_distance': tpm.get('avg_edit_distance', 0.0),
        'tool_path_exact_match_ratio': tpm.get('tool_path_exact_match_ratio', 0.0),
    }
    out: Dict[str, float] = {}
    for k, v in vals.items():
        try:
            f = float(v if v is not None else 0.0)
            if math.isnan(f) or math.isinf(f):
                print(f"[WARN] Invalid numeric value (stim) for {k}: {v}. Coerced to 0.0")
                f = 0.0
            out[k] = f
        except Exception:
            print(f"[WARN] Non-numeric value (stim) for {k}: {v}. Coerced to 0.0")
            out[k] = 0.0
    return out


def _count_gt_unblocked(full_data: Dict[str, Any]) -> int:
    try:
        results = full_data.get('results', []) or []
        return sum(1 for item in results if bool(item.get('gt_unblocked', False)))
    except Exception:
        return 0


# =============================
# Block-scaling specialized plots
# =============================
def _sorted_block_points(entries_by_bn: Dict[int, Dict[str, Any]]) -> Tuple[List[int], List[Dict[str, Any]]]:
    try:
        items = sorted(entries_by_bn.items(), key=lambda x: int(x[0]))
    except Exception:
        items = sorted(entries_by_bn.items(), key=lambda x: x[0])
    block_nums = [int(k) for k, _ in items]
    datas = [v for _, v in items]
    return block_nums, datas


def _prepare_all_block_scale_entries(
    model_names: List[str],
    experiment_result_dir: str,
    refinement_level: int,
    block_nums: Optional[List[int]] = None,
    block_types: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Dict[int, Dict[str, Any]]]]:
    """For each model, collect latest runs per block_num and per block_type at a fixed RL.

    Special case: block_num == 0 maps to the latest unblocked run at that RL.

    Returns mapping: model_name -> { block_type -> { block_num -> full_json_data } }
    """
    modes_all = ['ban_tool', 'cost_change', 'preference_change', 'steplen_change']
    modes_filter = [m for m in (block_types or modes_all) if m in modes_all]
    if not modes_filter:
        modes_filter = modes_all

    result: Dict[str, Dict[str, Dict[int, Dict[str, Any]]]] = {}
    for m in model_names:
        try:
            runs = _discover_runs(m, experiment_result_dir)
            if not runs:
                print(f"[WARN] No runs discovered for model: {m} (skipped)")
                continue
            # Prepare storage: per mode per block_num latest
            latest_by_mode_bn: Dict[str, Dict[int, Tuple[int, Dict[str, Any]]]] = {mode: {} for mode in modes_filter}
            latest_unblocked: Optional[Tuple[int, Dict[str, Any]]] = None
            for stats, ts, _fp, full in runs:
                try:
                    rlv = int(stats.get('refinement_level', -1))
                    if rlv != int(refinement_level):
                        continue
                    use_blocker = bool(stats.get('use_blocker'))
                    if not use_blocker:
                        # candidate for unblocked baseline at this RL
                        if (latest_unblocked is None) or (ts > latest_unblocked[0]):
                            latest_unblocked = (ts, full)
                        continue
                    mode = stats.get('block_mode')
                    if mode not in modes_filter:
                        continue
                    bn = int(stats.get('block_num', -1))
                    if block_nums is not None and bn not in block_nums:
                        continue
                    prev = latest_by_mode_bn[mode].get(bn)
                    if (prev is None) or (ts > prev[0]):
                        latest_by_mode_bn[mode][bn] = (ts, full)
                except Exception:
                    continue

            # Build final mapping per model
            model_map: Dict[str, Dict[int, Dict[str, Any]]] = {}
            for mode in modes_filter:
                bn_map_raw = latest_by_mode_bn.get(mode, {})
                bn_map: Dict[int, Dict[str, Any]] = {bn: tpl[1] for bn, tpl in bn_map_raw.items()}
                # inject bn=0 from unblocked when requested by block_nums
                if (block_nums is None or 0 in block_nums) and latest_unblocked is not None:
                    bn_map[0] = latest_unblocked[1]
                if bn_map:
                    model_map[mode] = bn_map
            if not model_map:
                print(f"[WARN] No block-scaling runs for model: {m} (skipped)")
                continue
            result[m] = model_map
        except Exception as e:
            print(f"[ERROR] Failed preparing all_block_scale entries for {m}: {e}")
    return result


def _plot_block_scale_per_model_metric(
    model_name: str,
    entries_by_mode_bn: Dict[str, Dict[int, Dict[str, Any]]],
    output_dir: str,
    metric_key: str,
    metric_title: str,
    file_tag: str,
    refinement_level: Optional[int] = None,
    normalize: bool = True,
    baseline_color_map: Optional[Dict[str, str]] = None,
) -> None:
    """Draw a single-metric line chart for one model, lines = block types, x = block_num.

    If normalize=True, values are min-max normalized across all available points.
    """
    if not entries_by_mode_bn:
        print(f"[WARN] _plot_block_scale_per_model_metric: empty for model {model_name}.")
        return

    modes_order = ['ban_tool', 'cost_change', 'preference_change', 'steplen_change']
    modes_present = [m for m in modes_order if m in entries_by_mode_bn]
    if not modes_present:
        print(f"[WARN] No block types present for model {model_name}.")
        return

    # union x ticks across modes
    all_block_nums: List[int] = []
    for mode in modes_present:
        for bn in entries_by_mode_bn.get(mode, {}).keys():
            if bn not in all_block_nums:
                all_block_nums.append(bn)
    all_block_nums.sort()
    if not all_block_nums:
        print(f"[WARN] No block nums for model {model_name}.")
        return

    # colors for modes
    mode_colors = {
        'ban_tool': '#2ca02c',
        'cost_change': '#d62728',
        'preference_change': '#9467bd',
        'steplen_change': '#1f77b4',
    }
    if baseline_color_map:
        for key, value in baseline_color_map.items():
            if key in mode_colors and value:
                mode_colors[key] = value

    # collect global mins/maxes for normalization
    vmin, vmax, denom = 0.0, 1.0, 1.0
    if normalize:
        vals: List[float] = []
        for mode in modes_present:
            for bn in all_block_nums:
                full = entries_by_mode_bn.get(mode, {}).get(bn)
                if not full:
                    continue
                perf = _extract_agent_perf(full.get('stats', {}))
                try:
                    v = float(perf.get(metric_key, np.nan))
                    if not (math.isnan(v) or math.isinf(v)):
                        vals.append(v)
                except Exception:
                    continue
        if not vals:
            print(f"[WARN] No values for {metric_key} in model {model_name}.")
            return
        vmin, vmax = min(vals), max(vals)
        denom = (vmax - vmin) if (vmax > vmin) else 1.0

    fig, ax = plt.subplots(figsize=(10, 5))
    for mode in modes_present:
        xs: List[int] = []
        ys_raw: List[float] = []
        for bn in all_block_nums:
            full = entries_by_mode_bn.get(mode, {}).get(bn)
            if not full:
                continue
            perf = _extract_agent_perf(full.get('stats', {}))
            try:
                val = float(perf.get(metric_key, np.nan))
            except Exception:
                val = np.nan
            if not (math.isnan(val) or math.isinf(val)):
                xs.append(int(bn))
                ys_raw.append(val)
        if xs and ys_raw:
            ys = [ (y - vmin) / denom for y in ys_raw ] if normalize else ys_raw
            ax.plot(
                xs,
                ys,
                marker='o',
                label=_pretty_baseline_name(mode),
                color=mode_colors.get(mode),
                linewidth=2,
            )

    ax.set_xlabel('Number of Blockings')
    ax.set_ylabel(metric_title)
    title_model = _ensure_gpt_upper(model_name)
    if title_model:
        title_model = title_model.title()
        # title() would lowercase GPT, fix back to uppercase if needed
        title_model = _ensure_gpt_upper(title_model)
    title_suffix = ' (Normalized 0–1)' if normalize else ''
    ax.set_title(f"{title_model}{title_suffix}", fontstyle='italic')
    ax.set_xticks(all_block_nums)
    ax.set_xticklabels([str(v) for v in all_block_nums])
    if normalize:
        # Dynamically set the y-axis range to the min/max of the plotted curves
        ymins: List[float] = []
        ymaxs: List[float] = []
        for line in ax.get_lines():
            data = line.get_ydata(orig=False)
            if data is None or len(data) == 0:
                continue
            arr = np.asarray(data, dtype=float)
            arr = arr[~np.isnan(arr) & ~np.isinf(arr)]
            if arr.size:
                ymins.append(float(np.min(arr)))
                ymaxs.append(float(np.max(arr)))
        if ymins and ymaxs:
            ylo, yhi = min(ymins), max(ymaxs)
            if yhi <= ylo:
                pad = 1e-6
                ax.set_ylim(ylo - pad, yhi + pad)
            else:
                ax.set_ylim(ylo, yhi)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rl_part = f"_rl{refinement_level}" if refinement_level is not None else ""
    clean_model_name = _ensure_gpt_upper(model_name).replace('/', '_').replace(' ', '_')
    norm_tag = 'normalized' if normalize else 'raw'
    out_file = output_path / f'{clean_model_name}{rl_part}_block_scale_{file_tag}_{norm_tag}.pdf'
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Saved per-model block scale {metric_key} ({'normalized' if normalize else 'raw'}) plot to: {out_file}")
    plt.close(fig)
def plot_block_scaling_costs(entries_by_bn: Dict[int, Dict[str, Any]], model_name: str, block_mode: str, output_dir: str, refinement_level: Optional[int] = None, filename_suffix: str = "block_scaling_costs") -> None:
    block_nums, datas = _sorted_block_points(entries_by_bn)
    agent_costs: List[float] = []
    gt_costs: List[float] = []
    stim_costs: List[float] = []
    for d in datas:
        stats = d.get('stats', {}) if isinstance(d, dict) else {}
        a, g, s = _extract_costs(stats)
        agent_costs.append(a)
        gt_costs.append(g)
        stim_costs.append(s)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(block_nums))
    width = 0.25
    b1 = ax.bar(x - width, agent_costs, width, label='Agent', color='#d62728', alpha=0.85)
    b2 = ax.bar(x, gt_costs, width, label='GT', color='#2ca02c', alpha=0.85)
    b3 = ax.bar(x + width, stim_costs, width, label='Stimulation', color='#1f77b4', alpha=0.85)
    ax.set_xlabel('Block Num')
    ax.set_ylabel('Cost')
    title_rl = f" | RL={refinement_level}" if refinement_level is not None else ""
    ax.set_title(f'Block Scaling - Costs ({block_mode}){title_rl}')
    ax.set_xticks(x)
    ax.set_xticklabels([str(bn) for bn in block_nums])
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    plt.tight_layout()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    clean_model_name = _ensure_gpt_upper(model_name).replace('/', '_').replace(' ', '_')
    rl_part = f"_rl{refinement_level}" if refinement_level is not None else ""
    filename = f"{clean_model_name}_{block_mode}{rl_part}_{filename_suffix}.pdf"
    plt.savefig(output_path / filename, dpi=300, bbox_inches='tight')
    print(f"Saved block-scaling costs plot to: {output_path / filename}")
    plt.close(fig)


def plot_all_block_comparison_across_models(
    entries_maps_by_model: Dict[str, Dict[str, Any]],
    output_dir: str,
    refinement_level: Optional[int] = None,
    normalize: bool = True,
) -> None:
    """Across multiple models, draw four figures (Accuracy, NED, ED, EM).

    X-axis baselines: unblocked + four blocking modes. Each model is a series.
    entries_maps_by_model: model_name -> entries_map (from prepare_blocked_comparison_entries)
    """
    # Enforce requested order for modes
    baselines = ['cost_change', 'ban_tool', 'preference_change', 'steplen_change', 'unblocked']
    mode_colors = {
        'cost_change': '#d62728',
        'ban_tool': '#2ca02c',
        'preference_change': '#9467bd',
        'steplen_change': '#1f77b4',
        'unblocked': '#8c564b',
    }
    metric_specs = [
        ('Final Answer Accuracy', 'final_answer_accuracy', 'all_block_accuracy'),
        ('Normalized Edit Distance', 'avg_normalized_edit_distance', 'all_block_ned'),
        ('Edit Distance', 'avg_edit_distance', 'all_block_ed'),
        ('Tool Path EMR', 'tool_path_exact_match_ratio', 'all_block_em'),
    ]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    title_rl = f" | RL={refinement_level}" if refinement_level is not None else ""
    rl_part = f"_rl{refinement_level}" if refinement_level is not None else ""

    model_names = list(entries_maps_by_model.keys())
    if not model_names:
        print("[WARN] plot_all_block_comparison_across_models: no models provided.")
        return

    # X-axis will be models; legend will be modes

    # Helper to shorten legend labels
    def pretty_model_label(name: str) -> str:
        try:
            short = name.split('/')[-1]
            label = (short[:1].upper() + short[1:]) if short else short
            return _ensure_gpt_upper(label)
        except Exception:
            label = (name[:1].upper() + name[1:]) if name else name
            return _ensure_gpt_upper(label)

    x = np.arange(len(model_names))

    for pretty, key, fname in metric_specs:
        fig, ax = plt.subplots(figsize=(20, 16))
        any_series = False
        # group bars per model; each model has bars for five modes
        width = max(0.08, 0.8 / max(1, len(baselines)))
        # collect values for normalization if requested
        vmin, vmax, denom = 0.0, 1.0, 1.0
        if normalize:
            vals: List[float] = []
            for m in model_names:
                entries_map = entries_maps_by_model.get(m, {})
                for b in baselines:
                    try:
                        stats = entries_map.get(b, {}).get('stats', {})
                        perf = _extract_agent_perf(stats)
                        val = float(perf.get('final_answer_accuracy' if key == 'final_answer_accuracy' else key, np.nan))
                        if not (math.isnan(val) or math.isinf(val)):
                            vals.append(val)
                    except Exception:
                        continue
            if vals:
                vmin, vmax = min(vals), max(vals)
                denom = (vmax - vmin) if (vmax > vmin) else 1.0
        for j, b in enumerate(baselines):
            ys: List[float] = []
            for m in model_names:
                entries_map = entries_maps_by_model.get(m, {})
                try:
                    stats = entries_map.get(b, {}).get('stats', {})
                    perf = _extract_agent_perf(stats)
                    raw_val = float(perf.get('final_answer_accuracy' if key == 'final_answer_accuracy' else key, 0.0))
                except Exception:
                    raw_val = np.nan
                if normalize:
                    if math.isnan(raw_val) or math.isinf(raw_val):
                        ys.append(np.nan)
                    else:
                        ys.append((raw_val - vmin) / denom)
                else:
                    ys.append(raw_val)
            if np.all(np.isnan(ys)):
                continue
            any_series = True
            offs = (j - (len(baselines)-1)/2) * width
            ax.bar(x + offs, ys, width, label=_pretty_baseline_name(b), color=mode_colors.get(b, None), alpha=0.9)

        if not any_series:
            plt.close(fig)
            print(f"[WARN] No data to plot for {pretty} across models.")
            continue

        ax.set_xticks(x)
        ax.set_xticklabels([pretty_model_label(m) for m in model_names], rotation=15, ha='center')
        for _txt in ax.get_xticklabels():
            _txt.set_fontstyle('italic')
        # strip textual elements per requirement
        ax.set_ylabel('')
        if normalize:
            # Dynamic y-axis range (based on the data plotted in this figure)
            ymins: List[float] = []
            ymaxs: List[float] = []
            for rect in ax.patches:
                try:
                    y = float(rect.get_height())
                except Exception:
                    continue
                if not (math.isnan(y) or math.isinf(y)):
                    ymins.append(y)
                    ymaxs.append(y)
            for line in ax.get_lines():
                data = line.get_ydata(orig=False)
                if data is None or len(data) == 0:
                    continue
                arr = np.asarray(data, dtype=float)
                arr = arr[~np.isnan(arr) & ~np.isinf(arr)]
                if arr.size:
                    ymins.append(float(np.min(arr)))
                    ymaxs.append(float(np.max(arr)))
            if ymins and ymaxs:
                ylo, yhi = min(ymins), max(ymaxs)
                if yhi <= ylo:
                    pad = 1e-6
                    ax.set_ylim(ylo - pad, yhi + pad)
                else:
                    ax.set_ylim(ylo, yhi)
        # ax.set_title(pretty)
        ax.set_title(pretty, fontsize=FONT_SIZE_TITLE)
        ax.grid(True, alpha=0.3, axis='y')
        leg = ax.get_legend()
        if leg:
            leg.remove()
        plt.tight_layout()

        # ensure single extension
        norm_tag = 'normalized' if normalize else 'raw'
        filename = f'all_models{rl_part}_{Path(fname).stem}_{norm_tag}.pdf'
        plt.savefig(output_path / filename, dpi=300, bbox_inches='tight')
        print(f"Saved all-model block comparison plot to: {output_path / filename}")
        plt.close(fig)


def plot_all_block_comparison_em_ned_combined(
    entries_maps_by_model: Dict[str, Dict[str, Any]],
    output_dir: str,
    refinement_level: Optional[int] = None,
    normalize: bool = True,
    baseline_color_map: Optional[Dict[str, Any]] = None,
) -> None:
    """In all_block_comparison mode, draw two side-by-side subplots (EM and NED) following the all_unblock_scale layout.

    - X-axis is the model; the legend lists the baseline modes (same as existing all_block_comparison).
    - The two subplots show EMR and ANED, respectively.
    - Each subplot applies global min-max normalization of the metric to the [0,1] range.
    """
    baselines = ['cost_change', 'ban_tool', 'preference_change', 'steplen_change', 'unblocked']
    mode_colors = {
        'cost_change': '#d62728',
        'ban_tool': '#2ca02c',
        'preference_change': "#987fae",
        'steplen_change': '#1f77b4',
        'unblocked': '#8c564b',
    }
    # Allow CLI-provided --colors to override baseline colors
    if baseline_color_map:
        for k, v in baseline_color_map.items():
            if k in mode_colors and v:
                mode_colors[k] = v

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rl_part = f"_rl{refinement_level}" if refinement_level is not None else ""

    model_names = list(entries_maps_by_model.keys())
    if not model_names:
        print("[WARN] plot_all_block_comparison_em_ned_combined: no models provided.")
        return

    def pretty_model_label(name: str) -> str:
        try:
            short = name.split('/')[-1]
            label = (short[:1].upper() + short[1:]) if short else short
            return _ensure_gpt_upper(label)
        except Exception:
            label = (name[:1].upper() + name[1:]) if name else name
            return _ensure_gpt_upper(label)

    # Collect the raw values of the specified metric (all models, all baselines) for min-max scaling
    def collect_raw_values(metric_key: str) -> List[float]:
        vals: List[float] = []
        for m in model_names:
            emap = entries_maps_by_model.get(m, {})
            for b in baselines:
                try:
                    stats = emap.get(b, {}).get('stats', {})
                    perf = _extract_agent_perf(stats)
                    v = float(perf.get(metric_key, np.nan))
                    if not (math.isnan(v) or math.isinf(v)):
                        vals.append(v)
                except Exception:
                    continue
        return vals

    # Create two side-by-side subplots: ANED on the left, EMR on the right
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    subplots_spec = [
        ("ANED", 'avg_normalized_edit_distance', 'em_ned_combined_left'),
        ("EMR", 'tool_path_exact_match_ratio', 'em_ned_combined_right'),
    ]

    x = np.arange(len(model_names))
    width = max(0.08, 0.8 / max(1, len(baselines)))

    # Note: colors are fixed per baseline; override with baseline_color_map when provided

    for ax, (title_txt, metric_key, _stub) in zip(axes, subplots_spec):
        if normalize:
            raw_vals = collect_raw_values(metric_key)
            if raw_vals:
                vmin, vmax = min(raw_vals), max(raw_vals)
            else:
                vmin, vmax = 0.0, 1.0
            denom = (vmax - vmin) if (vmax > vmin) else 1.0
        else:
            vmin, denom = 0.0, 1.0

        any_series = False
        for j, b in enumerate(baselines):
            ys: List[float] = []
            for m in model_names:
                try:
                    stats = entries_maps_by_model.get(m, {}).get(b, {}).get('stats', {})
                    perf = _extract_agent_perf(stats)
                    val = float(perf.get(metric_key, np.nan))
                    if math.isnan(val) or math.isinf(val):
                        ys.append(np.nan)
                    else:
                        ys.append((val - vmin) / denom if normalize else val)
                except Exception:
                    ys.append(np.nan)
            if np.all(np.isnan(ys)):
                continue
            any_series = True
            offs = (j - (len(baselines)-1)/2) * width
            ax.bar(x + offs, ys, width, label=_pretty_baseline_name(b), color=mode_colors.get(b, None), alpha=0.9)

        if not any_series:
            print(f"[WARN] No data to plot for {title_txt} (combined).")
            continue

        ax.set_xticks(x)
        ax.set_xticklabels([pretty_model_label(m) for m in model_names], rotation=15, ha='center')
        for _txt in ax.get_xticklabels():
            _txt.set_fontstyle('italic')
        ax.set_ylabel('')
        ax.set_title(title_txt)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()

    plt.tight_layout()
    norm_tag = 'normalized' if normalize else 'raw'
    filename = f'all_models{rl_part}_em_ned_combined_{norm_tag}.pdf'
    plt.savefig(output_path / filename, dpi=300, bbox_inches='tight')
    print(f"Saved all-model EM/NED combined plot to: {output_path / filename}")
    plt.close(fig)

def prepare_unblocked_scaling_entries(
    model_names: List[str],
    experiment_result_dir: str,
    refinement_levels: Optional[List[int]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Collect the latest unblocked runs per refinement level for multiple models (with auto-discovered paths).

    - Reuse the directory discovery logic from blocked comparisons (supports base_dir and subdirectories).
    - Optionally filter by the provided refinement_levels.
    Returns: model_name -> [stats dict sorted by refinement_level]
    """
    result: Dict[str, List[Dict[str, Any]]] = {}
    for m in model_names:
        try:
            runs = _discover_runs(m, experiment_result_dir)
            if not runs:
                print(f"[WARN] No runs discovered for model: {m} (skipped)")
                continue
            latest_unblocked_by_ref: Dict[int, Tuple[Dict[str, Any], int]] = {}
            for stats, ts, _fp, _full in runs:
                try:
                    if bool(stats.get('use_blocker')):
                        continue
                    rlv = int(stats.get('refinement_level', -1))
                    if refinement_levels is not None and rlv not in refinement_levels:
                        continue
                    prev = latest_unblocked_by_ref.get(rlv)
                    if (prev is None) or (ts > prev[1]):
                        latest_unblocked_by_ref[rlv] = (stats, ts)
                except Exception:
                    continue
            if not latest_unblocked_by_ref:
                print(f"[WARN] No unblocked runs for model: {m} (skipped)")
                continue
            sorted_stats = [v[0] for k, v in sorted(latest_unblocked_by_ref.items(), key=lambda x: x[0])]
            result[m] = sorted_stats
        except Exception as e:
            print(f"[ERROR] Failed preparing unblocked scaling entries for {m}: {e}")
    return result


def plot_all_unblocked_scaling_across_models(
    entries_unblocked_by_model: Dict[str, List[Dict[str, Any]]],
    output_dir: str,
    normalize: bool = True,
    custom_colors: Optional[List[str]] = None,
) -> None:
    """Plot side-by-side bar charts for unblocked metrics (ANED and EM) across models.

    - X-axis: models (tick labels use the model short name, capitalize the first letter, force GPT prefix to uppercase, italicized).
    - Within each model: grouped bars per refinement level, mapped to task sequence = refinement_level + 3.
    - Optional custom_colors list overrides the palette for each task sequence in order.
    - When normalize=True, apply global min-max normalization of each metric to [0,1].
    """
    if not entries_unblocked_by_model:
        print("[WARN] plot_all_unblocked_scaling_across_models: empty entries.")
        return

    # Preserve the input order without additional sorting
    model_names = list(entries_unblocked_by_model.keys())
    if not model_names:
        print("[WARN] plot_all_unblocked_scaling_across_models: no models provided.")
        return

    # Gather all refinement levels (ascending)
    all_ref_levels: List[int] = []
    for m in model_names:
        for stats in entries_unblocked_by_model.get(m, []):
            try:
                rl = int(stats.get('refinement_level', -1))
            except Exception:
                continue
            if rl < 0:
                continue
            if rl not in all_ref_levels:
                all_ref_levels.append(rl)
    all_ref_levels.sort()
    if not all_ref_levels:
        print("[WARN] No refinement levels found for all_unblock_scale plot.")
        return

    # Compute the task sequence (RL + 3) mapping
    task_seq_labels = {rl: rl + 3 for rl in all_ref_levels}

    # Color mapping: prefer the custom list, otherwise fall back to tab10
    base_cmap = plt.get_cmap('tab10')
    rl_colors: Dict[int, Any] = {}
    for idx, rl in enumerate(all_ref_levels):
        if custom_colors and idx < len(custom_colors):
            rl_colors[rl] = custom_colors[idx]
        elif custom_colors and len(custom_colors) > 0:
            # If there are fewer provided colors than levels, cycle through the list
            rl_colors[rl] = custom_colors[idx % len(custom_colors)]
        else:
            rl_colors[rl] = base_cmap(idx % 10)

    # Build a mapping of per-model -> refinement level -> stats for quick lookup
    stats_by_model_rl: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for m in model_names:
        stats_map: Dict[int, Dict[str, Any]] = {}
        for stats in entries_unblocked_by_model.get(m, []):
            try:
                rl = int(stats.get('refinement_level', -1))
            except Exception:
                continue
            if rl < 0:
                continue
            stats_map[rl] = stats
        stats_by_model_rl[m] = stats_map

    def pretty_model_label(name: str) -> str:
        try:
            short = name.split('/')[-1]
            label = (short[:1].upper() + short[1:]) if short else short
        except Exception:
            label = (name[:1].upper() + name[1:]) if name else name
        return _ensure_gpt_upper(label)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Metrics to visualize
    metric_specs = [
        ('ANED', 'avg_normalized_edit_distance'),
        ('Tool Path EMR', 'tool_path_exact_match_ratio'),
    ]

    # Pre-collect raw values for normalization
    raw_values_by_metric: Dict[str, List[float]] = {key: [] for _, key in metric_specs}
    for _label, metric_key in metric_specs:
        for m in model_names:
            stats_map = stats_by_model_rl.get(m, {})
            for rl in all_ref_levels:
                stats = stats_map.get(rl)
                if not stats:
                    continue
                perf = _extract_agent_perf(stats)
                try:
                    v = float(perf.get(metric_key, np.nan))
                except Exception:
                    v = np.nan
                if math.isnan(v) or math.isinf(v):
                    continue
                raw_values_by_metric[metric_key].append(v)

    fig, axes = plt.subplots(1, 2, figsize=(20, 5))
    x_positions = np.arange(len(model_names))
    group_width = 0.8
    bar_width = group_width / max(len(all_ref_levels), 1)

    legend_entries: Dict[int, Any] = {}

    for ax, (y_label, metric_key) in zip(axes, metric_specs):
        if normalize and raw_values_by_metric.get(metric_key):
            vals = raw_values_by_metric[metric_key]
            vmin = min(vals)
            vmax = max(vals)
            denom = (vmax - vmin) if (vmax > vmin) else 1.0
        elif normalize:
            vmin, denom = 0.0, 1.0
        else:
            vmin, denom = 0.0, 1.0

        for idx, rl in enumerate(all_ref_levels):
            heights: List[float] = []
            for m in model_names:
                stats = stats_by_model_rl.get(m, {}).get(rl)
                if not stats:
                    heights.append(np.nan)
                    continue
                perf = _extract_agent_perf(stats)
                try:
                    raw_val = float(perf.get(metric_key, np.nan))
                except Exception:
                    raw_val = np.nan
                if normalize:
                    if math.isnan(raw_val) or math.isinf(raw_val):
                        heights.append(np.nan)
                    else:
                        heights.append((raw_val - vmin) / denom)
                else:
                    heights.append(raw_val)

            offset = (idx - (len(all_ref_levels) - 1) / 2.0) * bar_width
            bars = ax.bar(
                x_positions + offset,
                heights,
                bar_width * 0.9,
                color=rl_colors.get(rl),
                label=f"task sequence = {task_seq_labels[rl]}",
                alpha=0.9,
            )
            if rl not in legend_entries:
                legend_entries[rl] = bars[0]

        ax.set_xticks(x_positions)
        tick_labels = [pretty_model_label(m) for m in model_names]
        ax.set_xticklabels(tick_labels)
        for tick in ax.get_xticklabels():
            tick.set_fontstyle('italic')
        ax.set_ylabel(y_label)
        ax.set_xlabel('')
        ax.grid(True, alpha=0.3, axis='y')

        if normalize:
            ymins: List[float] = []
            ymaxs: List[float] = []
            for rect in ax.patches:
                try:
                    height = float(rect.get_height())
                except Exception:
                    continue
                if math.isnan(height) or math.isinf(height):
                    continue
                ymins.append(height)
                ymaxs.append(height)
            if ymins and ymaxs:
                ylo, yhi = min(ymins), max(ymaxs)
                if yhi <= ylo:
                    pad = 1e-6
                    ax.set_ylim(ylo - pad, yhi + pad)
                else:
                    ax.set_ylim(ylo, yhi)

    # Place a shared legend at the top
    legend_handles = [legend_entries[rl] for rl in all_ref_levels if rl in legend_entries]
    legend_labels = [f"task sequence = {task_seq_labels[rl]}" for rl in all_ref_levels if rl in legend_entries]
    if legend_handles and legend_labels:
        fig.legend(legend_handles, legend_labels, loc='upper center', ncol=min(len(legend_labels), 4), frameon=False, fontsize=FONT_SIZE_LEGEND)

    plt.tight_layout(rect=(0, 0, 1, 0.92))

    norm_tag = 'normalized' if normalize else 'raw'
    out_file = output_path / f'all_models_unblocked_combined_{norm_tag}.pdf'
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Saved unblocked scaling ({'normalized' if normalize else 'raw'}) combined plot to: {out_file}")
    plt.close(fig)

def _parse_noise_value_from_filename(path: Path) -> Optional[str]:
    """Parse the noise value from the filename, e.g., *_noise-5.0.json → '5.0'.

    Return None if no match is found.
    """
    try:
        m = re.search(r"noise-([0-9]+(?:\.[0-9]+)?)", path.name)
        if not m:
            return None
        return m.group(1)
    except Exception:
        return None


def prepare_noise_entries_across_models(
    model_names: List[str],
    experiment_result_dir: str,
    refinement_level: int,
    noise_values: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Collect the latest EM/NED metrics per noise level for each model at the specified refinement level.

    Returns: model_name -> noise(str) -> { 'em': float, 'ned': float }
    - Reuse the blocked-comparison directory discovery logic (supports base_dir and subdirectories) via _discover_runs.
    - For each noise value, keep only the run with the newest timestamp.
    - EM = tool_path_exact_match_ratio, NED = avg_normalized_edit_distance.
    """
    result: Dict[str, Dict[str, Dict[str, float]]] = {}
    base_dir = Path(experiment_result_dir)
    for mname in model_names:
        runs = _discover_runs(mname, experiment_result_dir)
        if not runs:
            print(f"[WARN] No noise runs discovered for model: {mname} (skipped)")
            continue
        latest_by_noise: Dict[str, Tuple[int, Dict[str, float]]] = {}
        for stats, ts, fp, _full in runs:
            try:
                rlv = int(stats.get('refinement_level', -1))
                if rlv != int(refinement_level):
                    continue
                noise_str = _parse_noise_value_from_filename(fp)
                if noise_str is None:
                    continue
                perf = _extract_agent_perf(stats)
                em = float(perf.get('tool_path_exact_match_ratio', 0.0))
                ned = float(perf.get('avg_normalized_edit_distance', 0.0))
                prev = latest_by_noise.get(noise_str)
                if (prev is None) or (ts > prev[0]):
                    latest_by_noise[noise_str] = (ts, {'em': em, 'ned': ned})
            except Exception:
                continue
        if not latest_by_noise:
            print(f"[WARN] No matching RL={refinement_level} noise runs for model: {mname} (skipped)")
            continue
        # Strip timestamps, keep values; optionally filter by noise_values and preserve the requested order
        if noise_values:
            filtered: Dict[str, Dict[str, float]] = {}
            # Build a numeric index mapping to resolve exact-match issues such as '1' vs '1.0'
            by_float: Dict[float, Tuple[int, str]] = {}
            for k, (ts_k, _vals_k) in latest_by_noise.items():
                try:
                    fk = float(k)
                except Exception:
                    # If the key cannot be parsed as a number (rare), skip numeric mapping and rely on direct string matches
                    continue
                prev = by_float.get(fk)
                if (prev is None) or (ts_k > prev[0]):
                    by_float[fk] = (ts_k, k)
            for nz in noise_values:
                if nz in latest_by_noise:
                    filtered[nz] = latest_by_noise[nz][1]
                    continue
                # Numeric equivalence match, for example '1' matching '1.0' in filenames
                try:
                    fnz = float(nz)
                except Exception:
                    fnz = None
                if fnz is not None and fnz in by_float:
                    matched_key = by_float[fnz][1]
                    filtered[nz] = latest_by_noise[matched_key][1]
            if filtered:
                result[mname] = filtered
            else:
                # Skip the model if none of the requested noise values are available
                print(f"[WARN] No runs for requested noise values for model: {mname} (skipped)")
        else:
            result[mname] = {k: v[1] for k, v in latest_by_noise.items()}
    return result


def _prepare_unblocked_run_full(
    model_name: str,
    experiment_result_dir: str,
    refinement_level: int,
) -> Optional[Dict[str, Any]]:
    """Return the latest unblocked run (full JSON) for the model at the specified refinement level.

    Return None if no match is found.
    """
    runs = _discover_runs(model_name, experiment_result_dir)
    if not runs:
        return None
    latest: Optional[Tuple[int, Dict[str, Any]]] = None
    for stats, ts, _fp, full in runs:
        try:
            if bool(stats.get('use_blocker')):
                continue
            rlv = int(stats.get('refinement_level', -1))
            if rlv != int(refinement_level):
                continue
            if (latest is None) or (ts > latest[0]):
                latest = (ts, full)
        except Exception:
            continue
    return None if latest is None else latest[1]


def _extract_error_tool_responses(full_data: Dict[str, Any]) -> List[str]:
    """Extract tool messages that start with "[ERROR]" from the full result JSON."""
    out: List[str] = []
    try:
        results = full_data.get('results', []) or []
    except Exception:
        results = []
    for item in results:
        try:
            history = item.get('conversation_history', []) or []
        except Exception:
            history = []
        for msg in history:
            try:
                if (msg.get('role') == 'tool'):
                    content = msg.get('content', '')
                    if isinstance(content, str):
                        # Allow leading whitespace
                        head = content.lstrip()
                        if head.startswith('[ERROR]'):
                            out.append(content)
            except Exception:
                continue
    return out


def _is_ban_tool_message(msg: Dict[str, Any]) -> bool:
    """Determine whether the tool message originates from the ban tool (via name/tool_name or content heuristics)."""
    try:
        name = msg.get('name') or msg.get('tool_name') or ''
        if isinstance(name, str):
            lower = name.lower()
            if 'ban_tool' in lower or 'ban tool' in lower:
                return True
        content = msg.get('content', '')
        if isinstance(content, str):
            lower_c = content.lower()
            if 'ban_tool' in lower_c or 'ban tool' in lower_c:
                return True
    except Exception:
        pass
    return False


def _extract_error_tool_responses_excluding_ban_tool(full_data: Dict[str, Any]) -> List[str]:
    """Extract "[ERROR]" tool outputs while excluding messages from the ban tool."""
    outputs: List[str] = []
    try:
        results = full_data.get('results', []) or []
    except Exception:
        results = []
    for item in results:
        try:
            history = item.get('conversation_history', []) or []
        except Exception:
            history = []
        for msg in history:
            try:
                if msg.get('role') != 'tool':
                    continue
                if _is_ban_tool_message(msg):
                    continue
                content = msg.get('content', '')
                if isinstance(content, str) and content.lstrip().startswith('[ERROR]'):
                    outputs.append(content)
            except Exception:
                continue
    return outputs


def _count_total_tool_calls(full_data: Dict[str, Any]) -> int:
    """Count the total number of tool messages observed in the conversations."""
    total = 0
    try:
        results = full_data.get('results', []) or []
    except Exception:
        results = []
    for item in results:
        try:
            history = item.get('conversation_history', []) or []
        except Exception:
            history = []
        for msg in history:
            try:
                if msg.get('role') == 'tool':
                    total += 1
            except Exception:
                continue
    return total


def plot_all_noise_vis_across_models(
    entries_by_model: Dict[str, Dict[str, Dict[str, float]]],
    output_dir: str,
    refinement_level: Optional[int] = None,
    noise_values_order: Optional[List[str]] = None,
    normalize: bool = True,
) -> None:
    """Plot two figures:
    - Figure 1: y = EM, x = noise (string values), grouped bars per noise with one bar per model.
    - Figure 2: y = NED, x = noise, identical layout.
    Colors are fixed per model and shared across both figures.
    """
    if not entries_by_model:
        print("[WARN] plot_all_noise_vis_across_models: empty entries.")
        return

    # Preserve the user-provided order; default to the first four models for the 2x2 layout
    model_names = list(entries_by_model.keys())
    if len(model_names) > 4:
        print(f"[INFO] More than 4 models provided ({len(model_names)}). Only the first 4 will be plotted.")
        model_names = model_names[:4]
    # Noise ticks: use the provided order if available; otherwise derive from the data and sort numerically
    if noise_values_order:
        all_noises: List[str] = [s for s in noise_values_order if s]
    else:
        all_noises = []
        for m in model_names:
            for nz in entries_by_model.get(m, {}).keys():
                if nz not in all_noises:
                    all_noises.append(nz)
        # Sort numerically when possible (otherwise lexicographically)
        def noise_sort_key(x: str):
            try:
                return (0, float(x))
            except Exception:
                return (1, x)
        all_noises.sort(key=noise_sort_key)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    base_colors = plt.get_cmap('tab10')
    color_map: Dict[str, Any] = {m: base_colors(i % 10) for i, m in enumerate(model_names)}

    title_rl = f" | RL={refinement_level}" if refinement_level is not None else ""
    rl_part = f"_rl{refinement_level}" if refinement_level is not None else ""

    # Helper: for each metric, generate a 2x2 grid with one subplot per model
    def _plot_metric(metric_key: str, pretty: str, filename_stub: str):
        # Compute the metric's global min/max across models and noise levels when normalization is enabled
        if normalize:
            raw_vals: List[float] = []
            for m in model_names:
                for nz in all_noises:
                    v = entries_by_model.get(m, {}).get(nz, {}).get(metric_key, None)
                    try:
                        fv = float(v)
                        if math.isnan(fv) or math.isinf(fv):
                            continue
                        raw_vals.append(fv)
                    except Exception:
                        continue
            if raw_vals:
                vmin, vmax = min(raw_vals), max(raw_vals)
            else:
                vmin, vmax = 0.0, 1.0
            denom = (vmax - vmin) if (vmax > vmin) else 1.0
        else:
            vmin, denom = 0.0, 1.0

        # Each subplot was previously (14, 6); keep the per-subplot size in the 2x2 layout → (28, 12) overall
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        axes_list = axes.flatten()
        x = np.arange(len(all_noises))

        for j, m in enumerate(model_names):
            ax = axes_list[j]
            ys: List[float] = []
            for nz in all_noises:
                v = entries_by_model.get(m, {}).get(nz, {}).get(metric_key, np.nan)
                try:
                    fv = float(v)
                    if math.isnan(fv) or math.isinf(fv):
                        ys.append(np.nan)
                    else:
                        ys.append(((fv - vmin) / denom) if normalize else fv)
                except Exception:
                    ys.append(np.nan)
            # Display the model name as the subplot title; omit the legend and plot a single line
            ax.plot(x, ys, marker='o', color=color_map[m], linewidth=2)
            ax.set_xticks(x)
            ax.set_xticklabels(all_noises)
            ax.set_xlabel('noise_std', fontsize=FONT_SIZE_LABEL)
            ax.set_ylabel(pretty, fontsize=FONT_SIZE_LABEL)
            ax.grid(True, alpha=0.3, axis='y')
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            try:
                short = m.split('/')[-1]
            except Exception:
                short = m
            title_text = _ensure_gpt_upper((short[:1].upper() + short[1:]) if short else short)
            ax.set_title(title_text, fontstyle='italic', fontsize=FONT_SIZE_TITLE)

        # Hide unused subplots if fewer than four models are provided
        for k in range(len(model_names), 4):
            axes_list[k].axis('off')

        # Under normalization, adjust each subplot's y-axis based on its data range
        if normalize:
            for ax in axes_list[:len(model_names)]:
                ymins: List[float] = []
                ymaxs: List[float] = []
                for line in ax.get_lines():
                    data = line.get_ydata(orig=False)
                    if data is None or len(data) == 0:
                        continue
                    arr = np.asarray(data, dtype=float)
                    arr = arr[~np.isnan(arr) & ~np.isinf(arr)]
                    if arr.size:
                        ymins.append(float(np.min(arr)))
                        ymaxs.append(float(np.max(arr)))
                if ymins and ymaxs:
                    ylo, yhi = min(ymins), max(ymaxs)
                    if yhi <= ylo:
                        pad = 1e-6
                        ax.set_ylim(ylo - pad, yhi + pad)
                    else:
                        ax.set_ylim(ylo, yhi)

        plt.tight_layout()

        norm_tag = 'normalized' if normalize else 'raw'
        fname = f"all_models{rl_part}_{filename_stub}_{norm_tag}.pdf"
        plt.savefig(output_path / fname, dpi=300, bbox_inches='tight')
        print(f"Saved noise comparison plot to: {output_path / fname}")
        plt.close(fig)

    # Figure 1: EMR; Figure 2: ANED
    _plot_metric('em', 'EMR', 'noise_em')
    _plot_metric('ned', 'ANED', 'noise_ned')

def plot_block_scaling_agent_performance(entries_by_bn: Dict[int, Dict[str, Any]], model_name: str, block_mode: str, output_dir: str, refinement_level: Optional[int] = None, filename_suffix: str = "block_scaling_agent_performance") -> None:
    block_nums, datas = _sorted_block_points(entries_by_bn)
    acc: List[float] = []
    ned: List[float] = []
    ed: List[float] = []
    em: List[float] = []
    for d in datas:
        perf = _extract_agent_perf((d.get('stats', {}) if isinstance(d, dict) else {}))
        acc.append(perf.get('final_answer_accuracy', 0.0))
        ned.append(perf.get('avg_normalized_edit_distance', 0.0))
        ed.append(perf.get('avg_edit_distance', 0.0))
        em.append(perf.get('tool_path_exact_match_ratio', 0.0))

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()
    series = [
        ('Final Answer Accuracy', acc),
        ('Normalized Edit Distance', ned),
        ('Edit Distance', ed),
        ('Tool Path EMR', em),
    ]
    colors = ['#ff7f0e', '#9467bd', '#8c564b', '#17becf']
    for i, (title, ys) in enumerate(series):
        ax = axes[i]
        x = np.arange(len(block_nums))
        bars = ax.bar(x, ys, color=colors[i], alpha=0.9)
        ax.set_title(title)
        ax.set_xlabel('Block Num')
        ax.set_ylabel('Value')
        ax.set_xticks(x)
        ax.set_xticklabels([str(bn) for bn in block_nums])
        ax.grid(True, alpha=0.3, axis='y')

    title_rl = f" | RL={refinement_level}" if refinement_level is not None else ""
    plt.suptitle(f'Block Scaling - Agent Performance ({block_mode}){title_rl}')
    plt.tight_layout()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    clean_model_name = _ensure_gpt_upper(model_name).replace('/', '_').replace(' ', '_')
    rl_part = f"_rl{refinement_level}" if refinement_level is not None else ""
    filename = f"{clean_model_name}_{block_mode}{rl_part}_{filename_suffix}.pdf"
    plt.savefig(output_path / filename, dpi=300, bbox_inches='tight')
    print(f"Saved block-scaling agent performance plot to: {output_path / filename}")
    plt.close(fig)


def plot_block_scaling_agent_vs_stim(entries_by_bn: Dict[int, Dict[str, Any]], model_name: str, block_mode: str, output_dir: str, refinement_level: Optional[int] = None, filename_suffix: str = "block_scaling_agent_vs_stim") -> None:
    block_nums, datas = _sorted_block_points(entries_by_bn)
    agent_ned: List[float] = []
    agent_ed: List[float] = []
    agent_em: List[float] = []
    stim_ned: List[float] = []
    stim_ed: List[float] = []
    stim_em: List[float] = []
    for d in datas:
        stats = d.get('stats', {}) if isinstance(d, dict) else {}
        a = _extract_agent_perf(stats)
        s = _extract_stim_path(stats)
        agent_ned.append(a.get('avg_normalized_edit_distance', 0.0))
        agent_ed.append(a.get('avg_edit_distance', 0.0))
        agent_em.append(a.get('tool_path_exact_match_ratio', 0.0))
        stim_ned.append(s.get('avg_normalized_edit_distance', 0.0))
        stim_ed.append(s.get('avg_edit_distance', 0.0))
        stim_em.append(s.get('tool_path_exact_match_ratio', 0.0))

    metrics = [
        ('Normalized Edit Distance', agent_ned, stim_ned),
        ('Edit Distance', agent_ed, stim_ed),
        ('EM Ratio', agent_em, stim_em),
    ]
    colors = ['#d62728', '#1f77b4']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, (title, ys_agent, ys_stim) in enumerate(metrics):
        ax = axes[i]
        x = np.arange(len(block_nums))
        width = 0.35
        b1 = ax.bar(x - width/2, ys_agent, width, label='Agent', color=colors[0], alpha=0.9)
        b2 = ax.bar(x + width/2, ys_stim, width, label='Stimulation', color=colors[1], alpha=0.9)
        ax.set_title(title)
        ax.set_xlabel('Block Num')
        ax.set_ylabel('Value')
        ax.set_xticks(x)
        ax.set_xticklabels([str(bn) for bn in block_nums])
        ax.grid(True, alpha=0.3, axis='y')
        if i == 0:
            ax.legend()
    title_rl = f" | RL={refinement_level}" if refinement_level is not None else ""
    plt.suptitle(f'Block Scaling - Agent vs Stimulation ({block_mode}){title_rl}')
    plt.tight_layout()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    clean_model_name = _ensure_gpt_upper(model_name).replace('/', '_').replace(' ', '_')
    rl_part = f"_rl{refinement_level}" if refinement_level is not None else ""
    filename = f"{clean_model_name}_{block_mode}{rl_part}_{filename_suffix}.pdf"
    plt.savefig(output_path / filename, dpi=300, bbox_inches='tight')
    print(f"Saved block-scaling agent vs stimulation plot to: {output_path / filename}")
    plt.close(fig)


def plot_block_scaling_validity(entries_by_bn: Dict[int, Dict[str, Any]], model_name: str, block_mode: str, output_dir: str, refinement_level: Optional[int] = None, filename_suffix: str = "block_scaling_validity") -> None:
    block_nums, datas = _sorted_block_points(entries_by_bn)
    gt_unblocked: List[float] = []
    valid_num: List[float] = []
    block_invalid_num: List[float] = []
    no_answer_num: List[float] = []
    no_goal_state_num: List[float] = []
    for d in datas:
        stats = d.get('stats', {}) if isinstance(d, dict) else {}
        scores = stats.get('scores', {}) or {}
        gt_unblocked.append(float(_count_gt_unblocked(d)))
        def to_f(x: Any) -> float:
            try:
                f = float(x if x is not None else 0.0)
                if math.isnan(f) or math.isinf(f):
                    return 0.0
                return f
            except Exception:
                return 0.0
        valid_num.append(to_f(scores.get('valid_num', 0)))
        block_invalid_num.append(to_f(scores.get('block_invalid_num', 0)))
        no_answer_num.append(to_f(scores.get('no_answer_num', 0)))
        no_goal_state_num.append(to_f(scores.get('no_goal_state_num', 0)))

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(block_nums))
    width = 0.16
    series = [
        ('GT Unblocked', gt_unblocked),
        ('Valid', valid_num),
        ('Block Invalid', block_invalid_num),
        ('No Answer', no_answer_num),
        ('No Goal State', no_goal_state_num),
    ]
    for i, (label, ys) in enumerate(series):
        offs = (i - (len(series)-1)/2) * width
        ax.bar(x + offs, ys, width, label=label, alpha=0.85)
    ax.set_xlabel('Block Num')
    ax.set_ylabel('Count')
    ax.set_xticks(x)
    ax.set_xticklabels([str(bn) for bn in block_nums])
    title_rl = f" | RL={refinement_level}" if refinement_level is not None else ""
    ax.set_title(f'Block Scaling - Validity Breakdown ({block_mode}){title_rl}')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    plt.tight_layout()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    clean_model_name = _ensure_gpt_upper(model_name).replace('/', '_').replace(' ', '_')
    rl_part = f"_rl{refinement_level}" if refinement_level is not None else ""
    filename = f"{clean_model_name}_{block_mode}{rl_part}_{filename_suffix}.pdf"
    plt.savefig(output_path / filename, dpi=300, bbox_inches='tight')
    print(f"Saved block-scaling validity breakdown plot to: {output_path / filename}")
    plt.close(fig)


def plot_block_scaling_paths_combined(
    entries_by_mode: Dict[str, Dict[int, Dict[str, Any]]],
    model_name: str,
    output_dir: str,
    refinement_level: Optional[int] = None,
    filename_suffixes: Dict[str, str] = None
) -> None:
    """Plot EM, ED, NED as three bar charts, each containing four block modes.

    entries_by_mode: mapping of block_mode -> (mapping of block_num -> full_json_data)
    """
    if filename_suffixes is None:
        filename_suffixes = {
            'tool_path_exact_match_ratio': 'block_scaling_em',
            'avg_edit_distance': 'block_scaling_ed',
            'avg_normalized_edit_distance': 'block_scaling_ned',
        }

    # Consistent order and colors per mode
    modes_order = ['ban_tool', 'cost_change', 'preference_change', 'steplen_change']
    mode_colors = {
        'ban_tool': '#2ca02c',
        'cost_change': '#d62728',
        'preference_change': '#9467bd',
        'steplen_change': '#1f77b4',
    }

    metrics_spec = [
        ('EM Ratio', 'tool_path_exact_match_ratio'),
        ('Edit Distance', 'avg_edit_distance'),
        ('Normalized Edit Distance', 'avg_normalized_edit_distance'),
    ]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    clean_model_name = _ensure_gpt_upper(model_name).replace('/', '_').replace(' ', '_')
    title_rl = f" | RL={refinement_level}" if refinement_level is not None else ""
    rl_part = f"_rl{refinement_level}" if refinement_level is not None else ""

    # For each metric, draw a figure with grouped bars per block_num and bars for modes
    for pretty_name, key in metrics_spec:
        fig, ax = plt.subplots(figsize=(14, 6))
        any_series = False
        # gather union of block_nums across modes
        all_block_nums: List[int] = []
        per_mode_values: Dict[str, Dict[int, float]] = {}
        for mode in modes_order:
            entries_by_bn = entries_by_mode.get(mode, {})
            if not entries_by_bn:
                continue
            block_nums, datas = _sorted_block_points(entries_by_bn)
            if not block_nums:
                continue
            any_series = True
            for bn, d in zip(block_nums, datas):
                perf = _extract_agent_perf((d.get('stats', {}) if isinstance(d, dict) else {}))
                per_mode_values.setdefault(mode, {})[bn] = float(perf.get(key, 0.0))
            for bn in block_nums:
                if bn not in all_block_nums:
                    all_block_nums.append(bn)
        if not any_series:
            plt.close(fig)
            print(f"[WARN] No data available to plot for {pretty_name}.")
            continue
        all_block_nums.sort()
        x = np.arange(len(all_block_nums))
        width = 0.18
        for i, mode in enumerate(modes_order):
            vals = [per_mode_values.get(mode, {}).get(bn, np.nan) for bn in all_block_nums]
            offs = (i - (len(modes_order)-1)/2) * width
            ax.bar(x + offs, vals, width, label=_pretty_baseline_name(mode), color=mode_colors.get(mode), alpha=0.9)

        ax.set_xlabel('Block Num')
        ax.set_ylabel('Value')
        ax.set_title(f'Block Scaling - {pretty_name}{title_rl}')
        ax.set_xticks(x)
        ax.set_xticklabels([str(bn) for bn in all_block_nums])
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()
        plt.tight_layout()

        filename = f"{clean_model_name}{rl_part}_{filename_suffixes[key]}.pdf"
        plt.savefig(output_path / filename, dpi=300, bbox_inches='tight')
        print(f"Saved combined block-scaling ({pretty_name}) plot to: {output_path / filename}")
        plt.close(fig)


def plot_blocked_costs_threeway(entries_map: Dict[str, Dict[str, Any]], model_name: str, output_dir: str, filename_suffix: str = "blocked_results_cost_comparison"):
    """Plot cost bars for Agent, GT, and Stimulation across modes (and optional 'unblocked')."""
    # Enforce requested order per model
    baselines = ['cost_change', 'ban_tool', 'preference_change', 'steplen_change', 'unblocked']
    labels = [b for b in baselines if b in entries_map]
    if not labels:
        print("No entries available for cost plotting.")
        return

    agent_costs = []
    gt_costs = []
    stim_costs = []
    for b in labels:
        try:
            full = entries_map[b]
            if not isinstance(full, dict) or not full:
                print(f"[WARN] Empty entry for baseline {b}, coerced zeros")
                a, g, s = 0.0, 0.0, 0.0
            else:
                stats = full.get('stats', {})
                a, g, s = _extract_costs(stats)
            agent_costs.append(a)
            gt_costs.append(g)
            stim_costs.append(s)
        except Exception as e:
            print(f"[ERROR] Failed extracting costs for baseline {b}: {e}. Coerced zeros")
            agent_costs.append(0.0)
            gt_costs.append(0.0)
            stim_costs.append(0.0)

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 7))
    bars1 = ax.bar(x - width, agent_costs, width, label='Agent', color='#d62728', alpha=0.85)
    bars2 = ax.bar(x, gt_costs, width, label='GT', color='#2ca02c', alpha=0.85)
    bars3 = ax.bar(x + width, stim_costs, width, label='Stimulation', color='#1f77b4', alpha=0.85)

    # value labels
    for bars in (bars1, bars2, bars3):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + max(1.0, 0.01*h), f'{h:.1f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([_pretty_baseline_name(lbl) for lbl in labels])
    ax.set_ylabel('Cost')
    ax.set_title('Cost Comparison: Agent vs GT vs Stimulation')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()

    plt.tight_layout()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    clean_model_name = _ensure_gpt_upper(model_name).replace('/', '_').replace(' ', '_')
    filename = f'{clean_model_name}_{filename_suffix}.pdf'
    plt.savefig(output_path / filename, dpi=300, bbox_inches='tight')
    print(f"Saved cost comparison (3-way) plot to: {output_path / filename}")
    plt.close(fig)


def plot_blocked_performance_agent(entries_map: Dict[str, Dict[str, Any]], model_name: str, output_dir: str, filename_suffix: str = "blocked_results_performance_comparison"):
    """Plot agent-only performance: acc, NED, ED, EM across baselines."""
    # Enforce requested order per model
    baselines = ['cost_change', 'ban_tool', 'preference_change', 'steplen_change', 'unblocked']
    labels = [b for b in baselines if b in entries_map]
    if not labels:
        print("No entries available for performance plotting.")
        return

    metrics = ['Final Answer Accuracy', 'Normalized Edit Distance', 'Edit Distance', 'Tool Path EMR']
    metric_keys = ['final_answer_accuracy', 'avg_normalized_edit_distance', 'avg_edit_distance', 'tool_path_exact_match_ratio']

    # collect values per baseline
    values = {mk: [] for mk in metric_keys}
    for b in labels:
        try:
            perf = _extract_agent_perf(entries_map[b].get('stats', {}))
        except Exception as e:
            print(f"[ERROR] Failed extracting agent perf for baseline {b}: {e}. Using zeros")
            perf = {mk: 0.0 for mk in metric_keys}
        for mk in metric_keys:
            values[mk].append(perf.get(mk, 0.0))

    x = np.arange(len(metrics))
    width = 0.12

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, b in enumerate(labels):
        offs = (i - (len(labels)-1)/2) * width
        bars = ax.bar(x + offs, [values[mk][i] for mk in metric_keys], width, label=_pretty_baseline_name(b), alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.01, f'{h:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=20, ha='right')
    ax.set_ylim(0, max(1.05, ax.get_ylim()[1]))
    ax.set_ylabel('Values')
    ax.set_title('Agent Performance Comparison across Modes/Unblocked')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    clean_model_name = _ensure_gpt_upper(model_name).replace('/', '_').replace(' ', '_')
    filename = f'{clean_model_name}_{filename_suffix}.pdf'
    plt.savefig(output_path / filename, dpi=300, bbox_inches='tight')
    print(f"Saved agent performance comparison plot to: {output_path / filename}")
    plt.close(fig)


def plot_agent_vs_stimulation(entries_map: Dict[str, Dict[str, Any]], model_name: str, output_dir: str, title: str, filename_suffix: str):
    """Plot 3 path metrics comparing Agent vs Stimulation across baselines.

    Metrics: NED, ED, EM ratio (no accuracy).
    """
    # Enforce requested order per model
    baselines = [k for k in ['cost_change', 'ban_tool', 'preference_change', 'steplen_change', 'unblocked'] if k in entries_map]
    if not baselines:
        print("No entries available for agent vs stimulation plotting.")
        return

    agent_vals = { 'avg_normalized_edit_distance': [], 'avg_edit_distance': [], 'tool_path_exact_match_ratio': [] }
    stim_vals = { 'avg_normalized_edit_distance': [], 'avg_edit_distance': [], 'tool_path_exact_match_ratio': [] }
    for b in baselines:
        try:
            perf_a = _extract_agent_perf(entries_map[b].get('stats', {}))
        except Exception as e:
            print(f"[ERROR] Failed extracting agent perf for {b}: {e}. Using zeros")
            perf_a = {k: 0.0 for k in agent_vals.keys()}
        try:
            perf_s = _extract_stim_path(entries_map[b].get('stats', {}))
        except Exception as e:
            print(f"[WARN] Failed extracting stimulation metrics for {b}: {e}. Using zeros")
            perf_s = {k: 0.0 for k in stim_vals.keys()}
        for k in agent_vals.keys():
            agent_vals[k].append(perf_a.get(k, 0.0))
            stim_vals[k].append(perf_s.get(k, 0.0))

    metrics = [('Normalized Edit Distance', 'avg_normalized_edit_distance'), ('Edit Distance', 'avg_edit_distance'), ('EM Ratio', 'tool_path_exact_match_ratio')]

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    for idx, (label, key) in enumerate(metrics):
        ax = axes[idx]
        x = np.arange(len(baselines))
        width = 0.35
        b1 = ax.bar(x - width/2, agent_vals[key], width, label='Agent', color='#d62728', alpha=0.85)
        b2 = ax.bar(x + width/2, stim_vals[key], width, label='Stimulation', color='#1f77b4', alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([_pretty_baseline_name(m) for m in baselines], rotation=20, ha='right')
        ax.set_title(label)
        ax.grid(True, alpha=0.3, axis='y')
        # value labels
        for bars in (b1, b2):
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., h + 0.01, f'{h:.3f}', ha='center', va='bottom', fontsize=8)
        if idx == 0:
            ax.legend()

    fig.suptitle(title)
    plt.tight_layout()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    clean_model_name = _ensure_gpt_upper(model_name).replace('/', '_').replace(' ', '_')
    filename = f'{clean_model_name}_{filename_suffix}.pdf'
    plt.savefig(output_path / filename, dpi=300, bbox_inches='tight')
    print(f"Saved agent vs stimulation comparison plot to: {output_path / filename}")
    plt.close(fig)


def plot_validity_breakdown(entries_map: Dict[str, Dict[str, Any]], model_name: str, output_dir: str, filename_suffix: str):
    """Plot grouped bars of validity/invalidity counts per baseline."""
    # Enforce requested order per model
    baselines = [k for k in ['cost_change', 'ban_tool', 'preference_change', 'steplen_change', 'unblocked'] if k in entries_map]
    if not baselines:
        print("No entries available for validity plotting.")
        return

    categories = ['gt_unblocked', 'valid_num', 'block_invalid_num', 'no_answer_num', 'no_goal_state_num']
    data_by_cat: Dict[str, List[float]] = {c: [] for c in categories}

    for b in baselines:
        try:
            full = entries_map[b]
            stats = full.get('stats', {}) if isinstance(full, dict) else {}
            scores = stats.get('scores', {}) or {}
            data_by_cat['gt_unblocked'].append(float(_count_gt_unblocked(full)))
            for key in ['valid_num', 'block_invalid_num', 'no_answer_num', 'no_goal_state_num']:
                v = scores.get(key, 0)
                try:
                    f = float(v if v is not None else 0.0)
                    if math.isnan(f) or math.isinf(f):
                        print(f"[WARN] Invalid numeric count {key} for {b}: {v}. Coerced to 0.0")
                        f = 0.0
                except Exception:
                    print(f"[WARN] Non-numeric count {key} for {b}: {v}. Coerced to 0.0")
                    f = 0.0
                data_by_cat[key].append(f)
        except Exception as e:
            print(f"[ERROR] Failed extracting validity counts for {b}: {e}. Using zeros")
            data_by_cat['gt_unblocked'].append(0.0)
            for key in ['valid_num', 'block_invalid_num', 'no_answer_num', 'no_goal_state_num']:
                data_by_cat[key].append(0.0)

    x = np.arange(len(categories))
    width = 0.12
    fig, ax = plt.subplots(figsize=(16, 6))
    for i, b in enumerate(baselines):
        offs = (i - (len(baselines)-1)/2) * width
        bars = ax.bar(x + offs, [data_by_cat[c][i] for c in categories], width, label=_pretty_baseline_name(b), alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + max(0.5, 0.02*h), f'{h:.0f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(['GT Unblocked', 'Valid', 'Block Invalid', 'No Answer', 'No Goal State'], rotation=20, ha='right')
    ax.set_ylabel('Count')
    ax.set_title('Validity/Invalidity Breakdown per Baseline')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    clean_model_name = model_name.replace('/', '_').replace(' ', '_')
    filename = f'{clean_model_name}_{filename_suffix}.pdf'
    plt.savefig(output_path / filename, dpi=300, bbox_inches='tight')
    print(f"Saved validity breakdown plot to: {output_path / filename}")
    plt.close(fig)

def plot_unblocked_results(unblocked_data: List[Dict[str, Any]], model_name: str, output_dir: str):
    """Plot 5 line charts for unblocked results"""
    # Extract refinement levels and sort data by refinement level
    sorted_data = sorted(unblocked_data, key=lambda x: x['refinement_level'])
    refinement_levels = [d['refinement_level'] for d in sorted_data]
    
    # Create figure with 2x3 subplots (5 plots + 1 empty)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Unblocked Results Analysis', fontsize=FONT_SIZE_TITLE)
    
    # 1. Final Answer Accuracy
    accuracies = [d['scores']['accuracy_metrics']['final_answer_accuracy'] for d in sorted_data]
    axes[0, 0].bar(np.arange(len(refinement_levels)), accuracies, color='#1f77b4', alpha=0.9)
    axes[0, 0].set_xticks(np.arange(len(refinement_levels)))
    axes[0, 0].set_xticklabels([str(r) for r in refinement_levels])
    axes[0, 0].set_title('Final Answer Accuracy')
    axes[0, 0].set_xlabel('Refinement Level')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 1)
    
    # 2. Average Costs (Agent vs GT)
    agent_costs = [d['scores']['cost_metrics']['avg_agent_cost'] for d in sorted_data]
    gt_costs = [d['scores']['cost_metrics']['avg_gt_cost'] for d in sorted_data]
    x = np.arange(len(refinement_levels))
    width = 0.35
    axes[0, 1].bar(x - width/2, agent_costs, width, label='Agent Cost', color='#d62728', alpha=0.9)
    axes[0, 1].bar(x + width/2, gt_costs, width, label='GT Cost', color='#2ca02c', alpha=0.9)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([str(r) for r in refinement_levels])
    axes[0, 1].set_title('Average Costs')
    axes[0, 1].set_xlabel('Refinement Level')
    axes[0, 1].set_ylabel('Cost')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Normalized Edit Distance
    edit_distances = [d['scores']['tool_path_metrics']['avg_normalized_edit_distance'] for d in sorted_data]
    axes[0, 2].bar(np.arange(len(refinement_levels)), edit_distances, color='#9467bd', alpha=0.9)
    axes[0, 2].set_xticks(np.arange(len(refinement_levels)))
    axes[0, 2].set_xticklabels([str(r) for r in refinement_levels])
    axes[0, 2].set_title('Normalized Edit Distance')
    axes[0, 2].set_xlabel('Refinement Level')
    axes[0, 2].set_ylabel('Edit Distance')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_ylim(0, 1)
    
    # 4. Tool Path EMR
    match_ratios = [d['scores']['tool_path_metrics']['tool_path_exact_match_ratio'] for d in sorted_data]
    axes[1, 0].bar(np.arange(len(refinement_levels)), match_ratios, color='#17becf', alpha=0.9)
    axes[1, 0].set_xticks(np.arange(len(refinement_levels)))
    axes[1, 0].set_xticklabels([str(r) for r in refinement_levels])
    axes[1, 0].set_title('Tool Path EMR')
    axes[1, 0].set_xlabel('Refinement Level')
    axes[1, 0].set_ylabel('Match Ratio')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1)
    
    # 5. Block Valid Ratio
    valid_ratios = [d['scores']['valid_num'] / d['scores']['total_queries'] for d in sorted_data]
    axes[1, 1].bar(np.arange(len(refinement_levels)), valid_ratios, color='#ff7f0e', alpha=0.9)
    axes[1, 1].set_xticks(np.arange(len(refinement_levels)))
    axes[1, 1].set_xticklabels([str(r) for r in refinement_levels])
    axes[1, 1].set_title('Block Valid Ratio')
    axes[1, 1].set_xlabel('Refinement Level')
    axes[1, 1].set_ylabel('Valid Ratio')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(0, 1)
    
    # 6. Edit Distance (raw)
    raw_edit_distances = [d['scores']['tool_path_metrics']['avg_edit_distance'] for d in sorted_data]
    axes[1, 2].bar(np.arange(len(refinement_levels)), raw_edit_distances, color='#bcbd22', alpha=0.9)
    axes[1, 2].set_xticks(np.arange(len(refinement_levels)))
    axes[1, 2].set_xticklabels([str(r) for r in refinement_levels])
    axes[1, 2].set_title('Edit Distance')
    axes[1, 2].set_xlabel('Refinement Level')
    axes[1, 2].set_ylabel('Edit Distance')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Clean model name for filename
    clean_model_name = model_name.replace('/', '_').replace(' ', '_')
    filename = f'{clean_model_name}_unblocked_results_analysis.pdf'
    plt.savefig(output_path / filename, dpi=300, bbox_inches='tight')
    print(f"Saved unblocked results plot to: {output_path / filename}")
    plt.close(fig)

def plot_blocked_results(blocked_data: List[Dict[str, Any]], model_name: str, output_dir: str):
    """Plot bar charts comparing different block modes"""
    # Separate blocked and unblocked data
    blocked_runs = [d for d in blocked_data if d.get('use_blocker')]
    unblocked_candidates = [d for d in blocked_data if not d.get('use_blocker')]
    unblocked_run = unblocked_candidates[0] if unblocked_candidates else None
    
    # Group by block mode
    block_modes = {}
    for run in blocked_runs:
        mode = run.get('block_mode')
        block_modes[mode] = run
    
    # Extract values for each baseline
    baselines = ['ban_tool', 'cost_change', 'preference_change', 'steplen_change', 'unblocked']
    baseline_data = {}
    
    for baseline in baselines:
        if baseline == 'unblocked':
            if unblocked_run is None:
                continue
            run_data = unblocked_run
        else:
            if baseline not in block_modes:
                continue
            run_data = block_modes[baseline]
        
        baseline_data[baseline] = {
            'final_answer_accuracy': run_data['scores']['accuracy_metrics']['final_answer_accuracy'],
            'avg_agent_cost': run_data['scores']['cost_metrics']['avg_agent_cost'],
            'avg_gt_cost': run_data['scores']['cost_metrics']['avg_gt_cost'],
            'avg_normalized_edit_distance': run_data['scores']['tool_path_metrics']['avg_normalized_edit_distance'],
            'tool_path_exact_match_ratio': run_data['scores']['tool_path_metrics']['tool_path_exact_match_ratio'],
            'block_valid_ratio': run_data['scores']['valid_num'] / run_data['scores']['total_queries']
        }
    
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Plot 1: Cost Metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    
    cost_metrics = ['Avg Agent Cost', 'Avg GT Cost']
    x = np.arange(len(cost_metrics))
    width = 0.15
    
    filtered_baselines = [b for b in baselines if b in baseline_data]
    for i, baseline in enumerate(filtered_baselines):
        offset = (i - 2) * width
        cost_values = [baseline_data[baseline]['avg_agent_cost'], baseline_data[baseline]['avg_gt_cost']]
        bars = ax.bar(x + offset, cost_values, width, 
                     label=_pretty_baseline_name(baseline), 
                     color=colors[i % len(colors)], alpha=0.8)
        
        # Add value labels on bars
        for bar, value in zip(bars, cost_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{value:.1f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Cost Metrics', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('Cost', fontsize=FONT_SIZE_LABEL)
    ax.set_title('Cost Comparison: Different Block Modes vs Unblocked', fontsize=FONT_SIZE_TITLE)
    ax.set_xticks(x)
    ax.set_xticklabels(cost_metrics)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Clean model name for filename
    clean_model_name = model_name.replace('/', '_').replace(' ', '_')
    filename = f'{clean_model_name}_blocked_results_cost_comparison.pdf'
    plt.savefig(output_path / filename, dpi=300, bbox_inches='tight')
    print(f"Saved blocked cost comparison plot to: {output_path / filename}")
    plt.close(fig)
    
    # Plot 2: Other Metrics
    fig, ax = plt.subplots(figsize=(12, 6))
    
    other_metrics = ['Final Answer Accuracy', 'Normalized Edit Distance', 'Tool Path EMR', 'Block Valid Ratio']
    x = np.arange(len(other_metrics))
    width = 0.15
    
    for i, baseline in enumerate(filtered_baselines):
        offset = (i - 2) * width
        other_values = [
            baseline_data[baseline]['final_answer_accuracy'],
            baseline_data[baseline]['avg_normalized_edit_distance'],
            baseline_data[baseline]['tool_path_exact_match_ratio'],
            baseline_data[baseline]['block_valid_ratio']
        ]
        bars = ax.bar(x + offset, other_values, width, 
                     label=_pretty_baseline_name(baseline), 
                     color=colors[i % len(colors)], alpha=0.8)
        
        # Add value labels on bars
        for bar, value in zip(bars, other_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Metrics', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('Values', fontsize=FONT_SIZE_LABEL)
    ax.set_title('Performance Comparison: Different Block Modes vs Unblocked', fontsize=FONT_SIZE_TITLE)
    ax.set_xticks(x)
    ax.set_xticklabels(other_metrics, rotation=20, ha='right')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.1)  # Since most metrics are ratios between 0-1
    
    plt.tight_layout()
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Clean model name for filename
    clean_model_name = model_name.replace('/', '_').replace(' ', '_')
    filename = f'{clean_model_name}_blocked_results_performance_comparison.pdf'
    plt.savefig(output_path / filename, dpi=300, bbox_inches='tight')
    print(f"Saved blocked performance comparison plot to: {output_path / filename}")
    plt.close(fig)

def main():
    """Main function to generate all plots"""
    parser = argparse.ArgumentParser(description='CostBench visualization')
    parser.add_argument('--model_name', type=str, required=False, help='Model name, e.g., Qwen/Qwen2.5-7B-Instruct or Qwen2.5-7B-Instruct')
    parser.add_argument('--experiment_result_dir', type=str, required=True, help='Experiment result directory containing results_*.json files')
    parser.add_argument('--output_dir', type=str, default='plots', help='Output directory for plots')
    parser.add_argument('--type', type=str, required=True, choices=['unblocked-refinement-scaling', 'blocked_comparison', 'block-scaling-comparison', 'all_block_comparison', 'all_unblock_scale', 'all_noise_vis', 'all_block_scale', 'error-analysis'], help='Plot type: unblocked-refinement-scaling / blocked_comparison / block-scaling-comparison / all_block_comparison / all_unblock_scale / all_noise_vis / all_block_scale / error-analysis')
    parser.add_argument('--refinement_level', type=int, default=None, help='Refinement level required when type=blocked_comparison or block-scaling-comparison')
    parser.add_argument('--block_num', type=int, default=None, help='Optional for type=blocked_comparison: filter a specific block_num')
    parser.add_argument('--block_mode', type=str, default=None, choices=['ban_tool', 'cost_change', 'preference_change', 'steplen_change'], help='Deprecated: legacy block-scaling-comparison required this block_mode')
    parser.add_argument('--model_names', type=str, default=None, help='Required for type=all_block_comparison or all_unblock_scale: comma-separated model names')
    parser.add_argument('--refinement_levels', type=str, default=None, help='Optional for type=all_unblock_scale: comma-separated refinement levels to filter')
    parser.add_argument('--noise_values', type=str, default=None, help='Optional for type=all_noise_vis: comma-separated noise values such as 0,0.5,5')
    parser.add_argument('--block_nums', type=str, default=None, help='Optional for type=all_block_scale: comma-separated block counts like 0,1,2,3 (0 represents unblocked)')
    parser.add_argument('--block_types', type=str, default=None, help='Optional for type=all_block_scale: comma-separated block types, e.g., ban_tool,cost_change,preference_change,steplen_change')
    parser.add_argument('--colors', type=str, default=None, help='Optional for type=all_block_comparison or all_unblock_scale: override baseline/unblocked or task-sequence colors; for all_block_comparison use baseline=color, for all_unblock_scale supply a comma-separated list (e.g., #c99b38,#00b0be,#eddca5,#8fd7d7)')
    parser.add_argument('--error_json', type=str, default=None, help='Required for type=error-analysis: output JSON path')
    args = parser.parse_args()

    # Argument validation
    if args.type in ['unblocked-refinement-scaling', 'blocked_comparison', 'block-scaling-comparison'] and not args.model_name:
        print("Please specify --model_name for the selected --type.")
        return
    if args.type == 'all_block_comparison' and not args.model_names:
        print("Please specify --model_names (comma-separated) for all_block_comparison.")
        return

    if args.type == 'unblocked-refinement-scaling':
        print("Loading data...")
        unblocked_data, _ = load_data(
            args.model_name,
            args.experiment_result_dir,
            None,
        )
        if not unblocked_data:
            print("No unblocked runs found. Please check --model_name and --experiment_result_dir.")
            return
        print("Plotting unblocked results...")
        plot_unblocked_results(unblocked_data, args.model_name, args.output_dir)
    elif args.type == 'blocked_comparison':
        if args.refinement_level is None:
            print("Please specify --refinement_level for blocked_comparison.")
            return
        # Build entries filtered by refinement level and optional block_num
        entries_map = prepare_blocked_comparison_entries(
            model_name=args.model_name,
            experiment_result_dir=args.experiment_result_dir,
            refinement_level=int(args.refinement_level),
            block_num=args.block_num,
        )
        if not entries_map:
            print("No matching runs found for blocked comparison.")
            return
        print("Plotting blocked results comparison...")
        # 1) Cost: Agent vs GT vs Stimulation
        plot_blocked_costs_threeway(entries_map, args.model_name, args.output_dir, filename_suffix='blocked_results_cost_comparison')
        # 2) Performance (Agent only): acc, NED, ED, EM
        plot_blocked_performance_agent(entries_map, args.model_name, args.output_dir, filename_suffix='blocked_results_performance_comparison')
        # 3) Agent vs Stimulation (3 path metrics)
        plot_agent_vs_stimulation(entries_map, args.model_name, args.output_dir, title='Agent vs Stimulation (Path Metrics)', filename_suffix='blocked_results_agent_vs_stimulation')
        # 4) Validity breakdown
        plot_validity_breakdown(entries_map, args.model_name, args.output_dir, filename_suffix='blocked_results_validity_breakdown')
    elif args.type == 'block-scaling-comparison':
        if args.refinement_level is None:
            print("Please specify --refinement_level for block-scaling-comparison.")
            return
        # Collect latest entries per block_num for each mode at the given refinement level
        modes = ['ban_tool', 'cost_change', 'preference_change', 'steplen_change']
        entries_by_mode: Dict[str, Dict[int, Dict[str, Any]]] = {}
        for m in modes:
            ebn = prepare_block_scaling_entries(
                model_name=args.model_name,
                experiment_result_dir=args.experiment_result_dir,
                refinement_level=int(args.refinement_level),
                block_mode=m
            )
            if ebn:
                entries_by_mode[m] = ebn
        if not entries_by_mode:
            print("No matching runs found for block-scaling-comparison across modes.")
            return
        rl = int(args.refinement_level) if args.refinement_level is not None else None
        # Combined EM/ED/NED plots (each figure includes all modes)
        plot_block_scaling_paths_combined(entries_by_mode, args.model_name, args.output_dir, refinement_level=rl)
    elif args.type == 'all_block_comparison':
        if args.refinement_level is None:
            print("Please specify --refinement_level for all_block_comparison.")
            return
        # Parse the model list
        models = [m.strip() for m in (args.model_names or '').split(',') if m.strip()]
        if not models:
            print("Parsed empty --model_names. Please provide at least one model.")
            return
        # Collect the latest baseline entries per model for the same refinement level (respecting optional block_num)
        entries_maps_by_model: Dict[str, Dict[str, Any]] = {}
        for m in models:
            emap = prepare_blocked_comparison_entries(
                model_name=m,
                experiment_result_dir=args.experiment_result_dir,
                refinement_level=int(args.refinement_level),
                block_num=args.block_num,
            )
            if not emap:
                print(f"[WARN] No matching runs found for model: {m} (skipped)")
                continue
            entries_maps_by_model[m] = emap
        if not entries_maps_by_model:
            print("No matching runs found for any provided models.")
            return
        # Produce only the raw plots: Accuracy, NED, ED, EM (X-axis = baselines, series = models)
        plot_all_block_comparison_across_models(
            entries_maps_by_model=entries_maps_by_model,
            output_dir=args.output_dir,
            refinement_level=int(args.refinement_level),
            normalize=False,
        )
        # Additionally render the side-by-side EM/NED combined plot (raw only)
        # Parse --colors entries: baseline=color with '=' or ':' separators
        baseline_color_map = None
        if getattr(args, 'colors', None):
            try:
                baseline_color_map = {}
                for part in [p for p in args.colors.split(',') if p.strip()]:
                    if ':' in part:
                        k, v = part.split(':', 1)
                    elif '=' in part:
                        k, v = part.split('=', 1)
                    else:
                        continue
                    k = k.strip()
                    v = v.strip()
                    if k and v:
                        baseline_color_map[k] = v
            except Exception:
                baseline_color_map = None
        plot_all_block_comparison_em_ned_combined(
            entries_maps_by_model=entries_maps_by_model,
            output_dir=args.output_dir,
            refinement_level=int(args.refinement_level),
            normalize=False,
            baseline_color_map=baseline_color_map,
        )
    elif args.type == 'all_unblock_scale':
        if not args.model_names:
            print("Please specify --model_names (comma-separated) for all_unblock_scale.")
            return
        models = [m.strip() for m in (args.model_names or '').split(',') if m.strip()]
        if not models:
            print("Parsed empty --model_names. Please provide at least one model.")
            return
        # Parse refinement_levels (optional)
        ref_levels: Optional[List[int]] = None
        if getattr(args, 'refinement_levels', None):
            try:
                ref_levels = [int(x.strip()) for x in args.refinement_levels.split(',') if x.strip()]
            except Exception:
                print("[WARN] Failed to parse --refinement_levels, ignore this filter.")
                ref_levels = None
        color_list: Optional[List[str]] = None
        if getattr(args, 'colors', None):
            try:
                parsed_colors = [c.strip() for c in args.colors.split(',') if c.strip()]
                if parsed_colors:
                    color_list = parsed_colors
            except Exception:
                print("[WARN] Failed to parse --colors for all_unblock_scale, fallback to default palette.")
                color_list = None
        entries_unblocked_by_model = prepare_unblocked_scaling_entries(
            model_names=models,
            experiment_result_dir=args.experiment_result_dir,
            refinement_levels=ref_levels,
        )
        if not entries_unblocked_by_model:
            print("No unblocked runs found for any provided models.")
            return
        for normalize in (True, False):
            plot_all_unblocked_scaling_across_models(
                entries_unblocked_by_model=entries_unblocked_by_model,
                output_dir=args.output_dir,
                normalize=normalize,
                custom_colors=color_list,
            )
    elif args.type == 'all_noise_vis':
        # Requires a model list and refinement level; experiment_result_dir should point to the noise results directory (e.g., noise/)
        if not args.model_names:
            print("Please specify --model_names (comma-separated) for all_noise_vis.")
            return
        if args.refinement_level is None:
            print("Please specify --refinement_level for all_noise_vis.")
            return
        models = [m.strip() for m in (args.model_names or '').split(',') if m.strip()]
        if not models:
            print("Parsed empty --model_names. Please provide at least one model.")
            return
        # Parse the custom noise value order (string format consistent with filename parsing)
        noise_values_list: Optional[List[str]] = None
        if getattr(args, 'noise_values', None):
            try:
                noise_values_list = [s.strip() for s in args.noise_values.split(',') if s.strip()]
            except Exception:
                noise_values_list = None
        entries_by_model = prepare_noise_entries_across_models(
            model_names=models,
            experiment_result_dir=args.experiment_result_dir,
            refinement_level=int(args.refinement_level),
            noise_values=noise_values_list,
        )
        if not entries_by_model:
            print("No noise runs found for any provided models.")
            return
        # Output only the raw EM and NED 2x2 plots
        plot_all_noise_vis_across_models(
            entries_by_model=entries_by_model,
            output_dir=args.output_dir,
            refinement_level=int(args.refinement_level),
            noise_values_order=noise_values_list,
            normalize=False,
        )
    elif args.type == 'all_block_scale':
        # Requirements: provide --model_names and --refinement_level; optional --block_nums and --block_types
        if not args.model_names:
            print("Please specify --model_names (comma-separated) for all_block_scale.")
            return
        if args.refinement_level is None:
            print("Please specify --refinement_level for all_block_scale.")
            return
        models = [m.strip() for m in (args.model_names or '').split(',') if m.strip()]
        if not models:
            print("Parsed empty --model_names. Please provide at least one model.")
            return
        block_nums_list: Optional[List[int]] = None
        if getattr(args, 'block_nums', None):
            try:
                block_nums_list = [int(x.strip()) for x in args.block_nums.split(',') if x.strip()]
            except Exception:
                print("[WARN] Failed to parse --block_nums, ignore this filter.")
                block_nums_list = None
        block_types_list: Optional[List[str]] = None
        if getattr(args, 'block_types', None):
            try:
                block_types_list = [s.strip() for s in args.block_types.split(',') if s.strip()]
            except Exception:
                print("[WARN] Failed to parse --block_types, ignore this filter.")
                block_types_list = None
        baseline_color_map: Optional[Dict[str, str]] = None
        if getattr(args, 'colors', None):
            try:
                parsed_map: Dict[str, str] = {}
                for part in [p for p in args.colors.split(',') if p.strip()]:
                    if ':' in part:
                        key, value = part.split(':', 1)
                    elif '=' in part:
                        key, value = part.split('=', 1)
                    else:
                        continue
                    key = key.strip()
                    value = value.strip()
                    if key and value:
                        parsed_map[key] = value
                if parsed_map:
                    baseline_color_map = parsed_map
            except Exception:
                print("[WARN] Failed to parse --colors for all_block_scale, ignore this setting.")
                baseline_color_map = None
        entries_by_model_mode_bn = _prepare_all_block_scale_entries(
            model_names=models,
            experiment_result_dir=args.experiment_result_dir,
            refinement_level=int(args.refinement_level),
            block_nums=block_nums_list,
            block_types=block_types_list,
        )
        if not entries_by_model_mode_bn:
            print("No block scale runs found for any provided models.")
            return
        for m in models:
            em = entries_by_model_mode_bn.get(m)
            if not em:
                print(f"[WARN] No entries for model {m} (skipped)")
                continue
            # six charts: NED/EM/ACC raw + normalized
            for normalize in (True, False):
                _plot_block_scale_per_model_metric(
                    model_name=m,
                    entries_by_mode_bn=em,
                    output_dir=args.output_dir,
                    metric_key='avg_normalized_edit_distance',
                    metric_title='ANED',
                    file_tag='ned',
                    refinement_level=int(args.refinement_level),
                    normalize=normalize,
                    baseline_color_map=baseline_color_map,
                )
                _plot_block_scale_per_model_metric(
                    model_name=m,
                    entries_by_mode_bn=em,
                    output_dir=args.output_dir,
                    metric_key='tool_path_exact_match_ratio',
                    metric_title='EMR',
                    file_tag='em',
                    refinement_level=int(args.refinement_level),
                    normalize=normalize,
                    baseline_color_map=baseline_color_map,
                )
                _plot_block_scale_per_model_metric(
                    model_name=m,
                    entries_by_mode_bn=em,
                    output_dir=args.output_dir,
                    metric_key='final_answer_accuracy',
                    metric_title='Final Answer Accuracy',
                    file_tag='acc',
                    refinement_level=int(args.refinement_level),
                    normalize=normalize,
                    baseline_color_map=baseline_color_map,
                )
    elif args.type == 'error-analysis':
        # Requirements: --model_names, --refinement_level, --error_json
        if not args.model_names:
            print("Please specify --model_names (comma-separated) for error-analysis.")
            return
        if args.refinement_level is None:
            print("Please specify --refinement_level for error-analysis.")
            return
        if not args.error_json:
            print("Please specify --error_json for error-analysis.")
            return
        models = [m.strip() for m in (args.model_names or '').split(',') if m.strip()]
        if not models:
            print("Parsed empty --model_names. Please provide at least one model.")
            return
        outputs_map: Dict[str, List[str]] = {}
        tool_calls_map: Dict[str, int] = {}
        for m in models:
            full = _prepare_unblocked_run_full(
                model_name=m,
                experiment_result_dir=args.experiment_result_dir,
                refinement_level=int(args.refinement_level),
            )
            if not full:
                print(f"[WARN] No unblocked run found for model={m} at RL={args.refinement_level}")
                outputs_map[m] = []
                tool_calls_map[m] = 0
                continue
            errs = _extract_error_tool_responses_excluding_ban_tool(full)
            outputs_map[m] = errs
            tool_calls_map[m] = _count_total_tool_calls(full)
        # Write the JSON (even if empty) using the specified structure
        try:
            err_path = Path(args.error_json)
            err_path.parent.mkdir(parents=True, exist_ok=True)
            with open(err_path, 'w', encoding='utf-8') as fw:
                json.dump({
                    "outputs": outputs_map,
                    "total_tool_calls": tool_calls_map,
                }, fw, ensure_ascii=False, indent=2)
            print(f"Saved error analysis JSON to: {err_path}")
        except Exception as e:
            print(f"[ERROR] Failed to write error JSON: {e}")
            return
    
    print("All plots generated successfully!")

if __name__ == "__main__":
    main()
    