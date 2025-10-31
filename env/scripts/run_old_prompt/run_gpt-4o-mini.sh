#!/usr/bin/env bash
# bash env/scripts/run_qwen3-8b.sh > env/logs/run_qwen3-8b.log 2>&1 &

# set -euo pipefail

model="Qwen/Qwen3-8B"
saved_model="Qwen3-8B"
max_tokens=8192
num_threads=100
end_index=-1
use_example=True
provide_composite_concept=True
block_modes=("cost_change" "preference_change" "ban_tool" "steplen_change")
refinement_levels=(0 1 2 3 4 5)

# Ensure required directories exist
mkdir -p "env/logs/${saved_model}" "outputs"

# Calculate total number of tests: for each refinement level, 1 unblocked + N block modes
num_refinement_levels=${#refinement_levels[@]}
num_block_modes=${#block_modes[@]}
total_tests=$(( num_refinement_levels * (1 + num_block_modes) ))

echo "Running Qwen3-8B model tests..."

test_num=0
for refinement_level in "${refinement_levels[@]}"; do

    test_num=$((test_num + 1))
    echo "Running $model with no blocking and refinement level $refinement_level"
    echo "Running $test_num/$total_tests"

    python env/run.py \
        --tool_creation_seed 42 \
        --tool_output_dir env/data/runtime/travel/tools \
        --refinement_level $refinement_level \
        --changed_tool_output_dir env/data/runtime/travel/changed_tools \
        --max_tool_steps 20 \
        --tool_mode memory \
        --min_atomic_cost 15 \
        --max_atomic_cost 25 \
        --noise_std 0.1 \
        --ban_longest_tool \
        --query_path env/data/runtime/queries/queries.json \
        --model_name $model \
        --temperature 0.6 \
        --max_tokens $max_tokens \
        --use_example \
        --provide_composite_concept \
        --start_index 0 \
        --end_index $end_index \
        --num_threads $num_threads \
        --output_dir outputs/ \
        --use_example \
        --provide_composite_concept \
        --provide_atomic_tool_sequence \
        --use_stimulation \
        --stimulation_num 1 > env/logs/${saved_model}/run_${saved_model}_unblocked_refinement_level_${refinement_level}.log 2>&1
    
    for block_mode in "${block_modes[@]}"; do

        test_num=$((test_num + 1))
        echo "Running $model with $block_mode and refinement level $refinement_level"
        echo "Running $test_num/$total_tests"

        python env/run.py \
            --tool_creation_seed 42 \
            --tool_output_dir env/data/runtime/travel/tools \
            --refinement_level $refinement_level \
            --changed_tool_output_dir env/data/runtime/travel/changed_tools \
            --max_tool_steps 20 \
            --tool_mode memory \
            --min_atomic_cost 15 \
            --max_atomic_cost 25 \
            --noise_std 0.1 \
            --use_blocker \
            --block_mode $block_mode \
            --block_num 1 \
            --ban_longest_tool \
            --query_path env/data/runtime/queries/queries.json \
            --model_name $model \
            --temperature 0.6 \
            --max_tokens $max_tokens \
            --use_example \
            --provide_composite_concept \
            --start_index 0 \
            --end_index $end_index \
            --num_threads $num_threads \
            --output_dir outputs/ \
            --use_example \
            --provide_composite_concept \
            --provide_atomic_tool_sequence \
            --use_stimulation \
            --stimulation_num 1 > env/logs/${saved_model}/run_${saved_model}_${block_mode}_refinement_level_${refinement_level}.log 2>&1
    
    done
done

# # Tool related
# args.add_argument("--tool_creation_seed", type=int, default=42, help="工具生成随机种子，控制每个query的工具creation random seed，还控制cost_change的下一批cost的生成")
# args.add_argument("--tool_output_dir", type=str, default="env/data/runtime/tools", help="工具输出目录")
# args.add_argument("--refinement_level", type=int, default=5, help="工具精炼级别")
# args.add_argument("--changed_tool_output_dir", type=str, default="env/data/runtime/changed_tools", help="缓存工具输出目录")
# args.add_argument("--max_tool_steps", type=int, default=5, help="最大工具调用步数")
# args.add_argument("--tool_mode", type=str, default="memory", choices=["memory", "file"], help="工具生成模式：memory 或 file，默认 memory")
# args.add_argument("--min_atomic_cost", type=int, default=19, help="原子工具成本最小值")
# args.add_argument("--max_atomic_cost", type=int, default=21, help="原子工具成本最大值")
# args.add_argument("--noise_std", type=float, default=0.1, help="复合工具噪声强度系数。实际噪声标准差 = noise_constant * 组件数量")

# # Blocker related
# args.add_argument("--use_blocker", action="store_true", help="是否使用Blocker")
# args.add_argument("--block_mode", type=str, default="mixed", choices=["mixed", "preference_change", "cost_change", "steplen_change", "ban_tool"], help="Blocker模式")
# args.add_argument("--block_num", type=int, default=3, help="每个查询的块数")
# args.add_argument("--ban_longest_tool", action="store_true", help="是否禁用能一步走完的工具")
# args.add_argument("--control_tool_length", action="store_true", help="是否控制工具长度")
# args.add_argument("--max_tool_length", type=int, default=8, help="最大工具长度")

# # Query related
# args.add_argument("--query_path", type=str, default="env/data/queries/travel_queries.json", help="查询文件路径")

# # Model related
# args.add_argument("--model_name", type=str, default="Qwen/Qwen3-32B", help="模型名称")
# args.add_argument("--temperature", type=float, default=0.0, help="采样温度")
# args.add_argument("--max_tokens", type=int, default=4096, help="最大生成长度")

# # Inference_utils related
# args.add_argument("--use_example", action="store_true", help="是否使用example")
# args.add_argument("--start_index", type=int, default=0, help="开始查询索引")
# args.add_argument("--end_index", type=int, default=-1, help="结束查询索引，-1表示到最后")
# args.add_argument("--num_threads", type=int, default=4, help="并发线程数")
# args.add_argument("--output_dir", type=str, default="results", help="结果输出目录")
# args.add_argument("--require_goal_state", action="store_true", help="若开启，则在达到目标数据类型前强制工具调用，并以达成goal state作为终止条件")
# args.add_argument("--print_tool_interface", action="store_true", help="若开启，则在每个工具调用前打印工具接口")