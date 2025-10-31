#!/usr/bin/env bash
# bash env/scripts/robustness/run_qwen3-14b.sh > env/logs/final_prompt/seed_qwen3-14b.log 2>&1 &

# set -euo pipefail

# 运行模式：仅 unblocked
run_mode="run_unblocked"

model="Qwen/Qwen3-14B"
saved_model="Qwen3-14B"
output_dir="seed/$saved_model"
max_tokens=8192
temperature=0.0
num_threads=76
end_index=-1
use_stimulation=True
stimulation_num=1
log_base_dir="seed"

# 噪声设置：在 unblocked 模式下测试不同 seed_std
seeds=(1000 2000 3000 4000 5000)

# refinement_levels 配置（仅 unblocked 使用）
run_unblocked_refinement_levels=(2)

# Ensure required directories exist
mkdir -p "$output_dir"

# 计算总测试数（unblocked x seeds）
num_refinement_levels_unblocked=${#run_unblocked_refinement_levels[@]}
num_seeds=${#seeds[@]}
total_tests=$(( num_refinement_levels_unblocked * num_seeds ))

echo "运行模式: ${run_modes[*]}"
echo "总测试数: $total_tests"

test_num=0

echo "开始运行模式: $run_mode"

# 设置当前模式的log目录
log_dir="env/logs/$log_base_dir/$run_mode/${saved_model}"
mkdir -p "$log_dir"

refinement_levels=("${run_unblocked_refinement_levels[@]}")

# run_unblocked 模式：遍历 refinement_levels 与 seeds
for refinement_level in "${refinement_levels[@]}"; do
    for seed in "${seeds[@]}"; do
        test_num=$((test_num + 1))
        echo "Running $model with no blocking, refinement level $refinement_level, seed_std $seed"
        echo "output_dir: $output_dir"
        echo "Running $test_num/$total_tests"

        python env/run_seed.py \
            --tool_creation_seed $seed \
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
            --temperature $temperature \
            --max_tokens $max_tokens \
            --start_index 0 \
            --end_index $end_index \
            --num_threads $num_threads \
            --output_dir $output_dir \
            ${use_stimulation:+--use_stimulation} \
            --stimulation_num $stimulation_num \
            --greedy \
            --provide_atomic_tool_sequence \
            --use_example > $log_dir/results_${saved_model}_unblocked_seed_${seed}_refinement_${refinement_level}.log 2>&1
    done
done

echo "运行完成"