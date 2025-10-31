#!/usr/bin/env bash
# bash env/scripts/final_prompt/longest_qwen3-14b.sh > env/logs/longest_tool/run_qwen3-14b.log 2>&1 &

# set -euo pipefail

# 运行模式选择: run_unblocked, run_blocked, run_block_scaling
# 可以选择多个模式，例如: ("run_unblocked" "run_blocked")
run_modes=("run_unblocked")  # 修改这里来选择运行模式，支持多个模式

model="Qwen/Qwen3-14B"
saved_model="Qwen3-14B"
output_dir="longest_tool/$saved_model"
max_tokens=8192
temperature=0.0
num_threads=76
end_index=-1
use_stimulation=True
stimulation_num=1
block_modes=("cost_change" "preference_change" "ban_tool" "steplen_change")
log_base_dir="longest_tool"

# 不同模式的refinement_levels配置
run_unblocked_refinement_levels=(2)
run_blocked_refinement_levels=(2)
run_block_scaling_refinement_levels=(4)

block_nums=(1 2 3) # 只在run_block_scaling模式下使用

# Ensure required directories exist
mkdir -p "$output_dir"

# 计算总测试数
num_refinement_levels_unblocked=${#run_unblocked_refinement_levels[@]}
num_refinement_levels_blocked=${#run_blocked_refinement_levels[@]}
num_refinement_levels_scaling=${#run_block_scaling_refinement_levels[@]}
num_block_modes=${#block_modes[@]}
num_block_nums=${#block_nums[@]}

total_tests=0
for run_mode in "${run_modes[@]}"; do
    case $run_mode in
        "run_unblocked")
            total_tests=$(( total_tests + num_refinement_levels_unblocked ))
            ;;
        "run_blocked")
            total_tests=$(( total_tests + num_refinement_levels_blocked * num_block_modes ))
            ;;
        "run_block_scaling")
            total_tests=$(( total_tests + num_refinement_levels_scaling * num_block_modes * num_block_nums ))
            ;;
        *)
            echo "错误：未知的运行模式 $run_mode"
            exit 1
            ;;
    esac
done

echo "运行模式: ${run_modes[*]}"
echo "总测试数: $total_tests"

test_num=0

# 遍历所有运行模式
for run_mode in "${run_modes[@]}"; do
    echo "开始运行模式: $run_mode"
    
    # 设置当前模式的log目录
    log_dir="env/logs/$log_base_dir/$run_mode/${saved_model}"
    mkdir -p "$log_dir"
    
    # 根据运行模式选择对应的refinement_levels
    case $run_mode in
        "run_unblocked")
            refinement_levels=("${run_unblocked_refinement_levels[@]}")
            ;;
        "run_blocked")
            refinement_levels=("${run_blocked_refinement_levels[@]}")
            ;;
        "run_block_scaling")
            refinement_levels=("${run_block_scaling_refinement_levels[@]}")
            ;;
    esac

    if [ "$run_mode" = "run_unblocked" ]; then
        # run_unblocked模式：只跑unblocked
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
                --use_example > $log_dir/results_${saved_model}_unblocked_refinement_${refinement_level}.log 2>&1
        
        done

    elif [ "$run_mode" = "run_blocked" ]; then
        # run_blocked模式：只跑blocked，block_num=1
        for refinement_level in "${refinement_levels[@]}"; do
            for block_mode in "${block_modes[@]}"; do
                test_num=$((test_num + 1))
                echo "Running $model with $block_mode (block_num: 1) and refinement level $refinement_level"
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
                    --use_example > $log_dir/results_${saved_model}_${block_mode}_1_refinement_${refinement_level}.log 2>&1
            done
        done

    elif [ "$run_mode" = "run_block_scaling" ]; then
        # run_block_scaling模式：跑所有block_nums的四种block type
        for refinement_level in "${refinement_levels[@]}"; do
            for block_mode in "${block_modes[@]}"; do
                for block_num in "${block_nums[@]}"; do
                    test_num=$((test_num + 1))
                    echo "Running $model with $block_mode (block_num: $block_num) and refinement level $refinement_level"
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
                        --block_num $block_num \
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
                        --use_example > $log_dir/results_${saved_model}_${block_mode}_${block_num}_refinement_${refinement_level}.log 2>&1
                done
            done
        done
    fi

done  # 结束运行模式循环

echo "运行完成"