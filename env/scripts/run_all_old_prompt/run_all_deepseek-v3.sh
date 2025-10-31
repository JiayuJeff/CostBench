#!/usr/bin/env bash
# bash env/scripts/run_all/run_all_deepseek-v3.sh > env/logs/run_all/run_all_deepseek-v3.log 2>&1 &

# set -euo pipefail

model="deepseek-v3"
saved_model="deepseek-v3"
output_dir="all_outputs/$saved_model"
max_tokens=8192
temperature=0.0
num_threads=381
end_index=1
use_example=True
provide_composite_concept=True
provide_atomic_tool_sequence=True
use_stimulation=True
stimulation_num=1
block_modes=("cost_change" "preference_change" "ban_tool" "steplen_change")
refinement_levels=(0 1 2 3 4 5)
less_block_num=1
block_nums=(2 3)
refinement_threshold=4

# Ensure required directories exist
mkdir -p "env/logs/run_all/${saved_model}"
mkdir -p "$output_dir"

# Calculate total number of tests with conditional block_num usage:
# for refinement levels < $refinement_threshold run only one block_num (less_block_num), otherwise run all block_nums
num_refinement_levels=${#refinement_levels[@]}
num_block_modes=${#block_modes[@]}
num_block_nums=${#block_nums[@]}
less_count=0
for rl in "${refinement_levels[@]}"; do
    if [ "$rl" -lt "$refinement_threshold" ]; then
        less_count=$((less_count + 1))
    fi
done
geq_count=$(( num_refinement_levels - less_count ))
total_tests=$(( less_count * (1 + num_block_modes * 1) + geq_count * (1 + num_block_modes * num_block_nums) ))

echo "Running deepseek-v3 model run_all..."       

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
        --temperature $temperature \
        --max_tokens $max_tokens \
        --start_index 0 \
        --end_index $end_index \
        --num_threads $num_threads \
        --output_dir $output_dir \
        --use_example \
        --provide_composite_concept \
        --provide_atomic_tool_sequence \
        --use_stimulation \
        --stimulation_num 1 \
        --greedy > env/logs/run_all/${saved_model}/results_${saved_model}_unblocked_refinement_level_${refinement_level}.log 2>&1
    
    for block_mode in "${block_modes[@]}"; do

        if [ "$refinement_level" -lt "$refinement_threshold" ]; then

            block_num="$less_block_num"
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
                --ban_longest_tool \
                --query_path env/data/runtime/queries/queries.json \
                --model_name $model \
                --temperature $temperature \
                --max_tokens $max_tokens \
                --start_index 0 \
                --end_index $end_index \
                --num_threads $num_threads \
                --output_dir $output_dir \
                --use_example \
                --provide_composite_concept \
                --provide_atomic_tool_sequence \
                --use_stimulation \
                --stimulation_num 1 \
                --greedy > env/logs/run_all/${saved_model}/results_${saved_model}_${block_mode}_${block_num}_refinement_level_${refinement_level}.log 2>&1

        else

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
                    --ban_longest_tool \
                    --query_path env/data/runtime/queries/queries.json \
                    --model_name $model \
                    --temperature $temperature \
                    --max_tokens $max_tokens \
                    --start_index 0 \
                    --end_index $end_index \
                    --num_threads $num_threads \
                    --output_dir $output_dir \
                    --use_example \
                    --provide_composite_concept \
                    --provide_atomic_tool_sequence \
                    --use_stimulation \
                    --stimulation_num 1 \
                    --greedy > env/logs/run_all/${saved_model}/results_${saved_model}_${block_mode}_${block_num}_refinement_level_${refinement_level}.log 2>&1

            done

        fi

    done
done
