# source env/scripts/test_run_qwen3-8b.sh > env/logs/run_qwen3-8b.log 2>&1 &

echo "Running Qwen3-8B model tests..."

# use blocker
# cost change
echo "Testing 1/10: Cost Change"
python env/run.py \
    --tool_creation_seed 42 \
    --tool_output_dir env/data/runtime/travel/tools \
    --refinement_level 5 \
    --changed_tool_output_dir env/data/runtime/travel/changed_tools \
    --max_tool_steps 20 \
    --tool_mode memory \
    --min_atomic_cost 15 \
    --max_atomic_cost 25 \
    --noise_std 0.1 \
    --use_blocker \
    --block_mode cost_change \
    --block_num 1 \
    --ban_longest_tool \
    --query_path env/data/runtime/queries/queries.json \
    --model_name Qwen/Qwen3-8B \
    --temperature 0.6 \
    --max_tokens 4096 \
    --start_index 0 \
    --end_index -1 \
    --num_threads 95 \
    --output_dir outputs/ > env/logs/run_qwen3-8b_cost_change.log 2>&1

# preference change
echo "Testing 2/10: Preference Change"
python env/run.py \
    --tool_creation_seed 42 \
    --tool_output_dir env/data/runtime/travel/tools \
    --refinement_level 5 \
    --changed_tool_output_dir env/data/runtime/travel/changed_tools \
    --max_tool_steps 20 \
    --tool_mode memory \
    --min_atomic_cost 15 \
    --max_atomic_cost 25 \
    --noise_std 0.1 \
    --use_blocker \
    --block_mode preference_change \
    --block_num 1 \
    --ban_longest_tool \
    --query_path env/data/runtime/queries/queries.json \
    --model_name Qwen/Qwen3-8B \
    --temperature 0.6 \
    --max_tokens 4096 \
    --start_index 0 \
    --end_index -1 \
    --num_threads 95 \
    --output_dir outputs/ > env/logs/run_qwen3-8b_preference_change.log 2>&1

# ban tool
echo "Testing 3/10: Ban Tool"
python env/run.py \
    --tool_creation_seed 42 \
    --tool_output_dir env/data/runtime/travel/tools \
    --refinement_level 5 \
    --changed_tool_output_dir env/data/runtime/travel/changed_tools \
    --max_tool_steps 20 \
    --tool_mode memory \
    --min_atomic_cost 15 \
    --max_atomic_cost 25 \
    --noise_std 0.1 \
    --use_blocker \
    --block_mode ban_tool \
    --block_num 1 \
    --ban_longest_tool \
    --query_path env/data/runtime/queries/queries.json \
    --model_name Qwen/Qwen3-8B \
    --temperature 0.6 \
    --max_tokens 4096 \
    --start_index 0 \
    --end_index -1 \
    --num_threads 95 \
    --output_dir outputs/ > env/logs/run_qwen3-8b_ban_tool.log 2>&1

# steplen change
echo "Testing 4/10: Step Length Change"
python env/run.py \
    --tool_creation_seed 42 \
    --tool_output_dir env/data/runtime/travel/tools \
    --refinement_level 5 \
    --changed_tool_output_dir env/data/runtime/travel/changed_tools \
    --max_tool_steps 20 \
    --tool_mode memory \
    --min_atomic_cost 15 \
    --max_atomic_cost 25 \
    --noise_std 0.1 \
    --use_blocker \
    --block_mode steplen_change \
    --block_num 1 \
    --ban_longest_tool \
    --query_path env/data/runtime/queries/queries.json \
    --model_name Qwen/Qwen3-8B \
    --temperature 0.6 \
    --max_tokens 4096 \
    --start_index 0 \
    --end_index -1 \
    --num_threads 95 \
    --output_dir outputs/ > env/logs/run_qwen3-8b_steplen_change.log 2>&1

# no use blocker
# refinement level 0
echo "Testing 5/10: Refinement Level 0"
python env/run.py \
    --tool_creation_seed 42 \
    --tool_output_dir env/data/runtime/travel/tools \
    --changed_tool_output_dir env/data/runtime/travel/changed_tools \
    --refinement_level 0 \
    --max_tool_steps 20 \
    --tool_mode memory \
    --min_atomic_cost 15 \
    --max_atomic_cost 25 \
    --noise_std 0.1 \
    --ban_longest_tool \
    --query_path env/data/runtime/queries/queries.json \
    --model_name Qwen/Qwen3-8B \
    --temperature 0.6 \
    --max_tokens 4096 \
    --start_index 0 \
    --end_index -1 \
    --num_threads 95 \
    --output_dir outputs/ > env/logs/run_qwen3-8b_refinement_level_0.log 2>&1

# refinement level 1
echo "Testing 6/10: Refinement Level 1"
python env/run.py \
    --tool_creation_seed 42 \
    --tool_output_dir env/data/runtime/travel/tools \
    --changed_tool_output_dir env/data/runtime/travel/changed_tools \
    --refinement_level 1 \
    --max_tool_steps 20 \
    --tool_mode memory \
    --min_atomic_cost 15 \
    --max_atomic_cost 25 \
    --noise_std 0.1 \
    --ban_longest_tool \
    --query_path env/data/runtime/queries/queries.json \
    --model_name Qwen/Qwen3-8B \
    --temperature 0.6 \
    --max_tokens 4096 \
    --start_index 0 \
    --end_index -1 \
    --num_threads 95 \
    --output_dir outputs/ > env/logs/run_qwen3-8b_refinement_level_1.log 2>&1

# refinement level 2
echo "Testing 7/10: Refinement Level 2"
python env/run.py \
    --tool_creation_seed 42 \
    --tool_output_dir env/data/runtime/travel/tools \
    --changed_tool_output_dir env/data/runtime/travel/changed_tools \
    --refinement_level 2 \
    --max_tool_steps 20 \
    --tool_mode memory \
    --min_atomic_cost 15 \
    --max_atomic_cost 25 \
    --noise_std 0.1 \
    --ban_longest_tool \
    --query_path env/data/runtime/queries/queries.json \
    --model_name Qwen/Qwen3-8B \
    --temperature 0.6 \
    --max_tokens 4096 \
    --start_index 0 \
    --end_index -1 \
    --num_threads 95 \
    --output_dir outputs/ > env/logs/run_qwen3-8b_refinement_level_2.log 2>&1

# refinement level 3
echo "Testing 8/10: Refinement Level 3"
python env/run.py \
    --tool_creation_seed 42 \
    --tool_output_dir env/data/runtime/travel/tools \
    --changed_tool_output_dir env/data/runtime/travel/changed_tools \
    --refinement_level 3 \
    --max_tool_steps 20 \
    --tool_mode memory \
    --min_atomic_cost 15 \
    --max_atomic_cost 25 \
    --noise_std 0.1 \
    --ban_longest_tool \
    --query_path env/data/runtime/queries/queries.json \
    --model_name Qwen/Qwen3-8B \
    --temperature 0.6 \
    --max_tokens 4096 \
    --start_index 0 \
    --end_index -1 \
    --num_threads 95 \
    --output_dir outputs/ > env/logs/run_qwen3-8b_refinement_level_3.log 2>&1

# refinement level 4
echo "Testing 9/10: Refinement Level 4"
python env/run.py \
    --tool_creation_seed 42 \
    --tool_output_dir env/data/runtime/travel/tools \
    --changed_tool_output_dir env/data/runtime/travel/changed_tools \
    --refinement_level 4 \
    --max_tool_steps 20 \
    --tool_mode memory \
    --min_atomic_cost 15 \
    --max_atomic_cost 25 \
    --noise_std 0.1 \
    --ban_longest_tool \
    --query_path env/data/runtime/queries/queries.json \
    --model_name Qwen/Qwen3-8B \
    --temperature 0.6 \
    --max_tokens 4096 \
    --start_index 0 \
    --end_index -1 \
    --num_threads 95 \
    --output_dir outputs/ > env/logs/run_qwen3-8b_refinement_level_4.log 2>&1

# refinement level 5
echo "Testing 10/10: Refinement Level 5"
python env/run.py \
    --tool_creation_seed 42 \
    --tool_output_dir env/data/runtime/travel/tools \
    --changed_tool_output_dir env/data/runtime/travel/changed_tools \
    --refinement_level 5 \
    --max_tool_steps 20 \
    --tool_mode memory \
    --min_atomic_cost 15 \
    --max_atomic_cost 25 \
    --noise_std 0.1 \
    --ban_longest_tool \
    --query_path env/data/runtime/queries/queries.json \
    --model_name Qwen/Qwen3-8B \
    --temperature 0.6 \
    --max_tokens 4096 \
    --start_index 0 \
    --end_index -1 \
    --num_threads 95 \
    --output_dir outputs/ > env/logs/run_qwen3-8b_refinement_level_5.log 2>&1