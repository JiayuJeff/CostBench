# source env/scripts/test_run_gpt-5.sh > env/logs/run_gpt-5.log 2>&1 &

echo "Running gpt-5 model tests..."

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
    --model_name openai/gpt-5 \
    --temperature 1.0 \
    --max_tokens 131072 \
    --start_index 0 \
    --end_index -1 \
    --num_threads 64 \
    --output_dir outputs/ > env/logs/run_gpt-5.log 

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
    --model_name openai/gpt-5 \
    --temperature 1.0 \
    --max_tokens 131072 \
    --start_index 0 \
    --end_index -1 \
    --num_threads 64 \
    --output_dir outputs/ > env/logs/run_gpt-5.log 

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
    --model_name openai/gpt-5 \
    --temperature 1.0 \
    --max_tokens 131072 \
    --start_index 0 \
    --end_index -1 \
    --num_threads 64 \
    --output_dir outputs/ > env/logs/run_gpt-5.log 

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
    --model_name openai/gpt-5 \
    --temperature 1.0 \
    --max_tokens 131072 \
    --start_index 0 \
    --end_index -1 \
    --num_threads 64 \
    --output_dir outputs/ > env/logs/run_gpt-5.log 

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
    --model_name openai/gpt-5 \
    --temperature 1.0 \
    --max_tokens 131072 \
    --start_index 0 \
    --end_index -1 \
    --num_threads 64 \
    --output_dir outputs/ > env/logs/run_gpt-5.log 

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
    --model_name openai/gpt-5 \
    --temperature 1.0 \
    --max_tokens 131072 \
    --start_index 0 \
    --end_index -1 \
    --num_threads 64 \
    --output_dir outputs/ > env/logs/run_gpt-5.log 

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
    --model_name openai/gpt-5 \
    --temperature 1.0 \
    --max_tokens 131072 \
    --start_index 0 \
    --end_index -1 \
    --num_threads 64 \
    --output_dir outputs/ > env/logs/run_gpt-5.log 

# refinement level 3
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
    --model_name openai/gpt-5 \
    --temperature 1.0 \
    --max_tokens 131072 \
    --start_index 0 \
    --end_index -1 \
    --num_threads 64 \
    --output_dir outputs/ > env/logs/run_gpt-5.log 

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
    --model_name openai/gpt-5 \
    --temperature 1.0 \
    --max_tokens 131072 \
    --start_index 0 \
    --end_index -1 \
    --num_threads 64 \
    --output_dir outputs/ > env/logs/run_gpt-5.log 

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
    --model_name openai/gpt-5 \
    --temperature 1.0 \
    --max_tokens 131072 \
    --start_index 0 \
    --end_index -1 \
    --num_threads 64 \
    --output_dir outputs/ > env/logs/run_gpt-5.log 