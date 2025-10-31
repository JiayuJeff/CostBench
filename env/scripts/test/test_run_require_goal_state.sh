# source env/scripts/test_run_require_goal_state.sh > env/logs/test_run_require_goal_state.log 2>&1 &

export VLLM_USE_V1=1
echo "Running require_goal_state function tests..."

# use blocker
# cost change
echo "Testing 1/10: Cost Change"
python env/run.py --tool_creation_seed 42 --tool_output_dir env/data/runtime/travel/tools --refinement_level 5 --changed_tool_output_dir env/data/runtime/travel/changed_tools --max_tool_steps 20 --tool_mode memory --use_blocker --block_mode cost_change --block_num 1 --query_path env/data/runtime/queries/queries.json --model_name emrecanacikgoz/Qwen2.5-7B-Instruct-ToolRL-grpo-cold --temperature 0.0 --max_tokens 8192 --start_index 0 --end_index 100 --num_threads 100 --output_dir test_outputs/ --require_goal_state --ban_longest_tool


# preference change
echo "Testing 2/10: Preference Change"
python env/run.py --tool_creation_seed 42 --tool_output_dir env/data/runtime/travel/tools --refinement_level 5 --changed_tool_output_dir env/data/runtime/travel/changed_tools --max_tool_steps 20 --use_blocker --block_mode preference_change --block_num 1 --query_path env/data/runtime/queries/queries.json --model_name emrecanacikgoz/Qwen2.5-7B-Instruct-ToolRL-grpo-cold --temperature 0.0 --max_tokens 8192 --start_index 0 --end_index 100 --num_threads 100 --output_dir test_outputs/ --require_goal_state --ban_longest_tool

# ban tool
echo "Testing 3/10: Ban Tool"
python env/run.py --tool_creation_seed 42 --tool_output_dir env/data/runtime/travel/tools --refinement_level 5 --changed_tool_output_dir env/data/runtime/travel/changed_tools --max_tool_steps 20 --use_blocker --block_mode ban_tool --block_num 1 --query_path env/data/runtime/queries/queries.json --model_name emrecanacikgoz/Qwen2.5-7B-Instruct-ToolRL-grpo-cold --temperature 0.0 --max_tokens 8192 --start_index 0 --end_index 100 --num_threads 100 --output_dir test_outputs/ --require_goal_state --ban_longest_tool

# steplen change
echo "Testing 4/10: Step Length Change"
python env/run.py --tool_creation_seed 42 --tool_output_dir env/data/runtime/travel/tools --refinement_level 5 --changed_tool_output_dir env/data/runtime/travel/changed_tools --max_tool_steps 20 --use_blocker --block_mode steplen_change --block_num 1 --query_path env/data/runtime/queries/queries.json --model_name emrecanacikgoz/Qwen2.5-7B-Instruct-ToolRL-grpo-cold --temperature 0.0 --max_tokens 8192 --start_index 0 --end_index 100 --num_threads 100 --output_dir test_outputs/ --require_goal_state --ban_longest_tool

# no use blocker
# refinement level 0
echo "Testing 5/10: Refinement Level 0"
python env/run.py --tool_creation_seed 42 --tool_output_dir env/data/runtime/travel/tools --refinement_level 0 --changed_tool_output_dir env/data/runtime/travel/changed_tools --max_tool_steps 20 --query_path env/data/runtime/queries/queries.json --model_name emrecanacikgoz/Qwen2.5-7B-Instruct-ToolRL-grpo-cold --temperature 0.0 --max_tokens 8192 --start_index 0 --end_index 100 --num_threads 100 --output_dir test_outputs/ --require_goal_state --ban_longest_tool

# refinement level 1
echo "Testing 6/10: Refinement Level 1"
python env/run.py --tool_creation_seed 42 --tool_output_dir env/data/runtime/travel/tools --refinement_level 1 --changed_tool_output_dir env/data/runtime/travel/changed_tools --max_tool_steps 20 --query_path env/data/runtime/queries/queries.json --model_name emrecanacikgoz/Qwen2.5-7B-Instruct-ToolRL-grpo-cold --temperature 0.0 --max_tokens 8192 --start_index 0 --end_index 100 --num_threads 100 --output_dir test_outputs/ --require_goal_state --ban_longest_tool

# refinement level 2
echo "Testing 7/10: Refinement Level 2"
python env/run.py --tool_creation_seed 42 --tool_output_dir env/data/runtime/travel/tools --refinement_level 2 --changed_tool_output_dir env/data/runtime/travel/changed_tools --max_tool_steps 20 --query_path env/data/runtime/queries/queries.json --model_name emrecanacikgoz/Qwen2.5-7B-Instruct-ToolRL-grpo-cold --temperature 0.0 --max_tokens 8192 --start_index 0 --end_index 100 --num_threads 100 --output_dir test_outputs/ --require_goal_state --ban_longest_tool

# refinement level 3
echo "Testing 8/10: Refinement Level 3"
python env/run.py --tool_creation_seed 42 --tool_output_dir env/data/runtime/travel/tools --refinement_level 3 --changed_tool_output_dir env/data/runtime/travel/changed_tools --max_tool_steps 20 --query_path env/data/runtime/queries/queries.json --model_name emrecanacikgoz/Qwen2.5-7B-Instruct-ToolRL-grpo-cold --temperature 0.0 --max_tokens 8192 --start_index 0 --end_index 100 --num_threads 100 --output_dir test_outputs/ --require_goal_state --ban_longest_tool

# refinement level 4
echo "Testing 9/10: Refinement Level 4"
python env/run.py --tool_creation_seed 42 --tool_output_dir env/data/runtime/travel/tools --refinement_level 4 --changed_tool_output_dir env/data/runtime/travel/changed_tools --max_tool_steps 20 --query_path env/data/runtime/queries/queries.json --model_name emrecanacikgoz/Qwen2.5-7B-Instruct-ToolRL-grpo-cold --temperature 0.0 --max_tokens 8192 --start_index 0 --end_index 100 --num_threads 100 --output_dir test_outputs/ --require_goal_state --ban_longest_tool

# refinement level 5
echo "Testing 10/10: Refinement Level 5"
python env/run.py --tool_creation_seed 42 --tool_output_dir env/data/runtime/travel/tools --refinement_level 5 --changed_tool_output_dir env/data/runtime/travel/changed_tools --max_tool_steps 20 --query_path env/data/runtime/queries/queries.json --model_name emrecanacikgoz/Qwen2.5-7B-Instruct-ToolRL-grpo-cold --temperature 0.0 --max_tokens 8192 --start_index 0 --end_index 100 --num_threads 100 --output_dir test_outputs/ --require_goal_state --ban_longest_tool