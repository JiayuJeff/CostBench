#!/usr/bin/env bash

# bash env/utils/analyze_coverage.sh \
#   --refinement_levels "2,3,4,5"

# models to process
models=(
    "gpt-5"
    "gemini-2.5-pro"
    "Qwen3-32B"
    "Qwen3-14B"
    "Qwen3-8B"
)
refinement_levels=(2 3 4 5)
test_num=0

# parse args: -l/--refinement_levels accepts comma or space-separated values
while [[ $# -gt 0 ]]; do
	case "$1" in
		-l|--refinement_levels)
			levels_arg="$2"
			shift 2
			;;
		-h|--help)
			echo "Usage: $0 [-l \"2,3,4\" | --refinement_levels \"2 3 4\"]"
			exit 0
			;;
		*)
			echo "Unknown argument: $1"
			echo "Use --help for usage."
			exit 1
			;;
	 esac
done

if [[ -n "${levels_arg:-}" ]]; then
	# normalize commas to spaces, then split into array
	levels_arg="${levels_arg//,/ }"
	# shell array assignment from words
	refinement_levels=( ${levels_arg} )
fi

# show how many settings will run (after args potentially override levels)
num_models="${#models[@]}"
num_levels="${#refinement_levels[@]}"
total_settings=$(( num_models * num_levels ))
echo "Total settings to run: ${total_settings} (${num_models} models × ${num_levels} refinement levels)"

# for each model and refinement level: prepare plans, then count coverage
for m in "${models[@]}"; do
	for level in "${refinement_levels[@]}"; do
		python env/utils/analysis.py \
			--type unblocked_coverage \
			--model final_prompt/${m} \
			--refinement_level ${level} \
			--output_path_path final_prompt/${m}/first_assistant_texts_refine_${level}.json

		python env/utils/llm_client.py \
			--input_path final_prompt/${m}/first_assistant_texts_refine_${level}.json \
			--output_path final_prompt/${m}/first_assistant_texts_refine_${level}.json \
			--task count_coverage \
			--model_name Qwen/Qwen3-14B \
			--think_len vanilla \
			--begins_at 0 \
			--ends_at 0
		
		test_num=$(( test_num + 1 ))
		echo "Processed ${test_num} jobs: ${m} refine=${level}"
	done
done