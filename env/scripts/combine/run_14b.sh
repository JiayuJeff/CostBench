# bash env/scripts/combine/run_14b.sh > env/scripts/combine/run_14b.log 2>&1 &

bash env/scripts/analysis/noise_qwen3-14b.sh > env/logs/final_prompt/noise_qwen3-14b.log 2>&1

bash env/scripts/final_prompt/run_qwen3-14b.sh > env/logs/final_prompt/run_qwen3-14b.log 2>&1 &