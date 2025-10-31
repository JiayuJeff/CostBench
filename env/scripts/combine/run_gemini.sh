# bash env/scripts/combine/run_gemini.sh > env/scripts/combine/run_gemini.log 2>&1 &

bash env/scripts/analysis/noise_gemini-2.5-pro.sh > env/logs/final_prompt/noise_gemini-2.5-pro.log 2>&1

bash env/scripts/final_prompt/run_gemini-2.5-pro.sh > env/logs/final_prompt/run_gemini-2.5-pro.log 2>&1 &