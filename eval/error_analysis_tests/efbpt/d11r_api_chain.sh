#!/usr/bin/env bash
# D11-R API FULL chain (docs/EFBPT_D11R_AMENDMENT_2.md, A2.2/A2.3).
# Same order as the 2026-11-07 chain: sets rdev300, dev200, eval458; models
# qwen3.8-max, deepseek-v4-flash, qwen3.7-flash, deepseek-v4-pro. Re-running
# resumes (ok records kept, sealed sets untouched). The chain continues after
# exit 3 (INCOMPLETE: scattered errors, resumable) and stops on exit 4 (circuit
# breaker: API unreachable) or any other non-zero exit. The key is read from
# ~/.urbench_api_env into the environment only and never printed.
# Usage: nohup bash eval/error_analysis_tests/efbpt/d11r_api_chain.sh > logs/d11r_api_full_resume.log 2>&1 &
set -u
cd "$(dirname "$0")/../../.." || exit 1
set -a; source ~/.urbench_api_env; set +a
PY=/mnt/home/user41/miniconda3/envs/urbench_eval/bin/python
echo "D11-R API FULL chain start $(date)  (Amendment 2: concurrency 2, circuit breaker)"
for set in rdev300 dev200 eval458; do
  for model in qwen3.8-max deepseek-v4-flash qwen3.7-flash deepseek-v4-pro; do
    echo "=== $(date +%T) START model=$model set=$set ==="
    "$PY" -B eval/error_analysis_tests/efbpt/d11r_api_reader.py --full --model "$model" --set "$set"
    rc=$?
    echo "=== $(date +%T) END model=$model set=$set exit=$rc ==="
    if [ "$rc" -ne 0 ] && [ "$rc" -ne 3 ]; then
      echo "D11-R API FULL chain STOP $(date): exit $rc (4 = circuit breaker); no further sets started"
      exit "$rc"
    fi
  done
done
echo "D11-R API FULL chain end $(date)"
