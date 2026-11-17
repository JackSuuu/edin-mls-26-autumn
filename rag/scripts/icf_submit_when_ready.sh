#!/usr/bin/env bash
# Usage: bash rag/scripts/icf_submit_when_ready.sh PREPARATION_PID
set -euo pipefail
cd "$HOME/edin-mls-26-autumn"
preparation_pid="${1:?pass the PID of the running preparation script}"
while kill -0 "$preparation_pid" 2>/dev/null; do
  sleep 15
done
if ! grep -q '^ICF_PREPARATION_COMPLETE$' rag/results/icf-prepare.log; then
  echo 'Preparation did not complete; GPU job was not submitted.'
  tail -n 30 rag/results/icf-prepare.log
  exit 1
fi
job_id=$(sbatch --parsable rag/scripts/icf_baseline.sbatch)
printf '%s\n' "$job_id" > rag/results/icf-baseline-job-id.txt
printf 'Submitted baseline job %s\n' "$job_id"
