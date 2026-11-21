#!/bin/bash
# Usage (from the repo root on your laptop): cluster/launch.sh JOB [JOB...]
# Ships code + data/clean to polaris:/tmp/ge, then starts one detached srun per JOB
# (cluster/runs/JOB.sh). Collect results later with cluster/fetch.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
tar czf /tmp/ge_in.tgz common.py neural.py data/clean data/raw/emotions.txt cluster/runs --transform 's,^cluster/,,'
ssh polaris 'mkdir -p /tmp/ge'
scp -q /tmp/ge_in.tgz cluster/job.sh polaris:/tmp/ge/
for J in "$@"; do
  ssh -f polaris "cd /tmp/ge && JOB=$J setsid nohup srun -A free -p gpu_preempt --gres=gpu:1 -c 4 --mem=32G -t 06:00:00 \
    --job-name=ge_$J --chdir=/tmp bash -c \"\$(cat job.sh)\" < ge_in.tgz > out_$J.tgz 2> log_$J.txt & disown; echo launched $J"
done
