#!/bin/bash
# Usage: cluster/fetch.sh JOB [JOB...]   - untars finished outputs into ./outputs
set -euo pipefail
cd "$(dirname "$0")/.."
for J in "$@"; do
  scp -q polaris:/tmp/ge/out_$J.tgz polaris:/tmp/ge/log_$J.txt /tmp/
  tar xzf /tmp/out_$J.tgz && mkdir -p outputs/cluster_logs && cp /tmp/log_$J.txt outputs/cluster_logs/ && echo "fetched $J"
done
