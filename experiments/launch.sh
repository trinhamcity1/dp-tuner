#!/bin/bash
# Start the benchmark runner detached from the shell; safe to re-run (it resumes and refuses duplicates).
cd "$(dirname "$0")/.." && source .venv/bin/activate
PID_FILE=experiments/logs/runner.pid
mkdir -p experiments/logs
if [ -f $PID_FILE ] && kill -0 "$(cat $PID_FILE)" 2>/dev/null; then echo "runner already running: $(cat $PID_FILE)"; exit 0; fi
setsid nohup python3 experiments/runner.py --workers "${WORKERS:-2}" "$@" >> experiments/logs/runner.out 2>&1 < /dev/null &
echo $! > $PID_FILE
echo "runner started: $!"
