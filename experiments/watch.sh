#!/bin/bash
# Keep-alive watcher: prints runner progress for up to ~28 minutes, then exits so the session re-arms it.
cd "$(dirname "$0")"
end=$(( $(date +%s) + 1680 ))
while [ $(date +%s) -lt $end ]; do
  kill -0 "$(cat logs/runner.pid)" 2>/dev/null || { echo "RUNNER DEAD"; break; }
  sleep 30
done
echo "results: $(wc -l < results.jsonl)"; tail -3 logs/run.log
