#!/bin/bash
SUMMARY=/media/volume/llm/unlearning/1.data-preparation/unlearn/logs_wikitext_unlearn_batch/_summary.csv
SENTINEL=/media/volume/llm/unlearning/scripts/.stage2_done
while true; do
  n=$(wc -l < "$SUMMARY" 2>/dev/null || echo 0)
  if [ "$n" -ge 101 ]; then
    date -Is > "$SENTINEL"
    echo "stage2 batch complete at $(date -Is), rows=$n" >> "$SENTINEL".log
    exit 0
  fi
  sleep 60
done
