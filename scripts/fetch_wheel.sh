#!/usr/bin/env bash
# Resumable wheel download. uv's 30s HTTP timeout restarts a 60 MB transfer from
# zero on this link; curl -C - resumes where it left off, so a stall costs
# seconds instead of the whole file.
set -u
URL="${1:?usage: fetch_wheel.sh <url> <out>}"
OUT="${2:?usage: fetch_wheel.sh <url> <out>}"
EXPECTED="${3:-0}"
for attempt in $(seq 1 200); do
  size=$(stat -c %s "$OUT" 2>/dev/null || echo 0)
  if [ "$EXPECTED" -gt 0 ] && [ "$size" -ge "$EXPECTED" ]; then
    echo "DONE: $OUT ($size bytes)"; exit 0
  fi
  echo "[attempt $attempt] at $size bytes"
  curl -L -C - --max-time 240 --retry 0 -sS -o "$OUT" "$URL" && {
    echo "DONE: $OUT ($(stat -c %s "$OUT") bytes)"; exit 0
  }
  sleep 2
done
echo "FAILED after 200 attempts" >&2; exit 1
