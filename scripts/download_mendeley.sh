#!/usr/bin/env bash
# Download the Mendeley "Video Dataset for Safe and Unsafe Behaviours" (xjmtb22pff v1).
#
# The public-api endpoint 302s to a presigned S3 URL that expires after ~300s,
# so a single curl cannot finish a 9.3 GB transfer. This loop re-resolves the
# redirect and resumes with a Range request until the file is complete.
set -u

DATASET_ID="${DATASET_ID:-xjmtb22pff}"
VERSION="${VERSION:-1}"
URL="https://data.mendeley.com/public-api/zip/${DATASET_ID}/download/${VERSION}"
OUT="${1:?usage: download_mendeley.sh <output.zip>}"
TOTAL="${TOTAL:-10002129420}"

# The link drops and DNS fails often here, so retry forever — the only real
# stop condition is the file being complete. Kill the process to abort.
attempt=0
while :; do
  attempt=$((attempt + 1))
  size=$(stat -c %s "$OUT" 2>/dev/null || echo 0)
  if [ "$size" -ge "$TOTAL" ]; then
    echo "DONE: $OUT ($size bytes)"
    exit 0
  fi
  before=$size
  pct=$(awk -v s="$size" -v t="$TOTAL" 'BEGIN{printf "%.2f", 100*s/t}')
  echo "[$(date +%H:%M:%S) attempt $attempt] at $size / $TOTAL (${pct}%)"
  curl -L -C - --max-time 280 --retry 0 -sS \
    -A "Mozilla/5.0" -o "$OUT" "$URL"
  after=$(stat -c %s "$OUT" 2>/dev/null || echo 0)
  echo "    +$(( (after - before) / 1024 )) KB this attempt"
  sleep 2
done
