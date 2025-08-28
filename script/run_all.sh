#!/bin/bash

set -euo pipefail

# Defaults
BASE_DIR="output"
FROM_DATE="2010-01-01"

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    -b|--base-dir)
      BASE_DIR="$2"; shift 2 ;;
    -f|--from-date)
      FROM_DATE="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [-b|--base-dir DIR] [-f|--from-date YYYY-MM-DD]"; exit 0 ;;
    *)
      echo "Unknown argument: $1"; exit 1 ;;
  esac
done

echo "[run_all] Using from-date=$FROM_DATE base-dir=$BASE_DIR"

uv run tennis_scrapper/main.py scrap-all --from-date "$FROM_DATE"
uv run tennis_scrapper/main.py generate-stats -o "$BASE_DIR"
uv run tennis_scrapper/train.py --base-dir "$BASE_DIR"
uv run tennis_scrapper/predict.py --base-dir "$BASE_DIR"