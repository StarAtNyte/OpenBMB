#!/usr/bin/env bash
# SOAR 2026 - Model Preprocessing Script
# Interface: bash prepare_model.sh --input <original_model_path> --output <processed_model_path>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[prepare_model] start $(date '+%F %T')"

# Parse arguments
INPUT_PATH=""
OUTPUT_PATH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --input)
            INPUT_PATH="$2"
            shift 2
            ;;
        --output)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [ -z "$INPUT_PATH" ] || [ -z "$OUTPUT_PATH" ]; then
    echo "Usage: bash prepare_model.sh --input <model_path> --output <output_path>"
    exit 1
fi

echo "[prepare_model] Input:  $INPUT_PATH"
echo "[prepare_model] Output: $OUTPUT_PATH"

# Run RTN W4A16 quantization
python3 "${SCRIPT_DIR}/preprocess_model.py" \
    --input "$INPUT_PATH" \
    --output "$OUTPUT_PATH" \
    --mode rtn \
    --bits 4 \
    --group-size 128

echo "[prepare_model] done"
