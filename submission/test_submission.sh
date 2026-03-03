#!/usr/bin/env bash
# Test script to simulate SOAR platform execution
# Run this inside Modal shell to debug submission issues

set -x  # Print commands as they execute

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_INPUT="/models/MiniCPM-SALA"
MODEL_OUTPUT="/tmp/processed_model"

echo "============================================"
echo "Testing SOAR Submission"
echo "Script dir: $SCRIPT_DIR"
echo "============================================"

# Step 1: Test prepare_env.sh (sourced, not executed)
echo ""
echo "=== Step 1: Sourcing prepare_env.sh ==="
cd "$SCRIPT_DIR"
source ./prepare_env.sh
echo "SGLANG_SERVER_ARGS: $SGLANG_SERVER_ARGS"

# Step 2: Test prepare_model.sh
echo ""
echo "=== Step 2: Running prepare_model.sh ==="
rm -rf "$MODEL_OUTPUT"
mkdir -p "$MODEL_OUTPUT"
bash ./prepare_model.sh --input "$MODEL_INPUT" --output "$MODEL_OUTPUT"
RESULT=$?
echo "prepare_model.sh exit code: $RESULT"

if [ $RESULT -ne 0 ]; then
    echo "FAILED: prepare_model.sh returned non-zero exit code"
    exit 1
fi

# Step 3: Verify output
echo ""
echo "=== Step 3: Verifying output ==="
ls -la "$MODEL_OUTPUT"/ | head -20
echo ""
echo "Config files:"
cat "$MODEL_OUTPUT/config.json" 2>/dev/null | head -20 || echo "No config.json"

# Step 4: Test SGLang server startup (optional)
echo ""
echo "=== Step 4: Test SGLang server (5 second test) ==="
echo "Would start: python -m sglang.launch_server --model-path $MODEL_OUTPUT $SGLANG_SERVER_ARGS"

echo ""
echo "============================================"
echo "Submission test completed successfully!"
echo "============================================"
