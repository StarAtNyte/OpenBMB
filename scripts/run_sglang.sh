#!/bin/bash
# SGLang Server Launch Script for SOAR 2026
# Matches competition evaluation environment settings
#
# Usage:
#   bash scripts/run_sglang.sh                              # Default
#   bash scripts/run_sglang.sh --quantization gptq_marlin   # With GPTQ

set -e

MODEL_PATH=${MODEL_PATH:-/models/MiniCPM-SALA}
PORT=${PORT:-30000}
HOST=${HOST:-0.0.0.0}

# Default args for MiniCPM-SALA model
# - minicpm_flashinfer: required for SALA hybrid sparse+linear attention
# - skip-server-warmup: prevents OOM during initialization
# - dense-as-sparse: enables sparse attention optimization
DEFAULT_ARGS=(
    "--disable-radix-cache"
    "--attention-backend" "minicpm_flashinfer"
    "--chunked-prefill-size" "8192"
    "--mem-fraction-static" "0.70"
    "--dense-as-sparse"
    "--skip-server-warmup"
)

# Parse SGLANG_SERVER_ARGS if set (for custom overrides)
if [ -n "$SGLANG_SERVER_ARGS" ]; then
    EXTRA_ARGS=($SGLANG_SERVER_ARGS)
else
    EXTRA_ARGS=()
fi

echo "=============================================="
echo "SOAR 2026 - SGLang Server"
echo "=============================================="
echo "Model:  $MODEL_PATH"
echo "Port:   $PORT"
echo "Host:   $HOST"
echo "Args:   ${DEFAULT_ARGS[*]} ${EXTRA_ARGS[*]} $*"
echo "=============================================="

python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    "${DEFAULT_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" \
    "$@"
