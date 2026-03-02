#!/usr/bin/env bash
# SOAR 2026 - Environment Setup Script
# Based on official demo-quant reference

echo "[prepare_env] start $(date '+%F %T')"

# ============================================================================
# Patch bfloat16 -> float16 for GPTQ compatibility
# ============================================================================

BACKEND_DIR=/opt/SGLang-MiniCPM-SALA/packages/sglang-minicpm/python/sglang/srt/layers/attention

if [ -d "$BACKEND_DIR" ]; then
    for pyfile in "$BACKEND_DIR"/minicpm_backend.py "$BACKEND_DIR"/minicpm_sparse_utils.py; do
        if [ -f "$pyfile" ]; then
            sed -i 's/torch\.bfloat16/torch.float16/g' "$pyfile" 2>/dev/null || true
            sed -i 's/"bfloat16"/"float16"/g' "$pyfile" 2>/dev/null || true
            echo "[prepare_env] patched $(basename "$pyfile")"
        fi
    done
    # Clear pycache
    find "$BACKEND_DIR/__pycache__" -name "minicpm_backend*" -o -name "minicpm_sparse_utils*" 2>/dev/null | xargs rm -f 2>/dev/null || true
else
    echo "[prepare_env] BACKEND_DIR not found, skipping patches"
fi

# ============================================================================
# SGLang Server Arguments for GPTQ
# ============================================================================

export SGLANG_SERVER_ARGS="--disable-radix-cache --attention-backend minicpm_flashinfer --chunked-prefill-size 8192 --skip-server-warmup --dense-as-sparse --quantization gptq --dtype float16 --disable-cuda-graph"

echo "[prepare_env] SGLANG_SERVER_ARGS=${SGLANG_SERVER_ARGS}"
echo "[prepare_env] done"
