#!/usr/bin/env bash

echo "[prepare_env] start $(date '+%F %T')"

# Resolve submission directory regardless of CWD (script is sourced by platform)
SUBMISSION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[prepare_env] submission dir: $SUBMISSION_DIR"

uv pip install --no-deps -e "$SUBMISSION_DIR/sglang/python"

export SGLANG_SERVER_ARGS="${SGLANG_SERVER_ARGS:-} --log-level info --attention-backend minicpm_flashinfer --chunked-prefill-size 8192 --mem-fraction-static 0.70 --dense-as-sparse --skip-server-warmup --quantization fp8"

echo "[prepare_env] done"
