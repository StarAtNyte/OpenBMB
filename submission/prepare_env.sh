#!/usr/bin/env bash

echo "[prepare_env] start $(date '+%F %T')"

# Resolve submission directory regardless of CWD (script is sourced by platform)
SUBMISSION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[prepare_env] submission dir: $SUBMISSION_DIR"

# Find the SALA sglang package directory
# Try the known SALA packages path first, then fall back to python import
SALA_SGLANG="/opt/SGLang-MiniCPM-SALA/packages/sglang-minicpm/python/sglang"
if [ ! -d "$SALA_SGLANG" ]; then
    SALA_SGLANG="$(python3 -c 'import sglang, os; print(os.path.dirname(sglang.__file__))' 2>/dev/null || echo '')"
fi
echo "[prepare_env] patching SALA SGLang at: $SALA_SGLANG"

if [ -z "$SALA_SGLANG" ] || [ ! -d "$SALA_SGLANG" ]; then
    echo "[prepare_env] ERROR: could not find sglang package directory"
    exit 1
fi

PATCH_FILES=(
    "srt/configs/model_config.py"
    "srt/layers/parameter.py"
    "srt/layers/linear.py"
    "srt/layers/vocab_parallel_embedding.py"
    "srt/layers/quantization/modelopt_quant.py"
    "srt/models/minicpm.py"
)

for rel in "${PATCH_FILES[@]}"; do
    src="$SUBMISSION_DIR/sglang/python/sglang/$rel"
    dst="$SALA_SGLANG/$rel"
    if [ -f "$src" ]; then
        mkdir -p "$(dirname "$dst")"
        cp "$src" "$dst"
        stem="$(basename "$dst" .py)"
        pyc_dir="$(dirname "$dst")/__pycache__"
        rm -f "$pyc_dir/${stem}".cpython-*.pyc 2>/dev/null || true
        python3 -c "import py_compile; py_compile.compile('$dst', doraise=False)"
        echo "[prepare_env]   patched + recompiled: $rel"
    else
        echo "[prepare_env]   WARNING: $src not found, skipping"
    fi
done

export SGLANG_FLASHINFER_FP4_GEMM_BACKEND="${SGLANG_FLASHINFER_FP4_GEMM_BACKEND:-cutlass}"
# --tokenizer-path needed because exported NVFP4 dir has custom MiniCPMSALAConfig that AutoTokenizer can't resolve
export SGLANG_SERVER_ARGS="${SGLANG_SERVER_ARGS:-} --log-level info --attention-backend minicpm_flashinfer --chunked-prefill-size 8192 --mem-fraction-static 0.70 --dense-as-sparse --skip-server-warmup --quantization modelopt_fp4 --disable-radix-cache --tokenizer-path /models/MiniCPM-SALA --model-path /workspace/nvfp4_model"

echo "[prepare_env] done"
