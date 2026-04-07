#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# NVFP4 quantization using ModelOpt export_hf_checkpoint
# Input: $MODEL_PATH (set by platform, default /models/MiniCPM-SALA)
# Output: /workspace/nvfp4_model (served with --quantization modelopt_fp4)

MODEL_INPUT="${MODEL_PATH:-/models/MiniCPM-SALA}"
MODEL_OUTPUT="/workspace/nvfp4_model"
CALIB_DATA="/opt/SOAR/eval_dataset/perf_public_set.jsonl"

echo "[prepare_model] Quantizing $MODEL_INPUT -> $MODEL_OUTPUT"

python3 "${SCRIPT_DIR}/preprocess_model_nvfp4_modelopt.py" \
    --input "$MODEL_INPUT" \
    --output "$MODEL_OUTPUT" \
    --config conservative \
    --calib-data "$CALIB_DATA" \
    --num-calib 32

# Ensure hf_quant_config.json has scale_fmt=ue8m0 and correct exclude_modules
python3 - <<'EOF'
import json, re
path = "/workspace/nvfp4_model/hf_quant_config.json"
try:
    with open(path) as f:
        cfg = json.load(f)
    section = cfg.get("quantization", cfg)
    if isinstance(section, dict):
        section["scale_fmt"] = "ue8m0"
        # ModelOpt exports per-layer q_proj/k_proj/v_proj exclusions.
        # SGLang uses fused names (qkv_proj), so we must convert.
        # Strategy: collect all layer prefixes that had q/k/v excluded,
        # then add qkv_proj and o_proj variants for those same layers.
        excl_raw = section.get("exclude_modules", [])
        # Find all layer prefixes (e.g. "model.layers.0.self_attn")
        layer_prefixes = set()
        for e in excl_raw:
            m = re.match(r'(model\.layers\.\d+\.self_attn)\.(q_proj|k_proj|v_proj|o_proj|qkv_proj)', e)
            if m:
                layer_prefixes.add(m.group(1))
        # Build new exclude list with fused SGLang names
        excl_new = []
        for prefix in sorted(layer_prefixes):
            excl_new.append(f"{prefix}.qkv_proj")
            excl_new.append(f"{prefix}.o_proj")
        section["exclude_modules"] = excl_new
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[prepare_model] patched hf_quant_config.json: scale_fmt=ue8m0, {len(excl_new)} exclude_modules (fused SGLang names)")
except FileNotFoundError:
    print(f"[prepare_model] WARNING: hf_quant_config.json not found at {path}")
EOF

# Also patch config.json quantization_config if present
python3 - <<'EOF'
import json, os, re
path = "/workspace/nvfp4_model/config.json"
if os.path.exists(path):
    with open(path) as f:
        cfg = json.load(f)
    qc = cfg.get("quantization_config")
    if isinstance(qc, dict):
        qc["scale_fmt"] = "ue8m0"
        # Convert ModelOpt's per-layer q/k/v_proj ignore entries to SGLang's fused qkv_proj names
        ignore_raw = qc.get("ignore", [])
        layer_prefixes = set()
        for e in ignore_raw:
            m = re.match(r'(model\.layers\.\d+\.self_attn)\.(q_proj|k_proj|v_proj|o_proj|qkv_proj)', e)
            if m:
                layer_prefixes.add(m.group(1))
        ignore_new = ["lm_head"]
        for prefix in sorted(layer_prefixes):
            ignore_new.append(f"{prefix}.qkv_proj")
            ignore_new.append(f"{prefix}.o_proj")
        qc["ignore"] = ignore_new
        with open(path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"[prepare_model] patched config.json: {len(ignore_new)} ignore entries (fused SGLang names)")
EOF

echo "[prepare_model] done: $MODEL_OUTPUT"
