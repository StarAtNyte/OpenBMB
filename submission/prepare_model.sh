#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# NVFP4 quantization using ModelOpt export_hf_checkpoint
# Input: $MODEL_PATH (set by platform, default /models/MiniCPM-SALA)
# Output: /workspace/nvfp4_model (served with --quantization modelopt_fp4)

# Accept --input and --output CLI args (used by simulate), fall back to env/defaults
MODEL_INPUT="${MODEL_PATH:-/models/MiniCPM-SALA}"
MODEL_OUTPUT="/workspace/nvfp4_model"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --input)  MODEL_INPUT="$2"; shift 2 ;;
        --output) MODEL_OUTPUT="$2"; shift 2 ;;
        *) shift ;;
    esac
done
CALIB_DATA="/opt/SOAR/eval_dataset/perf_public_set.jsonl"

echo "[prepare_model] Quantizing $MODEL_INPUT -> $MODEL_OUTPUT"

# Use 32 full-context calibration samples at 16384 tokens.
# Full context (document+question) is required to calibrate input_scale correctly —
# the per-block FP4 scale is stored as FP8-E4M3 and overflows if input_scale_inv
# is too large (calibrated on short 512-token questions vs 50k-token eval inputs).
python3 "${SCRIPT_DIR}/preprocess_model_nvfp4_modelopt.py" \
    --input "$MODEL_INPUT" \
    --output "$MODEL_OUTPUT" \
    --config conservative \
    --calib-data "$CALIB_DATA" \
    --num-calib 32

# Ensure hf_quant_config.json has scale_fmt=ue8m0 and correct exclude_modules
python3 - "$MODEL_OUTPUT" <<'EOF'
import json, re, sys
path = sys.argv[1] + "/hf_quant_config.json"

# Boundary MLP layers kept BF16 (must match preprocess_model_nvfp4_modelopt.py)
BOUNDARY_MLP_LAYERS = list(range(16)) + list(range(36, 52))

try:
    with open(path) as f:
        cfg = json.load(f)
    section = cfg.get("quantization", cfg)
    if isinstance(section, dict):
        section["scale_fmt"] = "ue8m0"
        # Preserve any exclusions ModelOpt set, converting q/k/v_proj to SGLang's fused qkv_proj
        excl_raw = section.get("exclude_modules", [])
        layer_prefixes = set()
        for e in excl_raw:
            m = re.match(r'(model\.layers\.\d+\.self_attn)\.(q_proj|k_proj|v_proj|o_proj|qkv_proj)', e)
            if m:
                layer_prefixes.add(m.group(1))
        excl_new = [e for e in excl_raw if not re.match(
            r'model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj)', e)]
        for prefix in sorted(layer_prefixes):
            excl_new.append(f"{prefix}.qkv_proj")
            excl_new.append(f"{prefix}.o_proj")
        for i in BOUNDARY_MLP_LAYERS:
            for proj in ("gate_proj", "up_proj", "down_proj"):
                excl_new.append(f"model.layers.{i}.mlp.{proj}")
            # All attention projections for boundary layers
            for proj in ("qkv_proj", "o_proj", "o_gate", "z_proj"):
                excl_new.append(f"model.layers.{i}.self_attn.{proj}")
        # ModelOpt's conservative config keeps qkv_proj and o_proj BF16 for ALL 52 layers,
        # but o_gate and z_proj are only BF16 on boundary layers (already covered above).
        # Layers 16-35: o_gate and z_proj ARE FP4 — do not exclude them.
        for i in range(52):
            for proj in ("qkv_proj", "o_proj"):
                excl_new.append(f"model.layers.{i}.self_attn.{proj}")
        excl_new = list(dict.fromkeys(excl_new))
        section["exclude_modules"] = excl_new
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[prepare_model] patched hf_quant_config.json: {len(excl_new)} exclude_modules (boundary MLP, all-layer attn BF16)")
except FileNotFoundError:
    print(f"[prepare_model] WARNING: hf_quant_config.json not found at {path}")
EOF

# Also patch config.json quantization_config if present
python3 - "$MODEL_OUTPUT" <<'EOF'
import json, os, re, sys
path = sys.argv[1] + "/config.json"

# Boundary MLP layers kept BF16 (must match preprocess_model_nvfp4_modelopt.py)
BOUNDARY_MLP_LAYERS = list(range(16)) + list(range(36, 52))

if os.path.exists(path):
    with open(path) as f:
        cfg = json.load(f)
    qc = cfg.get("quantization_config")
    if isinstance(qc, dict):
        qc["scale_fmt"] = "ue8m0"
        ignore_new = ["lm_head"]
        for i in BOUNDARY_MLP_LAYERS:
            for proj in ("gate_proj", "up_proj", "down_proj"):
                ignore_new.append(f"model.layers.{i}.mlp.{proj}")
            for proj in ("qkv_proj", "o_proj", "o_gate", "z_proj"):
                ignore_new.append(f"model.layers.{i}.self_attn.{proj}")
        # ModelOpt's conservative config keeps qkv_proj and o_proj BF16 for ALL 52 layers,
        # but o_gate and z_proj are only BF16 on boundary layers (0-15, 36-51).
        # Layers 16-35: o_gate and z_proj ARE FP4 in the checkpoint — do not add them to ignore.
        for i in range(52):
            for proj in ("qkv_proj", "o_proj"):
                ignore_new.append(f"model.layers.{i}.self_attn.{proj}")
        ignore_new = list(dict.fromkeys(ignore_new))
        qc["ignore"] = ignore_new
        for group in qc.get("config_groups", {}).values():
            ia = group.get("input_activations")
            if isinstance(ia, dict):
                ia["dynamic"] = True
                print(f"[prepare_model] Set input_activations.dynamic=True in config.json")
        with open(path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"[prepare_model] patched config.json: {len(ignore_new)} ignore entries (boundary MLP, all-layer attn BF16, scale_fmt=ue8m0)")
EOF

echo "[prepare_model] done: $MODEL_OUTPUT"
