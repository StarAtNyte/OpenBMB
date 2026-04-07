"""
NVFP4 model preprocessing using NVIDIA ModelOpt — SOAR 2026.

Uses nvidia-modelopt (already installed on SOAR container) to quantize
MiniCPM-SALA weights to NVFP4 E2M1 format with proper dual-scale mechanism.
ModelOpt produces the exact checkpoint format expected by SGLang's
ModelOptFp4LinearMethod (--quantization modelopt_fp4).

Configs available (choose based on accuracy/speed tradeoff):
  NVFP4_DEFAULT_CFG          — standard NVFP4, all linear layers
  MAMBA_MOE_NVFP4_CONSERVATIVE_CFG — Mamba-aware, conservative (higher accuracy)
  MAMBA_MOE_NVFP4_AGGRESSIVE_CFG  — Mamba-aware, aggressive (higher speed)

Usage:
    python preprocess_model_nvfp4_modelopt.py --input /path/to/model --output /path/to/output
    python preprocess_model_nvfp4_modelopt.py --input /path/to/model --output /path/to/output --config conservative
    python preprocess_model_nvfp4_modelopt.py --input /path/to/model --output /path/to/output --calib-data /path/to/data.jsonl
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


# Verify required deps are available (expected to run under SALA venv which has both)
def _ensure_deps():
    try:
        import flash_attn  # noqa
    except ImportError:
        print("[setup] Installing flash-attn...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "flash-attn", "--no-build-isolation", "-q"],
            check=True,
        )
    try:
        import modelopt  # noqa
        import modelopt.torch.quantization as _mtq  # noqa
        print(f"[setup] modelopt available: {modelopt.__version__}")
    except ImportError:
        print("[setup] Installing nvidia-modelopt...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "nvidia-modelopt", "-q"],
            check=True,
        )


_DEFAULT_CALIB_PROMPTS = [
    "What is the capital of France?",
    "Explain the Pythagorean theorem.",
    "What causes rainbows?",
    "Write a short poem about the ocean.",
    "What is quantum entanglement?",
    "Solve: 2x + 5 = 13. What is x?",
    "What are the main differences between Python and C++?",
    "Describe the process of photosynthesis.",
    "What is the significance of the Turing test?",
    "How does a neural network learn?",
    "What is the speed of light in vacuum?",
    "Explain Newton's second law of motion.",
    "What is the difference between DNA and RNA?",
    "Describe the water cycle.",
    "What is the Higgs boson?",
    "How does a computer CPU work?",
    "What is the theory of relativity?",
    "Explain the concept of entropy in thermodynamics.",
    "What is a black hole?",
    "Describe the structure of an atom.",
    "What is the central limit theorem?",
    "How does HTTPS encryption work?",
    "What is a Fourier transform?",
    "Explain gradient descent in machine learning.",
    "What is the P vs NP problem?",
    "How does the immune system work?",
    "What is CRISPR gene editing?",
    "Explain the concept of a derivative in calculus.",
    "What is the difference between supervised and unsupervised learning?",
    "How does a transformer neural network work?",
    "What is spontaneous symmetry breaking in physics?",
    "Explain the concept of wave-particle duality.",
]

CONFIG_MAP = {
    "default":      "NVFP4_DEFAULT_CFG",
    "conservative": "MAMBA_MOE_NVFP4_CONSERVATIVE_CFG",
    "aggressive":   "MAMBA_MOE_NVFP4_AGGRESSIVE_CFG",
    "mlp-only":     "NVFP4_MLP_ONLY_CFG",
}


def load_calib_prompts(calib_data: str | None, num_calib: int) -> list[str]:
    if calib_data and Path(calib_data).exists():
        prompts = []
        with open(calib_data) as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    q = item.get("question", item.get("text", ""))
                    if q:
                        prompts.append(q)
                    if len(prompts) >= num_calib:
                        break
        print(f"[calib] Loaded {len(prompts)} prompts from {calib_data}")
        return prompts
    else:
        prompts = _DEFAULT_CALIB_PROMPTS[:num_calib]
        print(f"[calib] Using {len(prompts)} built-in prompts")
        return prompts


def main():
    parser = argparse.ArgumentParser(description="MiniCPM-SALA NVFP4 quantization via ModelOpt")
    parser.add_argument("--input",     required=True, help="Original model directory")
    parser.add_argument("--output",    required=True, help="Output directory")
    parser.add_argument("--config",    default="conservative",
                        choices=list(CONFIG_MAP.keys()),
                        help="ModelOpt quantization config (default: conservative)")
    parser.add_argument("--calib-data", default=None,
                        help="JSONL file with calibration prompts (field: 'question')")
    parser.add_argument("--num-calib", type=int, default=32,
                        help="Number of calibration prompts (default: 32)")
    parser.add_argument("--device",    default="cuda")
    args = parser.parse_args()

    src = Path(args.input).resolve()
    dst = Path(args.output).resolve()

    if not src.is_dir():
        raise FileNotFoundError(f"Input model dir not found: {src}")
    dst.mkdir(parents=True, exist_ok=True)

    print(f"[nvfp4-modelopt] input:  {src}")
    print(f"[nvfp4-modelopt] output: {dst}")
    print(f"[nvfp4-modelopt] config: {args.config} ({CONFIG_MAP[args.config]})")

    _ensure_deps()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import modelopt.torch.quantization as mtq
    from modelopt.torch.export import export_hf_checkpoint

    # Load calibration prompts
    prompts = load_calib_prompts(args.calib_data, args.num_calib)

    # Load tokenizer
    print("\n[nvfp4-modelopt] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(src), trust_remote_code=True)

    # Load model in BF16
    print("[nvfp4-modelopt] Loading BF16 model...")
    model = AutoModelForCausalLM.from_pretrained(
        str(src),
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    ).to(args.device)
    model.eval()
    print(f"[nvfp4-modelopt] Model loaded. Params: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    # Build calibration dataloader
    def calib_dataloader():
        for prompt in prompts:
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            yield {k: v.to(args.device) for k, v in inputs.items()}

    # Get ModelOpt config
    cfg_name = CONFIG_MAP[args.config]
    quant_cfg = getattr(mtq, cfg_name)
    print(f"\n[nvfp4-modelopt] Quantizing with {cfg_name}...")

    # --- Calibrate input_scale via forward hooks on BF16 model BEFORE quantization ---
    # ModelOpt's conservative config uses dynamic (per-token) activation quantization —
    # it does NOT store static amaxes. We collect them ourselves via hooks.
    print("[nvfp4-modelopt] Collecting activation amaxes via forward hooks...")
    import safetensors.torch as st
    import torch.nn as nn

    act_amaxes: dict[str, torch.Tensor] = {}
    hooks = []

    def _make_hook(layer_name: str):
        def hook(module, inp, out):
            x = inp[0] if isinstance(inp, tuple) else inp
            if x is None:
                return
            amax_val = x.detach().float().abs().max()
            if layer_name not in act_amaxes:
                act_amaxes[layer_name] = amax_val
            else:
                act_amaxes[layer_name] = torch.max(act_amaxes[layer_name], amax_val)
        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(_make_hook(name)))

    print(f"[nvfp4-modelopt] Registered hooks on {len(hooks)} linear layers")

    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(args.device) for k, v in inputs.items()}
            model(**inputs)

    for h in hooks:
        h.remove()

    print(f"[nvfp4-modelopt] Collected {len(act_amaxes)} activation amaxes")
    if act_amaxes:
        sample_items = list(act_amaxes.items())[:3]
        for k, v in sample_items:
            print(f"  {k}: {v.item():.6f}")

    # Quantize
    mtq.quantize(model, quant_cfg, forward_loop=calib_dataloader)
    print("[nvfp4-modelopt] Quantization complete.")

    # Build input_scales dict from the hook-collected amaxes
    # (populated after export so we can filter to only quantized layers)
    input_scales_raw = {
        name: val.reshape(1).cpu()
        for name, val in act_amaxes.items()
        if val.item() > 0
    }
    print(f"[nvfp4-modelopt] {len(input_scales_raw)} raw activation amaxes collected")

    # Export to HF checkpoint format for SGLang --quantization modelopt_fp4.
    dst.mkdir(parents=True, exist_ok=True)
    print(f"[nvfp4-modelopt] Exporting HF checkpoint to {dst}...")
    export_hf_checkpoint(model, export_dir=str(dst))
    print(f"[nvfp4-modelopt] Export complete.")

    # Load the exported index to discover which layers were actually FP4-quantized.
    # Only inject input_scale for layers that have a corresponding weight in the checkpoint.
    # This avoids KeyError when SGLang tries to load input_scale for non-quantized layers
    # like lm_head that have no input_scale parameter registered.
    index_path = dst / "model.safetensors.index.json"
    # Debug: print any suspicious keys in the exported index
    if index_path.exists():
        with open(index_path) as _f:
            _idx = json.load(_f)
        _suspicious = [k for k in _idx.get("weight_map", {}) if "gate_gate" in k or "proj_proj" in k]
        if _suspicious:
            print(f"[nvfp4-modelopt] WARNING: suspicious exported keys: {_suspicious[:5]}")
        _gate_keys = sorted(k for k in _idx.get("weight_map", {}) if "gate" in k and "layers.0" in k)
        print(f"[nvfp4-modelopt] DEBUG layer[0] gate keys in index: {_gate_keys}")
    # Use weight_scale_2 as the indicator of actual FP4 quantization.
    # Only layers with weight_scale_2 in the checkpoint are truly FP4-quantized
    # and will have input_scale registered in SGLang's model definition.
    # (Excluded layers like o_proj, qkv_proj have only .weight but no weight_scale_2)
    quantized_layer_names: set[str] = set()
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        for key in index.get("weight_map", {}):
            if key.endswith(".weight_scale_2"):
                layer_name = key[: -len(".weight_scale_2")]
                quantized_layer_names.add(layer_name)
                # ModelOpt may keep the unfused name (gate_proj / up_proj) in the
                # index even after fusion. Derive the fused name so gate_up_proj
                # and qkv_proj are also accepted.
                # Only remap if the layer itself is not already a fused name.
                proj = layer_name.split(".")[-1]
                base = ".".join(layer_name.split(".")[:-1])
                if proj == "gate_proj" or proj == "up_proj":
                    quantized_layer_names.add(f"{base}.gate_up_proj")
                elif proj == "q_proj" or proj == "k_proj" or proj == "v_proj":
                    quantized_layer_names.add(f"{base}.qkv_proj")
    print(f"[nvfp4-modelopt] {len(quantized_layer_names)} FP4-quantized layers (have weight_scale_2)")

    # ModelOpt fuses projections on export:
    #   q_proj + k_proj + v_proj  →  qkv_proj  (SGLang's fused QKV)
    #   gate_proj + up_proj       →  gate_up_proj
    # We collected amaxes on the BF16 unfused layers, so we must remap them.
    # Strategy: for fused layers, take max(amax) of constituent projections.
    def _base(name: str) -> str:
        """Return layer base path: 'model.layers.0.self_attn.q_proj' → 'model.layers.0.self_attn'"""
        return ".".join(name.split(".")[:-1])

    # Build fused-name → amax mapping
    fused_amaxes: dict[str, torch.Tensor] = {}
    for name, val in input_scales_raw.items():
        proj = name.split(".")[-1]
        base = _base(name)
        if proj in ("q_proj", "k_proj", "v_proj"):
            fused = f"{base}.qkv_proj"
        elif proj in ("gate_proj", "up_proj"):
            fused = f"{base}.gate_up_proj"
        elif proj in ("gate_up_proj", "qkv_proj"):
            # BF16 model already has fused projections — use as-is
            fused = name
        else:
            fused = name  # o_proj, down_proj, z_proj, o_gate — not fused

        if fused not in fused_amaxes:
            fused_amaxes[fused] = val
        else:
            fused_amaxes[fused] = torch.max(fused_amaxes[fused], val)

    print(f"[nvfp4-modelopt] {len(fused_amaxes)} fused-layer amaxes (from {len(input_scales_raw)} raw)")

    # SGLang's load_weights uses stacked_params_mapping to shard fused projections.
    # For gate_up_proj, it iterates: gate_proj→shard_id=0, up_proj→shard_id=1.
    # Each call to _load_into_shard_id expects loaded_weight.shape[0] == 1 (scalar).
    # So we must save input_scale under the UNFUSED names (gate_proj, up_proj, q_proj, ...)
    # each with shape (1,). minicpm.py's load_weights remaps gate_proj→gate_up_proj and
    # calls weight_loader(param, scale, shard_id=0), which slices param_data[0].
    _FUSED_SPLITS = {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    }

    # Only keep layers that:
    # 1. appear in the exported index (i.e. ModelOpt actually quantized them), AND
    # 2. are inside model.layers.* (exclude lm_head, embed_tokens, etc.)
    input_scales = {}
    for name, val in fused_amaxes.items():
        if name not in quantized_layer_names or not name.startswith("model.layers."):
            continue
        proj = name.split(".")[-1]
        base = _base(name)
        split_names = _FUSED_SPLITS.get(proj)
        if split_names:
            # gate_up_proj / qkv_proj: skip input_scale injection.
            # When input_scale=0 the kernel defaults to dynamic scaling (effective 1.0),
            # which gave better accuracy than static calibrated scales from 32 prompts.
            # The static scale over-compresses activations and causes reasoning loops.
            pass
        else:
            input_scales[f"{name}.input_scale"] = val.reshape(1).clone()
    skipped = len(fused_amaxes) - len(input_scales)
    print(f"[nvfp4-modelopt] {len(input_scales)} input_scales to inject ({skipped} non-quantized/non-layer layers skipped)")

    # Sanity check: flag any suspicious double-component names before saving
    bad_keys = [k for k in input_scales if "gate_gate" in k or "up_up" in k or "proj_proj" in k]
    if bad_keys:
        print(f"[nvfp4-modelopt] ERROR: malformed input_scale keys detected: {bad_keys[:5]}")
        raise RuntimeError(f"Malformed input_scale keys: {bad_keys[:5]}")
    # Also print all keys for debugging
    print(f"[nvfp4-modelopt] input_scale keys sample: {sorted(input_scales.keys())[:6]}")

    if input_scales:
        # Save input_scales to a dedicated shard and update the index.
        scales_fname = "input_scales.safetensors"
        scales_path = str(dst / scales_fname)
        st.save_file(input_scales, scales_path)
        print(f"[nvfp4-modelopt] Saved {len(input_scales)} input_scales to {scales_path}")

        if index_path.exists():
            with open(index_path) as f:
                index = json.load(f)
            for key in input_scales:
                index["weight_map"][key] = scales_fname
            with open(index_path, "w") as f:
                json.dump(index, f, indent=2)
            print(f"[nvfp4-modelopt] Updated index with {len(input_scales)} input_scale entries")
    else:
        print(f"[nvfp4-modelopt] WARNING: no input_scale amaxes found — will default to 1.0 at inference")

    print(f"\n[nvfp4-modelopt] Done!")
    print(f"  SGLang: python -m sglang.launch_server --model-path {dst} --quantization modelopt_fp4")


if __name__ == "__main__":
    main()
