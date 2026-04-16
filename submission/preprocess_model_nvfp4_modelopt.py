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
        # Load all items grouped by task type to ensure balanced distribution
        # The eval set has 5 task types (mcq, niah, qa, fwe, cwe) x 30 each.
        # Matching this distribution in calibration improves quantization quality.
        # Also skip low-semantic tasks (niah/fwe/cwe) which contain random words/UUIDs
        # that can negatively affect calibration per competition champion notes.
        from collections import defaultdict
        by_task: dict = defaultdict(list)
        no_task: list = []
        with open(calib_data) as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                # Use full context+question — matches what eval_model.py sends.
                # The SOAR eval items have a long document/context field (~50k tokens)
                # plus a short question. Calibrating on just the question (100-500 tok)
                # underestimates activation magnitudes, causing FP8-E4M3 scale overflow
                # during eval and breaking per-block FP4 quantization accuracy.
                parts = []
                for field in ("context", "document", "passage", "content", "input", "text"):
                    v = item.get(field, "")
                    if v and isinstance(v, str) and len(v) > 100:
                        parts.append(v)
                        break  # take the first substantial context field
                q = item.get("question", "")
                if q:
                    parts.append(q)
                full_text = "\n\n".join(parts) if parts else q
                if not full_text:
                    continue
                task = item.get("task", "")
                if task:
                    by_task[task].append(full_text)
                else:
                    no_task.append(full_text)

        if by_task:
            # Prioritize high-semantic tasks (mcq, qa) over low-semantic (niah, fwe, cwe)
            HIGH_SEMANTIC = ["mcq", "qa"]
            LOW_SEMANTIC = ["niah", "fwe", "cwe"]
            # Allocate 60% to high-semantic, 40% to low-semantic
            n_high = int(num_calib * 0.6)
            n_low = num_calib - n_high
            prompts = []
            # Round-robin from high-semantic tasks
            high_pools = [by_task[t] for t in HIGH_SEMANTIC if t in by_task]
            low_pools = [by_task[t] for t in LOW_SEMANTIC if t in by_task]
            # Fill remaining from unlabeled
            for pool, n in [(high_pools, n_high), (low_pools, n_low)]:
                i = 0
                added = 0
                while added < n and any(pool):
                    src = pool[i % len(pool)] if pool else []
                    if src:
                        prompts.append(src[added // len(pool) if len(pool) > 0 else 0])
                    i += 1
                    added += 1
                    if added >= sum(len(p) for p in pool):
                        break
            # Fallback: just take sequentially if round-robin logic under-fills
            if len(prompts) < num_calib:
                all_prompts = [q for qs in by_task.values() for q in qs] + no_task
                prompts = all_prompts[:num_calib]
        else:
            prompts = no_task[:num_calib]

        print(f"[calib] Loaded {len(prompts)} prompts from {calib_data} (task-balanced)")
        return prompts
    else:
        prompts = _DEFAULT_CALIB_PROMPTS[:num_calib]
        print(f"[calib] Using {len(prompts)} built-in prompts")
        return prompts


def format_calib_chat(tokenizer, prompt: str) -> str:
    """Format a prompt as a chat message with thinking enabled, matching eval_model.py's template."""
    messages = [{"role": "user", "content": prompt}]
    try:
        # apply_chat_template with add_generation_prompt=True + <think> prefix
        # mirrors what eval_model.py does with enable_thinking=True
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        # Append <think> to trigger reasoning mode, matching SGLang's serving_chat.py behavior
        if not text.endswith("<think>"):
            text = text + "<think>"
        return text
    except Exception:
        return prompt


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

    # Load tokenizer
    print("\n[nvfp4-modelopt] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(src), trust_remote_code=True)

    # Load calibration prompts as bare questions (no chat template)
    # Chat-formatted prompts with <think> caused runaway thinking generation at inference
    prompts = load_calib_prompts(args.calib_data, args.num_calib)
    print(f"[calib] Loaded {len(prompts)} calibration prompts")
    print(f"[calib] Sample (first 200 chars): {repr(prompts[0][:200])}")

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

    # Use 16384 tokens to capture long-context activation magnitudes.
    # The per-block FP4 scale is stored as FP8-E4M3 (range ~2^-9 to 448).
    # It's computed as: SFValue = input_scale_inv * vecMax / 6.
    # If input_scale_inv is calibrated on short prompts (small vecMax),
    # long-context eval activations (large vecMax) cause SFValue to overflow
    # FP8-E4M3, clamping the scale and breaking quantization accuracy.
    # Using 16384-token sequences ensures input_scale reflects long-context magnitudes.
    def calib_dataloader():
        for prompt in prompts:
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=16384,
            )
            yield {k: v.to(args.device) for k, v in inputs.items()}

    # Get ModelOpt config
    cfg_name = CONFIG_MAP[args.config]
    quant_cfg = getattr(mtq, cfg_name)
    print(f"\n[nvfp4-modelopt] Quantizing with {cfg_name}...")

    import copy
    quant_cfg = copy.deepcopy(quant_cfg)

    # Mixed-precision: keep boundary layers in BF16 for accuracy.
    # Mixed-precision: keep boundary layers (both MLP+attn) in BF16 for accuracy.
    # History: 16→73.42%, 32→76.49% (best), 40→invalid(sh mismatch), 40+longcalib→73.27%,
    #          all-attn-BF16+all-MLP-FP4→75.47% (decode 2x slower, not worth it).
    # Best config: 32 boundary layers (0-15, 36-51), keeping middle 20 layers (16-35) as FP4.
    BOUNDARY_MLP_LAYERS = list(range(16)) + list(range(36, 52))

    inner = quant_cfg["quant_cfg"]
    print(f"[nvfp4-modelopt] quant_cfg['quant_cfg'] type: {type(inner).__name__}, len={len(inner)}")
    # All attention-related projections for boundary layers, including MiniCPM-SALA's
    # novel InfLLMv2 projections (o_gate, z_proj) that don't exist in standard models.
    ATTN_PROJS = ("q_proj", "k_proj", "v_proj", "o_proj", "o_gate", "z_proj")
    MLP_PROJS  = ("gate_proj", "up_proj", "down_proj")

    if isinstance(inner, list):
        for i in BOUNDARY_MLP_LAYERS:
            for proj in MLP_PROJS + ATTN_PROJS:
                inner.append({"quantizer_name": f"*layers.{i}.*{proj}*", "enable": False})
        print(f"[nvfp4-modelopt] Mixed-precision (list format): boundary {BOUNDARY_MLP_LAYERS} MLP+attn BF16")
    elif isinstance(inner, dict):
        for i in BOUNDARY_MLP_LAYERS:
            for proj in MLP_PROJS + ATTN_PROJS:
                inner[f"*layers.{i}.*{proj}*"] = {"enable": False}
        print(f"[nvfp4-modelopt] Mixed-precision (dict format): boundary {BOUNDARY_MLP_LAYERS} MLP+attn BF16")
    else:
        print(f"[nvfp4-modelopt] WARNING: unknown quant_cfg structure, skipping mixed-precision")


    # Quantize
    mtq.quantize(model, quant_cfg, forward_loop=calib_dataloader)
    print("[nvfp4-modelopt] Quantization complete.")

    # Export to HF checkpoint format for SGLang --quantization modelopt_fp4.
    dst.mkdir(parents=True, exist_ok=True)
    print(f"[nvfp4-modelopt] Exporting HF checkpoint to {dst}...")
    export_hf_checkpoint(model, export_dir=str(dst))
    print(f"[nvfp4-modelopt] Export complete.")

    index_path = dst / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as _f:
            _idx = json.load(_f)
        _gate_keys = sorted(k for k in _idx.get("weight_map", {}) if "gate" in k and "layers.0" in k)
        print(f"[nvfp4-modelopt] DEBUG layer[0] gate keys in index: {_gate_keys}")
    # Do NOT inject static input_scales.
    # config.json has input_activations.dynamic=True so SGLang computes per-token scales
    # at runtime. Static scales calibrated on short prompts (max 2048 tokens) would severely
    # hurt accuracy on the long reasoning chain inputs used in SOAR eval (often 50k+ tokens),
    # because the activation magnitudes differ dramatically between short and long contexts.
    print(f"[nvfp4-modelopt] Skipping static input_scale injection (dynamic=True in config.json)")

    print(f"\n[nvfp4-modelopt] Done!")
    print(f"  SGLang: python -m sglang.launch_server --model-path {dst} --quantization modelopt_fp4")


if __name__ == "__main__":
    main()
