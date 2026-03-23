"""
Model preprocessing for MiniCPM-SALA — SOAR 2026.

Copies the model and patches config.json to enable FP8 online quantization.
SGLang loads FP16 weights and quantizes to FP8 at model load time.

No pre-serialized FP8 tensors needed — avoids shape mismatch issues with
merged projections (gate_proj + up_proj fused in SGLang's MergedColumnParallelLinear).

SGLang flag: --quantization fp8
Effect:      weights quantized fp8 e4m3fn per-channel at load time (~1-2 min)
             activations dynamically quantized fp8 per-token at inference time
             2x compute throughput vs fp16 on Blackwell tensor cores

Usage:
    python preprocess_model.py --input /path/to/model --output /path/to/output
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniCPM-SALA model preprocessing")
    parser.add_argument("--input",  required=True, help="Original model directory")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    src = Path(args.input).resolve()
    dst = Path(args.output).resolve()

    if not src.is_dir():
        raise FileNotFoundError(f"Input model dir not found: {src}")

    dst.mkdir(parents=True, exist_ok=True)
    print(f"[preprocess] input:  {src}")
    print(f"[preprocess] output: {dst}")

    # -------------------------------------------------------------------------
    # Copy all model files (skip existing to avoid re-copying large shards)
    # config.json is always overwritten below
    # -------------------------------------------------------------------------
    count = 0
    for f in sorted(src.iterdir()):
        if f.name.startswith(".") or f.name == "config.json":
            continue
        target = dst / f.name
        if target.exists():
            continue
        if f.is_dir():
            shutil.copytree(f, target)
        else:
            shutil.copy2(f, target)
        count += 1

    print(f"[preprocess] copied {count} files")

    # -------------------------------------------------------------------------
    # Always write config.json with fp8 online quantization
    # SGLang reads this and uses Fp8Config (online per-channel weight quantization)
    # -------------------------------------------------------------------------
    with open(src / "config.json") as f:
        config = json.load(f)

    config["quantization_config"] = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "is_checkpoint_fp8_serialized": False,
    }
    with open(dst / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("[preprocess] wrote config.json: quant_method=fp8 (online quantization at load time)")

    print("[preprocess] done — SGLang flag: --quantization fp8")


if __name__ == "__main__":
    main()
