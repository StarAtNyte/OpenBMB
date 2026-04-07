"""
Model preprocessing for MiniCPM-SALA — SOAR 2026.

Copies BF16 model to output dir and patches config.json with FP8 metadata.
SGLang loads BF16 weights and quantizes to FP8 W8A8 at load time using
per-channel scales internally (is_checkpoint_fp8_serialized=False).
This gives ~83% accuracy (same as BF16) with ~2x compute throughput.

SGLang flag: --quantization fp8
Effect:      2x compute throughput vs fp16 on Blackwell tensor cores
             Accuracy: same as BF16 baseline (~83%)

Usage:
    python preprocess_model.py --input /path/to/model --output /path/to/output
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniCPM-SALA FP8 model preprocessing")
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

    # Copy all files except config.json (patched below)
    for f in sorted(src.iterdir()):
        if f.name.startswith(".") or f.name == "config.json":
            continue
        target = dst / f.name
        if not target.exists():
            print(f"[preprocess] copying {f.name} ...")
            shutil.copy2(f, target)

    # Patch config.json: set fp8 quant config with is_checkpoint_fp8_serialized=False
    # SGLang will quantize weights from BF16 to FP8 at load time (per-channel, full accuracy)
    with open(src / "config.json") as f:
        config = json.load(f)

    config.pop("quantization_config", None)
    config["quantization_config"] = {
        "quant_method": "fp8",
        "is_checkpoint_fp8_serialized": False,
        "activation_scheme": "dynamic",
    }

    with open(dst / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n[preprocess] done!")
    print(f"  BF16 weights copied; SGLang will quantize to FP8 at load time")
    print(f"  SGLang flag: --quantization fp8")


if __name__ == "__main__":
    main()
