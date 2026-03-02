#!/usr/bin/env python3
"""
SOAR 2026 - RTN W4A16 Quantization for MiniCPM-SALA

RTN (Round-To-Nearest) quantization that works directly on tensors.
No calibration forward passes needed - avoids flash_attention_2 compatibility issues.
Outputs GPTQ-compatible format for SGLang.

Based on official SOAR demo-quant reference implementation.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import torch
from pathlib import Path
from safetensors.torch import load_file, save_file


def quantize_weight_rtn(weight: torch.Tensor, bits: int = 4, group_size: int = 128):
    """
    RTN quantize a single linear layer weight to GPTQ-compatible packed format.

    weight shape : [out_features, in_features]
    Returns      : qweight, scales, qzeros, g_idx
    """
    out_features, in_features = weight.shape
    pack_factor = 32 // bits
    maxq = 2 ** bits - 1

    if group_size <= 0:
        group_size = in_features
    num_groups = (in_features + group_size - 1) // group_size

    # Pad if needed
    if in_features % group_size != 0:
        pad = group_size * num_groups - in_features
        weight = torch.nn.functional.pad(weight, (0, pad))
        in_features = weight.shape[1]

    w = weight.float()
    w_grouped = w.reshape(out_features, num_groups, group_size)
    w_min = w_grouped.min(dim=2, keepdim=True).values
    w_max = w_grouped.max(dim=2, keepdim=True).values

    scales = (w_max - w_min).clamp(min=1e-10) / maxq
    zeros = -w_min / scales

    qw = torch.clamp(torch.round(w_grouped / scales + zeros), 0, maxq).to(torch.int32)
    qw = qw.reshape(out_features, in_features)

    # qweight: [in_features // pack_factor, out_features]
    qw_t = qw.t().contiguous()
    qweight = torch.zeros(in_features // pack_factor, out_features, dtype=torch.int32)
    for j in range(pack_factor):
        qweight |= qw_t[j::pack_factor, :] << (bits * j)

    # scales: [num_groups, out_features]
    scales_out = scales.squeeze(2).t().contiguous().half()

    # qzeros: [num_groups, out_features // pack_factor]  (packed along output dim)
    zeros_int = torch.clamp(torch.round(zeros.squeeze(2)), 0, maxq).to(torch.int32)
    zeros_t = zeros_int.t().contiguous()
    qzeros = torch.zeros(num_groups, out_features // pack_factor, dtype=torch.int32)
    for j in range(pack_factor):
        qzeros |= zeros_t[:, j::pack_factor] << (bits * j)

    g_idx = torch.tensor([i // group_size for i in range(in_features)], dtype=torch.int32)

    return qweight, scales_out, qzeros, g_idx


# Linear layers to quantize - covers both attention types and MLP
LINEAR_SUFFIXES = (
    # Attention layers (works for both MiniCPMInfLLMv2Attention and LightningAttention)
    ".q_proj.weight", ".k_proj.weight", ".v_proj.weight",
    ".o_proj.weight", ".o_gate.weight", ".z_proj.weight",
    # MLP layers
    ".gate_proj.weight", ".up_proj.weight", ".down_proj.weight",
)


def copy_model(input_path: str, output_path: str):
    """Simple copy without modification (baseline)."""
    print(f"Copying model from {input_path} to {output_path}")

    input_dir = Path(input_path)
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    for item in input_dir.iterdir():
        dest = output_dir / item.name
        if item.is_file():
            shutil.copy2(item, dest)
        elif item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)

    print(f"Model copied to {output_path}")


def quantize_rtn(
    input_path: str,
    output_path: str,
    bits: int = 4,
    group_size: int = 128,
):
    """
    RTN W4A16 quantization for MiniCPM-SALA.

    Works directly on tensors without model forward passes.
    Saves in GPTQ-compatible format for SGLang.
    """
    start_time = time.time()

    print("=" * 60)
    print("  RTN W4A16 Quantization for MiniCPM-SALA")
    print("=" * 60)
    print(f"Input:      {input_path}")
    print(f"Output:     {output_path}")
    print(f"Bits:       {bits}")
    print(f"Group size: {group_size}")
    print("=" * 60)

    src = Path(input_path)
    dst = Path(output_path)
    dst.mkdir(parents=True, exist_ok=True)

    # Copy non-weight files
    print("\n[1/4] Copying config files...")
    for f in src.iterdir():
        if f.suffix in (".json", ".py", ".model", ".txt") or f.name in ("tokenizer.json", "tokenizer_config.json"):
            shutil.copy2(f, dst / f.name)
            print(f"  Copied: {f.name}")

    # Also copy any tokenizer files
    for pattern in ["tokenizer*", "special_tokens*", "added_tokens*"]:
        for f in src.glob(pattern):
            if f.is_file() and not (dst / f.name).exists():
                shutil.copy2(f, dst / f.name)
                print(f"  Copied: {f.name}")

    # Patch config.json with quantization info
    print("\n[2/4] Patching config.json...")
    config_path = src / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

        config["quantization_config"] = {
            "bits": bits,
            "group_size": group_size,
            "quant_method": "gptq",
            "desc_act": False,
            "sym": False,
        }
        config["torch_dtype"] = "float16"

        with open(dst / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        print("  Added quantization_config to config.json")

    # Write quantize_config.json (SGLang reads this separately)
    print("\n[3/4] Writing quantize_config.json...")
    quant_config = {
        "bits": bits,
        "group_size": group_size,
        "desc_act": False,
        "sym": False,
        "lm_head": False,
        "dynamic": {},
        "quant_method": "gptq",
    }
    with open(dst / "quantize_config.json", "w") as f:
        json.dump(quant_config, f, indent=2)
    print(f"  Written: quantize_config.json")

    # Quantize weight shards
    print("\n[4/4] Quantizing weight shards...")
    index_path = src / "model.safetensors.index.json"

    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
    else:
        # Single file model
        shard_files = list(src.glob("*.safetensors"))
        if not shard_files:
            shard_files = list(src.glob("*.bin"))
        shard_files = [f.name for f in shard_files]

    new_weight_map = {}
    total_quantized = 0
    total_skipped = 0

    for shard_name in shard_files:
        shard_path = src / shard_name
        if not shard_path.exists():
            print(f"  Warning: {shard_name} not found, skipping")
            continue

        print(f"\n  Processing {shard_name}...")

        try:
            shard = load_file(str(shard_path))
        except Exception as e:
            print(f"    Error loading {shard_name}: {e}")
            # Try copying as-is
            shutil.copy2(shard_path, dst / shard_name)
            continue

        new_tensors = {}

        for name, tensor in shard.items():
            # Check if this is a linear layer to quantize
            if name.endswith(LINEAR_SUFFIXES):
                try:
                    w = tensor.to(torch.float16)
                    qweight, scales, qzeros, g_idx = quantize_weight_rtn(w, bits, group_size)

                    base = name.removesuffix(".weight")
                    for suffix, t in [(".qweight", qweight), (".scales", scales),
                                       (".qzeros", qzeros), (".g_idx", g_idx)]:
                        new_tensors[base + suffix] = t
                        new_weight_map[base + suffix] = shard_name

                    total_quantized += 1
                    print(f"    Quantized: {name} {list(w.shape)} -> qweight {list(qweight.shape)}")
                except Exception as e:
                    # If quantization fails for this layer, keep original
                    print(f"    Warning: Failed to quantize {name}: {e}")
                    new_tensors[name] = tensor.to(torch.float16)
                    new_weight_map[name] = shard_name
                    total_skipped += 1
            else:
                # Keep non-quantized layers in float16
                new_tensors[name] = tensor.to(torch.float16)
                new_weight_map[name] = shard_name

        save_file(new_tensors, str(dst / shard_name))

    # Write updated index
    if index_path.exists():
        with open(dst / "model.safetensors.index.json", "w") as f:
            json.dump({"metadata": {"total_size": 0}, "weight_map": new_weight_map}, f, indent=2)

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"  Quantization Complete!")
    print("=" * 60)
    print(f"  Quantized layers: {total_quantized}")
    print(f"  Skipped layers:   {total_skipped}")
    print(f"  Time:             {elapsed:.1f}s")
    print(f"  Output:           {output_path}")
    print("=" * 60)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="SOAR 2026 Model Preprocessing - RTN W4A16 Quantization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # RTN quantization (recommended)
  python preprocess_model.py --input /models/MiniCPM-SALA --output /output/model --mode rtn

  # With custom group size (smaller = better quality, larger = faster)
  python preprocess_model.py --input /models/MiniCPM-SALA --output /output/model --mode rtn --group-size 128

  # Simple copy (baseline)
  python preprocess_model.py --input /models/MiniCPM-SALA --output /output/model --mode copy
        """
    )
    parser.add_argument("--input", required=True, help="Input model path")
    parser.add_argument("--output", required=True, help="Output model path")
    parser.add_argument("--mode", default="rtn", choices=["copy", "rtn", "gptq"],
                        help="Preprocessing mode (default: rtn)")
    parser.add_argument("--bits", type=int, default=4,
                        help="Quantization bits (default: 4)")
    parser.add_argument("--group-size", type=int, default=128,
                        help="Quantization group size (default: 128, official demo uses 1024)")

    args = parser.parse_args()

    print(f"Mode: {args.mode}")

    if args.mode == "copy":
        copy_model(args.input, args.output)
    elif args.mode in ("rtn", "gptq"):
        # Both modes use RTN now - GPTQ library doesn't work with SALA
        quantize_rtn(
            args.input,
            args.output,
            bits=args.bits,
            group_size=args.group_size,
        )


if __name__ == "__main__":
    main()
