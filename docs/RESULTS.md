# SOAR 2026 - Optimization Results

Track progress across optimization iterations.

## Scoring Formula

```
Final Score = Performance Score × C

Performance Score = S₁ × 40% + S₈ × 30% + Smax × 30%
S_N = (Duration_best / Duration_yours) × 100

C Coefficient (based on relative accuracy = raw/80×100):
  ≤97%: 0 (eliminated)  |  97-98%: 0.92  |  98-99%: 0.96  |  99-100%: 1.0
```

## Bounty Targets

| Level | Target Duration | Weight |
|-------|-----------------|--------|
| S1 (concurrent=1) | 420s | 40% |
| S8 (concurrent=8) | 420s | 30% |
| Smax (no limit) | 720s | 30% |

---

## Baseline (Stock Kernel) - H100 Reference

**Date**: 2026-03-02
**Environment**: Modal H100 (NOT final eval hardware)
**Kernel**: Stock (soar-toolkit image default)
**Config**: `--attention-backend minicpm_flashinfer --chunked-prefill-size 8192 --mem-fraction-static 0.70 --dense-as-sparse --skip-server-warmup`

> ⚠️ **Note**: These are H100 results for development reference only. Final evaluation runs on RTX PRO 6000 which will be slower. Use these to measure relative improvements.

### Correctness

| Metric | Value |
|--------|-------|
| Raw Accuracy | 81.31% |
| Baseline | 80% |
| Relative Score | 101.64% |
| C Coefficient | **1.0** ✓ |

### Speed (150 eval samples)

| Level | Duration | Target | Status | Weight |
|-------|----------|--------|--------|--------|
| S1 | 761.13s | 420s | ❌ Need -45% | 40% |
| S8 | 169.24s | 420s | ✅ Beats by 60% | 30% |
| Smax | 168.99s | 720s | ✅ Beats by 77% | 30% |

### Analysis

- **S1 is the bottleneck** - single request latency is 1.8x the target
- S8 and Smax already beat bounty targets significantly
- Focus optimization on prefill/decode latency for single requests
- Throughput at ~361 tokens/s output

### Optimization Opportunities

1. **Quantization (GPTQ W4A16)** - Reduce memory bandwidth, faster GEMM
2. **FP8 KV Cache** - Help with long context (32K-128K inputs are 35% of workload)
3. **Kernel fusion** - Reduce kernel launch overhead
4. **Speculative decoding** - Faster decode phase

---

## Iteration 1: Quantization Attempt

**Date**: 2026-03-02
**Changes**:
- [x] Implemented multi-method quantization (llm-compressor -> GPTQ -> bitsandbytes -> RTN)
- [x] Added FP8 KV Cache support
- [ ] BLOCKED: Standard GPTQ libraries fail with SALA's flash_attention_2 requirement

### Status

**BLOCKED**: MiniCPM-SALA requires `flash_attention_2` for sparse attention. Standard GPTQ/AutoGPTQ
libraries don't support this properly because they need to control the attention implementation
during calibration forward passes.

```
AssertionError: Only flash_attention_2 is supported for sparse attention
```

### Workarounds Attempted

1. **llm-compressor**: Not installed in base image, may work better
2. **bitsandbytes NF4**: Simpler approach, loads model with flash_attention_2 then quantizes
3. **RTN fallback**: Works but only pseudo-quantization (quantize/dequantize in float16)

### Next Steps

Option A: **Custom GPTQ for SALA** (complex)
- Quantize only MLP layers (skip attention projections)
- Keep attention layers in float16 for flash_attention_2 compatibility

Option B: **Focus on other optimizations** (pragmatic)
- S1 is 1.8x target, need other speedups
- EAGLE3 speculative decoding for faster decode
- Kernel fusion for prefill optimization

---

## Iteration 2: RTN W4A16 Quantization (GPTQ mode)

**Date**: 2026-03-02
**Environment**: Modal H100
**Config**: `--quantization gptq --dtype float16 --disable-cuda-graph --disable-radix-cache --attention-backend minicpm_flashinfer --chunked-prefill-size 8192 --dense-as-sparse`

**Changes**:
- [x] RTN W4A16 quantization (manual tensor-level, GPTQ-compatible format)
- [x] Patched minicpm_backend.py: bfloat16 → float16
- [x] Using `--quantization gptq` (not Marlin)
- [x] CUDA graph disabled for stability

### Correctness

| Metric | Value | vs Baseline |
|--------|-------|-------------|
| Raw Accuracy | 81.18% | -0.13% |
| Relative Score | 101.48% | -0.16% |
| C Coefficient | **1.0** ✓ | Same |

### Speed

| Level | Duration | vs Baseline | Target | Status |
|-------|----------|-------------|--------|--------|
| S1 | 2034.40s | **2.7x SLOWER** | 420s | ❌ |
| S8 | 545.35s | **3.2x SLOWER** | 420s | ❌ |
| Smax | 406.29s | **2.4x SLOWER** | 720s | ✅ |

### Analysis

**REGRESSION**: Quantization made things SLOWER, not faster!

Likely causes:
1. **`--disable-cuda-graph`** - Major performance hit, but was needed for stability
2. **`--quantization gptq`** - Uses slow dequantization without Marlin kernels
3. **RTN quality** - Round-to-nearest may produce suboptimal weights

Eval throughput was 407 tokens/s (vs 361 baseline = +13%), but benchmark duration regressed significantly. The `--disable-cuda-graph` flag is likely the main culprit.

### Next Steps

1. **Try Marlin kernels**: `--quantization gptq_marlin` (faster dequant GEMM)
2. **Enable CUDA graph**: Remove `--disable-cuda-graph`
3. **If still slow**: Abandon quantization, focus on other optimizations

---

## Iteration 3: RTN W4A16 + Marlin + FP8 KV + CUDA Graph

**Date**: 2026-03-03
**Environment**: Modal H100
**Config**: `--quantization gptq_marlin --dtype float16 --kv-cache-dtype fp8_e5m2 --disable-radix-cache --attention-backend minicpm_flashinfer --chunked-prefill-size 8192 --dense-as-sparse`

**Changes**:
- [x] RTN W4A16 quantization (based on official demo-quant)
- [x] Marlin kernels (`--quantization gptq_marlin`)
- [x] FP8 KV cache (`--kv-cache-dtype fp8_e5m2`)
- [x] CUDA graph enabled (removed `--disable-cuda-graph`)
- [x] Patched minicpm_backend.py: bfloat16 → float16

### Speed

| Level | Duration | vs Baseline | vs Iter 2 | Target | Status |
|-------|----------|-------------|-----------|--------|--------|
| S1 | 1124.94s | **+48% SLOWER** | -45% (better) | 420s | ❌ |
| S8 | 337.50s | **+99% SLOWER** | -38% (better) | 420s | ✅ |
| Smax | 301.66s | **+79% SLOWER** | -26% (better) | 720s | ✅ |

### Analysis

**Mixed Results**:
- ✅ **Marlin + CUDA graph improved** vs Iteration 2 (GPTQ without Marlin)
- ❌ **Still slower than FP16 baseline** - quantization overhead not compensated
- ✅ S8 and Smax beat bounty targets
- ❌ S1 still 2.7x the target (1125s vs 420s)

**Why quantization is slower**:
1. RTN quality may be suboptimal (no calibration data)
2. Dequantization overhead on H100 (even with Marlin)
3. FP16 baseline might already be well-optimized
4. Quantization helps more on memory-constrained GPUs

**Output throughput**: ~305 tokens/s (vs 361 baseline, 407 in Iteration 2 eval)

### Next Steps

**Option A: Abandon quantization, optimize FP16 baseline**
- Focus on kernel fusion, prefill optimization
- Tune chunked-prefill-size, memory settings
- Try speculative decoding (EAGLE3)

**Option B: Try different quantization approach**
- AWQ instead of GPTQ/RTN
- Calibration-based GPTQ (if SALA compatibility can be solved)
- FP8 instead of W4A16

**Option C: Submit current version**
- S8 and Smax beat targets
- Focus on improving S1 through other means

---

## Iteration 4: [Pending]

**Date**: YYYY-MM-DD
**Changes**:
- [ ] TBD based on Iteration 3 analysis

---

## Quick Reference

### Run Commands

```powershell
# Quick eval (10 samples, ~5 min)
modal run modal_app.py --cmd eval --num-samples 10

# Full eval (150 samples, ~40 min)
modal run modal_app.py --cmd eval

# Quick baseline (20 samples each, ~20 min)
modal run modal_app.py --cmd quick

# Full baseline (150 eval + speed benchmarks)
modal run modal_app.py --cmd baseline

# With custom kernel
modal run modal_app.py --cmd baseline --use-custom
```

### Key Files to Modify

```
sglang/sgl-kernel/csrc/
├── attention/           # Attention kernels
├── elementwise/         # Fused ops (RMSNorm, etc.)
├── gemm/marlin/         # GPTQ Marlin kernels
└── moe/                 # MoE kernels

sglang/python/sglang/srt/
├── models/minicpm_sala.py           # Model definition
├── layers/quantization/gptq.py      # GPTQ implementation
└── layers/attention/                # Attention backends
```
