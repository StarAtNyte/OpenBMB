# SOAR 2026 — Optimization Roadmap

**Hardware**: NVIDIA RTX PRO 6000 Blackwell Server Edition
**SM**: 12.0 | **VRAM**: 95 GB | **Memory BW**: ~1360 GB/s measured
**Stack**: SGLang 0.0.0.dev0 · FlashInfer 0.5.3 · TileLang 0.1.8 · Triton 3.5.1

---

## Optimization Plan

| # | Optimization | Expected Gain | Effort | Status |
|---|---|---|---|---|
| 1 | **FP8 W8A8 weight quantization** | 2x prefill throughput, 2x weight memory BW | Medium | 🔄 In Progress |
| 2 | Enable TMA in tilelang (`TL_DISABLE_TMA_LOWER=False`) | Hide memory latency in prefill kernels | Low | ⏳ Pending |
| 3 | `--mem-fraction-static 0.88` + `--chunked-prefill-size 32768` | More KV cache, faster long-context (128K-160K) | Trivial | ⏳ Pending |
| 4 | CUDA graphs for decode (`CUDAGraphBatchDecodeWithPagedKVCacheWrapper`) | 4x kernel launch overhead reduction | Low-medium | ⏳ Pending |
| 5 | Enable warp specialization (`TL_DISABLE_WARP_SPECIALIZED=False`) | Better SM utilization via async prefetch | Low | ⏳ Pending |
| 6 | NVFP4 quantization (if accuracy holds) | Up to 4x FP16 (Blackwell-exclusive) | High | ⏳ Pending |
| 7 | KV cache page sizing for L2 reuse | Faster decode latency (L2=128MB, 4500 GB/s) | Medium | ⏳ Pending |
| 8 | Replace any `torch.sort` with `torch.topk` | 2.8x on sparse block selection ops | Trivial | ⏳ Pending |

---

## #1 — FP8 W8A8 Quantization

### Why
From hardware probe:
- FP16 GEMM prefill M=2048: **324 TFLOPS**
- FP8  GEMM prefill M=2048: **691 TFLOPS** → **2.13x faster**
- FP8 is Blackwell-native (`tgv_gemm_sm100` in FlashInfer, `w8a8_fp8` in SGLang)
- Weight size halved: less VRAM bandwidth in memory-bound decode

### How (W8A8 Dynamic)
- **Weights**: static per-channel FP8 e4m3 quantization (offline, in `preprocess_model.py`)
- **Activations**: dynamic per-token FP8 quantization at runtime (zero calibration cost)
- SGLang flag: `--quantization w8a8_fp8`
- Format: weights as `float8_e4m3fn` + `weight_scale` per output-channel as float32

### What gets quantized
All linear projection weights:
- Attention: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `o_gate`, `z_proj`
- FFN: `gate_proj`, `up_proj`, `down_proj`
- Embeddings and LM head: **NOT quantized** (kept fp16)

### Files changed
- `submission/preprocess_model.py` — replaced copy with FP8 quantization
- `submission/prepare_env.sh` — add `--quantization w8a8_fp8` to server args

### Accuracy risk
- Dynamic activation quantization has minimal accuracy loss (<0.5% on most benchmarks)
- Competition threshold: 97% accuracy — FP8 dynamic is safe
- If accuracy drops below 98%: fall back to static activation calibration

---

## #2 — TMA in TileLang

### Why
TMA (Tensor Memory Accelerator) on Blackwell allows async global→shared memory copies,
overlapping memory transfers with tensor core computation. Currently disabled:
```python
tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True   # in minicpm_fuse_kernel.py
```

### How
Set `TL_DISABLE_TMA_LOWER: False` in `minicpm_fuse_kernel.py` `_pass_configs`.
Also test `TL_DISABLE_WARP_SPECIALIZED: False` for producer/consumer warp pattern.

### Risk
May produce incorrect results or fail to compile on SM12 — test carefully with eval.

---

## #3 — Server Args Tuning

### Why
- 95GB VRAM vs assumed 48GB → can push much higher `--mem-fraction-static`
- Larger chunked prefill = fewer pipeline bubbles for long inputs

### Changes to `prepare_env.sh`
```bash
# Before
--mem-fraction-static 0.70 --chunked-prefill-size 8192

# After
--mem-fraction-static 0.88 --chunked-prefill-size 32768
```

---

## #4 — CUDA Graphs for Decode

### Why
From hardware probe:
```
10 eager kernel launches:    0.050 ms  (5 us each)
10 via CUDA graph replay:    0.012 ms  → 4x speedup
```
Decode is dominated by many sequential small kernels. CUDA graph capture + replay
eliminates per-launch overhead. FlashInfer provides `CUDAGraphBatchDecodeWithPagedKVCacheWrapper`.

SGLang already supports CUDA graphs — ensure `--disable-cuda-graph` is NOT set.

---

## #5 — Warp Specialization

### Why
Blackwell SM12 (like Hopper SM9) supports producer/consumer warp specialization,
where "producer" warps prefetch data via TMA while "consumer" warps execute tensor cores.
Effectively doubles utilization by pipelining memory and compute.

### How
In `minicpm_fuse_kernel.py`:
```python
tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False  # currently True
```

---

## #6 — NVFP4 Quantization

### Why
FlashInfer on this build exposes:
- `flashinfer.fp4_quantize`, `nvfp4_quantize`, `nvfp4_batched_quantize`
- `flashinfer.mm_fp4`, `mxfp4_quantize`
- SGLang has `mxfp4.py`, `mxfp4_tensor.py`, `kvfp4_tensor.py`

FP4 = 4x weight compression vs FP16 = 4x less memory bandwidth in decode.
Blackwell-exclusive (SM12 only).

### Risk
High accuracy risk — FP4 is aggressive. Only attempt after FP8 is validated.

---

## #7 — KV Cache L2 Optimization

### Why
L2 cache: **128 MB** at **4500 GB/s** (vs GDDR7 at 1360 GB/s).
If KV blocks for active decode requests fit in L2, decode bandwidth is 3x better.

### How
- Tune `--page-size` / KV block size to match L2 capacity for expected batch sizes
- For S1 (single request decode): single request's KV at seq=160K in fp16 = 160K * 40 * 2 * 8 * 128 * 2 bytes ≈ 13GB → won't fit L2
- For decode KV of recently accessed tokens (sliding window): tune SWA window size to fit in L2

---

## #8 — Replace sort with topk

### Why
```
Topk [4096x4096] dim=-1:  0.14 ms
Sort [4096x4096] dim=-1:  0.39 ms   (2.8x slower)
```
Grep for `torch.sort` in sparse attention path and replace with `torch.topk`.

---

## Benchmark Targets

From competition:
| Concurrency | Bounty target | Weight |
|---|---|---|
| S1  (--max-concurrent 1) | 420s | 40% |
| S8  (--max-concurrent 8) | 420s | 30% |
| Smax (no limit)          | 720s | 30% |

Current baseline (from OPTIMIZATION_RESEARCH.md): S1 = ~761s
Need: **45% reduction on S1** just to hit bounty target.
FP8 alone should deliver 30-40% on prefill-heavy workloads.

---

## Hardware Reference (from probe)

```
GPU:          NVIDIA RTX PRO 6000 Blackwell Server Edition
SM:           12.0  |  188 SMs  |  24,064 CUDA cores
VRAM:         94.97 GB
Mem BW:       ~1360 GB/s measured  (2396 GB/s theoretical GDDR7 384-bit)
L2 cache:     128 MB  @  ~4500 GB/s
Shared mem:   100 KB/SM  (99 KB opt-in per block)
Max warps/SM: 48  (1536 threads)
PCIe:         Gen5 x16  |  51.7 GB/s H2D  |  53.2 GB/s D2H

Peak FP16 TC (measured): ~375 TFLOPS
Peak FP8  TC (measured): ~691 TFLOPS  (2.13x FP16)

Kernel launch:  ~5 us overhead
CUDA graph:     4x speedup on 10-op chains

Triton target:  GPUTarget(backend='cuda', arch=120, warp_size=32)  ✓ SM12 confirmed
TMA available:  Yes (SM12 >= SM9 threshold)
FP8 available:  Yes (SM12 >= SM8.9 threshold)
FP4 available:  Yes (Blackwell-exclusive, via FlashInfer nvfp4)
```
