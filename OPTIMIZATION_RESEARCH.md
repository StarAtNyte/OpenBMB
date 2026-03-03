# SOAR 2026 - LLM Inference Optimization Research

Comprehensive research on optimization techniques for MiniCPM-SALA inference.
Last updated: March 2026

## Table of Contents

1. [Current Status](#current-status)
2. [Quantization Techniques](#quantization-techniques)
3. [Speculative Decoding](#speculative-decoding)
4. [Attention Optimizations](#attention-optimizations)
5. [KV Cache Optimization](#kv-cache-optimization)
6. [Prefill/Decode Optimization](#prefilldecode-optimization)
7. [Kernel-Level Optimizations](#kernel-level-optimizations)
8. [Hardware-Specific Optimizations](#hardware-specific-optimizations)
9. [Recommended Implementation Order](#recommended-implementation-order)

---

## Current Status

| Optimization | Status | Impact | Notes |
|--------------|--------|--------|-------|
| RTN W4A16 Quantization | ✅ Done | +13% throughput | 81.18% accuracy maintained |
| FP8 KV Cache | ✅ Enabled | Helps long context | `--kv-cache-dtype fp8_e5m2` |
| Marlin Kernels | 🔄 Testing | +10-20% expected | `--quantization gptq_marlin` |
| CUDA Graph | 🔄 Testing | Reduce overhead | Removed `--disable-cuda-graph` |
| EAGLE3 Speculative | ⏳ Pending | 2-3x decode speed | Complex for SALA |

**Bottleneck**: S1 (single request) = 761s, target = 420s (need 45% reduction)

---

## Quantization Techniques

### 1. GPTQ W4A16 (Current)

**Status**: ✅ Implemented via RTN

- Round-To-Nearest quantization (no calibration needed)
- 4-bit weights, 16-bit activations
- Marlin kernels for fast dequantization GEMM

**References**:
- [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer)
- [500+ LLM Inference Optimization Techniques](https://www.aussieai.com/blog/llm-inference-optimization)

### 2. FP8 Quantization

**Potential**: Native FP8 on Hopper/Blackwell GPUs

- E4M3 format for weights
- E5M2 format for KV cache
- ~2x memory reduction vs FP16

**Implementation**:
```bash
--dtype float8_e4m3fn --kv-cache-dtype fp8_e5m2
```

**References**:
- [NVIDIA FP8 Introduction](https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/)
- [Optimizing LLMs with Post-Training Quantization](https://developer.nvidia.com/blog/optimizing-llms-for-performance-and-accuracy-with-post-training-quantization/)

### 3. NVFP4 (Blackwell Only)

**Potential**: 3.5x memory reduction vs FP16

- Micro-block scaling (blocks of 16 values)
- FP8 E4M3 scaling factors
- <1% accuracy loss on benchmarks

**Note**: Requires NVIDIA Blackwell GPUs (RTX PRO 6000 may not support)

**References**:
- [NVFP4 for Efficient Inference](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
- [NVFP4 KV Cache Optimization](https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/)

---

## Speculative Decoding

### EAGLE-3 (Recommended Path)

**Potential**: 2-3x decode speedup

EAGLE-3 uses a lightweight draft head attached to the target model's internal layers:
- Draft head = 2-5% of target model size
- No separate draft model needed
- Reuses target model's internal features
- Multi-layer feature fusion via "training-time test"

**Key Innovation**: Direct token prediction instead of feature prediction

**Challenges for MiniCPM-SALA**:
- Lightning Attention layers don't support tree-shaped causal masks
- Need creative solutions for verification with linear attention
- SGLang has preliminary integration in `hybrid_linear_attn_backend.py`

**Implementation Steps**:
1. Train EAGLE3 draft head using hidden states from SALA
2. Create EAGLE3 model file (reference: `llama_eagle3.py`)
3. Modify verification logic for Lightning Attention
4. Launch with: `--speculative-algorithm EAGLE3 --speculative-draft-model-path <path>`

**References**:
- [EAGLE-3 Paper (NeurIPS'25)](https://arxiv.org/abs/2503.01840)
- [EAGLE GitHub](https://github.com/SafeAILab/EAGLE)
- [SpecForge Training Framework](https://lmsys.org/blog/2025-07-25-spec-forge/)
- [EAGLE-3 on Vertex](https://lmsys.org/blog/2025-12-01-eagle3-vertex/)

### SpecForge Integration

SpecForge is the official training framework for EAGLE3:
- Tightly integrated with SGLang
- Zero-Overhead Overlap Scheduler for spec decoding
- Seamless training to deployment pipeline

---

## Attention Optimizations

### 1. FlashInfer (Current)

**Status**: ✅ Using `minicpm_flashinfer` backend

FlashInfer provides:
- Block-sparse format for KV cache
- JIT compilation for customization
- Load-balanced scheduling with CUDAGraph compatibility
- 29-69% latency reduction vs compiler backends

**References**:
- [FlashInfer Paper (MLSys 2025)](https://arxiv.org/abs/2501.01005)
- [FlashInfer GitHub](https://github.com/flashinfer-ai/flashinfer)
- [NVIDIA FlashInfer Blog](https://developer.nvidia.com/blog/run-high-performance-llm-inference-kernels-from-nvidia-using-flashinfer/)

### 2. Native Sparse Attention (NSA)

**Potential**: Optimized for SALA's sparse attention

Recent developments:
- Hardware-aligned kernel design
- Eliminates redundant KV transfers via group-wise sharing
- Balanced compute workloads across GPU streaming multiprocessors
- Optimized for Tensor Core utilization

**References**:
- [Native Sparse Attention Paper](https://arxiv.org/pdf/2502.11089)
- [Flash Sparse Attention (FSA)](https://arxiv.org/pdf/2508.18224)

### 3. FlashMLA (DeepSeek)

**Potential**: 93.3% KV cache reduction

Multi-head Latent Attention:
- Up to 3000 GB/s memory bandwidth on H800
- Up to 660 TFlops compute
- 57x KV cache reduction vs standard MHA

**Note**: Designed for Hopper GPUs, may need adaptation

**References**:
- [FlashMLA GitHub](https://github.com/deepseek-ai/FlashMLA)
- [FlashMLA Analysis](https://blog.gopenai.com/comprehensive-analysis-of-deepseeks-open-sourced-flashmla-83b8f590d804)

### 4. Linear Attention Optimization

**Relevant for**: SALA's Lightning Attention layers

Recent research (Oct 2025):
- 3.3x speed improvement over state-of-the-art
- 3.6x memory reduction
- O(ND²) complexity vs O(N²D) for softmax attention

**References**:
- [Linear Attention GPU Kernel](https://arxiv.org/abs/2510.21956)
- [Attention Optimization Research](https://www.aussieai.com/research/attention)

---

## KV Cache Optimization

### 1. FP8 KV Cache (Current)

**Status**: ✅ Enabled via `--kv-cache-dtype fp8_e5m2`

- 50% memory reduction vs FP16
- Critical for 32K-128K inputs (35% of workload)
- Minimal accuracy impact

### 2. KV Cache Compression Techniques

| Technique | Compression | Notes |
|-----------|-------------|-------|
| Quantization (AsymKV) | 2-4x | Layer-wise asymmetric quantization |
| ChunkKV | Variable | Semantic chunk-based compression |
| SqueezeAttention | 30-70% | 2D optimization (sequence + layer) |
| KVTC (Transform Coding) | Up to 20x | PCA + adaptive quantization + entropy |
| H2O | >40% throughput | Submodular eviction optimization |

**References**:
- [KV Cache Compression Survey](https://arxiv.org/pdf/2508.06297)
- [ChunkKV](https://openreview.net/forum?id=20JDhbJqn3)
- [SqueezeAttention](https://openreview.net/forum?id=9HK2rHNAhd)
- [KVTC](https://arxiv.org/abs/2511.01815)

### 3. Paged Attention

**Status**: Built into SGLang

- Virtual memory abstraction for KV cache
- Eliminates memory fragmentation
- Dynamic page allocation/reuse
- 23x throughput improvement in vLLM

---

## Prefill/Decode Optimization

### 1. Chunked Prefill (Current)

**Status**: ✅ Using `--chunked-prefill-size 8192`

Benefits:
- +50% throughput in benchmarks
- Compute-intensive prefill + memory-bound decode in parallel
- Decode requests "piggyback" on prefill batches

**Trade-off**: Increased KV cache loading (quadratic vs linear)

**Tuning**: Experiment with chunk sizes (4096, 8192, 16384, 32768)

**References**:
- [Chunked Prefill Optimization](https://donmoon.medium.com/llm-inference-optimizations-2-chunked-prefill-764407b3a67a)
- [TensorRT-LLM Chunked Prefill](https://developer.nvidia.com/blog/streamlining-ai-inference-performance-and-deployment-with-nvidia-tensorrt-llm-chunked-prefill/)

### 2. Prefill-Decode Disaggregation

**Potential**: Further throughput optimization

- Separate prefill and decode to different GPU resources
- Maximizes "goodput" (useful output tokens)
- DistServe implementation available

**References**:
- [DistServe](https://haoailab.com/blogs/distserve/)
- [Prefill-Decode Disaggregation](https://bentoml.com/llm/inference-optimization/prefill-decode-disaggregation)

### 3. POD-Attention (ASPLOS'25)

**Potential**: 35% energy reduction

- Full prefill-decode overlap
- Presented at ASPLOS March 2025

**References**:
- [POD-Attention Paper](https://akkamath.github.io/files/ASPLOS25_POD.pdf)

---

## Kernel-Level Optimizations

### 1. Kernel Fusion

**Impact**: Reduce kernel launch overhead

Fuse operations:
- LayerNorm + Linear
- RMSNorm + Activation
- Attention score + Softmax

**Key Files** (sgl-kernel):
```
csrc/elementwise/   # RMSNorm, activations
csrc/attention/     # Attention kernels
csrc/gemm/marlin/   # GPTQ Marlin kernels
```

### 2. Custom CUDA Kernels

**Opportunities for MiniCPM-SALA**:

1. **Sparse Attention Kernel**: Optimize for SALA's sparse pattern
2. **Lightning Attention Kernel**: Optimize GDN recurrence (`cutedsl_gdn.py`)
3. **Fused MLP**: gate_proj + up_proj + activation + down_proj

### 3. Triton Kernels

**Advantages**: Easier development than CUDA

SGLang uses Triton for:
- Linear attention chunks (`fla/chunk.py`)
- Fused recurrent operations (`fused_recurrent.py`)
- EAGLE3 utilities (`multi_layer_eagle_utils.py`)

**References**:
- [Triton Attention Kernel Anatomy](https://arxiv.org/html/2511.11581v1)
- [PyTorch FlexAttention](https://pytorch.org/blog/flexattention/)

### 4. Automated Kernel Generation

**Emerging**: LLM-assisted kernel optimization

- K-Search: Co-evolving world model for kernel generation
- Feature Search and Reinforcement (FSR) framework
- Up to 1.3x speedup on optimized baselines

**References**:
- [K-Search](https://arxiv.org/html/2602.19128v1)
- [CUDA Kernel Generation](https://www.emergentmind.com/topics/cuda-kernel-generation-and-optimization)

---

## Hardware-Specific Optimizations

### RTX PRO 6000 (Target Hardware)

**Specs to optimize for**:
- Ada Lovelace architecture
- FP8 Tensor Cores
- Large L2 cache
- High memory bandwidth

**Key considerations**:
- Not Hopper/Blackwell - some optimizations may not apply
- Focus on FP8 (not FP4)
- Marlin kernels optimized for consumer GPUs

### Memory Bandwidth Optimization

**Critical for decode phase**:
- Minimize KV cache reads
- Use FP8 KV cache
- Optimize memory access patterns
- Consider memory pooling

---

## Recommended Implementation Order

### Quick Wins (Low Effort, Immediate Impact)

| Priority | Optimization | Expected Gain | Effort |
|----------|--------------|---------------|--------|
| 1 | Marlin kernels (`gptq_marlin`) | +10-20% | Trivial |
| 2 | Enable CUDA Graph | +5-10% | Trivial |
| 3 | Tune chunked-prefill-size | +5-15% | Low |
| 4 | Memory fraction tuning | +5-10% | Low |

### Medium Term (Moderate Effort)

| Priority | Optimization | Expected Gain | Effort |
|----------|--------------|---------------|--------|
| 5 | EAGLE3 speculative decoding | +100-200% decode | High |
| 6 | Custom sparse attention kernel | +10-30% | Medium |
| 7 | Fused MLP kernel | +5-15% | Medium |

### Advanced (High Effort)

| Priority | Optimization | Expected Gain | Effort |
|----------|--------------|---------------|--------|
| 8 | Linear attention optimization | +20-40% | High |
| 9 | KV cache compression (KVTC) | 20x compression | High |
| 10 | Custom Lightning Attention kernel | Variable | Very High |

---

## Key Files to Modify

### SGLang Python
```
python/sglang/srt/models/minicpm_sala.py          # Model definition
python/sglang/srt/layers/attention/               # Attention backends
python/sglang/srt/layers/quantization/gptq.py    # GPTQ + Marlin
python/sglang/srt/speculative/                    # EAGLE3 integration
python/sglang/jit_kernel/cutedsl_gdn.py          # GDN CUDA kernel
```

### sgl-kernel CUDA
```
csrc/attention/                  # Attention kernels
csrc/elementwise/                # Fused ops (RMSNorm, etc.)
csrc/gemm/marlin/                # GPTQ Marlin kernels
csrc/speculative/                # EAGLE3 kernels
```

---

## References Summary

### Papers
- [EAGLE-3 (NeurIPS'25)](https://arxiv.org/abs/2503.01840)
- [FlashInfer (MLSys 2025)](https://arxiv.org/abs/2501.01005)
- [Native Sparse Attention](https://arxiv.org/pdf/2502.11089)
- [Linear Attention Kernel](https://arxiv.org/abs/2510.21956)
- [KV Cache Compression Survey](https://arxiv.org/pdf/2508.06297)
- [Survey on Efficient LLM Inference](https://arxiv.org/pdf/2404.14294)

### Tools & Frameworks
- [SGLang](https://github.com/sgl-project/sglang)
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer)
- [EAGLE](https://github.com/SafeAILab/EAGLE)
- [FlashMLA](https://github.com/deepseek-ai/FlashMLA)
- [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer)

### Blogs & Guides
- [NVIDIA Mastering LLM Inference](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)
- [500+ LLM Optimization Techniques](https://www.aussieai.com/blog/llm-inference-optimization)
- [New LLM Inference Stack 2025](https://www.stixor.com/blogs/new-inference-stack-2025)
- [SpecForge](https://lmsys.org/blog/2025-07-25-spec-forge/)
