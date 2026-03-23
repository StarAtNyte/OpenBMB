# MiniCPM-SALA Prefill Kernel Tuning Guide

## Target File
`sglang/python/sglang/srt/layers/attention/minicpm_fuse_kernel.py`

## Key Parameters in `fused_attn_pooling_online_topk_prefill`

### 1. Block Sizes (Lines 36-38)
```python
block_M: int = 16,    # Query block size
block_N: int = 64,    # Key block size
```

**What they control**:
- How many queries/keys are processed per thread block
- Affects register usage and occupancy

**RTX 6000 Optimization**:
- RTX 6000 has 84 SMs, 128 KB L1 cache per SM
- Try larger blocks to improve data reuse:
  - `block_M = 32` (2x more queries per block)
  - `block_N = 128` (2x more keys per block)

**How to change**:
1. Edit line 36-38 in minicpm_fuse_kernel.py
2. Rebuild sgl-kernel: `pip install --no-deps -e ./sglang/python`
3. Test with benchmark

### 2. Pipeline Stages (Line 49)
```python
num_stages: int = 0,  # DISABLED!
```

**What it controls**:
- Software pipelining - overlaps memory loads with computation
- 0 = disabled (current), 2-3 = typical for good performance

**Recommendation**:
```python
num_stages: int = 2,  # Enable 2-stage pipelining
```

**Expected gain**: 20-30% on long sequences

### 3. Threads Per Block (Line 50)
```python
threads: int = 128,
```

**What it controls**:
- Number of threads per CUDA block
- Affects occupancy and register pressure

**RTX 6000 Optimization**:
- Try: 128 (current), 256, 512
- 256 often optimal for Ampere/Ada architecture

### 4. Data Type (Line 51)
```python
dtype_str: str = "bfloat16",
```

**Current**: Uses bfloat16
**Alternative**: "float16" (might be faster on RTX 6000)

## Tuning Process

### Quick Test (10 min)
1. Change ONE parameter at a time
2. Run: `modal run modal_app.py --cmd eval --num-samples 10`
3. Compare time per sample

### Full Test (1-2 hours)
1. Once you find good params, run full baseline
2. `modal run modal_app.py --cmd baseline`

## Best Practices

1. **Start with num_stages = 2** - easiest win
2. **Then try block_M = 32** - better for long contexts
3. **Then try block_N = 128** - better memory coalescing
4. **Finally tune threads** - if still not optimal

## Expected Total Improvement
- num_stages: 2 → **+20-30%**
- Larger blocks → **+10-20%**
- Thread tuning → **+5-10%**

**Combined**: **30-50% faster prefill** → Significant S1/S8/Smax improvement!
