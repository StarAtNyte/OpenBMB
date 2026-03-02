# SOAR 2026 Competition - MiniCPM-SALA Optimization

Optimizing inference performance for MiniCPM-SALA (Sparse + Linear Attention) on SGLang.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  LOCAL (Windows)                                                │
│  - Edit code: sglang/sgl-kernel/, submission/                   │
│  - No GPU needed, just code editing                             │
└─────────────────────────┬───────────────────────────────────────┘
                          │ modal run modal_app.py --cmd sync
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  MODAL LABS (H100 GPU)                                          │
│  - Build sgl-kernel wheel                                       │
│  - Run correctness evaluation                                   │
│  - Run speed benchmarks (S1, S8, Smax)                          │
│  - Create submission package                                    │
└─────────────────────────┬───────────────────────────────────────┘
                          │ Download .tar.gz, upload to SOAR
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  SOAR PLATFORM (RTX PRO 6000)                                   │
│  - Final evaluation with private dataset                        │
│  - Official scoring and ranking                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

```powershell
# 1. Install Modal CLI (one time)
pip install modal
modal setup

# 2. Download model to Modal volume (one time)
modal run modal_app.py --cmd download

# 3. Sync your local code to Modal
modal run modal_app.py --cmd sync

# 4. Quick baseline test (~20 min)
modal run modal_app.py --cmd quick

# 5. When ready, create submission package
modal run modal_app.py --cmd package
```

## Project Structure

```
soar-optimization/
├── modal_app.py              # Modal development functions
├── submission/               # Submission package files
│   ├── prepare_env.sh        # Environment setup (required)
│   ├── prepare_model.sh      # Model preprocessing (optional)
│   └── preprocess_model.py   # Quantization script
├── scripts/
│   ├── run_sglang.sh         # Server launcher
│   ├── benchmark.sh          # Benchmark runner
│   ├── build_wheel.sh        # Wheel builder
│   ├── fast_dev_build.sh     # Fast incremental build
│   └── create_submission.py  # Package creator (Windows)
├── sglang/                   # SGLang repo (modify sgl-kernel/)
└── SOAR/                     # SOAR Toolkit (eval scripts)
```

## Commands Reference

| Command | Description |
|---------|-------------|
| `modal run modal_app.py --cmd help` | Show all commands |
| `modal run modal_app.py --cmd download` | Download model (once) |
| `modal run modal_app.py --cmd sync` | Sync local code to Modal |
| `modal run modal_app.py --cmd build` | Build sgl-kernel wheel |
| `modal run modal_app.py --cmd eval` | Run correctness evaluation |
| `modal run modal_app.py --cmd eval --num-samples 10` | Quick eval |
| `modal run modal_app.py --cmd benchmark` | Run speed benchmark |
| `modal run modal_app.py --cmd baseline` | Full baseline (eval + bench) |
| `modal run modal_app.py --cmd quick` | Quick baseline (~20 min) |
| `modal run modal_app.py --cmd package` | Create submission .tar.gz |
| `modal shell modal_app.py::dev_shell` | Interactive H100 shell |

## Development Workflow

### Standard Flow (Edit Local → Test on Modal)

```powershell
# 1. Edit code locally (Windows)
code sglang/sgl-kernel/csrc/attention/cascade.cu

# 2. Sync to Modal
modal run modal_app.py --cmd sync

# 3. Build kernel on H100
modal run modal_app.py --cmd build

# 4. Quick eval (10 samples, ~5 min)
modal run modal_app.py --cmd eval --num-samples 10

# 5. Full baseline when ready (~1-2 hours)
modal run modal_app.py --cmd baseline
```

### Interactive Shell (Fastest for Rapid Iteration)

```powershell
# Start H100 shell (4hr session)
modal shell modal_app.py::dev_shell
```

Inside the shell:
```bash
cd /workspace/sglang/sgl-kernel
vim csrc/attention/cascade.cu           # edit directly
bash /workspace/scripts/fast_dev_build.sh  # <2min rebuild
python -c "import sgl_kernel; print('OK')"

# Run eval without re-syncing
python /opt/SOAR/eval_model.py --api_base http://localhost:30000 ...
```

### Creating & Downloading Submission

```powershell
# Create package on Modal
modal run modal_app.py --cmd package

# Download to local machine
modal volume get soar-workspace /workspace/packages ./packages

# Submit the .tar.gz to SOAR platform
```

## Submission Format (New!)

The SOAR platform now uses a new submission format:

```
submission.tar.gz
├── prepare_env.sh          # Required - environment setup
├── prepare_model.sh        # Optional - model preprocessing
├── preprocess_model.py     # Optional - quantization script
├── sglang/python/          # Optional - custom sglang code
└── wheels/*.whl            # Optional - custom kernel wheel
```

### Create Submission Package

**On Windows:**
```powershell
python scripts/create_submission.py
# or
python scripts/create_submission.py --output my_submission.tar.gz
```

**On Modal:**
```bash
modal run modal_app.py --cmd package
modal volume get soar-workspace /workspace/packages ./packages
```

## Scoring Formula

```
Final Score = Performance Score × Correctness Coefficient C

Performance Score = S₁ × 40% + S₈ × 30% + S_∞ × 30%
```

### Correctness Requirements

| Accuracy | Coefficient C |
|----------|---------------|
| ≤ 97%    | 0 (ELIMINATED) |
| 97-98%   | 0.92 |
| 98-99%   | 0.96 |
| 99-100%  | 1.0 |

### Bounty Targets (Extra Prize)

| Concurrency | Target Duration |
|-------------|-----------------|
| S1 (--max-concurrent 1) | 420s |
| S8 (--max-concurrent 8) | 420s |
| Smax (no limit) | 720s |

## Technical Optimization Paths

### Path 1: GPTQ Quantization (Recommended)

**W4A16 + Marlin Kernel + FP8 KV Cache**

1. Quantize model to 4-bit (W4A16) using GPTQModel
2. Use Marlin high-performance GEMM kernel
3. Enable FP8 KV cache for memory savings

```bash
# In prepare_env.sh:
export SGLANG_SERVER_ARGS="--quantization gptq_marlin --kv-cache-dtype fp8_e5m2 --dtype float16"

# In prepare_model.sh:
python preprocess_model.py --input $INPUT --output $OUTPUT --mode gptq --bits 4
```

Key files to modify:
- `sglang/srt/models/minicpm_sala.py` - SALA model definition
- `sglang/srt/layers/quantization/gptq.py` - GPTQ linear layer
- `sgl-kernel/csrc/gemm/marlin/` - Marlin GEMM kernels

### Path 2: Speculative Decoding (Advanced)

**EAGLE3 with Lightning Attention compatibility**

Challenge: Lightning Attention uses linear attention which doesn't support tree-shaped causal masks.

Key files:
- `sglang/srt/speculative/multilayer_eagleworker.py`
- `sglang/srt/layers/attention/hybrid_linear_attn_backend.py`

## Input/Output Distribution

Optimize for these distributions (32K-128K inputs are 35% weight!):

**Input Lengths:**
- 0-4K: ~25%
- 4K-16K: ~10%
- 16K-32K: ~15%
- **32K-128K: ~35%** ← Focus here!
- 128K-160K: ~15%

**Output Lengths:**
- 0-512: ~35%
- 512-2K: ~25%
- 2K-4K: ~10%
- 4K-16K: ~15%
- 16K-32K: ~15%

## Environment Differences

| | Modal (Development) | SOAR (Submission) |
|---|---|---|
| GPU | H100 80GB | RTX PRO 6000 |
| Package Manager | pip | **uv pip** |
| Container | ghcr.io/openbmb/soar-toolkit:latest | Same |
| Max Submission Size | N/A | 2GB |
| Time Limit | N/A | 5 hours |

## Submission Checklist

- [ ] `prepare_env.sh` is present and uses `uv pip install`
- [ ] Total package size < 2GB
- [ ] Accuracy > 97% on public eval set
- [ ] Code is Apache 2.0 licensed
- [ ] All args use hyphens (`--dense-as-sparse`), not underscores

## Tips

1. **Use `--dense-as-sparse`** for SALA model sparse attention
2. **FP8 KV Cache** significantly helps long context performance
3. **Test on H100 first**, but remember RTX 6000 has different perf characteristics
4. **Quantization must run on single GPU** within submission time limit
5. **Don't submit quantized models** - submit quantization scripts instead

## Resources

- [SGLang GitHub](https://github.com/sgl-project/sglang)
- [SOAR Toolkit](https://github.com/OpenBMB/SOAR-Toolkit)
- [GPTQModel](https://github.com/ModelCloud/GPTQModel)
- [Competition FAQ](https://soar.openbmb.cn/faq)

## Contact

Competition issues: contact@openbmb.cn
