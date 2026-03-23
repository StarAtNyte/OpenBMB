"""
RTX PRO 6000 Hardware Probe — SOAR 2026

Collects everything useful about the GPU and software environment
so we can make informed optimization decisions.

Usage:
    modal run probe_hardware.py
"""

import subprocess
import modal

app = modal.App("soar-probe")

probe_image = (
    modal.Image.from_registry("ghcr.io/openbmb/soar-toolkit:latest")
    .pip_install("tabulate", "psutil")
)


def _run(cmd, shell=False):
    try:
        result = subprocess.run(
            cmd, shell=shell, capture_output=True, text=True, timeout=30
        )
        return (result.stdout + result.stderr).strip()
    except Exception as e:
        return f"[error: {e}]"


@app.function(image=probe_image, gpu="RTX-PRO-6000", timeout=600)
def probe():
    import os
    import sys
    import time
    import math

    import torch
    import psutil

    lines = []

    def section(title):
        lines.append("")
        lines.append("=" * 70)
        lines.append(f"  {title}")
        lines.append("=" * 70)

    def kv(key, value):
        lines.append(f"  {key:<44} {value}")

    def prop(obj, attr, fmt=None, fallback="N/A"):
        try:
            v = getattr(obj, attr)
            return fmt(v) if fmt else v
        except AttributeError:
            return fallback

    def bench(fn, warmup=10, iters=50):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000 / iters  # ms

    # =========================================================================
    # 1. GPU IDENTITY
    # =========================================================================
    section("GPU IDENTITY")
    props = torch.cuda.get_device_properties(0)
    kv("Name",                  props.name)
    kv("Total VRAM",            f"{props.total_memory / 2**30:.2f} GB  ({props.total_memory:,} bytes)")
    kv("CUDA Capability (SM)",  f"{props.major}.{props.minor}")
    kv("Multiprocessors (SMs)", props.multi_processor_count)
    kv("Max threads per SM",    prop(props, "max_threads_per_multi_processor"))
    kv("Warp size",             prop(props, "warp_size", fallback=32))
    kv("Registers per SM",      prop(props, "regs_per_multiprocessor"))
    kv("Shared mem / block (default)", prop(props, "shared_memory_per_block",
                                           fmt=lambda v: f"{v} bytes  ({v//1024} KB)"))
    kv("Shared mem / block (opt-in)",  prop(props, "shared_memory_per_block_optin",
                                           fmt=lambda v: f"{v} bytes  ({v//1024} KB)"))
    kv("Shared mem / SM",       prop(props, "shared_memory_per_multiprocessor",
                                     fmt=lambda v: f"{v} bytes  ({v//1024} KB)"))
    kv("L2 cache size",         prop(props, "L2_cache_size",
                                     fmt=lambda v: f"{v // 2**20} MB  ({v:,} bytes)"))

    # =========================================================================
    # 2. ALL torch device property attributes (raw dump)
    # =========================================================================
    section("ALL torch.cuda.get_device_properties ATTRIBUTES")
    for attr in sorted(a for a in dir(props) if not a.startswith("_")):
        lines.append(f"  {attr:<48} {prop(props, attr)}")

    # =========================================================================
    # 3. CLOCKS & BANDWIDTH (from nvidia-smi — torch doesn't expose on SM12)
    # =========================================================================
    section("CLOCKS & BANDWIDTH (nvidia-smi)")

    def smi_q(field):
        return _run(["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits"]).strip()

    sm_clock_max   = smi_q("clocks.max.sm")
    sm_clock_cur   = smi_q("clocks.current.sm")
    mem_clock_max  = smi_q("clocks.max.memory")
    mem_clock_cur  = smi_q("clocks.current.memory")
    gr_clock_max   = smi_q("clocks.max.gr")
    gr_clock_cur   = smi_q("clocks.current.gr")

    kv("SM clock  (max / current)",   f"{sm_clock_max} MHz  /  {sm_clock_cur} MHz")
    kv("GR clock  (max / current)",   f"{gr_clock_max} MHz  /  {gr_clock_cur} MHz")
    kv("Mem clock (max / current)",   f"{mem_clock_max} MHz  /  {mem_clock_cur} MHz")
    kv("Power limit / draw",          f"{smi_q('power.limit')} W  /  {smi_q('power.draw')} W")
    kv("Temperature (GPU)",           f"{smi_q('temperature.gpu')} C")
    kv("ECC mode",                    smi_q("ecc.mode.current"))
    kv("Compute mode",                smi_q("compute_mode"))
    kv("MIG mode",                    smi_q("mig.mode.current"))

    sm_count = props.multi_processor_count
    warp_sz  = prop(props, "warp_size", fallback=32)
    try:
        clock_ghz    = float(sm_clock_max) / 1000
        mem_clock_mhz = float(mem_clock_max)
        # Blackwell RTX PRO 6000: GDDR7, 384-bit bus, QDR (4 transfers/cycle)
        mem_bw_gb_s  = (384 / 8) * mem_clock_mhz * 4 / 1000
    except ValueError:
        clock_ghz, mem_clock_mhz, mem_bw_gb_s = 2.43, 12481, 1792.0

    cuda_cores_per_sm = 128          # Blackwell SM12: 128 FP32 CUDA cores / SM
    total_cuda_cores  = sm_count * cuda_cores_per_sm

    section("DERIVED PEAK METRICS")
    kv("Architecture",                      f"Blackwell SM{props.major}.{props.minor}")
    kv("CUDA cores/SM  (Blackwell)",        cuda_cores_per_sm)
    kv("Total CUDA cores",                  total_cuda_cores)
    kv("Peak FP32  (dense)",               f"{(total_cuda_cores * 2 * clock_ghz):.1f} TFLOPS")
    kv("Peak FP32  (2:4 sparse)",          f"{(total_cuda_cores * 4 * clock_ghz):.1f} TFLOPS")
    kv("Peak FP16  (tensor core, dense)",  f"{(total_cuda_cores * 4 * clock_ghz):.1f} TFLOPS")
    kv("Peak BF16  (tensor core, dense)",  f"{(total_cuda_cores * 4 * clock_ghz):.1f} TFLOPS")
    kv("Peak FP8   (tensor core, dense)",  f"{(total_cuda_cores * 8 * clock_ghz):.1f} TFLOPS")
    kv("Peak FP8   (2:4 sparse)",          f"{(total_cuda_cores * 16 * clock_ghz):.1f} TFLOPS")
    kv("Peak INT8  (tensor core, dense)",  f"{(total_cuda_cores * 8 * clock_ghz):.1f} TOPS")
    kv("Mem bandwidth (theoretical)",      f"{mem_bw_gb_s:.0f} GB/s")
    kv("Arith intensity crossover FP32",   f"{(total_cuda_cores * 2 * clock_ghz * 1e12) / (mem_bw_gb_s * 1e9):.0f} FLOP/byte")
    kv("Arith intensity crossover FP16 TC",f"{(total_cuda_cores * 4 * clock_ghz * 1e12) / (mem_bw_gb_s * 1e9):.0f} FLOP/byte")

    threads_per_sm = prop(props, "max_threads_per_multi_processor", fallback=1536)
    if isinstance(threads_per_sm, int):
        kv("Max warps / SM",               f"{threads_per_sm // warp_sz}  ({threads_per_sm} threads)")
        kv("Max concurrent warps (all SM)",f"{threads_per_sm // warp_sz * sm_count:,}")

    # =========================================================================
    # 4. MEMORY SUBSYSTEM
    # =========================================================================
    section("MEMORY SUBSYSTEM")
    free, total = torch.cuda.mem_get_info(0)
    kv("VRAM total",  f"{total / 2**30:.2f} GB")
    kv("VRAM free",   f"{free  / 2**30:.2f} GB")
    kv("VRAM used",   f"{(total-free) / 2**30:.2f} GB")

    # KV cache budget for MiniCPM-SALA
    # Model: 52 layers (40 sparse SWA + 12 linear/no-KV), 8 GQA kv-heads, head_dim=128, fp16
    # Only sparse layers need KV cache; linear layers don't store KV
    # This is an estimate — actual numbers depend on model config
    SPARSE_LAYERS = 40
    KV_HEADS      = 8
    HEAD_DIM      = 128
    BYTES_PER_EL  = 2   # fp16
    bytes_per_tok = SPARSE_LAYERS * 2 * KV_HEADS * HEAD_DIM * BYTES_PER_EL  # K+V

    section("KV CACHE BUDGET (MiniCPM-SALA est: 40 sparse layers, 8 GQA heads, dim=128, fp16)")
    for frac in [0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.95]:
        pool_bytes  = total * frac
        pool_gb     = pool_bytes / 2**30
        max_tokens  = int(pool_bytes / bytes_per_tok)
        kv(f"  --mem-fraction-static {frac:.2f}",
           f"{pool_gb:.1f} GB pool  => ~{max_tokens:,} tokens KV cache")

    # =========================================================================
    # 5. CPU & HOST SYSTEM
    # =========================================================================
    section("CPU & HOST SYSTEM")
    kv("CPU logical cores",  psutil.cpu_count(logical=True))
    kv("CPU physical cores", psutil.cpu_count(logical=False))
    ram = psutil.virtual_memory()
    kv("RAM total",          f"{ram.total    / 2**30:.1f} GB")
    kv("RAM available",      f"{ram.available/ 2**30:.1f} GB")
    kv("CPU freq (current)", f"{psutil.cpu_freq().current:.0f} MHz" if psutil.cpu_freq() else "N/A")

    # PCIe bandwidth (host <-> device)
    # Benchmark CPU->GPU and GPU->CPU pinned transfer
    h2d_size = 512 * 2**20  # 512 MB
    cpu_buf  = torch.randn(h2d_size // 4, dtype=torch.float32).pin_memory()
    gpu_buf  = torch.empty(h2d_size // 4, dtype=torch.float32, device="cuda")
    # H2D
    for _ in range(3): gpu_buf.copy_(cpu_buf)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10): gpu_buf.copy_(cpu_buf)
    torch.cuda.synchronize()
    h2d_bw = (h2d_size * 10) / (time.perf_counter() - t0) / 2**30
    # D2H
    for _ in range(3): cpu_buf.copy_(gpu_buf)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10): cpu_buf.copy_(gpu_buf)
    torch.cuda.synchronize()
    d2h_bw = (h2d_size * 10) / (time.perf_counter() - t0) / 2**30
    kv("PCIe H2D bandwidth (pinned, 512MB)", f"{h2d_bw:.1f} GB/s")
    kv("PCIe D2H bandwidth (pinned, 512MB)", f"{d2h_bw:.1f} GB/s")
    del cpu_buf, gpu_buf

    # =========================================================================
    # 6. SOFTWARE VERSIONS
    # =========================================================================
    section("SOFTWARE VERSIONS")
    kv("Python",  sys.version.split()[0])
    kv("PyTorch", torch.__version__)
    kv("CUDA (torch built with)", torch.version.cuda)
    kv("cuDNN",   str(torch.backends.cudnn.version()))
    kv("Driver",  smi_q("driver_version"))
    kv("nvcc",    _run(["nvcc", "--version"]).split("\n")[-1].strip())

    for pkg in ["sglang", "flashinfer", "triton", "tilelang", "xformers", "vllm",
                "cutlass", "bitsandbytes", "autoawq", "gptqmodel"]:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "installed")
        except ImportError as e:
            ver = f"[not found]"
        except Exception as e:
            ver = f"[{e}]"
        kv(pkg, ver)

    # =========================================================================
    # 7. CUDA / HARDWARE FEATURES
    # =========================================================================
    section("CUDA / HARDWARE FEATURES")
    kv("TF32 matmul",           torch.backends.cuda.matmul.allow_tf32)
    kv("TF32 cuDNN",            torch.backends.cudnn.allow_tf32)
    kv("BF16 native",           torch.cuda.is_bf16_supported())
    kv("Flash SDPA",            torch.backends.cuda.flash_sdp_enabled())
    kv("Math SDPA",             torch.backends.cuda.math_sdp_enabled())
    kv("Mem-efficient SDPA",    torch.backends.cuda.mem_efficient_sdp_enabled())
    fp8_ok = props.major > 8 or (props.major == 8 and props.minor >= 9)
    kv("FP8 (SM >= 8.9)",       fp8_ok)
    # Blackwell-specific
    kv("TMA  (SM >= 9.0)",      props.major >= 9)
    kv("Warp specialization (SM >= 9.0)", props.major >= 9)
    kv("Thread block clusters (SM >= 9.0)", props.major >= 9)
    kv("Async copy (SM >= 8.0)", props.major >= 8)
    kv("LDSM / ldmatrix (SM >= 8.0)", props.major >= 8)
    kv("Tensor cores (SM >= 7.0)", props.major >= 7)

    # Check torch.compile / dynamo
    try:
        import torch._dynamo
        kv("torch.compile available", True)
        backends = torch._dynamo.list_backends()
        kv("torch.compile backends", str(backends))
    except Exception as e:
        kv("torch.compile", f"[{e}]")

    # Check CUDA graphs
    try:
        g = torch.cuda.CUDAGraph()
        kv("CUDA graphs", "available")
        del g
    except Exception as e:
        kv("CUDA graphs", f"[{e}]")

    # FlashInfer backend details
    try:
        import flashinfer
        section("FLASHINFER DETAILS")
        kv("Version", flashinfer.__version__)
        for attr in sorted(a for a in dir(flashinfer) if not a.startswith("_")):
            val = getattr(flashinfer, attr, None)
            if callable(val) or isinstance(val, (str, int, float, bool)):
                kv(f"  flashinfer.{attr}", str(val)[:80])
    except Exception as e:
        kv("FlashInfer details", f"[{e}]")

    # Triton backend details
    try:
        import triton
        section("TRITON DETAILS")
        kv("Version", triton.__version__)
        # Check if SM12 is in supported targets
        try:
            import triton.backends
            kv("Backends", str(dir(triton.backends)))
        except Exception as e2:
            kv("Backends", f"[{e2}]")
        # GPU target string triton would use
        try:
            import triton.runtime
            kv("CUDA runtime", str(triton.runtime.driver.active.get_current_target()))
        except Exception as e2:
            kv("Target (runtime)", f"[{e2}]")
    except Exception as e:
        kv("Triton details", f"[{e}]")

    # TileLang details
    try:
        import tilelang
        section("TILELANG DETAILS")
        kv("Version", getattr(tilelang, "__version__", "installed"))
        for key in dir(tilelang):
            if "PassConfig" in key or "Target" in key or "arch" in key.lower():
                val = getattr(tilelang, key, None)
                kv(f"  tilelang.{key}", str(val)[:80])
        try:
            from tilelang import PassConfigKey
            kv("PassConfigKeys", str([k for k in dir(PassConfigKey) if not k.startswith("_")]))
        except Exception as e2:
            kv("PassConfigKey", f"[{e2}]")
    except Exception as e:
        kv("TileLang details", f"[{e}]")

    # =========================================================================
    # 8. SHARED MEMORY BANDWIDTH (smem r/w via triton)
    # =========================================================================
    section("SHARED MEMORY BANDWIDTH (Triton kernel)")
    try:
        import triton
        import triton.language as tl

        @triton.jit
        def smem_bw_kernel(out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
            pid   = tl.program_id(0)
            buf   = tl.zeros([BLOCK], dtype=tl.float32)
            # write then read shared memory BLOCK elements, 32 times
            for _ in tl.static_range(32):
                buf = buf + 1.0
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            tl.store(out_ptr + offs, buf)

        BLOCK = 1024
        grid_sz = 188 * 4  # 4 blocks per SM
        out = torch.empty(grid_sz * BLOCK, device="cuda", dtype=torch.float32)
        fn = lambda: smem_bw_kernel[(grid_sz,)](out, grid_sz * BLOCK, BLOCK)
        ms = bench(fn, warmup=5, iters=20)
        kv("Triton smem kernel (sanity)", f"{ms:.4f} ms  (warmup ok = triton compiles SM12)")
    except Exception as e:
        kv("Triton smem kernel", f"[{e}]")

    # =========================================================================
    # 9. MEMORY BANDWIDTH — multiple buffer sizes
    # =========================================================================
    section("MEMORY BANDWIDTH vs BUFFER SIZE")
    for size_mb in [1, 4, 16, 64, 256, 512, 1024, 4096]:
        try:
            n_fp16 = size_mb * 2**20 // 2
            src = torch.randn(n_fp16, device="cuda", dtype=torch.float16)
            dst = torch.empty_like(src)
            ms  = bench(lambda: dst.copy_(src), warmup=5, iters=20)
            bw  = (size_mb * 2 / 1024) / (ms / 1000)  # read + write, GB/s
            kv(f"  {size_mb:>5} MB buffer", f"{bw:>8.0f} GB/s   ({ms:.4f} ms)")
            del src, dst
        except Exception as e:
            kv(f"  {size_mb:>5} MB buffer", f"[{e}]")
        torch.cuda.empty_cache()

    # =========================================================================
    # 10. COMPUTE BENCHMARKS — shapes relevant to LLM inference
    # =========================================================================
    section("GEMM BENCHMARKS (inference-relevant shapes, FP16)")
    # Decode: M=1 or small batch, large N/K (weight matrix sizes)
    # Prefill: M=seqlen, large N/K
    shapes = [
        # (M,    N,    K,    label)
        (1,    4096, 4096, "decode  M=1   (single token, hidden=4096)"),
        (1,    4096,14336, "decode  M=1   (ffn gate,  hidden->intermediate)"),
        (1,   14336, 4096, "decode  M=1   (ffn down,  intermediate->hidden)"),
        (8,    4096, 4096, "decode  M=8   (batch=8)"),
        (16,   4096, 4096, "decode  M=16  (batch=16)"),
        (32,   4096, 4096, "decode  M=32  (batch=32)"),
        (128,  4096, 4096, "prefill M=128"),
        (512,  4096, 4096, "prefill M=512"),
        (2048, 4096, 4096, "prefill M=2048"),
        (4096, 4096, 4096, "prefill M=4096 (square)"),
        (4096, 4096,14336, "prefill M=4096 (ffn gate)"),
    ]
    for M, N, K, label in shapes:
        try:
            a = torch.randn(M, K, device="cuda", dtype=torch.float16)
            b = torch.randn(K, N, device="cuda", dtype=torch.float16)
            ms = bench(lambda: torch.matmul(a, b), warmup=10, iters=50)
            tflops = 2 * M * N * K / 1e12 / (ms / 1000)
            kv(f"  {label}", f"{ms:.4f} ms  {tflops:6.1f} TFLOPS")
            del a, b
        except Exception as e:
            kv(f"  {label}", f"[{e}]")
        torch.cuda.empty_cache()

    section("GEMM BENCHMARKS (BF16)")
    for M, N, K, label in [(1, 4096, 4096, "decode M=1"), (2048, 4096, 4096, "prefill M=2048"),
                            (4096, 4096, 4096, "square")]:
        try:
            a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
            ms = bench(lambda: torch.matmul(a, b), warmup=10, iters=50)
            tflops = 2 * M * N * K / 1e12 / (ms / 1000)
            kv(f"  {label}", f"{ms:.4f} ms  {tflops:6.1f} TFLOPS")
            del a, b
        except Exception as e:
            kv(f"  {label}", f"[{e}]")
        torch.cuda.empty_cache()

    # FP8 GEMM (Blackwell-native)
    section("GEMM BENCHMARKS (FP8 e4m3 — Blackwell native)")
    try:
        for M, N, K, label in [(1, 4096, 4096, "decode M=1"), (2048, 4096, 4096, "prefill M=2048"),
                                (4096, 4096, 4096, "square")]:
            a = torch.randn(M, K, device="cuda").to(torch.float8_e4m3fn)
            b = torch.randn(N, K, device="cuda").to(torch.float8_e4m3fn)  # transposed
            scale_a = torch.tensor(1.0, device="cuda")
            scale_b = torch.tensor(1.0, device="cuda")
            out = torch.empty(M, N, device="cuda", dtype=torch.float16)
            ms = bench(lambda: torch._scaled_mm(a, b.t(), scale_a, scale_b, out_dtype=torch.float16), warmup=10, iters=50)
            tflops = 2 * M * N * K / 1e12 / (ms / 1000)
            kv(f"  {label}", f"{ms:.4f} ms  {tflops:6.1f} TFLOPS")
            del a, b, out
            torch.cuda.empty_cache()
    except Exception as e:
        kv("FP8 GEMM", f"[{e}]")

    # =========================================================================
    # 11. ATTENTION-SPECIFIC BENCHMARKS
    # =========================================================================
    section("ATTENTION BENCHMARKS (torch SDPA)")
    # Shapes: (batch, heads, seqlen, head_dim)
    attn_shapes = [
        (1,  32, 512,   128, "bs=1  heads=32 seq=512   dim=128"),
        (1,  32, 4096,  128, "bs=1  heads=32 seq=4K    dim=128"),
        (1,  32, 16384, 128, "bs=1  heads=32 seq=16K   dim=128"),
        (1,  32, 32768, 128, "bs=1  heads=32 seq=32K   dim=128"),
        (8,  32, 512,   128, "bs=8  heads=32 seq=512   dim=128"),
        (8,  32, 4096,  128, "bs=8  heads=32 seq=4K    dim=128"),
    ]
    for B, H, S, D, label in attn_shapes:
        try:
            q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
            k = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
            v = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
            ms = bench(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True),
                       warmup=5, iters=20)
            # attn flops: 4 * B * H * S^2 * D (QK + softmax + AV)
            flops = 4 * B * H * S * S * D / 1e12
            tflops = flops / (ms / 1000)
            kv(f"  {label}", f"{ms:.4f} ms  {tflops:.2f} TFLOPS")
            del q, k, v
        except Exception as e:
            kv(f"  {label}", f"[{e}]")
        torch.cuda.empty_cache()

    # =========================================================================
    # 12. ELEMENTWISE / POINTWISE OPS (memory-bound bottlenecks)
    # =========================================================================
    section("ELEMENTWISE / REDUCTION OPS (memory-bound)")
    x4k = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    x2M = torch.randn(2 * 2**20, device="cuda", dtype=torch.float16)

    for label, fn in [
        ("Softmax     [4096x4096] FP16",   lambda: torch.softmax(x4k, dim=-1)),
        ("RMSNorm     [4096x4096] FP16",   lambda: x4k / (x4k.pow(2).mean(-1, keepdim=True) + 1e-6).sqrt()),
        ("SiLU        [4096x4096] FP16",   lambda: torch.nn.functional.silu(x4k)),
        ("GELU        [4096x4096] FP16",   lambda: torch.nn.functional.gelu(x4k)),
        ("Elementwise add [4096x4096]",    lambda: x4k + x4k),
        ("Elementwise mul [4096x4096]",    lambda: x4k * x4k),
        ("Sum reduce  [4096x4096] dim=-1", lambda: x4k.sum(dim=-1)),
        ("Max reduce  [4096x4096] dim=-1", lambda: x4k.max(dim=-1).values),
        ("Topk k=64   [4096x4096] dim=-1", lambda: torch.topk(x4k, 64, dim=-1)),
        ("Sort        [4096x4096] dim=-1", lambda: x4k.sort(dim=-1)),
        ("Gather      [2M] FP16",          lambda: x2M[torch.randint(0, x2M.shape[0], (x2M.shape[0],), device="cuda")]),
    ]:
        try:
            ms = bench(fn, warmup=5, iters=30)
            kv(f"  {label}", f"{ms:.4f} ms")
        except Exception as e:
            kv(f"  {label}", f"[{e}]")

    # LayerNorm
    ln = torch.nn.LayerNorm(4096).cuda().half()
    x_ln = torch.randn(2048, 4096, device="cuda", dtype=torch.float16)
    kv("  LayerNorm   [2048x4096] FP16", f"{bench(lambda: ln(x_ln)):.4f} ms")

    # =========================================================================
    # 13. KERNEL LAUNCH OVERHEAD
    # =========================================================================
    section("KERNEL LAUNCH OVERHEAD")
    dummy = torch.ones(1, device="cuda")
    # Sequential launches
    ms_seq = bench(lambda: dummy + dummy, warmup=20, iters=1000)
    kv("Single kernel launch overhead", f"{ms_seq:.5f} ms  ({ms_seq*1000:.2f} us)")

    # Back-to-back launches (pipeline depth)
    def many_launches():
        x = dummy
        for _ in range(10):
            x = x + dummy
        return x
    ms_chain = bench(many_launches, warmup=10, iters=200)
    kv("10 chained launches", f"{ms_chain:.5f} ms  ({ms_chain/10*1000:.2f} us avg)")

    # CUDA graph capture benefit
    try:
        g = torch.cuda.CUDAGraph()
        x = dummy.clone()
        with torch.cuda.graph(g):
            for _ in range(10):
                x = x + dummy
        ms_graph = bench(g.replay, warmup=10, iters=500)
        kv("10 ops via CUDA graph replay", f"{ms_graph:.5f} ms  (vs {ms_chain:.5f} ms eager)")
        kv("  CUDA graph speedup (10 ops)", f"{ms_chain/ms_graph:.1f}x")
        del g
    except Exception as e:
        kv("CUDA graph", f"[{e}]")

    # =========================================================================
    # 14. CONCURRENCY & OCCUPANCY
    # =========================================================================
    section("OCCUPANCY ESTIMATES (for common kernel configs)")
    try:
        import triton
        import triton.language as tl

        @triton.jit
        def occupancy_probe_kernel(out, N: tl.constexpr, BLOCK: tl.constexpr, USE_SMEM: tl.constexpr):
            pid = tl.program_id(0)
            if USE_SMEM:
                buf = tl.zeros([BLOCK], dtype=tl.float32)
                buf = buf + 1.0
            else:
                buf = tl.full([BLOCK], 1.0, dtype=tl.float32)
            tl.store(out + pid * BLOCK + tl.arange(0, BLOCK), buf)

        for block_size in [64, 128, 256, 512, 1024]:
            try:
                n = sm_count * 8 * block_size
                out = torch.empty(n, device="cuda", dtype=torch.float32)
                fn = lambda: occupancy_probe_kernel[(n // block_size,)](out, n, block_size, True)
                ms = bench(fn, warmup=3, iters=10)
                kv(f"  Triton block={block_size:4d}, smem, {sm_count*8} blocks", f"{ms:.4f} ms")
            except Exception as e2:
                kv(f"  Triton block={block_size}", f"[{e2}]")
    except Exception as e:
        kv("Occupancy probe", f"[{e}]")

    # =========================================================================
    # 15. WARP-LEVEL PRIMITIVES
    # =========================================================================
    section("WARP-LEVEL PRIMITIVES (Triton)")
    try:
        import triton
        import triton.language as tl

        @triton.jit
        def warp_reduce_kernel(out, N: tl.constexpr, BLOCK: tl.constexpr):
            pid  = tl.program_id(0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            x    = tl.load(out + offs).to(tl.float32)
            # warp reduction
            x = tl.sum(x, axis=0)
            if pid == 0:
                tl.store(out, x)

        n_el = sm_count * 128 * 32  # enough to saturate
        buf  = torch.ones(n_el, device="cuda", dtype=torch.float32)
        fn   = lambda: warp_reduce_kernel[(n_el // 32,)](buf, n_el, 32)
        ms   = bench(fn, warmup=5, iters=50)
        kv("Warp reduce (sum, BLOCK=32)", f"{ms:.4f} ms  (SM12 warp shuffle ok)")
    except Exception as e:
        kv("Warp reduce kernel", f"[{e}]")

    # =========================================================================
    # 16. SPARSE / TOPK ops (critical for MiniCPM-SALA sparse attention)
    # =========================================================================
    section("SPARSE & TOPK OPS (MiniCPM-SALA specific)")
    for N, K, label in [
        (4096,  64, "topk k=64  from N=4K  (pool output)"),
        (4096, 128, "topk k=128 from N=4K"),
        (16384, 64, "topk k=64  from N=16K"),
        (16384,128, "topk k=128 from N=16K"),
        (65536, 64, "topk k=64  from N=64K (128K seqlen / block64)"),
        (65536,128, "topk k=128 from N=64K"),
    ]:
        try:
            batch = 32
            x = torch.randn(batch, N, device="cuda", dtype=torch.float16)
            ms = bench(lambda: torch.topk(x, K, dim=-1), warmup=5, iters=50)
            kv(f"  {label}", f"{ms:.4f} ms  (batch={batch})")
            del x
        except Exception as e:
            kv(f"  {label}", f"[{e}]")
        torch.cuda.empty_cache()

    # Sparse scatter/gather (index_select, used in sparse attn)
    for seq_len, topk in [(4096, 64), (32768, 128), (131072, 128)]:
        try:
            # Simulate selecting topk KV vectors from full sequence
            kv_cache = torch.randn(seq_len, 128, device="cuda", dtype=torch.float16)
            idx      = torch.randint(0, seq_len, (topk,), device="cuda")
            ms = bench(lambda: kv_cache[idx], warmup=5, iters=50)
            kv(f"  index_select seq={seq_len:<7} topk={topk}", f"{ms:.4f} ms")
            del kv_cache, idx
        except Exception as e:
            kv(f"  index_select seq={seq_len}", f"[{e}]")
        torch.cuda.empty_cache()

    # =========================================================================
    # 17. QUANTIZATION SUPPORT
    # =========================================================================
    section("QUANTIZATION SUPPORT")
    kv("FP8 e4m3 dtype available", hasattr(torch, "float8_e4m3fn"))
    kv("FP8 e5m2 dtype available", hasattr(torch, "float8_e5m2"))
    kv("INT8 quantization",        "Yes (SM >= 6.1)")
    kv("INT4 (custom kernels)",    "Yes via bitsandbytes / GPTQ / AWQ")

    # Test FP8 cast overhead
    try:
        x = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
        ms_cast = bench(lambda: x.to(torch.float8_e4m3fn), warmup=5, iters=50)
        kv("FP8 e4m3 cast [4096x4096] FP16->FP8", f"{ms_cast:.4f} ms")
        x8 = x.to(torch.float8_e4m3fn)
        ms_back = bench(lambda: x8.to(torch.float16), warmup=5, iters=50)
        kv("FP8 e4m3 cast [4096x4096] FP8->FP16", f"{ms_back:.4f} ms")
        del x, x8
    except Exception as e:
        kv("FP8 cast benchmark", f"[{e}]")

    # =========================================================================
    # 18. ASYNC / PIPELINE FEATURES
    # =========================================================================
    section("ASYNC & PIPELINE FEATURES")
    # CUDA streams
    try:
        s1 = torch.cuda.Stream()
        s2 = torch.cuda.Stream()
        a  = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
        b  = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
        def dual_stream():
            with torch.cuda.stream(s1):
                c = torch.matmul(a, b)
            with torch.cuda.stream(s2):
                d = torch.matmul(b, a)
            torch.cuda.synchronize()
        ms_dual = bench(dual_stream, warmup=5, iters=30)
        ms_seq_gemm = bench(lambda: (torch.matmul(a,b), torch.matmul(b,a)), warmup=5, iters=30)
        kv("Dual-stream 2x GEMM[1K]", f"{ms_dual:.4f} ms  vs sequential {ms_seq_gemm:.4f} ms")
        del s1, s2, a, b
    except Exception as e:
        kv("Dual-stream test", f"[{e}]")

    # Non-blocking host<->device
    try:
        cpu_buf = torch.randn(64 * 2**20 // 4, dtype=torch.float32).pin_memory()
        gpu_buf = torch.empty_like(cpu_buf, device="cuda")
        ms_async = bench(lambda: gpu_buf.copy_(cpu_buf, non_blocking=True), warmup=3, iters=10)
        kv("Async H2D copy non_blocking (64MB)", f"{ms_async:.4f} ms")
        del cpu_buf, gpu_buf
    except Exception as e:
        kv("Async H2D copy", f"[{e}]")

    # =========================================================================
    # 19. NVIDIA-SMI (full)
    # =========================================================================
    section("NVIDIA-SMI FULL")
    for l in _run(["nvidia-smi"]).split("\n"):
        lines.append("  " + l)

    section("NVIDIA-SMI DETAILED QUERY")
    fields = [
        "name", "driver_version", "compute_cap", "pci.bus_id",
        "memory.total", "memory.free", "memory.used",
        "clocks.max.sm", "clocks.max.memory", "clocks.max.gr",
        "clocks.current.sm", "clocks.current.memory",
        "clocks.applications.gr", "clocks.applications.memory",
        "power.limit", "power.draw", "power.default_limit", "power.max_limit",
        "temperature.gpu", "temperature.memory",
        "fan.speed",
        "utilization.gpu", "utilization.memory",
        "ecc.mode.current", "mig.mode.current",
        "pcie.link.gen.current", "pcie.link.width.current",
        "pcie.link.gen.max", "pcie.link.width.max",
    ]
    rows = _run(["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv"]).split("\n")
    hdrs = [h.strip() for h in rows[0].split(",")]
    vals = [v.strip() for v in rows[1].split(",")] if len(rows) > 1 else []
    for h, v in zip(hdrs, vals):
        kv(f"  {h}", v)

    # PCIe gen/width matters for CPU<->GPU overlap
    section("PCIE DETAILS")
    for f in ["pcie.link.gen.current", "pcie.link.width.current",
              "pcie.link.gen.max", "pcie.link.width.width.max"]:
        kv(f"  {f}", smi_q(f))

    # =========================================================================
    # 20. OPTIMIZATION CHEATSHEET (summary)
    # =========================================================================
    section("OPTIMIZATION CHEATSHEET (Blackwell SM12.0 — RTX PRO 6000)")
    smem = prop(props, "shared_memory_per_multiprocessor", fallback=102400)
    smem_kb = smem // 1024 if isinstance(smem, int) else 100
    threads_sm = prop(props, "max_threads_per_multi_processor", fallback=1536)
    warps_sm   = threads_sm // 32 if isinstance(threads_sm, int) else 48

    notes = [
        ("VRAM",                  f"{props.total_memory/2**30:.0f} GB — fit model + large KV cache easily"),
        ("mem-fraction-static",   "Try 0.85-0.90 (huge VRAM vs H100 80GB)"),
        ("chunked-prefill-size",  "Try 32768 or 65536 (more VRAM = bigger chunks OK)"),
        ("dtype recommendation",  "FP8 e4m3 for weights+activations (Blackwell-native, 2x FP16 TC throughput)"),
        ("Sparse 2:4",            "Enabled on Blackwell (2x compute for eligible weight mats)"),
        ("TMA",                   "Available SM>=9.0 — currently DISABLED in tilelang, consider enabling"),
        ("Warp specialization",   "Available SM>=9.0 — producer/consumer warp pattern (async prefetch)"),
        ("Thread block clusters", "Available SM>=9.0 — distributed shared mem across SM cluster"),
        ("Shared mem / SM",       f"{smem_kb} KB  |  opt-in per block: {prop(props, 'shared_memory_per_block_optin', fmt=lambda v: f'{v//1024} KB')}"),
        ("Max warps / SM",        f"{warps_sm}  =>  up to {warps_sm * sm_count:,} concurrent warps"),
        ("L2 cache",              f"{prop(props, 'L2_cache_size', fmt=lambda v: f'{v//2**20} MB')} — cache reuse key for decode"),
        ("PCIe",                  "Check gen/width above — affects CPU<->GPU KV offload speed"),
        ("FP8 GEMM tile",         "16x16x32 (2x FP16 tile capacity)"),
        ("Kernel launch",         "~few us overhead — batch small ops or use CUDA graphs"),
        ("Triton SM12 support",   f"SM12.0 — verify triton>=3.5 compiles correctly (see above)"),
        ("tilelang TMA flag",     "TL_DISABLE_TMA_LOWER=True currently — flip to False and test"),
        ("Memory BW (theoretical)",f"{mem_bw_gb_s:.0f} GB/s (GDDR7 384-bit)  vs measured: see section 9"),
        ("Arith intensity",       f"FP16 TC crossover ~{int((total_cuda_cores*4*clock_ghz*1e12)/(mem_bw_gb_s*1e9))} FLOP/byte"),
    ]
    for k, v in notes:
        kv(f"  {k}", v)

    # =========================================================================
    output = "\n".join(lines)
    print(output)
    return output


@app.local_entrypoint()
def main():
    result = probe.remote()
    out_path = "hardware_report.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(result)
    print(f"\nReport saved to {out_path}")
