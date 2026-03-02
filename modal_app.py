"""
SOAR 2026 Competition - Modal Labs Development Environment

Development on Modal H100 GPUs, submission for SOAR RTX PRO (6000) GPUs.

Workflow:
1. Edit code locally in sglang/sgl-kernel/ or submission/
2. Sync to Modal: modal run modal_app.py --cmd sync
3. Build and test on Modal H100
4. Create submission package: modal run modal_app.py --cmd package
5. Submit .tar.gz to SOAR platform
"""

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import modal

# Modal App
app = modal.App("soar-2026")

# Base image matching SOAR platform
# Note: gptqmodel requires GPU to build, so install at runtime if needed
soar_image = (
    modal.Image.from_registry("ghcr.io/openbmb/soar-toolkit:latest")
    .apt_install("git", "wget", "curl", "htop", "vim", "ccache")
    .pip_install("httpx", "tqdm", "rich", "bitsandbytes")
    # Note: flash-attn and gptqmodel require GPU, installed at runtime
    .run_commands(
        [
            "git clone https://github.com/OpenBMB/SOAR-Toolkit.git /opt/SOAR",
        ]
    )
)

# Image with GPU-dependent packages pre-built
soar_image_with_flash = (
    soar_image
    .run_commands(
        [
            # Install flash-attn and quantization libraries with GPU
            "pip install flash-attn --no-build-isolation",
            "pip install gptqmodel",
            "pip install autoawq",  # AWQ quantization (SGLang compatible)
        ],
        gpu="H100",  # GPU required for compilation
    )
)

# Persistent volumes
model_volume = modal.Volume.from_name("soar-models", create_if_missing=True)
workspace_volume = modal.Volume.from_name("soar-workspace", create_if_missing=True)


# ============================================================================
# Helper Functions
# ============================================================================
def wait_for_server(host="localhost", port=30000, timeout=600, server_proc=None):
    """Wait for SGLang server to be ready."""
    import httpx

    start = time.time()
    while time.time() - start < timeout:
        # Check if server process died
        if server_proc and server_proc.poll() is not None:
            print(f"Server process died with code {server_proc.returncode}")
            # Try to get any remaining output
            try:
                remaining = server_proc.stdout.read()
                if remaining:
                    print(f"Server output: {remaining[:2000]}")
            except:
                pass
            raise RuntimeError(f"Server process died with code {server_proc.returncode}")

        try:
            resp = httpx.get(f"http://{host}:{port}/health", timeout=5)
            if resp.status_code == 200:
                print(f"Server ready in {time.time() - start:.1f}s")
                return True
        except Exception:
            pass
        time.sleep(5)
        elapsed = time.time() - start
        if int(elapsed) % 30 == 0:
            print(f"Waiting for server... ({elapsed:.0f}s)")

    raise TimeoutError(f"Server not ready after {timeout}s")


def start_sglang_server(
    model_path: str = "/models/MiniCPM-SALA",
    port: int = 30000,
    extra_args: str = "",
    quantization: str = None,
):
    """Start SGLang server as background process."""
    # Default args for MiniCPM-SALA model (from working baseline)
    # - minicpm_flashinfer: required for SALA hybrid sparse+linear attention
    # - skip-server-warmup: prevents OOM during initialization
    # - dense-as-sparse: enables sparse attention optimization
    default_args = (
        "--disable-radix-cache "
        "--attention-backend minicpm_flashinfer "
        "--chunked-prefill-size 8192 "
        "--mem-fraction-static 0.70 "
        "--dense-as-sparse "
        "--skip-server-warmup"
    )

    cmd = [
        "python", "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--host", "0.0.0.0",
        "--port", str(port),
    ] + default_args.split()

    if quantization:
        cmd.extend(["--quantization", quantization, "--dtype", "float16"])

    if extra_args:
        cmd.extend(extra_args.split())

    print(f"Starting SGLang: {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    return proc


def run_streaming(cmd, env=None, label="", timeout=None):
    """Run a subprocess with real-time output streaming."""
    captured = []
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    try:
        for line in proc.stdout:
            line = line.rstrip("\n")
            prefix = f"[{label}] " if label else ""
            print(f"{prefix}{line}", flush=True)
            captured.append(line)
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        captured.append(f"[TIMEOUT after {timeout}s]")

    return "\n".join(captured), proc.returncode


def generate_speed_data(output_path: str, num_samples: int = 50):
    """Generate synthetic speed test data matching competition distribution."""
    import random

    def random_input_length():
        r = random.random()
        if r < 0.25:
            return random.randint(100, 4000)
        elif r < 0.35:
            return random.randint(4000, 16000)
        elif r < 0.50:
            return random.randint(16000, 32000)
        elif r < 0.85:
            return random.randint(32000, 128000)
        else:
            return random.randint(128000, 160000)

    def random_output_length():
        r = random.random()
        if r < 0.35:
            return random.randint(50, 512)
        elif r < 0.60:
            return random.randint(512, 2000)
        elif r < 0.70:
            return random.randint(2000, 4000)
        elif r < 0.85:
            return random.randint(4000, 16000)
        else:
            return random.randint(16000, 32000)

    samples = []
    for _ in range(num_samples):
        input_len = random_input_length()
        output_len = random_output_length()
        samples.append({
            "question": "x " * (input_len // 2),
            "model_response": "y " * (output_len // 2),
            "_input_len": input_len,
            "_output_len": output_len,
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for sample in samples:
            # Remove internal fields before writing
            write_sample = {k: v for k, v in sample.items() if not k.startswith("_")}
            f.write(json.dumps(write_sample) + "\n")

    input_lens = [s["_input_len"] for s in samples]
    output_lens = [s["_output_len"] for s in samples]
    print(f"Generated {num_samples} samples to {output_path}")
    print(f"  Input:  min={min(input_lens)}, max={max(input_lens)}, avg={sum(input_lens)//len(input_lens)}")
    print(f"  Output: min={min(output_lens)}, max={max(output_lens)}, avg={sum(output_lens)//len(output_lens)}")

    return output_path


# ============================================================================
# Model Download
# ============================================================================
@app.function(
    image=soar_image,
    gpu="H100",
    timeout=3600,
    volumes={"/models": model_volume},
)
def download_model():
    """Download MiniCPM-SALA model to persistent volume (run once)."""
    model_path = Path("/models/MiniCPM-SALA")
    if model_path.exists() and any(model_path.iterdir()):
        print("Model already exists, skipping download")
        return str(model_path)

    env = os.environ.copy()
    env["HF_HOME"] = "/models/.cache"

    print("Downloading MiniCPM-SALA from HuggingFace...")
    subprocess.run(
        [
            "huggingface-cli", "download",
            "OpenBMB/MiniCPM-SALA",
            "--local-dir", "/models/MiniCPM-SALA",
        ],
        env=env,
        check=True,
    )

    model_volume.commit()
    return "Model downloaded to /models/MiniCPM-SALA"


# ============================================================================
# Workspace Management
# ============================================================================
@app.function(
    image=soar_image,
    volumes={"/workspace": workspace_volume},
    timeout=300,
)
def check_workspace():
    """Check what's currently in the workspace volume."""
    print("Workspace contents:")
    for root, dirs, files in os.walk("/workspace"):
        level = root.replace("/workspace", "").count(os.sep)
        indent = " " * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        if level < 2:
            subindent = " " * 2 * (level + 1)
            for file in files[:10]:
                print(f"{subindent}{file}")
            if len(files) > 10:
                print(f"{subindent}... and {len(files) - 10} more files")


# ============================================================================
# Build & Install Custom Kernel
# ============================================================================
@app.function(
    image=soar_image,
    gpu="H100",
    timeout=7200,
    volumes={"/models": model_volume, "/workspace": workspace_volume},
)
def build_kernel(
    kernel_path: str = "/workspace/sglang/sgl-kernel",
    for_submission: bool = False,
):
    """
    Build custom sgl-kernel wheel.

    Args:
        kernel_path: Path to sgl-kernel source
        for_submission: If True, patch CMakeLists.txt for competition compliance
    """
    import glob

    # Find kernel path
    possible_paths = [
        "/workspace/sglang/sgl-kernel",
        "/workspace/sgl-kernel",
        "/workspace/soar-optimization/sglang/sgl-kernel",
    ]

    for p in possible_paths:
        if os.path.exists(p):
            kernel_path = p
            break
    else:
        return {"success": False, "error": f"Kernel path not found. Did you sync your code?"}

    os.chdir(kernel_path)
    print(f"Building kernel from: {kernel_path}")

    # Patch CMakeLists.txt for submission if needed
    if for_submission:
        cmake_path = Path(kernel_path) / "CMakeLists.txt"
        if cmake_path.exists():
            content = cmake_path.read_text()
            # Comment out third-party installs
            patterns = [
                r'install\(TARGETS deep_gemm_cpp[^\)]*\)',
                r'install\(DIRECTORY "\$\{repo-deepgemm_SOURCE_DIR\}[^\)]*\)',
                r'install\(DIRECTORY "\$\{repo-triton_SOURCE_DIR\}[^\)]*\)',
                r'install\(DIRECTORY "\$\{FLASH_ATTN_CUTE_DST\}[^\)]*\)',
                r'install\(FILES \$\{CMAKE_CURRENT_BINARY_DIR\}/deepgemm_pkg_init[^\)]*\)',
                r'install\(DIRECTORY "\$\{repo-cutlass_SOURCE_DIR\}/include/cute/"[^\)]*\)',
                r'install\(DIRECTORY "\$\{repo-cutlass_SOURCE_DIR\}/include/cutlass/"[^\)]*\)',
            ]
            for pattern in patterns:
                content = re.sub(pattern, lambda m: f"# SOAR_DISABLED: {m.group(0)}", content, flags=re.MULTILINE)
            cmake_path.write_text(content)
            print("Patched CMakeLists.txt for competition compliance")

    # Clean and build
    subprocess.run(["rm", "-rf", "build/", "dist/"], check=False)
    subprocess.run(["pip", "install", "build", "wheel", "scikit-build-core", "cmake", "ninja"], check=True)

    print("\nBuilding wheel...")
    result = subprocess.run(["python", "-m", "build", "--wheel", "--no-isolation"])

    if result.returncode != 0:
        return {"success": False, "error": f"Build failed with code {result.returncode}"}

    # Find wheel
    wheels = glob.glob("dist/*.whl")
    if not wheels:
        return {"success": False, "error": "No wheel produced"}

    wheel_path = wheels[0]
    print(f"\nBuilt: {wheel_path}")

    # Validate wheel contents
    import zipfile
    with zipfile.ZipFile(wheel_path, "r") as zf:
        top_level = set(name.split("/")[0] for name in zf.namelist() if name.split("/")[0])
        violations = [d for d in top_level if not d.startswith("sgl_kernel") and not d.endswith(".dist-info")]

        if violations:
            print(f"\n⚠️  WARNING: Disallowed directories: {violations}")
            if for_submission:
                print("This will FAIL competition validation!")

    # Install
    subprocess.run(["pip", "install", "--force-reinstall", wheel_path], check=True)
    workspace_volume.commit()

    return {"success": True, "wheel": wheel_path, "violations": violations if violations else None}


# ============================================================================
# Correctness Evaluation
# ============================================================================
@app.function(
    image=soar_image,
    gpu="H100",
    timeout=7200,
    volumes={"/models": model_volume, "/workspace": workspace_volume},
)
def run_eval(
    num_samples: int = None,
    concurrency: int = 32,
    extra_args: str = "",
    use_custom_kernel: bool = True,
    quantization: str = None,
):
    """
    Run correctness evaluation on public dataset.

    Must maintain >97% accuracy to not be eliminated.
    """
    import glob
    import threading

    # Install custom kernel if available
    if use_custom_kernel:
        for wheel_dir in ["/workspace/sglang/sgl-kernel/dist", "/workspace/wheels"]:
            if os.path.exists(wheel_dir):
                wheels = glob.glob(f"{wheel_dir}/*.whl")
                if wheels:
                    print(f"Installing custom kernel: {wheels[0]}")
                    subprocess.run(["pip", "install", "--force-reinstall", wheels[0]], check=True)
                    break

    proc = start_sglang_server(extra_args=extra_args, quantization=quantization)

    # Stream server logs in background so we can see startup errors
    def stream_server_logs():
        for line in proc.stdout:
            print(f"[server] {line.rstrip()}", flush=True)
    log_thread = threading.Thread(target=stream_server_logs, daemon=True)
    log_thread.start()

    try:
        wait_for_server(server_proc=proc)

        cmd = [
            "python3", "/opt/SOAR/eval_model.py",
            "--api_base", "http://localhost:30000",
            "--model_path", "/models/MiniCPM-SALA",
            "--data_path", "/opt/SOAR/eval_dataset/perf_public_set.jsonl",
            "--concurrency", str(concurrency),
        ]
        if num_samples:
            cmd.extend(["--num_samples", str(num_samples)])

        print(f"Running: {' '.join(cmd)}")
        output, returncode = run_streaming(cmd, label="eval")

        # Parse accuracy
        acc_match = re.search(r"(?:Average Score|Accuracy|overall_accuracy):\s*([0-9.]+)%?", output, re.IGNORECASE)
        accuracy = float(acc_match.group(1)) if acc_match else None

        if accuracy is not None:
            # Calculate relative score (raw_accuracy / baseline * 100)
            # Baseline is 80 points per toolkit
            baseline = 80.0
            relative_score = (accuracy / baseline) * 100

            print(f"\n{'=' * 50}")
            print(f"  Raw Accuracy:    {accuracy:.2f}%")
            print(f"  Baseline:        {baseline:.0f}%")
            print(f"  Relative Score:  {relative_score:.2f}%")

            if relative_score <= 97:
                print(f"  Status:          ⚠️  ELIMINATED (relative < 97%)")
                print(f"  C Coefficient:   0")
            elif relative_score <= 98:
                print(f"  Status:          ✓ PASSING")
                print(f"  C Coefficient:   0.92")
            elif relative_score <= 99:
                print(f"  Status:          ✓ PASSING")
                print(f"  C Coefficient:   0.96")
            else:
                print(f"  Status:          ✓ PASSING")
                print(f"  C Coefficient:   1.0")
            print(f"{'=' * 50}")

        return {"stdout": output, "returncode": returncode, "accuracy": accuracy}

    finally:
        proc.terminate()
        proc.wait()


# ============================================================================
# Speed Benchmark
# ============================================================================
@app.function(
    image=soar_image,
    gpu="H100",
    timeout=14400,  # 4 hours
    volumes={"/models": model_volume, "/workspace": workspace_volume},
)
def run_benchmark(
    num_samples: int = 50,
    levels: str = "S1,S8,Smax",
    extra_args: str = "",
    use_custom_kernel: bool = True,
    quantization: str = None,
):
    """
    Run speed benchmark at specified concurrency levels.

    Scoring formula:
    - S1 (--max-concurrent 1): 40% weight
    - S8 (--max-concurrent 8): 30% weight
    - Smax (no limit): 30% weight

    Bounty targets: S1=420s, S8=420s, Smax=720s
    """
    import glob
    import threading

    # Install custom kernel if available
    if use_custom_kernel:
        for wheel_dir in ["/workspace/sglang/sgl-kernel/dist", "/workspace/wheels"]:
            if os.path.exists(wheel_dir):
                wheels = glob.glob(f"{wheel_dir}/*.whl")
                if wheels:
                    print(f"Installing custom kernel: {wheels[0]}")
                    subprocess.run(["pip", "install", "--force-reinstall", wheels[0]], check=True)
                    break

    # Generate test data
    speed_data_path = "/tmp/speed_test_data.jsonl"
    generate_speed_data(speed_data_path, num_samples)

    # Start server
    proc = start_sglang_server(extra_args=extra_args, quantization=quantization)

    # Stream server logs in background
    def stream_server_logs():
        for line in proc.stdout:
            print(f"[server] {line.rstrip()}", flush=True)
    log_thread = threading.Thread(target=stream_server_logs, daemon=True)
    log_thread.start()

    try:
        wait_for_server(server_proc=proc)

        results = {"S1": None, "S8": None, "Smax": None}
        levels_to_run = [l.strip() for l in levels.split(",")]

        for level in levels_to_run:
            if level not in results:
                print(f"Unknown level: {level}, skipping")
                continue

            print(f"\n{'=' * 60}")
            print(f"  BENCHMARK: {level}")
            concurrency_desc = {
                "S1": "--max-concurrent 1 (40% weight, bounty: 420s)",
                "S8": "--max-concurrent 8 (30% weight, bounty: 420s)",
                "Smax": "no limit (30% weight, bounty: 720s)",
            }
            print(f"  {concurrency_desc.get(level, '')}")
            print(f"{'=' * 60}\n")

            # Set environment for bench_serving.sh
            env = os.environ.copy()
            env_var = {"S1": "SPEED_DATA_S1", "S8": "SPEED_DATA_S8", "Smax": "SPEED_DATA_SMAX"}[level]
            env[env_var] = speed_data_path

            # Clear other env vars to run only this level
            for v in ["SPEED_DATA_S1", "SPEED_DATA_S8", "SPEED_DATA_SMAX"]:
                if v != env_var:
                    env.pop(v, None)

            output, returncode = run_streaming(
                ["bash", "/opt/SOAR/bench_serving.sh", "http://localhost:30000"],
                env=env,
                label=level,
            )

            # Parse duration
            duration_match = re.search(r"Benchmark duration \(s\):\s+([0-9.]+)", output)
            duration = float(duration_match.group(1)) if duration_match else None
            results[level] = duration

            if duration:
                bounty = {"S1": 420, "S8": 420, "Smax": 720}[level]
                status = "✓ BEATS BOUNTY" if duration <= bounty else f"(need: {bounty}s)"
                print(f"\n→ {level}: {duration:.2f}s {status}")

        # Summary
        print(f"\n{'=' * 60}")
        print("  BENCHMARK SUMMARY")
        print(f"{'=' * 60}")
        for level, duration in results.items():
            if duration:
                bounty = {"S1": 420, "S8": 420, "Smax": 720}[level]
                weight = {"S1": "40%", "S8": "30%", "Smax": "30%"}[level]
                status = "✓" if duration <= bounty else "✗"
                print(f"  {level}: {duration:.2f}s [{weight}] {status}")

        return results

    finally:
        proc.terminate()
        proc.wait()


# ============================================================================
# Full Baseline (Eval + Benchmark)
# ============================================================================
@app.function(
    image=soar_image,
    gpu="H100",
    timeout=18000,  # 5 hours
    volumes={"/models": model_volume, "/workspace": workspace_volume},
)
def run_baseline(
    num_eval_samples: int = None,
    num_speed_samples: int = 50,
    extra_args: str = "",
    use_custom_kernel: bool = False,
    quantization: str = None,
    quick: bool = False,
):
    """
    Run full baseline: correctness eval + speed benchmark at all levels.

    Args:
        quick: If True, use reduced samples for faster iteration
    """
    import glob
    import threading

    if quick:
        num_eval_samples = num_eval_samples or 20
        num_speed_samples = 20
        print(f"[QUICK MODE] {num_eval_samples} eval samples, {num_speed_samples} speed samples\n")

    # Install custom kernel if requested
    if use_custom_kernel:
        for wheel_dir in ["/workspace/sglang/sgl-kernel/dist", "/workspace/wheels"]:
            if os.path.exists(wheel_dir):
                wheels = glob.glob(f"{wheel_dir}/*.whl")
                if wheels:
                    print(f"Installing custom kernel: {wheels[0]}")
                    subprocess.run(["pip", "install", "--force-reinstall", wheels[0]], check=True)
                    break

    # Generate speed data
    speed_data_path = "/tmp/baseline_speed.jsonl"
    generate_speed_data(speed_data_path, num_speed_samples)

    # Start server
    proc = start_sglang_server(extra_args=extra_args, quantization=quantization)

    # Stream server logs in background
    def stream_logs():
        for line in proc.stdout:
            print(f"[server] {line.rstrip()}", flush=True)

    log_thread = threading.Thread(target=stream_logs, daemon=True)
    log_thread.start()

    results = {
        "accuracy": None,
        "c_coefficient": None,
        "speed": {"S1": None, "S8": None, "Smax": None},
    }

    try:
        wait_for_server()

        # Step 1: Correctness
        print(f"\n{'=' * 60}")
        print("  STEP 1: CORRECTNESS EVALUATION")
        print(f"{'=' * 60}\n")

        eval_cmd = [
            "python3", "/opt/SOAR/eval_model.py",
            "--api_base", "http://localhost:30000",
            "--model_path", "/models/MiniCPM-SALA",
            "--data_path", "/opt/SOAR/eval_dataset/perf_public_set.jsonl",
            "--concurrency", "32",
        ]
        if num_eval_samples:
            eval_cmd.extend(["--num_samples", str(num_eval_samples)])

        eval_output, _ = run_streaming(eval_cmd, label="eval")

        acc_match = re.search(r"(?:Average Score|Accuracy|overall_accuracy):\s*([0-9.]+)%?", eval_output, re.IGNORECASE)
        accuracy = float(acc_match.group(1)) if acc_match else None
        results["accuracy"] = accuracy

        # Calculate relative score and C coefficient
        # Relative score = (raw_accuracy / baseline) * 100, baseline = 80
        if accuracy is not None:
            baseline = 80.0
            relative_score = (accuracy / baseline) * 100
            results["relative_score"] = relative_score

            if relative_score <= 97:
                results["c_coefficient"] = 0
            elif relative_score <= 98:
                results["c_coefficient"] = 0.92
            elif relative_score <= 99:
                results["c_coefficient"] = 0.96
            else:
                results["c_coefficient"] = 1.0

        # Steps 2-4: Speed benchmarks
        for step, (level, env_var) in enumerate([
            ("S1", "SPEED_DATA_S1"),
            ("S8", "SPEED_DATA_S8"),
            ("Smax", "SPEED_DATA_SMAX"),
        ], start=2):
            print(f"\n{'=' * 60}")
            print(f"  STEP {step}: SPEED BENCHMARK [{level}]")
            print(f"{'=' * 60}\n")

            env = os.environ.copy()
            for v in ["SPEED_DATA_S1", "SPEED_DATA_S8", "SPEED_DATA_SMAX"]:
                env.pop(v, None)
            env[env_var] = speed_data_path

            output, _ = run_streaming(
                ["bash", "/opt/SOAR/bench_serving.sh", "http://localhost:30000"],
                env=env,
                label=level,
            )

            duration_match = re.search(r"Benchmark duration \(s\):\s+([0-9.]+)", output)
            results["speed"][level] = float(duration_match.group(1)) if duration_match else None

        # Summary
        print(f"\n{'╔' + '═' * 58 + '╗'}")
        print(f"{'║'}{'BASELINE RESULTS'.center(58)}{'║'}")
        print(f"{'╠' + '═' * 58 + '╣'}")

        if results["accuracy"] is not None:
            c = results["c_coefficient"]
            rel = results.get("relative_score", 0)
            status = "⚠️ ELIMINATED" if c == 0 else "✓ PASSING"
            print(f"{'║'}  Raw Accuracy:   {results['accuracy']:.2f}%".ljust(59) + "║")
            print(f"{'║'}  Relative Score: {rel:.2f}% (baseline=80)".ljust(59) + "║")
            print(f"{'║'}  Status: {status}  (C = {c})".ljust(59) + "║")
        else:
            print(f"{'║'}  Accuracy: PARSE ERROR".ljust(59) + "║")

        print(f"{'║' + '─' * 58 + '║'}")

        bounties = {"S1": 420, "S8": 420, "Smax": 720}
        weights = {"S1": "40%", "S8": "30%", "Smax": "30%"}

        for level in ["S1", "S8", "Smax"]:
            dur = results["speed"][level]
            if dur:
                status = "✓ BOUNTY" if dur <= bounties[level] else f"need {bounties[level]}s"
                line = f"  {level}: {dur:.2f}s [{weights[level]}] {status}"
            else:
                line = f"  {level}: FAILED [{weights[level]}]"
            print(f"{'║'}{line}".ljust(59) + "║")

        print(f"{'╚' + '═' * 58 + '╝'}")

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"/workspace/baseline_results/{timestamp}"
        os.makedirs(results_dir, exist_ok=True)

        with open(f"{results_dir}/summary.json", "w") as f:
            json.dump({
                "timestamp": timestamp,
                **results,
                "config": {
                    "num_eval_samples": num_eval_samples,
                    "num_speed_samples": num_speed_samples,
                    "use_custom_kernel": use_custom_kernel,
                    "quantization": quantization,
                },
            }, f, indent=2)

        workspace_volume.commit()
        print(f"\nResults saved to: {results_dir}")

        return results

    finally:
        proc.terminate()
        proc.wait()


# ============================================================================
# Create Submission Package
# ============================================================================
@app.function(
    image=soar_image,
    gpu="H100",
    timeout=3600,
    volumes={"/models": model_volume, "/workspace": workspace_volume},
)
def create_submission(
    output_name: str = "submission",
    include_quantization: bool = False,
):
    """
    Create .tar.gz submission package for SOAR platform.

    Package structure:
    - prepare_env.sh (required)
    - prepare_model.sh (optional)
    - preprocess_model.py (for quantization)
    - sglang/python/ (custom sglang code)
    - wheels/*.whl (custom kernel wheel)
    """
    import glob
    import tarfile

    submission_dir = "/workspace/submission"
    output_dir = "/workspace/packages"
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(submission_dir):
        return {"success": False, "error": "No submission directory found. Sync first!"}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tarball_path = f"{output_dir}/{output_name}_{timestamp}.tar.gz"

    print(f"Creating submission package: {tarball_path}")

    with tarfile.open(tarball_path, "w:gz") as tar:
        # Add submission scripts
        for script in ["prepare_env.sh", "prepare_model.sh", "preprocess_model.py"]:
            script_path = f"{submission_dir}/{script}"
            if os.path.exists(script_path):
                tar.add(script_path, arcname=script)
                print(f"  Added: {script}")

        # Add custom sglang if present
        sglang_python = "/workspace/sglang/python"
        if os.path.exists(sglang_python):
            tar.add(sglang_python, arcname="sglang/python")
            print(f"  Added: sglang/python/")

        # Add wheel if present
        wheel_dir = "/workspace/sglang/sgl-kernel/dist"
        if os.path.exists(wheel_dir):
            wheels = glob.glob(f"{wheel_dir}/*.whl")
            if wheels:
                os.makedirs(f"{submission_dir}/wheels", exist_ok=True)
                for wheel in wheels:
                    wheel_name = os.path.basename(wheel)
                    tar.add(wheel, arcname=f"wheels/{wheel_name}")
                    print(f"  Added: wheels/{wheel_name}")

    # Check size
    size_mb = os.path.getsize(tarball_path) / (1024 * 1024)
    print(f"\nPackage size: {size_mb:.2f} MB")

    if size_mb > 2048:
        print("⚠️  WARNING: Package exceeds 2GB limit!")
    else:
        print("✓ Package is within 2GB limit")

    workspace_volume.commit()
    return {"success": True, "path": tarball_path, "size_mb": size_mb}


# ============================================================================
# Test GPTQ Quantization
# ============================================================================
@app.function(
    image=soar_image_with_flash,  # Uses image with flash-attn pre-built
    gpu="H100",
    timeout=14400,  # 4 hours for quantization
    volumes={"/models": model_volume, "/workspace": workspace_volume},
)
def test_gptq_quantization(
    calibration_samples: int = 256,  # GPTQModel recommends >= 256
    bits: int = 4,
    group_size: int = 128,
):
    """
    Test W4 quantization on H100.

    This tries multiple quantization methods (llm-compressor, gptqmodel, bitsandbytes)
    and runs a quick eval to verify accuracy.
    """
    import threading

    # Check pre-installed libraries (installed in Modal image)
    print("Checking quantization libraries...")

    try:
        import flash_attn
        print(f"  ✓ flash-attn {flash_attn.__version__} available")
    except ImportError:
        print("  ⚠ flash-attn not in image, installing (this takes 10-30 min)...")
        subprocess.run(
            ["pip", "install", "flash-attn", "--no-build-isolation", "--no-cache-dir"],
            capture_output=True,
        )

    # These should be pre-installed in the Modal image
    for lib in ["gptqmodel", "bitsandbytes", "awq"]:
        try:
            __import__(lib)
            print(f"  ✓ {lib} available")
        except ImportError:
            print(f"  Installing {lib}...")
            pkg = "autoawq" if lib == "awq" else lib
            subprocess.run(["pip", "install", pkg, "--no-cache-dir"], capture_output=True)

    # Run quantization (will try multiple methods with fallbacks)
    quant_output = "/workspace/quantized_model"
    print(f"\nQuantizing to {quant_output}...")

    quant_cmd = [
        "python3", "/workspace/submission/preprocess_model.py",
        "--input", "/models/MiniCPM-SALA",
        "--output", quant_output,
        "--mode", "gptq",
        "--bits", str(bits),
        "--group-size", str(group_size),
        "--calibration-samples", str(calibration_samples),
    ]

    result = subprocess.run(quant_cmd, capture_output=False)
    if result.returncode != 0:
        return {"success": False, "error": "All quantization methods failed"}

    # Detect which quantization method succeeded
    quant_method = "unknown"
    quant_config_path = f"{quant_output}/quantize_config.json"
    if os.path.exists(quant_config_path):
        with open(quant_config_path) as f:
            config = json.load(f)
            quant_method = config.get("quant_method", "unknown")

    print(f"\nQuantization method used: {quant_method}")

    # Determine server args based on quantization method
    base_args = (
        "--disable-radix-cache "
        "--attention-backend minicpm_flashinfer "
        "--chunked-prefill-size 8192 "
        "--mem-fraction-static 0.70 "
        "--dense-as-sparse "
        "--skip-server-warmup"
    )

    quant_args = ""
    if quant_method == "gptq":
        quant_args = "--quantization gptq_marlin --dtype float16"
    elif quant_method == "awq":
        quant_args = "--quantization awq_marlin --dtype float16"
    elif quant_method == "rtn":
        # RTN saves as float16, no special quantization args
        quant_args = ""

    server_args = f"{base_args} {quant_args}"

    # Start server with quantized model
    print(f"\nStarting server with quantized model...")
    print(f"Server args: {server_args}")

    proc = subprocess.Popen(
        [
            "python", "-m", "sglang.launch_server",
            "--model-path", quant_output,
            "--host", "0.0.0.0",
            "--port", "30000",
        ] + server_args.split(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def stream_logs():
        for line in proc.stdout:
            print(f"[server] {line.rstrip()}", flush=True)
    log_thread = threading.Thread(target=stream_logs, daemon=True)
    log_thread.start()

    try:
        wait_for_server(server_proc=proc)

        # Quick eval
        print("\nRunning quick eval (20 samples)...")
        eval_cmd = [
            "python3", "/opt/SOAR/eval_model.py",
            "--api_base", "http://localhost:30000",
            "--model_path", quant_output,
            "--data_path", "/opt/SOAR/eval_dataset/perf_public_set.jsonl",
            "--concurrency", "16",
            "--num_samples", "20",
        ]

        output, _ = run_streaming(eval_cmd, label="eval")

        # Parse accuracy
        acc_match = re.search(r"Average Score:\s*([0-9.]+)%?", output)
        accuracy = float(acc_match.group(1)) if acc_match else None

        if accuracy:
            baseline = 80.0
            relative = (accuracy / baseline) * 100
            print(f"\n{'=' * 50}")
            print(f"  Quantization Test Results ({quant_method})")
            print(f"{'=' * 50}")
            print(f"  Method:          {quant_method}")
            print(f"  Raw Accuracy:    {accuracy:.2f}%")
            print(f"  Relative Score:  {relative:.2f}%")
            print(f"  Status:          {'✓ PASSING' if relative > 97 else '⚠️ FAILING'}")
            print(f"{'=' * 50}")

        workspace_volume.commit()
        return {
            "success": True,
            "accuracy": accuracy,
            "quant_method": quant_method,
            "quantized_model": quant_output,
        }

    finally:
        proc.terminate()
        proc.wait()


# ============================================================================
# Interactive Development Shell
# ============================================================================
@app.function(
    image=soar_image,
    gpu="H100",
    timeout=14400,  # 4 hours
    volumes={"/models": model_volume, "/workspace": workspace_volume},
)
def dev_shell():
    """Interactive development shell on H100."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║              SOAR 2026 Development Environment (H100)                ║
╠══════════════════════════════════════════════════════════════════════╣
║  Model:      /models/MiniCPM-SALA                                    ║
║  Your code:  /workspace/sglang/sgl-kernel                            ║
║  Submission: /workspace/submission                                   ║
║  SOAR Kit:   /opt/SOAR                                               ║
╠══════════════════════════════════════════════════════════════════════╣
║  Quick commands:                                                     ║
║    bash /workspace/scripts/fast_dev_build.sh    # fast rebuild       ║
║    python /opt/SOAR/eval_model.py --help        # eval help          ║
║    bash /opt/SOAR/bench_serving.sh --help       # bench help         ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    if os.path.exists("/workspace/sglang/sgl-kernel"):
        print("✓ sglang code found")
    else:
        print("⚠ No sglang code - run: modal run modal_app.py --cmd sync")

    if os.path.exists("/workspace/submission"):
        print("✓ submission directory found")
    else:
        print("⚠ No submission directory - run: modal run modal_app.py --cmd sync")

    os.chdir("/workspace")
    subprocess.run(["/bin/bash"])


# ============================================================================
# Entry Point
# ============================================================================
@app.local_entrypoint()
def main(
    cmd: str = "help",
    num_samples: int = None,
    quick: bool = False,
    levels: str = "S1,S8,Smax",
    use_custom: bool = False,
    quantization: str = None,
):
    """
    SOAR 2026 Modal Development CLI.

    Commands:
        help        Show this help
        sync        Sync local code to Modal
        download    Download model (run once)
        check       Check workspace contents
        build       Build sgl-kernel wheel
        eval        Run correctness evaluation
        benchmark   Run speed benchmark
        baseline    Full baseline (eval + benchmark)
        quick       Quick baseline (~20 min)
        package     Create submission package
        shell       Start interactive dev shell
    """
    if cmd == "help":
        print("""
SOAR 2026 - Modal Development Environment

SETUP (run once):
  modal run modal_app.py --cmd download      # Download model
  modal run modal_app.py --cmd sync          # Sync local code

DEVELOPMENT:
  modal run modal_app.py --cmd build         # Build kernel
  modal run modal_app.py --cmd eval          # Full eval (~150 samples)
  modal run modal_app.py --cmd eval --num-samples 10   # Quick eval
  modal run modal_app.py --cmd benchmark     # Speed benchmark
  modal run modal_app.py --cmd baseline      # Full baseline
  modal run modal_app.py --cmd quick         # Quick baseline (~20 min)

OPTIMIZATION:
  modal run modal_app.py --cmd gptq          # Test GPTQ quantization (1-2hr)

SUBMISSION:
  modal run modal_app.py --cmd package       # Create .tar.gz

INTERACTIVE:
  modal shell modal_app.py::dev_shell        # H100 shell (4hr)

OPTIONS:
  --num-samples N     Limit samples for eval/benchmark
  --quick             Use quick mode (fewer samples)
  --levels S1,S8,Smax Which benchmark levels to run
  --use-custom        Use custom kernel (if built)
  --quantization X    Use quantization (gptq_marlin, etc.)

WORKFLOW:
  1. Edit code locally: soar-optimization/sglang/sgl-kernel/
  2. Sync:              modal run modal_app.py --cmd sync
  3. Build:             modal run modal_app.py --cmd build
  4. Test:              modal run modal_app.py --cmd quick
  5. Package:           modal run modal_app.py --cmd package
  6. Submit .tar.gz to SOAR platform
""")
        return

    if cmd == "sync":
        local_paths = ["./sglang", "./submission", "./scripts"]
        for local_path in local_paths:
            if Path(local_path).exists():
                print(f"Syncing {local_path} -> Modal volume...")
                result = subprocess.run(
                    ["modal", "volume", "put", "soar-workspace", local_path, "/", "--force"]
                )
                if result.returncode == 0:
                    print(f"✓ Synced {local_path}")
                else:
                    print(f"✗ Failed to sync {local_path}")

        print("\nNext: modal run modal_app.py --cmd build")
        return

    if cmd == "download":
        result = download_model.remote()
        print(f"\nResult: {result}")

    elif cmd == "check":
        check_workspace.remote()

    elif cmd == "build":
        result = build_kernel.remote(for_submission=True)
        print(f"\nResult: {result}")

    elif cmd == "eval":
        result = run_eval.remote(
            num_samples=num_samples,
            use_custom_kernel=use_custom,
            quantization=quantization,
        )
        print(f"\nAccuracy: {result.get('accuracy')}")

    elif cmd == "benchmark":
        result = run_benchmark.remote(
            num_samples=num_samples or 50,
            levels=levels,
            use_custom_kernel=use_custom,
            quantization=quantization,
        )
        print(f"\nResults: {result}")

    elif cmd == "baseline":
        result = run_baseline.remote(
            num_eval_samples=num_samples,
            use_custom_kernel=use_custom,
            quantization=quantization,
            quick=quick,
        )
        print(f"\nResults: {result}")

    elif cmd == "quick":
        result = run_baseline.remote(quick=True, use_custom_kernel=use_custom)
        print(f"\nResults: {result}")

    elif cmd == "package":
        result = create_submission.remote()
        if result.get("success"):
            print(f"\n✓ Package created: {result['path']}")
            print(f"  Size: {result['size_mb']:.2f} MB")
            print(f"\nDownload with:")
            print(f"  modal volume get soar-workspace /workspace/packages ./packages")
        else:
            print(f"\n✗ Failed: {result.get('error')}")

    elif cmd == "gptq":
        print("Testing GPTQ W4A16 quantization...")
        print("This will take 1-2 hours (quantization + eval)\n")
        result = test_gptq_quantization.remote(
            calibration_samples=num_samples or 256,  # GPTQModel recommends >= 256
        )
        print(f"\nResult: {result}")

    elif cmd == "shell":
        print("Use: modal shell modal_app.py::dev_shell")

    else:
        print(f"Unknown command: {cmd}")
        print("Run: modal run modal_app.py --cmd help")
