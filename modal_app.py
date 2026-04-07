"""
SOAR 2026 Competition - Modal Labs Development Environment

Development on Modal RTX PRO 6000 GPUs (same as SOAR evaluation hardware).

Workflow:
1. Edit code locally in sglang/sgl-kernel/ or submission/
2. Sync to Modal: modal run modal_app.py --cmd sync
3. Build and test on Modal RTX PRO 6000
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
soar_image_with_flash = soar_image.run_commands(
    [
        # Install flash-attn and quantization libraries with GPU
        "pip install flash-attn --no-build-isolation",
        "pip install gptqmodel",
        "pip install autoawq",  # AWQ quantization (SGLang compatible)
    ],
    gpu="RTX-PRO-6000",  # GPU required for compilation
)

# Persistent volumes
model_volume = modal.Volume.from_name("soar-models", create_if_missing=True)
workspace_volume = modal.Volume.from_name("soar-workspace", create_if_missing=True)

# Image variant that injects local submission/sglang/python at /mnt/local_sglang.
soar_image_nvfp4 = soar_image.add_local_dir(
    "./submission/sglang/python",
    remote_path="/mnt/local_sglang",
)


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
            raise RuntimeError(
                f"Server process died with code {server_proc.returncode}"
            )

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
        "python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_path,
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
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
        samples.append(
            {
                "question": "x " * (input_len // 2),
                "model_response": "y " * (output_len // 2),
                "_input_len": input_len,
                "_output_len": output_len,
            }
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for sample in samples:
            # Remove internal fields before writing
            write_sample = {k: v for k, v in sample.items() if not k.startswith("_")}
            f.write(json.dumps(write_sample) + "\n")

    input_lens = [s["_input_len"] for s in samples]
    output_lens = [s["_output_len"] for s in samples]
    print(f"Generated {num_samples} samples to {output_path}")
    print(
        f"  Input:  min={min(input_lens)}, max={max(input_lens)}, avg={sum(input_lens) // len(input_lens)}"
    )
    print(
        f"  Output: min={min(output_lens)}, max={max(output_lens)}, avg={sum(output_lens) // len(output_lens)}"
    )

    return output_path


# ============================================================================
# Model Download
# ============================================================================
@app.function(
    image=soar_image,
    gpu="RTX-PRO-6000",
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
            "huggingface-cli",
            "download",
            "OpenBMB/MiniCPM-SALA",
            "--local-dir",
            "/models/MiniCPM-SALA",
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
    gpu="RTX-PRO-6000",
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
        return {
            "success": False,
            "error": f"Kernel path not found. Did you sync your code?",
        }

    os.chdir(kernel_path)
    print(f"Building kernel from: {kernel_path}")

    # Patch CMakeLists.txt for submission if needed
    if for_submission:
        cmake_path = Path(kernel_path) / "CMakeLists.txt"
        if cmake_path.exists():
            content = cmake_path.read_text()
            # Comment out third-party installs
            patterns = [
                r"install\(TARGETS deep_gemm_cpp[^\)]*\)",
                r'install\(DIRECTORY "\$\{repo-deepgemm_SOURCE_DIR\}[^\)]*\)',
                r'install\(DIRECTORY "\$\{repo-triton_SOURCE_DIR\}[^\)]*\)',
                r'install\(DIRECTORY "\$\{FLASH_ATTN_CUTE_DST\}[^\)]*\)',
                r"install\(FILES \$\{CMAKE_CURRENT_BINARY_DIR\}/deepgemm_pkg_init[^\)]*\)",
                r'install\(DIRECTORY "\$\{repo-cutlass_SOURCE_DIR\}/include/cute/"[^\)]*\)',
                r'install\(DIRECTORY "\$\{repo-cutlass_SOURCE_DIR\}/include/cutlass/"[^\)]*\)',
            ]
            for pattern in patterns:
                content = re.sub(
                    pattern,
                    lambda m: f"# SOAR_DISABLED: {m.group(0)}",
                    content,
                    flags=re.MULTILINE,
                )
            cmake_path.write_text(content)
            print("Patched CMakeLists.txt for competition compliance")

    # Clean and build
    subprocess.run(["rm", "-rf", "build/", "dist/"], check=False)
    subprocess.run(
        ["pip", "install", "build", "wheel", "scikit-build-core", "cmake", "ninja"],
        check=True,
    )

    print("\nBuilding wheel...")
    result = subprocess.run(["python", "-m", "build", "--wheel", "--no-isolation"])

    if result.returncode != 0:
        return {
            "success": False,
            "error": f"Build failed with code {result.returncode}",
        }

    # Find wheel
    wheels = glob.glob("dist/*.whl")
    if not wheels:
        return {"success": False, "error": "No wheel produced"}

    wheel_path = wheels[0]
    print(f"\nBuilt: {wheel_path}")

    # Validate wheel contents
    import zipfile

    with zipfile.ZipFile(wheel_path, "r") as zf:
        top_level = set(
            name.split("/")[0] for name in zf.namelist() if name.split("/")[0]
        )
        violations = [
            d
            for d in top_level
            if not d.startswith("sgl_kernel") and not d.endswith(".dist-info")
        ]

        if violations:
            print(f"\n⚠️  WARNING: Disallowed directories: {violations}")
            if for_submission:
                print("This will FAIL competition validation!")

    # Install
    subprocess.run(["pip", "install", "--force-reinstall", wheel_path], check=True)
    workspace_volume.commit()

    return {
        "success": True,
        "wheel": wheel_path,
        "violations": violations if violations else None,
    }


# ============================================================================
# Correctness Evaluation
# ============================================================================
@app.function(
    image=soar_image,
    gpu="RTX-PRO-6000",
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
                    subprocess.run(
                        ["pip", "install", "--force-reinstall", wheels[0]], check=True
                    )
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
            "python3",
            "/opt/SOAR/eval_model.py",
            "--api_base",
            "http://localhost:30000",
            "--model_path",
            "/models/MiniCPM-SALA",
            "--data_path",
            "/opt/SOAR/eval_dataset/perf_public_set.jsonl",
            "--concurrency",
            str(concurrency),
        ]
        if num_samples:
            cmd.extend(["--num_samples", str(num_samples)])

        print(f"Running: {' '.join(cmd)}")
        output, returncode = run_streaming(cmd, label="eval")

        # Parse accuracy
        acc_match = re.search(
            r"(?:Average Score|Accuracy|overall_accuracy):\s*([0-9.]+)%?",
            output,
            re.IGNORECASE,
        )
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
    gpu="RTX-PRO-6000",
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
                    subprocess.run(
                        ["pip", "install", "--force-reinstall", wheels[0]], check=True
                    )
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
            env_var = {
                "S1": "SPEED_DATA_S1",
                "S8": "SPEED_DATA_S8",
                "Smax": "SPEED_DATA_SMAX",
            }[level]
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
                status = (
                    "✓ BEATS BOUNTY" if duration <= bounty else f"(need: {bounty}s)"
                )
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
    gpu="RTX-PRO-6000",
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
        print(
            f"[QUICK MODE] {num_eval_samples} eval samples, {num_speed_samples} speed samples\n"
        )

    # Install custom kernel if requested
    if use_custom_kernel:
        for wheel_dir in ["/workspace/sglang/sgl-kernel/dist", "/workspace/wheels"]:
            if os.path.exists(wheel_dir):
                wheels = glob.glob(f"{wheel_dir}/*.whl")
                if wheels:
                    print(f"Installing custom kernel: {wheels[0]}")
                    subprocess.run(
                        ["pip", "install", "--force-reinstall", wheels[0]], check=True
                    )
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
            "python3",
            "/opt/SOAR/eval_model.py",
            "--api_base",
            "http://localhost:30000",
            "--model_path",
            "/models/MiniCPM-SALA",
            "--data_path",
            "/opt/SOAR/eval_dataset/perf_public_set.jsonl",
            "--concurrency",
            "32",
        ]
        if num_eval_samples:
            eval_cmd.extend(["--num_samples", str(num_eval_samples)])

        eval_output, _ = run_streaming(eval_cmd, label="eval")

        acc_match = re.search(
            r"(?:Average Score|Accuracy|overall_accuracy):\s*([0-9.]+)%?",
            eval_output,
            re.IGNORECASE,
        )
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
        for step, (level, env_var) in enumerate(
            [
                ("S1", "SPEED_DATA_S1"),
                ("S8", "SPEED_DATA_S8"),
                ("Smax", "SPEED_DATA_SMAX"),
            ],
            start=2,
        ):
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
            results["speed"][level] = (
                float(duration_match.group(1)) if duration_match else None
            )

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
                status = (
                    "✓ BOUNTY" if dur <= bounties[level] else f"need {bounties[level]}s"
                )
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
            json.dump(
                {
                    "timestamp": timestamp,
                    **results,
                    "config": {
                        "num_eval_samples": num_eval_samples,
                        "num_speed_samples": num_speed_samples,
                        "use_custom_kernel": use_custom_kernel,
                        "quantization": quantization,
                    },
                },
                f,
                indent=2,
            )

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
    gpu="RTX-PRO-6000",
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
        for script in [
            "prepare_env.sh",
            "prepare_model.sh",
            "preprocess_model.py",
            "preprocess_model_nvfp4_modelopt.py",
            "quantize_gptq.py",
        ]:
            script_path = f"{submission_dir}/{script}"
            if os.path.exists(script_path):
                tar.add(script_path, arcname=script)
                print(f"  Added: {script}")

        # Add custom sglang if present
        sglang_python = f"{submission_dir}/sglang/python"
        if not os.path.exists(sglang_python):
            sglang_python = "/workspace/sglang/python"
        if os.path.exists(sglang_python):
            tar.add(sglang_python, arcname="sglang/python")
            print(f"  Added: sglang/python/ (from {sglang_python})")

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
# Check Model Architecture
# ============================================================================
@app.function(
    image=soar_image,
    gpu=None,  # No GPU needed, just read config
    timeout=60,
    volumes={"/models": model_volume},
)
def check_model_architecture():
    """Check if MiniCPM-SALA supports hybrid_linear_attn backend."""
    import json
    from pathlib import Path

    model_path = Path("/models/MiniCPM-SALA")
    config_path = model_path / "config.json"

    if not config_path.exists():
        return {"error": f"Config not found at {config_path}"}

    with open(config_path) as f:
        config = json.load(f)

    output = []
    output.append("=== MiniCPM-SALA Model Config ===")
    output.append(f"Model Type: {config.get('model_type', 'N/A')}")
    output.append(f"Architecture: {config.get('architectures', ['N/A'])[0]}")
    output.append(f"Num Layers: {config.get('num_hidden_layers', 'N/A')}")

    # Check for hybrid indicators
    output.append("\n=== Hybrid Architecture Check ===")

    if config.get("model_type") == "bailing_hybrid":
        output.append(
            "[YES] Model type is 'bailing_hybrid' - HYBRID BACKEND SUPPORTED!"
        )
        output.append(f"   Layer group size: {config.get('layer_group_size', 'N/A')}")
        hybrid_supported = True
    else:
        # Check for other hybrid indicators
        hybrid_keys = [
            "hybrid_config",
            "linear_attention_layers",
            "mambaish_config",
            "layer_group_size",
        ]
        found = [k for k in hybrid_keys if k in config]

        if found:
            output.append(f"[YES] Found hybrid indicators: {', '.join(found)}")
            hybrid_supported = True
        else:
            output.append("[NO] No hybrid architecture indicators found")
            output.append("   Will use standard minicpm_flashinfer backend")
            hybrid_supported = False

    # Show relevant config keys
    output.append("\n=== Key Config Fields ===")
    relevant_keys = [
        "model_type",
        "architectures",
        "num_hidden_layers",
        "hidden_size",
        "num_attention_heads",
        "layer_group_size",
        "linear_silu",
    ]
    for key in relevant_keys:
        if key in config:
            output.append(f"  {key}: {config[key]}")

    return {
        "hybrid_supported": hybrid_supported,
        "model_type": config.get("model_type"),
        "output": "\n".join(output),
    }


# ============================================================================
# FP8 W8A8 Quantization + Eval
# ============================================================================
@app.function(
    image=soar_image_nvfp4,
    gpu="RTX-PRO-6000",
    timeout=7200,  # 2 hours: quant ~20min + eval
    volumes={"/models": model_volume, "/workspace": workspace_volume},
)
def test_fp8_quantization(num_samples: int = 50):
    """
    Quantize MiniCPM-SALA to FP8 W8A8, start server, run eval.

    Uses preprocess_model.py (copies model + patches config for online FP8).
    SGLang loads FP16 weights and quantizes to FP8 at load time (~1-2 min overhead).
    SGLang flag: --quantization fp8
    Expected: ~2x prefill throughput vs FP16 baseline.
    """
    import shutil as _shutil
    import threading

    quant_output = "/workspace/fp8_model"

    # Step 1: Quantize (always re-quantize to pick up latest preprocess_model.py changes)
    print("=" * 60)
    print("Step 1: FP8 W8A8 Quantization")
    print("=" * 60)

    if os.path.exists(quant_output):
        _shutil.rmtree(quant_output)
        print(f"[quant] Removed old {quant_output}")

    quant_cmd = [
        "/opt/SGLang-MiniCPM-SALA/sglang_minicpm_sala_env/bin/python3",
        "/workspace/submission/preprocess_model.py",
        "--input",
        "/models/MiniCPM-SALA",
        "--output",
        quant_output,
    ]
    out, rc = run_streaming(quant_cmd, label="quant")
    if rc != 0:
        return {
            "success": False,
            "error": "FP8 quantization failed",
            "log": out[-2000:],
        }
    print(f"\nQuantization done. Output: {quant_output}")

    # Step 2: Start SGLang server with fp8
    print("\n" + "=" * 60)
    print("Step 2: Starting SGLang server (w8a8_fp8)")
    print("=" * 60)

    server_args = (
        "--disable-radix-cache "
        "--attention-backend minicpm_flashinfer "
        "--chunked-prefill-size 8192 "
        "--mem-fraction-static 0.70 "
        "--dense-as-sparse "
        "--skip-server-warmup "
        "--quantization fp8"
    )
    proc = subprocess.Popen(
        [
            "/opt/SGLang-MiniCPM-SALA/sglang_minicpm_sala_env/bin/python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            quant_output,
            "--tokenizer-path",
            "/models/MiniCPM-SALA",
            "--host",
            "0.0.0.0",
            "--port",
            "30000",
        ]
        + server_args.split(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def stream_logs():
        for line in proc.stdout:
            print(f"[server] {line.rstrip()}", flush=True)

    threading.Thread(target=stream_logs, daemon=True).start()

    try:
        wait_for_server(server_proc=proc)

        # Step 3: Eval
        print("\n" + "=" * 60)
        print(f"Step 3: Correctness eval ({num_samples} samples)")
        print("=" * 60)
        t0 = time.time()
        eval_out, _ = run_streaming(
            [
                "python3",
                "/opt/SOAR/eval_model.py",
                "--api_base",
                "http://localhost:30000",
                "--model_path",
                quant_output,
                "--data_path",
                "/opt/SOAR/eval_dataset/perf_public_set.jsonl",
                "--num_samples",
                str(num_samples),
                "--concurrency",
                "32",
            ],
            label="eval",
        )
        t_eval = time.time() - t0
        s1_est = t_eval * (150 / num_samples)
        print(f"\n[speed] {num_samples} samples in {t_eval:.1f}s")
        print(f"[speed] S1 estimate: {s1_est:.1f}s  (target: 420s)")

        # Step 4: Quick benchmark (S1 only for speed)
        print("\n" + "=" * 60)
        print("Step 4: Speed benchmark (S1 — single request)")
        print("=" * 60)
        bench_data = "/tmp/bench_s1.jsonl"
        generate_speed_data(bench_data, num_samples=30)
        bench_out, _ = run_streaming(
            [
                "bash",
                "/opt/SOAR/bench_serving.sh",
                "http://localhost:30000",
            ],
            label="bench",
            env={
                **os.environ,
                "SPEED_DATA_S1": bench_data,
            },
        )

        return {
            "success": True,
            "quant_output": quant_output,
            "eval_log": eval_out[-3000:],
            "bench_log": bench_out[-3000:],
        }

    finally:
        proc.terminate()


# ============================================================================
# Test GPTQ+Marlin Quantization
# ============================================================================
@app.function(
    image=soar_image_nvfp4,
    gpu="RTX-PRO-6000",
    timeout=14400,
    volumes={"/models": model_volume, "/workspace": workspace_volume},
)
def test_gptq_quantization(
    num_samples: int = 10,
    group_size: int = 128,
):
    """
    W4A16 GPTQ quantization + Marlin kernel via SGLang --quantization gptq_marlin.
    Uses quantize_gptq.py (RTN, no calibration needed, fast).
    Expected: ~2x decode speedup vs BF16, accuracy ~80%.
    """
    import shutil as _shutil
    import threading

    quant_output = "/workspace/gptq_model"

    print("=" * 60)
    print("Step 1: W4A16 GPTQ quantization (RTN)")
    print("=" * 60)

    existing = os.path.exists(f"{quant_output}/quantize_config.json")
    if existing:
        print(f"[quant] Reusing existing GPTQ export: {quant_output}")
    else:
        if os.path.exists(quant_output):
            _shutil.rmtree(quant_output)
        sala_python = "/opt/SGLang-MiniCPM-SALA/sglang_minicpm_sala_env/bin/python3"
        quant_cmd = [
            sala_python,
            "/workspace/submission/quantize_gptq.py",
            "--input",
            "/models/MiniCPM-SALA",
            "--output",
            quant_output,
            "--group-size",
            str(group_size),
        ]
        out, rc = run_streaming(quant_cmd, label="quant")
        if rc != 0:
            return {
                "success": False,
                "error": "GPTQ quantization failed",
                "log": out[-2000:],
            }
        workspace_volume.commit()

    print(f"\n[quant] Done: {quant_output}")

    print("\n" + "=" * 60)
    print("Step 2: Starting SGLang server (--quantization gptq)")
    print("=" * 60)

    # Patch SGLang files from live mount
    _patch_sglang_files()

    # Ensure gptq model config uses bfloat16 (not float16) to match attention kernels
    import json as _json

    cfg_path = f"{quant_output}/config.json"
    with open(cfg_path) as _f:
        _cfg = _json.load(_f)
    if _cfg.get("torch_dtype") == "float16":
        _cfg["torch_dtype"] = "bfloat16"
        with open(cfg_path, "w") as _f:
            _json.dump(_cfg, _f, indent=2)
        print(f"[patch] Patched {cfg_path}: torch_dtype -> bfloat16")

    server_cmd = [
        "/opt/SGLang-MiniCPM-SALA/sglang_minicpm_sala_env/bin/python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        quant_output,
        "--tokenizer-path",
        "/models/MiniCPM-SALA",
        "--host",
        "0.0.0.0",
        "--port",
        "30000",
        "--quantization",
        "gptq",
        "--attention-backend",
        "minicpm_flashinfer",
        "--chunked-prefill-size",
        "8192",
        "--mem-fraction-static",
        "0.70",
        "--dense-as-sparse",
        "--skip-server-warmup",
        "--disable-radix-cache",
    ]

    proc = subprocess.Popen(
        server_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def stream_logs():
        for line in proc.stdout:
            print(f"[server] {line.rstrip()}", flush=True)

    threading.Thread(target=stream_logs, daemon=True).start()

    try:
        wait_for_server(server_proc=proc)

        print("\n" + "=" * 60)
        print(f"Step 3: Eval ({num_samples} samples, concurrency=32)")
        print("=" * 60)
        t0 = time.time()
        eval_out, _ = run_streaming(
            [
                "python3",
                "/opt/SOAR/eval_model.py",
                "--api_base",
                "http://localhost:30000",
                "--model_path",
                quant_output,
                "--data_path",
                "/opt/SOAR/eval_dataset/perf_public_set.jsonl",
                "--num_samples",
                str(num_samples),
                "--concurrency",
                "32",
            ],
            label="eval",
        )
        t_eval = time.time() - t0
        s1_est = t_eval * (150 / num_samples)

        acc_match = re.search(r"Average Score:\s*([0-9.]+)%?", eval_out)
        accuracy = float(acc_match.group(1)) if acc_match else None

        print(f"\n[speed] {num_samples} samples in {t_eval:.1f}s")
        print(f"[speed] S1 estimate: {s1_est:.1f}s  (target: 420s)")
        if accuracy:
            print(f"[accuracy] {accuracy:.1f}% (baseline ~83%)")

        return {
            "success": True,
            "accuracy": accuracy,
            "s1_estimate_s": round(s1_est, 1),
        }
    finally:
        proc.terminate()


def _patch_sglang_files():
    """Patch SGLang files from /mnt/local_sglang (live mount) or workspace volume."""
    import shutil as _shutil

    sglang_bases = [
        "/opt/SGLang-MiniCPM-SALA/packages/sglang-minicpm/python",
        "/opt/SGLang-MiniCPM-SALA/sglang_minicpm_sala_env/lib/python3.10/site-packages",
    ]
    patch_files = [
        "sglang/srt/configs/model_config.py",
        "sglang/srt/layers/parameter.py",
        "sglang/srt/layers/quantization/modelopt_quant.py",
    ]
    for rel in patch_files:
        src = None
        for base in ["/mnt/local_sglang", "/workspace/submission/sglang/python"]:
            candidate = f"{base}/{rel}"
            if os.path.exists(candidate):
                src = candidate
                break
        if src is None:
            continue
        for sglang_base in sglang_bases:
            dst = f"{sglang_base}/{rel}"
            if os.path.exists(os.path.dirname(dst)):
                _shutil.copy2(src, dst)
                # Nuke pyc
                stem = os.path.basename(dst)[:-3]
                pyc_dir = os.path.join(os.path.dirname(dst), "__pycache__")
                import glob as _g

                for pyc in _g.glob(f"{pyc_dir}/{stem}.cpython-*.pyc"):
                    os.remove(pyc)
                print(f"[patch] {rel} -> {dst}")


# ============================================================================
# (old broken GPTQ stub removed)
# ============================================================================
def _old_gptq_stub():
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
            subprocess.run(
                ["pip", "install", pkg, "--no-cache-dir"], capture_output=True
            )

    # Run quantization (will try multiple methods with fallbacks)
    quant_output = "/workspace/quantized_model"
    print(f"\nQuantizing to {quant_output}...")

    quant_cmd = [
        "python3",
        "/workspace/submission/preprocess_model.py",
        "--input",
        "/models/MiniCPM-SALA",
        "--output",
        quant_output,
        "--mode",
        "gptq",
        "--bits",
        str(bits),
        "--group-size",
        str(group_size),
        "--calibration-samples",
        str(calibration_samples),
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
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            quant_output,
            "--host",
            "0.0.0.0",
            "--port",
            "30000",
        ]
        + server_args.split(),
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
            "python3",
            "/opt/SOAR/eval_model.py",
            "--api_base",
            "http://localhost:30000",
            "--model_path",
            quant_output,
            "--data_path",
            "/opt/SOAR/eval_dataset/perf_public_set.jsonl",
            "--concurrency",
            "16",
            "--num_samples",
            "20",
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
# NVFP4 Quantization + Eval
# ============================================================================
@app.function(
    image=soar_image_nvfp4,
    gpu="RTX-PRO-6000",
    timeout=7200,
    volumes={"/models": model_volume, "/workspace": workspace_volume},
)
def test_nvfp4_quantization(num_samples: int = 10, force_requantize: bool = True):
    """
    Quantize MiniCPM-SALA to NVFP4 (E2M1 + dual-scale), start server, run eval.

    NVFP4 = 4-bit float, Blackwell-native hardware acceleration.
    Dual-scale: per-16-weight FP8 local scale + global FP32 scale.
    SGLang flag: --quantization modelopt_fp4
    Expected: ~3-4x decode throughput vs FP16 (4x memory bandwidth reduction).
    """
    import glob as _pyc_glob
    import json as _json

    # Patch model_config.py and parameter.py into ALL known sglang install locations.
    # The server subprocess may use a different Python than the current process, so we
    # can't rely on importlib to find the right location — patch all candidates directly.
    import py_compile
    import shutil
    import threading

    def _patch_into(src, dst):
        if not os.path.exists(dst):
            return False
        shutil.copy(src, dst)
        # Aggressively nuke ALL pyc files for this module so no stale cache is used
        pyc_dir = os.path.join(os.path.dirname(dst), "__pycache__")
        stem = os.path.splitext(os.path.basename(dst))[0]
        for pyc in _pyc_glob.glob(f"{pyc_dir}/{stem}.cpython-*.pyc"):
            try:
                os.remove(pyc)
            except Exception:
                pass
        # Also try to remove the whole __pycache__ dir if it exists and we can
        # (avoids stale .pyc from previous runs with different Python minor versions)
        for pyc in _pyc_glob.glob(f"{pyc_dir}/{stem}*.pyc"):
            try:
                os.remove(pyc)
            except Exception:
                pass
        py_compile.compile(dst, doraise=False)
        print(f"[patch] patched {dst}")
        return True

    # All candidate sglang base dirs (cover current-process install + SALA venv + SALA packages)
    _sglang_bases = [
        "/opt/SGLang-MiniCPM-SALA/packages/sglang-minicpm/python",
        "/opt/SGLang-MiniCPM-SALA/sglang_minicpm_sala_env/lib/python3.10/site-packages",
        "/opt/SGLang-MiniCPM-SALA/sglang_minicpm_sala_env/lib/python3.11/site-packages",
        "/opt/SGLang-MiniCPM-SALA/sglang_minicpm_sala_env/lib/python3.12/site-packages",
    ]
    # Also add wherever the current Python has sglang installed
    try:
        import importlib

        import sglang.srt.layers.parameter as _p

        _sglang_bases.insert(
            0,
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(_p.__file__)))
            ),
        )
    except Exception:
        pass

    for _rel, _sub in [
        ("srt/configs/model_config.py", "srt/configs/model_config.py"),
        ("srt/layers/parameter.py", "srt/layers/parameter.py"),
        (
            "srt/layers/quantization/modelopt_quant.py",
            "srt/layers/quantization/modelopt_quant.py",
        ),
        ("srt/models/minicpm.py", "srt/models/minicpm.py"),
    ]:
        # Prefer the live mount (/mnt/local_sglang) which reflects local edits without
        # a volume sync.  Fall back to the workspace volume copy if the mount isn't there.
        _mount_src = f"/mnt/local_sglang/sglang/{_rel}"
        _vol_src = f"/workspace/submission/sglang/python/sglang/{_rel}"
        if os.path.exists(_mount_src):
            _src = _mount_src
            print(f"[patch] using live mount: {_mount_src}")
        elif os.path.exists(_vol_src):
            _src = _vol_src
            print(f"[patch] using volume copy: {_vol_src}")
        else:
            print(f"[patch] WARNING: {_rel} not found in mount or volume, skipping")
            continue
        patched = 0
        for base in _sglang_bases:
            dst = f"{base}/sglang/{_sub}"
            if _patch_into(_src, dst):
                patched += 1
        print(f"[patch] {_rel}: patched {patched} location(s)")

    quant_output = "/workspace/nvfp4_model"

    if force_requantize and os.path.exists(quant_output):
        shutil.rmtree(quant_output)
        print(f"[quant] force_requantize=True — deleted {quant_output}")

    # Step 1: Quantize + export (skip if already done)
    print("=" * 60)
    print("Step 1: NVFP4 Quantization + export (E2M1 + dual-scale)")
    print("=" * 60)
    import glob as _glob

    existing = _glob.glob(f"{quant_output}/*.safetensors") + _glob.glob(
        f"{quant_output}/*.bin"
    )
    if existing:
        # Check if input_scale was injected (new export) — if not, force re-quantize
        try:
            from safetensors import safe_open

            has_input_scale = False
            # Check all safetensors files (input_scales may be in a separate shard)
            for sf in existing:
                with safe_open(sf, framework="pt") as f:
                    if any("input_scale" in k for k in f.keys()):
                        has_input_scale = True
                        break
            # Also check dedicated input_scales.safetensors (but verify lm_head is not in it)
            import os as _os

            if not has_input_scale and _os.path.exists(
                f"{quant_output}/input_scales.safetensors"
            ):
                with safe_open(
                    f"{quant_output}/input_scales.safetensors", framework="pt"
                ) as f:
                    keys = list(f.keys())
                # Valid export: has input_scale keys but NOT lm_head (non-quantized layer)
                if any("input_scale" in k for k in keys) and not any(
                    "lm_head" in k for k in keys
                ):
                    has_input_scale = True
                else:
                    print(
                        f"[quant] input_scales.safetensors has invalid keys (lm_head present or empty) — re-quantizing"
                    )
            if not has_input_scale:
                print(
                    f"[quant] Existing export missing input_scale — re-quantizing for accuracy fix"
                )
                shutil.rmtree(quant_output)
                existing = []
            else:
                print(
                    f"[quant] Reusing existing export: {quant_output} ({len(existing)} weight files, input_scale present)"
                )
        except Exception as e:
            print(
                f"[quant] Reusing existing export: {quant_output} ({len(existing)} weight files)"
            )
    if not existing:
        if os.path.exists(quant_output):
            shutil.rmtree(quant_output)
        # Use SALA's venv Python so modelopt version matches SALA's SGLang loader
        sala_python = "/opt/SGLang-MiniCPM-SALA/sglang_minicpm_sala_env/bin/python3"
        quant_cmd = [
            sala_python,
            "/workspace/submission/preprocess_model_nvfp4_modelopt.py",
            "--input",
            "/models/MiniCPM-SALA",
            "--output",
            quant_output,
            "--config",
            "conservative",
            "--calib-data",
            "/opt/SOAR/eval_dataset/perf_public_set.jsonl",
            "--num-calib",
            "32",
        ]
        out, rc = run_streaming(quant_cmd, label="quant")
        if rc != 0:
            return {
                "success": False,
                "error": "NVFP4 quantization failed",
                "log": out[-2000:],
            }

    print(f"\n[quant] Done: {quant_output}")

    # Inspect first weight tensor to understand dtype/values
    import glob as _glob2

    st_files = _glob2.glob(f"{quant_output}/*.safetensors")
    if st_files:
        try:
            import torch
            from safetensors import safe_open

            with safe_open(st_files[0], framework="pt") as f:
                keys = list(f.keys())
            # Print all keys for layers.0 to understand checkpoint structure
            layer0_keys = [k for k in keys if "layers.0." in k]
            print(f"[inspect] layers.0 keys ({len(layer0_keys)}): {layer0_keys}")
            # Also show layers.0.mlp.down_proj keys specifically (input_scale, weight_scale_2?)
            down_keys = [k for k in keys if "layers.0.mlp.down_proj" in k]
            print(f"[inspect] down_proj keys: {down_keys}")
            with safe_open(st_files[0], framework="pt") as f_sc:
                for dk in down_keys:
                    t = f_sc.get_tensor(dk)
                    print(
                        f"[inspect] {dk}: dtype={t.dtype} shape={t.shape} val={t.reshape(-1)[:4].tolist()}"
                    )
            # Find a quantized linear weight (not embed, not scale) — prefer q_proj/k_proj in layers.0
            weight_key = next(
                (
                    k
                    for k in keys
                    if "layers.0." in k
                    and ".weight" in k
                    and "scale" not in k
                    and "norm" not in k
                ),
                None,
            )
            if weight_key is None:
                weight_key = next(
                    (
                        k
                        for k in keys
                        if ".weight" in k
                        and "scale" not in k
                        and "embed" not in k
                        and "lm_head" not in k
                    ),
                    None,
                )
            if weight_key is None:
                weight_key = next(
                    (k for k in keys if "weight" in k and "scale" not in k), keys[0]
                )
            # Find matching scale (same layer prefix)
            layer_prefix = ".".join(
                weight_key.split(".")[:-1]
            )  # e.g. model.layers.0.mlp.down_proj
            scale_key = next(
                (k for k in keys if k.startswith(layer_prefix) and "weight_scale" in k),
                None,
            )
            print(f"[inspect] weight_key={weight_key}, scale_key={scale_key}")
            with safe_open(st_files[0], framework="pt") as f:
                w = f.get_tensor(weight_key)
                sc = f.get_tensor(scale_key) if scale_key else None
            print(f"[inspect] key={weight_key} dtype={w.dtype} shape={w.shape}")
            print(f"[inspect] first 8 values: {w.reshape(-1)[:8].tolist()}")
            print(
                f"[inspect] unique values count: {w.reshape(-1).float().unique().numel()}"
            )
            if sc is not None:
                print(
                    f"[inspect] scale key={scale_key} dtype={sc.dtype} shape={sc.shape}"
                )
                print(f"[inspect] scale first 4: {sc.reshape(-1)[:4].tolist()}")
                # Try dequant: divide weight by scale to see if we get FP4 values
                # scale shape is [out, in/group_size], weight shape [out, in]
                group_size = w.shape[1] // sc.shape[1] if sc.dim() == 2 else None
                if group_size:
                    sc_expanded = (
                        sc.to(torch.float32)
                        .unsqueeze(2)
                        .expand(-1, -1, group_size)
                        .reshape(w.shape[0], -1)
                    )
                    dequant = w.to(torch.float32) / sc_expanded[:, : w.shape[1]]
                    print(
                        f"[inspect] dequant first 8: {dequant.reshape(-1)[:8].tolist()}"
                    )
                    print(
                        f"[inspect] dequant unique count: {dequant.reshape(-1).unique().numel()}"
                    )
        except Exception as e:
            print(f"[inspect] error: {e}")

    # Repair hf_quant_config.json:
    # 1. Add scale_fmt=ue8m0 (ModelOpt export omits it → DeepGemm warning)
    # 2. Add unquantized attention layers to exclude_modules so SGLang uses BF16 for them.
    #    The checkpoint has no weight_scale for q/k/v/o_proj — they were skipped by the
    #    quantizer.  Without this, SGLang allocates uint8 [out, in//2] for them but the
    #    checkpoint has BF16 [out, in], causing a shape mismatch on load.
    # MiniCPM fuses q/k/v into qkv_proj (QKVParallelLinear). The checkpoint has no
    # weight_scale for q/k/v/o_proj — only for mlp and o_gate.
    # is_layer_skipped() uses substring matching: any(ignored in prefix).
    # So "self_attn.qkv_proj" matches "model.layers.N.self_attn.qkv_proj" for all N.
    _unquant_attn = [
        "self_attn.qkv_proj",
        "self_attn.o_proj",
    ]
    hf_quant_path = f"{quant_output}/hf_quant_config.json"
    if os.path.exists(hf_quant_path):
        with open(hf_quant_path) as f:
            hf_quant = _json.load(f)
        quant_section = hf_quant.get("quantization", hf_quant)
        if isinstance(quant_section, dict):
            # Always overwrite scale_fmt and exclude_modules with the correct values.
            # Keep the original per-layer excludes from ModelOpt plus our fused-layer wildcards.
            quant_section["scale_fmt"] = "ue8m0"
            existing_excl = quant_section.get("exclude_modules", [])
            # Remove any stale wildcard patterns from prior runs, then add current ones
            cleaned = [m for m in existing_excl if not m.startswith("*")]
            quant_section["exclude_modules"] = cleaned + _unquant_attn
            with open(hf_quant_path, "w") as f:
                _json.dump(hf_quant, f, indent=2)
            print(
                f"[repair] Wrote hf_quant_config.json: scale_fmt=ue8m0, exclude_modules wildcards={_unquant_attn}"
            )
    else:
        with open(hf_quant_path, "w") as f:
            _json.dump(
                {
                    "quant_algo": "NVFP4",
                    "scale_fmt": "ue8m0",
                    "exclude_modules": ["lm_head"] + _unquant_attn,
                },
                f,
                indent=2,
            )
        print(f"[repair] Created hf_quant_config.json")

    # Also patch config.json's quantization_config if present — SGLang reads it first
    # and may bypass hf_quant_config.json entirely.
    config_json_path = f"{quant_output}/config.json"
    if os.path.exists(config_json_path):
        with open(config_json_path) as f:
            cfg = _json.load(f)
        qc = cfg.get("quantization_config")
        if isinstance(qc, dict):
            changed = False
            if qc.get("scale_fmt") != "ue8m0":
                qc["scale_fmt"] = "ue8m0"
                changed = True
            existing_excl = qc.get("exclude_modules", qc.get("ignore", []))
            to_add = [m for m in _unquant_attn if m not in existing_excl]
            if to_add:
                # Use whichever key already exists
                key = "exclude_modules" if "exclude_modules" in qc else "ignore"
                qc[key] = existing_excl + to_add
                changed = True
            if changed:
                with open(config_json_path, "w") as f:
                    _json.dump(cfg, f, indent=2)
                print(
                    f"[repair] Patched config.json quantization_config: scale_fmt=ue8m0, excl+={to_add}"
                )
            else:
                print(f"[repair] config.json quantization_config already up to date")
        else:
            print(f"[repair] config.json has no quantization_config, skipping")

    # Verify key fields
    print(
        f"[repair] config.json quant_config: {_json.load(open(config_json_path)).get('quantization_config', {}).get('exclude_modules', 'N/A')[:3] if os.path.exists(config_json_path) else 'no file'}"
    )

    # Step 2: Start SGLang server with exported NVFP4 checkpoint
    # Use original model path for tokenizer (exported dir has custom config AutoTokenizer can't resolve)
    print("\n" + "=" * 60)
    print("Step 2: Starting SGLang server (--quantization modelopt_fp4)")
    print("=" * 60)
    server_env = {**os.environ, "SGLANG_FLASHINFER_FP4_GEMM_BACKEND": "cutlass"}
    server_cmd = [
        "python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        quant_output,
        "--tokenizer-path",
        "/models/MiniCPM-SALA",
        "--quantization",
        "modelopt_fp4",
        "--host",
        "0.0.0.0",
        "--port",
        "30000",
        "--disable-radix-cache",
        "--trust-remote-code",
        "--attention-backend",
        "minicpm_flashinfer",
        "--chunked-prefill-size",
        "8192",
        "--mem-fraction-static",
        "0.70",
        "--dense-as-sparse",
        "--skip-server-warmup",
    ]
    proc = subprocess.Popen(
        server_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=server_env,
    )
    threading.Thread(
        target=lambda: [
            print(f"[server] {l.rstrip()}", flush=True) for l in proc.stdout
        ],
        daemon=True,
    ).start()

    try:
        wait_for_server(server_proc=proc)

        # Sanity check
        import urllib.request

        req = urllib.request.Request(
            "http://localhost:30000/v1/chat/completions",
            data=_json.dumps(
                {
                    "model": quant_output,
                    "messages": [
                        {"role": "user", "content": "What is 1+1? Answer briefly."}
                    ],
                    "max_tokens": 512,
                    "temperature": 0.0,
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                resp = _json.loads(r.read())
                msg = resp["choices"][0]["message"]
                # MiniCPM-SALA separates reasoning from final answer
                reasoning = msg.get("reasoning_content") or ""
                content = msg.get("content") or ""
                finish = resp["choices"][0].get("finish_reason", "?")
                print(f"[sanity] finish={finish}")
                print(
                    f"[sanity] reasoning ({len(reasoning)} chars): {repr(reasoning[:500])}"
                )
                print(f"[sanity] content ({len(content)} chars): {repr(content[:300])}")
                print(f"[sanity] raw msg keys: {list(msg.keys())}")
                print(f"[sanity] usage: {resp.get('usage')}")
                # Show raw bytes to diagnose whitespace/garbage tokens
                raw_text = reasoning + content
                print(f"[sanity] raw combined repr: {repr(raw_text[:200])}")
                print(
                    f"[sanity] raw combined bytes: {raw_text[:100].encode('utf-8', errors='replace').hex()}"
                )
        except Exception as e:
            print(f"[sanity] FAILED: {e}")
            print(f"[sanity] raw resp: {resp if 'resp' in dir() else 'no response'}")

        # Step 3: Eval
        print("\n" + "=" * 60)
        print(f"Step 3: Eval ({num_samples} samples, concurrency=32)")
        print("=" * 60)
        t0 = time.time()
        eval_out, _ = run_streaming(
            [
                "python3",
                "/opt/SOAR/eval_model.py",
                "--api_base",
                "http://localhost:30000",
                "--model_path",
                "/models/MiniCPM-SALA",  # tokenizer needs original model dir
                "--data_path",
                "/opt/SOAR/eval_dataset/perf_public_set.jsonl",
                "--num_samples",
                str(num_samples),
                "--concurrency",
                "32",
            ],
            label="eval",
        )
        t_eval = time.time() - t0
        s1_est = t_eval * (150 / num_samples)
        print(f"\n[speed] {num_samples} samples in {t_eval:.1f}s")
        print(f"[speed] S1 estimate: {s1_est:.1f}s  (target: 420s)")

        return {
            "success": True,
            "quant_output": quant_output,
            "eval_log": eval_out[-3000:],
            "s1_estimate_s": round(s1_est, 1),
            "eval_time_s": round(t_eval, 1),
        }
    finally:
        proc.terminate()


# ============================================================================
# Interactive Development Shell
# ============================================================================
@app.function(
    image=soar_image,
    gpu="RTX-PRO-6000",
    timeout=14400,  # 4 hours
    volumes={"/models": model_volume, "/workspace": workspace_volume},
)
def dev_shell():
    """Interactive development shell on RTX PRO 6000."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║          SOAR 2026 Development Environment (RTX PRO 6000)            ║
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


@app.function(
    image=soar_image,
    gpu="RTX-PRO-6000",
    timeout=300,
    volumes={"/models": model_volume, "/workspace": workspace_volume},
)
def patch_quant_configs():
    """Apply the fixed exclude_modules patch to the cached quantized model without re-quantizing."""
    import json
    import re

    results = {}
    for model_dir in ["/workspace/nvfp4_model", "/workspace/nvfp4_model_simulate"]:
        if not os.path.exists(model_dir):
            results[model_dir] = "not found"
            continue

        # Patch hf_quant_config.json
        hf_path = f"{model_dir}/hf_quant_config.json"
        if os.path.exists(hf_path):
            with open(hf_path) as f:
                cfg = json.load(f)
            section = cfg.get("quantization", cfg)
            excl_raw = section.get("exclude_modules", [])
            layer_prefixes = set()
            for e in excl_raw:
                m = re.match(r'(model\.layers\.\d+\.self_attn)\.(q_proj|k_proj|v_proj|o_proj|qkv_proj)', e)
                if m:
                    layer_prefixes.add(m.group(1))
            excl_new = []
            for prefix in sorted(layer_prefixes):
                excl_new.append(f"{prefix}.qkv_proj")
                excl_new.append(f"{prefix}.o_proj")
            section["scale_fmt"] = "ue8m0"
            section["exclude_modules"] = excl_new
            with open(hf_path, "w") as f:
                json.dump(cfg, f, indent=2)
            print(f"[patch] {hf_path}: {len(excl_new)} exclude_modules")
            results[f"{model_dir}/hf_quant_config.json"] = excl_new[:4]

        # Patch config.json
        cfg_path = f"{model_dir}/config.json"
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
            qc = cfg.get("quantization_config")
            if isinstance(qc, dict):
                ignore_raw = qc.get("ignore", [])
                layer_prefixes = set()
                for e in ignore_raw:
                    m = re.match(r'(model\.layers\.\d+\.self_attn)\.(q_proj|k_proj|v_proj|o_proj|qkv_proj)', e)
                    if m:
                        layer_prefixes.add(m.group(1))
                ignore_new = ["lm_head"]
                for prefix in sorted(layer_prefixes):
                    ignore_new.append(f"{prefix}.qkv_proj")
                    ignore_new.append(f"{prefix}.o_proj")
                qc["ignore"] = ignore_new
                qc["scale_fmt"] = "ue8m0"
                with open(cfg_path, "w") as f:
                    json.dump(cfg, f, indent=2)
                print(f"[patch] {cfg_path}: {len(ignore_new)} ignore entries")
                results[f"{model_dir}/config.json"] = ignore_new[:4]

    workspace_volume.commit()
    return results


@app.function(
    image=soar_image,
    gpu="RTX-PRO-6000",
    timeout=600,
    volumes={"/models": model_volume, "/workspace": workspace_volume},
)
def inspect_quant_model():
    """Inspect the exported NVFP4 model configs and weight_scale shapes without re-quantizing."""
    import json
    from safetensors import safe_open

    results = {}

    for model_dir in ["/workspace/nvfp4_model", "/workspace/nvfp4_model_simulate"]:
        if not os.path.exists(model_dir):
            results[model_dir] = "not found"
            continue

        info = {}

        # Print config files
        for cfg in ["hf_quant_config.json", "config.json"]:
            p = f"{model_dir}/{cfg}"
            if os.path.exists(p):
                with open(p) as f:
                    info[cfg] = json.load(f)
                print(f"\n[{model_dir}] {cfg}:")
                print(json.dumps(info[cfg], indent=2)[:2000])

        # Inspect weight_scale shapes for layers 0 and 1
        index_path = f"{model_dir}/model.safetensors.index.json"
        if os.path.exists(index_path):
            with open(index_path) as f:
                index = json.load(f)
            weight_map = index.get("weight_map", {})

            # Find weight_scale keys for first 2 layers
            target_keys = [k for k in weight_map if "weight_scale" in k and
                          any(f"layers.{i}." in k for i in range(2))]
            target_keys.sort()

            print(f"\n[{model_dir}] weight_scale keys for layers 0-1:")
            seen_files = {}
            for key in target_keys[:20]:
                shard_file = f"{model_dir}/{weight_map[key]}"
                if shard_file not in seen_files:
                    seen_files[shard_file] = safe_open(shard_file, framework="pt")
                f = seen_files[shard_file]
                if key in f.keys():
                    t = f.get_tensor(key)
                    print(f"  {key}: shape={t.shape} dtype={t.dtype}")
                    info[key] = {"shape": list(t.shape), "dtype": str(t.dtype)}

        results[model_dir] = info

    return results


@app.function(
    image=soar_image,
    gpu="RTX-PRO-6000",
    timeout=21600,  # 6 hours — matches SOAR 5hr limit + buffer
    volumes={"/models": model_volume, "/workspace": workspace_volume},
)
def simulate_soar(num_samples: int = None):
    """
    Full SOAR platform simulation: prepare_env -> prepare_model -> server -> eval.
    Times each phase so you can see exactly where time is spent.
    """
    import shlex
    import shutil
    import threading
    import time

    submission_dir = "/workspace/submission"
    model_input = "/models/MiniCPM-SALA"
    model_output = "/workspace/nvfp4_model_simulate"

    if not os.path.exists(submission_dir):
        return {"success": False, "error": f"No submission dir at {submission_dir}. Run: modal run modal_app.py --cmd sync"}

    timings = {}

    # ── Phase 1: prepare_env.sh ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("Phase 1: prepare_env.sh")
    print("="*60)
    t0 = time.time()
    # source the script in a bash subprocess and dump exported env vars
    env_script = os.path.join(submission_dir, "prepare_env.sh")
    if not os.path.exists(env_script):
        return {"success": False, "error": "prepare_env.sh not found in submission dir"}

    # Source prepare_env.sh and capture the resulting environment
    env_dump_cmd = f'cd {submission_dir} && source {env_script} && env'
    env_result = subprocess.run(
        ["bash", "-c", env_dump_cmd],
        capture_output=True, text=True
    )
    if env_result.returncode != 0:
        print(f"[prepare_env] stderr: {env_result.stderr[-2000:]}")
        return {"success": False, "error": "prepare_env.sh failed"}
    print(env_result.stdout[-3000:])

    # Parse exported environment from the dump
    extra_env = {}
    for line in env_result.stdout.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            extra_env[k] = v
    server_args_str = extra_env.get("SGLANG_SERVER_ARGS", "")
    print(f"\n[simulate] SGLANG_SERVER_ARGS = {server_args_str!r}")
    timings["prepare_env_s"] = round(time.time() - t0, 1)
    print(f"[simulate] Phase 1 done in {timings['prepare_env_s']}s")

    # ── Inspect existing quantized model configs ──────────────────────────────
    for cfg_path in ["/workspace/nvfp4_model/hf_quant_config.json",
                     "/workspace/nvfp4_model/config.json",
                     "/workspace/nvfp4_model_simulate/hf_quant_config.json",
                     "/workspace/nvfp4_model_simulate/config.json"]:
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                import json as _json
                content = _json.load(f)
            print(f"\n[inspect] {cfg_path}:")
            print(_json.dumps(content.get("quantization_config", content), indent=2)[:1000])

    # ── Phase 2: prepare_model.sh ─────────────────────────────────────────────
    print("\n" + "="*60)
    print("Phase 2: prepare_model.sh")
    print("="*60)
    t0 = time.time()
    model_script = os.path.join(submission_dir, "prepare_model.sh")
    if os.path.exists(model_script):
        # Clean output dir so we always requantize (matches SOAR cold start)
        if os.path.exists(model_output):
            shutil.rmtree(model_output)
        quant_cmd = ["bash", model_script, "--input", model_input, "--output", model_output]
        print(f"[simulate] Running: {' '.join(quant_cmd)}")
        out, rc = run_streaming(quant_cmd, label="prepare_model")
        timings["prepare_model_s"] = round(time.time() - t0, 1)
        print(f"[simulate] Phase 2 done in {timings['prepare_model_s']}s")
        if rc != 0:
            return {"success": False, "error": "prepare_model.sh failed", "timings": timings}
    else:
        print("[simulate] No prepare_model.sh — skipping")
        timings["prepare_model_s"] = 0

    # ── Phase 3: Start SGLang server using SGLANG_SERVER_ARGS ─────────────────
    print("\n" + "="*60)
    print("Phase 3: SGLang server startup")
    print("="*60)
    t0 = time.time()

    server_cmd = ["python3", "-m", "sglang.launch_server"] + shlex.split(server_args_str)
    print(f"[simulate] Server cmd: {' '.join(server_cmd)}")

    merged_env = {**os.environ, **extra_env}
    server_proc = subprocess.Popen(
        server_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=merged_env,
        cwd=submission_dir,
    )

    def stream_server_logs():
        for line in server_proc.stdout:
            print(f"[server] {line.rstrip()}", flush=True)

    log_thread = threading.Thread(target=stream_server_logs, daemon=True)
    log_thread.start()

    try:
        wait_for_server(server_proc=server_proc)
        timings["server_startup_s"] = round(time.time() - t0, 1)
        print(f"[simulate] Phase 3 done in {timings['server_startup_s']}s")

        # ── Phase 4: eval_model.py with concurrency=32 ────────────────────────
        print("\n" + "="*60)
        print("Phase 4: eval_model.py (concurrency=32, matching SOAR)")
        print("="*60)
        t0 = time.time()

        eval_cmd = [
            "python3", "/opt/SOAR/eval_model.py",
            "--api_base", "http://localhost:30000",
            "--model_path", model_input,
            "--data_path", "/opt/SOAR/eval_dataset/perf_public_set.jsonl",
            "--concurrency", "32",
        ]
        if num_samples:
            eval_cmd.extend(["--num_samples", str(num_samples)])

        print(f"[simulate] Running: {' '.join(eval_cmd)}")
        eval_out, eval_rc = run_streaming(eval_cmd, label="eval")
        timings["eval_s"] = round(time.time() - t0, 1)
        timings["total_s"] = sum(timings.values())
        print(f"[simulate] Phase 4 done in {timings['eval_s']}s")

    finally:
        server_proc.terminate()
        server_proc.wait()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("SOAR SIMULATION SUMMARY")
    print("="*60)
    for k, v in timings.items():
        print(f"  {k:<25} {v:.0f}s  ({v/60:.1f} min)")
    print(f"  {'SOAR 5hr limit':<25} 18000s (300 min)")
    budget_used = timings.get("total_s", 0) / 18000 * 100
    print(f"  {'Budget used':<25} {budget_used:.1f}%")

    acc_match = re.search(r"Average Score:\s*([0-9.]+)%?", eval_out)
    accuracy = float(acc_match.group(1)) if acc_match else None
    print(f"\n  Accuracy: {accuracy}%")

    return {"success": True, "timings": timings, "accuracy": accuracy}


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
  modal run modal_app.py --cmd fp8           # FP8 W8A8 quant + eval (~20min)
  modal run modal_app.py --cmd nvfp4         # NVFP4 E2M1 quant + eval (~30min) ← Blackwell
  modal run modal_app.py --cmd gptq          # Test GPTQ quantization (1-2hr)

SUBMISSION:
  modal run modal_app.py --cmd simulate      # Full SOAR simulation (prepare_env + prepare_model + server + eval)
  modal run modal_app.py --cmd simulate --num-samples 10  # Quick simulation
  modal run modal_app.py --cmd package       # Create .tar.gz

INTERACTIVE:
  modal shell modal_app.py::dev_shell        # RTX PRO 6000 shell (4hr)

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
                    [
                        "modal",
                        "volume",
                        "put",
                        "soar-workspace",
                        local_path,
                        "/",
                        "--force",
                    ]
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
            print(f"  modal volume get soar-workspace /packages ./packages --force")
        else:
            print(f"\n✗ Failed: {result.get('error')}")

    elif cmd == "fp8":
        print(
            "Testing FP8 W8A8 quantization (per-channel static weights, dynamic activations)..."
        )
        print("Expected: ~20min quantization + eval\n")
        result = test_fp8_quantization.remote(num_samples=num_samples or 50)
        if result.get("success"):
            print("\n✓ FP8 quantization succeeded")
            print("\n--- EVAL ---")
            print(result.get("eval_log", ""))
            print("\n--- BENCH ---")
            print(result.get("bench_log", ""))
        else:
            print(f"\n✗ Failed: {result.get('error')}")
            print(result.get("log", ""))

    elif cmd == "gptq":
        print("Testing GPTQ W4A16 quantization...")
        print("This will take 1-2 hours (quantization + eval)\n")
        result = test_gptq_quantization.remote(
            num_samples=num_samples or 10,
        )
        print(f"\nResult: {result}")

    elif cmd == "nvfp4":
        print("Testing NVFP4 (E2M1 + dual-scale) quantization + eval...")
        print("Expected: ~30min quantization + eval\n")
        result = test_nvfp4_quantization.remote(
            num_samples=num_samples or 10, force_requantize=False
        )
        if result.get("success"):
            print("\n✓ NVFP4 succeeded")
            print(f"  S1 estimate: {result['s1_estimate_s']}s  (target: 420s)")
            print("\n--- EVAL ---")
            print(result.get("eval_log", ""))
        else:
            print(f"\n✗ Failed: {result.get('error')}")
            print(result.get("log", ""))

    elif cmd == "check-model":
        print("Checking MiniCPM-SALA model config...\n")
        result = check_model_architecture.remote()
        print(f"\n{result.get('output', str(result))}")
        print(f"\nHybrid Supported: {result.get('hybrid_supported', 'Unknown')}")

    elif cmd == "simulate":
        result = simulate_soar.remote(num_samples=num_samples)
        print(f"\nResult: {result}")

    elif cmd == "inspect-quant":
        result = inspect_quant_model.remote()
        print(f"\nResult: {result}")

    elif cmd == "patch-quant":
        result = patch_quant_configs.remote()
        print(f"\nResult: {result}")

    elif cmd == "shell":
        print("Use: modal shell modal_app.py::dev_shell")

    else:
        print(f"Unknown command: {cmd}")
        print("Run: modal run modal_app.py --cmd help")
