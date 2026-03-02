#!/bin/bash
# Fast development build for SOAR 2026
#
# Instead of rebuilding the entire wheel (600+ compilation units, ~2 hours),
# this script only rebuilds common_ops for SM90 (H100) or SM89 (RTX 6000)
# and copies the .so into the installed package.
#
# First build: ~10-15 minutes
# Incremental: <2 minutes (with ccache)
#
# Usage (inside Modal dev shell or GPU container):
#   bash /workspace/scripts/fast_dev_build.sh              # incremental
#   bash /workspace/scripts/fast_dev_build.sh full         # clean build
#   bash /workspace/scripts/fast_dev_build.sh rtx6000      # build for RTX 6000

set -euo pipefail

# Find kernel directory
KERNEL_DIR=""
for path in "/workspace/sglang/sgl-kernel" "/workspace/sgl-kernel" "./sglang/sgl-kernel"; do
    if [ -d "$path" ]; then
        KERNEL_DIR="$path"
        break
    fi
done

if [ -z "$KERNEL_DIR" ]; then
    echo "ERROR: sgl-kernel directory not found"
    exit 1
fi

BUILD_DIR="${KERNEL_DIR}/_dev_build"

# Get installed sgl_kernel path
INSTALL_DIR=$(python3 -c "import sgl_kernel; import os; print(os.path.dirname(sgl_kernel.__file__))" 2>/dev/null || echo "")

if [ -z "$INSTALL_DIR" ]; then
    echo "WARNING: sgl_kernel not installed, will only build"
fi

MODE="${1:-incremental}"

# Detect GPU architecture
GPU_ARCH=""
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    case "$GPU_NAME" in
        *H100*|*H200*)
            GPU_ARCH="sm90"
            ;;
        *RTX*6000*|*RTX*5000*|*RTX*4090*|*RTX*4080*)
            GPU_ARCH="sm89"
            ;;
        *A100*|*A6000*)
            GPU_ARCH="sm80"
            ;;
        *)
            GPU_ARCH="sm90"  # default
            ;;
    esac
fi

# Override for RTX 6000 target
if [ "$MODE" = "rtx6000" ]; then
    GPU_ARCH="sm89"
    MODE="incremental"
fi

echo "=== SOAR Fast Dev Build ==="
echo "Kernel dir:  $KERNEL_DIR"
echo "Build dir:   $BUILD_DIR"
echo "Install dir: ${INSTALL_DIR:-'(not installed)'}"
echo "GPU arch:    $GPU_ARCH"
echo "Mode:        $MODE"
echo ""

# Setup ccache
if command -v ccache &> /dev/null; then
    export CCACHE_DIR="${CCACHE_DIR:-/workspace/.ccache}"
    mkdir -p "$CCACHE_DIR"
    echo "ccache enabled: $CCACHE_DIR"
    ccache --show-stats 2>/dev/null | head -3 || true
fi

if [ "$MODE" = "full" ]; then
    echo ""
    echo "=== Clean build ==="
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"

# Configure CMake (only if not already configured or full build)
if [ ! -f "$BUILD_DIR/CMakeCache.txt" ] || [ "$MODE" = "full" ]; then
    echo ""
    echo "=== Configuring CMake (${GPU_ARCH} only, no extras) ==="

    # Get torch cmake prefix
    TORCH_CMAKE=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")

    CMAKE_ARGS=(
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_PREFIX_PATH="$TORCH_CMAKE"
        -DSGL_KERNEL_COMPILE_THREADS=32
        -DENABLE_CCACHE=ON
    )

    # Architecture-specific flags
    case "$GPU_ARCH" in
        sm90)
            CMAKE_ARGS+=(
                -DENABLE_BELOW_SM90=OFF
                -DSGL_KERNEL_ENABLE_FA3=OFF
                -DSGL_KERNEL_ENABLE_SM100A=OFF
                -DSGL_KERNEL_ENABLE_FP4=OFF
            )
            ;;
        sm89)
            CMAKE_ARGS+=(
                -DENABLE_BELOW_SM90=ON
                -DSGL_KERNEL_ENABLE_FA3=OFF
                -DSGL_KERNEL_ENABLE_SM100A=OFF
                -DSGL_KERNEL_ENABLE_FP4=OFF
                -DCMAKE_CUDA_ARCHITECTURES=89
            )
            ;;
        sm80)
            CMAKE_ARGS+=(
                -DENABLE_BELOW_SM90=ON
                -DSGL_KERNEL_ENABLE_FA3=OFF
                -DSGL_KERNEL_ENABLE_SM100A=OFF
                -DCMAKE_CUDA_ARCHITECTURES=80
            )
            ;;
    esac

    cmake -S "$KERNEL_DIR" -B "$BUILD_DIR" "${CMAKE_ARGS[@]}" 2>&1 | tail -20

    echo ""
    echo "✓ CMake configured for $GPU_ARCH"
fi

# Build target based on architecture
BUILD_TARGET=""
case "$GPU_ARCH" in
    sm90)
        BUILD_TARGET="common_ops_sm90_build"
        SO_SUBDIR="sm90"
        ;;
    sm89|sm80)
        BUILD_TARGET="common_ops_below_sm90_build"
        SO_SUBDIR="below_sm90"
        ;;
esac

echo ""
echo "=== Building $BUILD_TARGET ==="
echo ""

time cmake --build "$BUILD_DIR" \
    --target "$BUILD_TARGET" \
    --parallel $(nproc) \
    2>&1

echo ""

# Find and install the built .so
SO_FILE=""
for pattern in "$BUILD_DIR/$SO_SUBDIR/common_ops*" "$BUILD_DIR/**/common_ops*"; do
    for f in $pattern; do
        if [ -f "$f" ] && [[ "$f" == *.so ]]; then
            SO_FILE="$f"
            break 2
        fi
    done
done

if [ -n "$SO_FILE" ] && [ -n "$INSTALL_DIR" ]; then
    mkdir -p "$INSTALL_DIR/$SO_SUBDIR"
    cp "$SO_FILE" "$INSTALL_DIR/$SO_SUBDIR/"
    echo "✓ Installed: $(basename $SO_FILE) -> $INSTALL_DIR/$SO_SUBDIR/"
elif [ -n "$SO_FILE" ]; then
    echo "✓ Built: $SO_FILE (not installed - sgl_kernel not found)"
else
    echo "✗ Build may have failed - no common_ops.so found"
    exit 1
fi

# Show ccache stats
if command -v ccache &> /dev/null && [ -d "${CCACHE_DIR:-}" ]; then
    echo ""
    echo "=== ccache stats ==="
    ccache --show-stats 2>/dev/null | head -5 || true
fi

echo ""
echo "=== Done! ==="
echo "Test with: python3 -c \"import sgl_kernel; print('OK')\""
echo ""
echo "For next changes, just run:"
echo "  bash /workspace/scripts/fast_dev_build.sh"
