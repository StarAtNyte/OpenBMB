#!/bin/bash
# Build sgl-kernel wheel for SOAR 2026 competition
#
# Competition Requirements:
# - Wheel must ONLY contain: sgl_kernel/, sgl_kernel.libs/, sgl_kernel-.dist-info/
# - No: deep_gemm/, flash_attn_origin/, triton_kernels/, include/, lib/

set -e

SGL_KERNEL_PATH=${1:-/workspace/sglang/sgl-kernel}
OUTPUT_DIR=${2:-/workspace/dist}

echo "=============================================="
echo "SOAR 2026 - sgl-kernel Wheel Builder"
echo "=============================================="
echo "Source:  $SGL_KERNEL_PATH"
echo "Output:  $OUTPUT_DIR"
echo "=============================================="

cd "$SGL_KERNEL_PATH"

# Backup original CMakeLists.txt
if [ ! -f CMakeLists.txt.orig ]; then
    cp CMakeLists.txt CMakeLists.txt.orig
fi

echo "Patching CMakeLists.txt for competition compliance..."

# Create a patched CMakeLists.txt that excludes third-party installations
# We need to keep FetchContent for build-time deps but not install them
python3 << 'EOF'
import re

with open('CMakeLists.txt', 'r') as f:
    content = f.read()

# Lines to comment out (these install third-party dirs to wheel)
patterns = [
    r'^install\(TARGETS deep_gemm_cpp.*$',
    r'^install\(\s*DIRECTORY \$\{repo-deepgemm_SOURCE_DIR\}.*$',
    r'^install\(DIRECTORY "\$\{repo-triton_SOURCE_DIR\}.*$',
    r'^install\(DIRECTORY "\$\{FLASH_ATTN_CUTE_DST\}.*$',
    r'^install\(\s*FILES \$\{CMAKE_CURRENT_BINARY_DIR\}/deepgemm_pkg_init\.py.*$',
    r'^install\(DIRECTORY "\$\{repo-cutlass_SOURCE_DIR\}/include/cute/".*$',
    r'^install\(DIRECTORY "\$\{repo-cutlass_SOURCE_DIR\}/include/cutlass/".*$',
]

for pattern in patterns:
    content = re.sub(pattern, r'# SOAR_REMOVED: \g<0>', content, flags=re.MULTILINE)

with open('CMakeLists.txt', 'w') as f:
    f.write(content)

print("CMakeLists.txt patched successfully")
EOF

# Clean previous builds
rm -rf build/ dist/ *.egg-info

# Install build dependencies
pip install build wheel scikit-build-core cmake ninja 2>/dev/null || true

echo "Building wheel..."
python -m build --wheel

# Verify wheel contents
echo ""
echo "Verifying wheel contents..."
WHEEL=$(ls dist/*.whl | head -1)

if [ -z "$WHEEL" ]; then
    echo "ERROR: No wheel produced!"
    exit 1
fi

echo "Built: $WHEEL"
echo ""
echo "Wheel contents:"
python3 << EOF
import zipfile
import sys

wheel_path = "$WHEEL"
with zipfile.ZipFile(wheel_path, 'r') as zf:
    top_level = set()
    for name in zf.namelist():
        top = name.split('/')[0]
        if top:
            top_level.add(top)

    allowed = {'sgl_kernel', 'sgl_kernel.libs'}

    print("Top-level directories:")
    violations = []
    for d in sorted(top_level):
        status = "✓" if (d in allowed or d.endswith('.dist-info') or d.startswith('sgl_kernel')) else "✗"
        if status == "✗":
            violations.append(d)
        print(f"  {status} {d}")

    if violations:
        print(f"\n⚠️  WARNING: Disallowed directories found: {violations}")
        print("This wheel will FAIL competition validation!")
        sys.exit(1)
    else:
        print("\n✅ Wheel is competition-compliant!")
EOF

# Copy to output directory
mkdir -p "$OUTPUT_DIR"
cp "$WHEEL" "$OUTPUT_DIR/"

echo ""
echo "Wheel copied to: $OUTPUT_DIR/$(basename $WHEEL)"
