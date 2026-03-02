#!/usr/bin/env python3
"""
Create SOAR 2026 submission package (.tar.gz)

Works on Windows, macOS, and Linux.

Usage:
    python scripts/create_submission.py
    python scripts/create_submission.py --output my_submission.tar.gz
    python scripts/create_submission.py --include-sglang  # include custom sglang
"""

import argparse
import os
import shutil
import tarfile
from datetime import datetime
from pathlib import Path


def create_submission(
    output_path: str = None,
    include_sglang: bool = True,
    include_wheel: bool = True,
):
    """Create submission .tar.gz package."""

    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    submission_dir = project_root / "submission"

    if not submission_dir.exists():
        print(f"ERROR: submission directory not found at {submission_dir}")
        return False

    # Output path
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = project_root / f"submission_{timestamp}.tar.gz"
    else:
        output_path = Path(output_path)

    print(f"Creating submission package: {output_path}")
    print(f"Source: {submission_dir}")
    print()

    with tarfile.open(output_path, "w:gz") as tar:
        # Required: prepare_env.sh
        prepare_env = submission_dir / "prepare_env.sh"
        if prepare_env.exists():
            tar.add(prepare_env, arcname="prepare_env.sh")
            print(f"  ✓ prepare_env.sh")
        else:
            print(f"  ✗ prepare_env.sh (REQUIRED - missing!)")
            return False

        # Optional: prepare_model.sh
        prepare_model = submission_dir / "prepare_model.sh"
        if prepare_model.exists():
            tar.add(prepare_model, arcname="prepare_model.sh")
            print(f"  ✓ prepare_model.sh")

        # Optional: preprocess_model.py
        preprocess = submission_dir / "preprocess_model.py"
        if preprocess.exists():
            tar.add(preprocess, arcname="preprocess_model.py")
            print(f"  ✓ preprocess_model.py")

        # Optional: custom sglang
        if include_sglang:
            sglang_python = project_root / "sglang" / "python"
            if sglang_python.exists():
                tar.add(sglang_python, arcname="sglang/python")
                print(f"  ✓ sglang/python/")

        # Optional: wheels
        if include_wheel:
            wheel_dirs = [
                submission_dir / "wheels",
                project_root / "sglang" / "sgl-kernel" / "dist",
            ]
            for wheel_dir in wheel_dirs:
                if wheel_dir.exists():
                    for wheel in wheel_dir.glob("*.whl"):
                        tar.add(wheel, arcname=f"wheels/{wheel.name}")
                        print(f"  ✓ wheels/{wheel.name}")
                    break

    # Check size
    size_bytes = output_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)

    print()
    print(f"Package created: {output_path}")
    print(f"Size: {size_mb:.2f} MB")

    if size_mb > 2048:
        print()
        print("⚠️  WARNING: Package exceeds 2GB limit!")
        print("   Remove unnecessary files or use more aggressive compression.")
        return False
    else:
        print("✓ Package is within 2GB limit")

    return True


def main():
    parser = argparse.ArgumentParser(description="Create SOAR 2026 submission package")
    parser.add_argument("--output", "-o", help="Output .tar.gz path")
    parser.add_argument(
        "--include-sglang",
        action="store_true",
        default=True,
        help="Include custom sglang code",
    )
    parser.add_argument(
        "--no-sglang", action="store_true", help="Exclude custom sglang code"
    )
    parser.add_argument(
        "--include-wheel",
        action="store_true",
        default=True,
        help="Include sgl-kernel wheel",
    )
    parser.add_argument(
        "--no-wheel", action="store_true", help="Exclude sgl-kernel wheel"
    )

    args = parser.parse_args()

    include_sglang = args.include_sglang and not args.no_sglang
    include_wheel = args.include_wheel and not args.no_wheel

    success = create_submission(
        output_path=args.output,
        include_sglang=include_sglang,
        include_wheel=include_wheel,
    )

    exit(0 if success else 1)


if __name__ == "__main__":
    main()
