#!/bin/bash
# SOAR 2026 Benchmark Runner
# Runs benchmarks at all three concurrency levels
#
# Usage:
#   ./benchmark.sh                           # All levels with generated data
#   ./benchmark.sh <api_base>                # Custom API base
#   ./benchmark.sh <api_base> <data.jsonl>   # Use provided data file
#
# Environment variables:
#   API_BASE         - SGLang server URL (default: http://localhost:30000)
#   SPEED_DATA_PATH  - Path to JSONL data file
#   NUM_SAMPLES      - Samples to generate if no data file (default: 50)
#
# Scoring weights:
#   S1 (--max-concurrent 1):  40%  (bounty: 420s)
#   S8 (--max-concurrent 8):  30%  (bounty: 420s)
#   Smax (no limit):          30%  (bounty: 720s)

set -e

API_BASE=${1:-${API_BASE:-http://localhost:30000}}
SPEED_DATA_PATH=${2:-${SPEED_DATA_PATH:-/tmp/soar_speed_data.jsonl}}
NUM_SAMPLES=${NUM_SAMPLES:-50}

# Find SOAR benchmark script
SOAR_BENCH=""
for path in "/opt/SOAR/bench_serving.sh" "/workspace/SOAR/bench_serving.sh" "./SOAR/bench_serving.sh"; do
    if [ -f "$path" ]; then
        SOAR_BENCH="$path"
        break
    fi
done

if [ -z "$SOAR_BENCH" ]; then
    echo "ERROR: SOAR bench_serving.sh not found"
    echo "Make sure you're running in the SOAR container environment"
    exit 1
fi

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              SOAR 2026 Speed Benchmark                       ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  API Base:    $API_BASE"
echo "║  Data Path:   $SPEED_DATA_PATH"
echo "║  Bench Script: $SOAR_BENCH"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check server health
echo "Checking server health..."
if ! curl -s "$API_BASE/health" > /dev/null 2>&1; then
    echo "ERROR: Server not responding at $API_BASE"
    echo "Start the server first with: bash scripts/run_sglang.sh"
    exit 1
fi
echo "Server is healthy ✓"
echo ""

# Generate test data if not exists
if [ ! -f "$SPEED_DATA_PATH" ]; then
    echo "Generating $NUM_SAMPLES test samples..."
    python3 << EOF
import json
import random

def random_input_length():
    r = random.random()
    if r < 0.25: return random.randint(100, 4000)
    elif r < 0.35: return random.randint(4000, 16000)
    elif r < 0.50: return random.randint(16000, 32000)
    elif r < 0.85: return random.randint(32000, 128000)
    else: return random.randint(128000, 160000)

def random_output_length():
    r = random.random()
    if r < 0.35: return random.randint(50, 512)
    elif r < 0.60: return random.randint(512, 2000)
    elif r < 0.70: return random.randint(2000, 4000)
    elif r < 0.85: return random.randint(4000, 16000)
    else: return random.randint(16000, 32000)

with open("$SPEED_DATA_PATH", "w") as f:
    for _ in range($NUM_SAMPLES):
        sample = {
            "question": "x " * (random_input_length() // 2),
            "model_response": "y " * (random_output_length() // 2),
        }
        f.write(json.dumps(sample) + "\n")

print(f"Generated $NUM_SAMPLES samples to $SPEED_DATA_PATH")
EOF
    echo ""
fi

# Store results
declare -A DURATIONS
DURATIONS=()

# Run each benchmark level
for LEVEL in S1 S8 Smax; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    case $LEVEL in
        S1)
            echo "Running $LEVEL benchmark (--max-concurrent 1) - 40% weight"
            echo "Bounty target: 420s"
            export SPEED_DATA_S1="$SPEED_DATA_PATH"
            unset SPEED_DATA_S8 SPEED_DATA_SMAX
            ;;
        S8)
            echo "Running $LEVEL benchmark (--max-concurrent 8) - 30% weight"
            echo "Bounty target: 420s"
            export SPEED_DATA_S8="$SPEED_DATA_PATH"
            unset SPEED_DATA_S1 SPEED_DATA_SMAX
            ;;
        Smax)
            echo "Running $LEVEL benchmark (no concurrency limit) - 30% weight"
            echo "Bounty target: 720s"
            export SPEED_DATA_SMAX="$SPEED_DATA_PATH"
            unset SPEED_DATA_S1 SPEED_DATA_S8
            ;;
    esac

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    # Run benchmark and capture output
    bash "$SOAR_BENCH" "$API_BASE" 2>&1 | tee /tmp/bench_${LEVEL}.log

    # Extract duration from output
    DURATION=$(grep -oP "Benchmark duration \(s\):\s+\K[0-9.]+" /tmp/bench_${LEVEL}.log 2>/dev/null || echo "FAILED")
    DURATIONS[$LEVEL]="$DURATION"

    echo ""
    echo "→ $LEVEL Duration: ${DURATION}s"
    echo ""
done

# Summary
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    Benchmark Summary                         ║"
echo "╠══════════════════════════════════════════════════════════════╣"

for LEVEL in S1 S8 Smax; do
    DUR="${DURATIONS[$LEVEL]}"

    case $LEVEL in
        S1)   BOUNTY=420; WEIGHT="40%" ;;
        S8)   BOUNTY=420; WEIGHT="30%" ;;
        Smax) BOUNTY=720; WEIGHT="30%" ;;
    esac

    if [ "$DUR" != "FAILED" ] && [ -n "$DUR" ]; then
        if (( $(echo "$DUR <= $BOUNTY" | bc -l 2>/dev/null || echo 0) )); then
            STATUS="✓ BEATS BOUNTY"
        else
            STATUS="(need: ${BOUNTY}s)"
        fi
        printf "║  %-5s: %8.2fs [%s] %s\n" "$LEVEL" "$DUR" "$WEIGHT" "$STATUS"
    else
        printf "║  %-5s: FAILED    [%s]\n" "$LEVEL" "$WEIGHT"
    fi
done

echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Note: These are synthetic test results. Final scoring uses  ║"
echo "║  the official private dataset on RTX PRO GPUs.               ║"
echo "╚══════════════════════════════════════════════════════════════╝"
