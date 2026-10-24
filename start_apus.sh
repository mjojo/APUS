#!/usr/bin/env bash
set -euo pipefail

# Always operate relative to the repository root where start_apus.sh is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export LANG="${LANG:-en_US.UTF-8}"
export LC_ALL="${LC_ALL:-en_US.UTF-8}"

APUS_BIN="target/release/apus"
MODE="${1:-}"

echo "==============================================================================="
echo "   APUS // ASYNC PAGED ULTRA-STREAMING MoE ENGINE v0.1.0 (by mjojo)"
echo "==============================================================================="

# 1. Ensure release binary is built
if [[ ! -x "$APUS_BIN" ]]; then
    echo "[APUS Launcher] Release binary not found. Building optimized binary..."
    cargo build --release --bin apus
fi

# 2. Ensure OOBE Hardware Wizard has initialized apus.toml
if [[ ! -f "apus.toml" ]]; then
    echo "[APUS Launcher] First launch detected. Running OOBE Hardware Wizard..."
    "./$APUS_BIN" wizard apus.toml
fi

run_full_test() {
    echo ""
    echo "[Step 1/7] Running OOBE Hardware & Disk Volume Audit..."
    "./$APUS_BIN" wizard apus.toml

    echo ""
    echo "[Step 2/7] Checking Frontier MoE Profiles Catalog..."
    "./$APUS_BIN" profiles

    echo ""
    echo "[Step 3/7] Testing GGUF v3 -> 4KB Page-Aligned .apus Importer..."
    local tmp_apus="${TMPDIR:-/tmp}/apus_batch_test_$$.apus"
    "./$APUS_BIN" import sample_moe.gguf "$tmp_apus" 2 4

    echo ""
    echo "[Step 4/7] Inspecting 512B Header, LBA Offset Table & BLAKE3 Root..."
    "./$APUS_BIN" inspect "$tmp_apus"

    echo ""
    echo "[Step 5/7] Running End-to-End SSD Streaming, SIMD & .roost Benchmark..."
    "./$APUS_BIN" bench "$tmp_apus"

    echo ""
    echo "[Step 6/7] Running Mathematical SQNR/Perplexity Audit & 8-Thread Stress Test..."
    "./$APUS_BIN" audit
    "./$APUS_BIN" stress

    echo ""
    echo "[Step 7/7] Verifying 60 FPS Boot Splash & Telemetry HUD Rendering..."
    "./$APUS_BIN" run --headless

    rm -f "$tmp_apus" "coding_rust.roost"

    echo ""
    echo "==============================================================================="
    echo " [APUS Launcher] ALL 7 END-TO-END DIAGNOSTIC STAGES PASSED SUCCESSFULLY!"
    echo "==============================================================================="
}

# 3. Dispatch non-interactive CLI modes if passed as arguments
case "${MODE,,}" in
    --test|test)
        run_full_test
        exit 0
        ;;
    --headless)
        exec "./$APUS_BIN" run --headless
        ;;
    tui|run)
        shift || true
        exec "./$APUS_BIN" run "$@"
        ;;
    serve)
        PORT="${2:-8080}"
        MODEL_ARG="${3:-}"
        if [[ -n "$MODEL_ARG" ]]; then
            exec "./$APUS_BIN" serve "$PORT" "$MODEL_ARG"
        else
            exec "./$APUS_BIN" serve "$PORT"
        fi
        ;;
    wizard)
        CUSTOM_DIR="${2:-}"
        if [[ -n "$CUSTOM_DIR" ]]; then
            exec "./$APUS_BIN" wizard apus.toml "$CUSTOM_DIR"
        else
            exec "./$APUS_BIN" wizard apus.toml
        fi
        ;;
    profiles)
        exec "./$APUS_BIN" profiles
        ;;
    pull)
        CHOICE_ARG="${2:-}"
        if [[ -n "$CHOICE_ARG" ]]; then
            exec "./$APUS_BIN" pull "$CHOICE_ARG"
        else
            exec "./$APUS_BIN" pull
        fi
        ;;
    bench)
        MODEL_ARG="${2:-}"
        if [[ -n "$MODEL_ARG" ]]; then
            exec "./$APUS_BIN" bench "$MODEL_ARG"
        else
            exec "./$APUS_BIN" bench
        fi
        ;;
esac

# 4. Interactive Launcher Menu
while true; do
    echo ""
    echo " Select launch mode:"
    echo "   [1] Launch 60 FPS Aerospace Telemetry TUI (apus run)"
    echo "   [2] Start OpenAI / Anthropic HTTP+SSE Server on port 8080 (apus serve)"
    echo "   [3] Run Hardware & Storage Auto-Tuner Wizard (apus wizard)"
    echo "   [4] List Supported Frontier MoE Profiles & Disk Readiness (apus profiles)"
    echo "   [5] Open Interactive Model Hub / Importer (apus pull)"
    echo "   [6] Run Full End-to-End Diagnostic & Benchmark Suite (apus --test)"
    echo "   [0] Exit"
    echo ""
    read -r -p "Enter selection [0-6]: " CHOICE

    case "$CHOICE" in
        1)
            exec "./$APUS_BIN" run
            ;;
        2)
            exec "./$APUS_BIN" serve 8080
            ;;
        3)
            exec "./$APUS_BIN" wizard apus.toml
            ;;
        4)
            exec "./$APUS_BIN" profiles
            ;;
        5)
            exec "./$APUS_BIN" pull
            ;;
        6)
            run_full_test
            exit 0
            ;;
        0)
            exit 0
            ;;
        *)
            echo "[APUS Launcher] Invalid option. Please select 0-6."
            ;;
    esac
done
