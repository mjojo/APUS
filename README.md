<div align="center">

```text
            ◣                  ▄▀█ █▀█ █ █ █▀   ASYNC PAGED ULTRA-STREAMING
           ██ ◣                █▀█ █▀▀ █▄█ ▄█   700B+ MoE Inference Runtime
     ◣     ████ ◣              ────────────────────────────────────────────
    ███▄▄███████▄▄             AUTHOR :: mjojo  │  KERNEL :: Zero-Copy DMA
     ▀▀██████████▀             FORMAT :: *.apus │  SESSION:: *.roost (MLA)
         ▀▀████▀
```

# APUS (`apus-moe`)

**Asynchronous Paged Ultra-Streaming Inference Runtime for 700B+ Mixture-of-Experts LLMs**

[![Rust Edition](https://img.shields.io/badge/Rust-2024_Edition-00F0FF?style=for-the-badge&logo=rust)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-7B61FF?style=for-the-badge)](LICENSE)
[![Format](https://img.shields.io/badge/Binary_Layout-4096B_Aligned_.apus-00E676?style=for-the-badge)](#binary-specifications-apus--roost)
[![Engineered By](https://img.shields.io/badge/Engineered_by-mjojo-FFB800?style=for-the-badge)](https://github.com/mjojo)

*Run frontier Mixture-of-Experts models (**GLM-5.2 744B**, **DeepSeek-V3 / R1 671B**, **Qwen-MoE**, **Mixtral 8x22B**) on consumer workstations with **24 GB RAM** and budget GPUs by streaming 4KB page-aligned expert blocks directly from NVMe storage.*

[Architecture](#system-architecture) •
[Key Innovations](#key-engineering-innovations) •
[Supported Profiles](#supported-frontier-moe-profiles-apus-profiles--apus-pull) •
[Binary Format (`*.apus` / `*.roost`)](#binary-specifications-apus--roost) •
[Quick Start](#quick-start) •
[Telemetry HUD](#aerospace-telemetry-tui) •
[OpenAI & Anthropic API](#openai--anthropic-compatible-httpsse-api)

</div>

---

## Overview

In biological taxonomy, both the hummingbird (*Colibri*) and the common swift (*Apus apus*) belong to the order **Apodiformes**. Where a hummingbird burns massive energy hovering in place, the common swift is engineered for high-speed, non-stop flight—staying airborne for up to 10 months without landing.

**APUS** (**A**synchronous **P**aged **U**ltra-**S**treaming) exploits the structural sparsity of modern Mixture-of-Experts (MoE) architectures. While a model such as **GLM-5.2** contains **744 billion total parameters** (~346 GB in INT4), only **~40B parameters** (Attention, Shared Experts, and Top-$K$ Routed Experts per layer) are activated for any given token.

APUS pins the dense backbone in VRAM/RAM and streams the active routed experts on-demand from NVMe SSDs via 4096-byte page-aligned DMA slots—bypassing the OS page cache, eliminating heap fragmentation, and protecting the host operating system from I/O starvation and thermal throttling.

---

## System Architecture

```text
                              ┌───────────────────────────────────────────────┐
                              │          Token Hidden State (Layer N)         │
                              └───────────────────────┬───────────────────────┘
                                                      │
                            ┌─────────────────────────┴─────────────────────────┐
                            ▼                                                   ▼
             ┌──────────────────────────────┐                    ┌──────────────────────────────┐
             │   Top-K Gating Router (N)    │                    │  Look-Ahead Predictor (N+1)  │
             │ Selects 8 / 288 Active Exps  │                    │  Issues Async NVMe Prefetch  │
             └──────────────┬───────────────┘                    └──────────────┬───────────────┘
                            │                                                   │
                            ▼                                                   │
      ┌───────────────────────────────────────────┐                             │
      │      3-Tier ARC Cache Manager (L1/L2)     │◄────────────────────────────┘
      │  • T1 (Recency) / T2 (Frequency) Balance  │
      │  • B1 / B2 Ghost Eviction History         │
      └──────┬──────────────────────┬─────────────┘
             │ Hit (L1/L2)          │ Cold Miss (L3 NVMe)
             │                      ▼
             │       ┌───────────────────────────────────────────┐
             │       │    OS Drive QoS & S.M.A.R.T. PID Guard    │
             │       │  • Idle I/O Priority (Bypass OS Freeze)   │
             │       │  • Token-Bucket Bandwidth Cap (80% Max)   │
             │       │  • PID Thermal Pacing (65°C) / Brake 70°C │
             │       └─────────────────────┬─────────────────────┘
             │                             │ 4096B Page-Aligned Read + BLAKE3 Verify
             │                             ▼
             │       ┌───────────────────────────────────────────┐
             │       │     4KB PageAlignedSlot (SlabArena)       │
             │       │     Pinned in Physical RAM (mlock/VLock)  │
             └──────►└─────────────────────┬─────────────────────┘
                                           │
                                           ▼
                     ┌───────────────────────────────────────────┐
                     │   On-the-Fly INT4 Register Dequantizer    │
                     │   SwiGLU GEMV + Dual-GPU P2P Pipeline     │
                     │   + MLA 576d KV-Cache + Native 2x MTP     │
                     └───────────────────────────────────────────┘
```

### Workspace Crate Topology

| Crate | Path | Responsibility |
| :--- | :--- | :--- |
| **`apus-core`** | [`crates/apus-core`](./crates/apus-core) | 4096B `PageAlignedSlot` (`VirtualLock` / `mlock` Anti-Swap), `SlabArena`, 3-tier `ArcCache` ($T_1/T_2/B_1/B_2$), `MlaKvCache` (576d) & `GqaKvCache`, `MoERouter`, `LookAheadPrefetcher`, dual-loop `MtpSpeculativeEngine` + `ExternalDraftSpeculativeVerifier`, `BpeTokenizer`, `GenerationSampler` (`DRY` / `Min-P` / `Top-P`), `GrammarConstraintEngine`, `HardwareTopologyScanner` (NUMA/CCD pinning), `detect_system_hardware` (`sysinfo` RAM auto-tuner), `ConcurrencyStressRunner`, and `MoEModelPreset` catalog. |
| **`apus-quant`** | [`crates/apus-quant`](./crates/apus-quant) | Zero-copy binary specifications for `*.apus` (512B header + 16B LBA Offset Table + BLAKE3 Merkle Tree) and `*.roost` (128B session snapshot), `SafetensorsMoEParser` & `GgufHeaderSummary` importers, `ApusConverter`, `ImatrixCalibrator` (`QuantizerInt8` / `QuantizerInt4` / `QuantizerInt2`), and `QuantizationQualityAuditor` (`SQNR dB` / `Cosine` / $\Delta\text{PPL}$). |
| **`apus-io`** | [`crates/apus-io`](./crates/apus-io) | Pluggable storage engines (`ApusBlockStorage` with unbuffered `FILE_FLAG_NO_BUFFERING` / `O_DIRECT` & SATA sequential LBA cluster sweep, `MmapStorage` with `madvise(MADV_WILLNEED)`, `KvCompressedStorage` with `KVZ4` lossless codec), `OsDriveQosController` (`QD <= 32` async semaphore, background thread I/O priority & bandwidth cap), `SmartThermalController` (PID thermal guard), and `ChunkedHubDownloader` (`*.apus.idx` manifest). |
| **`apus-backends`** | [`crates/apus-backends`](./crates/apus-backends) | Runtime CPU SIMD capability dispatcher (`AVX-512 VNNI`, `AVX2 + FMA3` hardware intrinsics, `ARM NEON`), on-the-fly INT4 register dequantization with SwiGLU GEMV execution, cross-platform `WGSL` GPU compute shader (`@compute @workgroup_size(64)`), and DirectStorage 1.2 `GDeflateCodec`. |
| **`apus-net`** | [`crates/apus-net`](./crates/apus-net) | `DualGpuPipeline` (2x 8GB VRAM asymmetric L1 cache + Chunked LM Head logits computation), P2P DMA / Pinned Host staging, `ExpertParallelReplicator` (Dynamic Replication of hot experts across 4x–8x GPUs), `IoAwareBatchScheduler` (Continuous Batching), and `LanRpcNode` (async TCP/RPC LAN ring with `Fp8ActivationPacket` compression). |
| **`apus-tui`** | [`crates/apus-tui`](./crates/apus-tui) | 60 FPS `mpsc`-decoupled Truecolor RGB Aerospace/Cyberpunk Terminal HUD (`ratatui`): animated Boot Splash screen, dynamic 8-dot Unicode Braille expert heatmap, live Markdown & Rust syntax highlighter, triple `L1/L2/Miss %` cache indicators, NVMe sparklines, and `[Tab]` panel navigation. |
| **`apus-cli`** | [`crates/apus-cli`](./crates/apus-cli) | Unified `apus` CLI (`convert`, `import`, `inspect`, `bench`, `audit`, `manifest`, `stress`, `profiles`, `wizard`, `run`, `serve`), end-to-end `ApusRuntimeSession`, and embedded `axum` HTTP/SSE server supporting both **OpenAI** (`/v1/chat/completions`, `/v1/models`) and **Anthropic** (`/v1/messages`) APIs. |

---

## Key Engineering Innovations

1. **Zero-Double-Caching, OS Page Pinning & 4KB Slab Arena (`SlabArena`)**
   Standard file reads pollute the OS Page Cache, duplicating expert weights in RAM and triggering swap thrashing. APUS opens containers with OS-native unbuffered flags (`FILE_FLAG_NO_BUFFERING` on Windows, `O_DIRECT` on Linux), aligns every expert block to **4096-byte sector boundaries**, and streams directly into pre-allocated `PageAlignedSlot` buffers locked in physical RAM via `VirtualLock` / `mlock`—guaranteeing zero heap fragmentation and zero pagefile swap over multi-day uptime.
2. **Adaptive Replacement Cache (`ArcCache`)**
   Unlike naive LRU caches that flush hot coding/math experts during a single topic shift, APUS implements a lock-free 3-tier **ARC** engine tracking both *Recency* ($T_1, B_1$) and *Frequency* ($T_2, B_2$) across **L1 VRAM** and **L2 Pinned Slab RAM** (verified at **1.74M+ ops/sec** across 8 OS threads).
3. **OS Drive Protection QoS, SATA AHCI Semaphore & S.M.A.R.T. PID Thermal Guard**
   Streaming gigabytes per second from a primary OS drive can starve the operating system and overheat NVMe controllers past 70°C. `apus-io` assigns background thread I/O priority (`THREAD_PRIORITY_LOWEST` / `IOPRIO_CLASS_IDLE`), enforces an async `tokio::sync::Semaphore` (`QD <= 32`) with contiguous LBA cluster reads for SATA SSDs, caps peak bandwidth (e.g., 5.5 GB/s out of 7.0 GB/s), and runs a closed-loop **PID thermal controller** that injects micro-sleeps at 65°C and trips an emergency thermal brake at 70°C.
4. **Hybrid MLA (576d) & GQA Paged KV-Cache + Dual-Loop Speculative Decoding**
   Supports both **Multi-Head Latent Attention (`MlaKvCache`)** compressing KV states to **576 latent values per token** in FP8/INT8 (~584 bytes/token) and classical **Grouped-Query Attention (`GqaKvCache`)**, paired with a dual speculative decoding loop: native **MTP heads** (`MtpSpeculativeEngine`) for GLM-5.2/DeepSeek-V3 and **Rejection Sampling (`ExternalDraftSpeculativeVerifier`)** for external draft models.
5. **Hardware `AVX2 + FMA3` Intrinsics, `WGSL` GPU Compute Shaders & `GDeflate`**
   Executes on-the-fly INT4 dequantization and SwiGLU GEMV directly in CPU vector registers (`std::arch::x86_64::_mm256_fmadd_ps`) or cross-platform Vulkan/Metal/DX12 `WGSL` compute shaders (`WGSL_INT4_SWIGLU_SHADER`), with lossless `GDeflateCodec` framing to reduce PCIe bus traffic.
6. **Multi-GPU Pipeline Parallelism, Server Dynamic Replication & LAN FP8 RPC**
   Supports `2x 8GB VRAM` budget rigs via `DualGpuPipeline` (asymmetric L1 caches + 16K Chunked LM Head logits computation), `4x–8x GPU` server rigs via `ExpertParallelReplicator` (automatic cloning of hot experts onto idle GPUs), and multi-node LAN clusters via `LanRpcNode` with `Fp8ActivationPacket` wire compression.

---

## Supported Frontier MoE Profiles (`apus profiles`)

| Model Preset ID | Total / Active Params | Topology (Layers × Experts) | Attention Mechanism | Native MTP | INT4 Disk Size | Min Slab RAM |
| :--- | :--- | :--- | :--- | :---: | :--- | :--- |
| **`GLM-5.2-744B-Q4_K_M`** | `744B` / `40B` | `75L × 288E` (`Top-8` + `2 Shared`) | `MLA (576d Latent KV)` | Yes (2x) | `346.4 GB` | `24.0 GB` |
| **`DeepSeek-V3-671B-Q4_K_M`** | `671B` / `37B` | `61L × 256E` (`Top-8` + `1 Shared`) | `MLA (576d Latent KV)` | Yes (2x) | `312.0 GB` | `20.0 GB` |
| **`Qwen3-235B-A22B-Q4_K_M`** | `235B` / `22B` | `94L × 128E` (`Top-8`) | `GQA (PagedAttention)` | Draft Verifier | `118.5 GB` | `16.0 GB` |
| **`Mixtral-8x22B-Instruct-Q4_K_M`** | `141B` / `39B` | `56L × 8E` (`Top-2`) | `GQA (PagedAttention)` | Draft Verifier | `74.2 GB` | `16.0 GB` |

---

## Binary Specifications (`*.apus` & `*.roost`)

### 1. Model Container Format: `*.apus` (v1.0)

Every `*.apus` file begins with a packed **512-byte header** verified before mounting:

| Byte Offset | Size | Type | Field | Description |
| :--- | :--- | :--- | :--- | :--- |
| `0x00..0x04` | 4 B | `[u8; 4]` | `magic` | Format magic bytes: strictly `b"APUS"` |
| `0x04..0x08` | 4 B | `u32` | `version` | Specification version (`0x00000001`) |
| `0x08..0x28` | 32 B | `[u8; 32]` | `author_sig` | Mandatory signature: `"apus-moe // authored by mjojo\0"` |
| `0x28..0x48` | 32 B | `[u8; 32]` | `model_id` | Null-terminated model identifier (e.g. `GLM-5.2-744B-Q4_K_M`) |
| `0x48..0x4C` | 4 B | `u32` | `total_layers` | Total transformer/MoE layers (e.g. `75`) |
| `0x4C..0x50` | 4 B | `u32` | `total_experts` | Total routed experts across all layers (e.g. `21504`) |
| `0x50..0x58` | 8 B | `u64` | `dense_size` | 4KB-aligned byte size of the resident dense backbone |
| `0x58..0x60` | 8 B | `u64` | `lba_table_offset` | Absolute byte offset of the LBA Expert Offset Table |
| `0x60..0x68` | 8 B | `u64` | `blake3_root_hash` | 64-bit root tag of the BLAKE3 Merkle integrity tree |
| `0x68..0x200` | 376 B | `[u8; 376]` | `reserved` | Zero-padded reserved expansion area |

Immediately following the 4KB-aligned dense weights is the **LBA Offset Table**, consisting of `total_experts` packed 16-byte entries (`layer_id: u16`, `expert_id: u16`, `file_offset: u64`, `compressed_size: u32`), where every `file_offset` and `compressed_size` is an exact multiple of `4096`.

### 2. Instant Session Snapshot Format: `*.roost` (v1.0)

Pressing `Ctrl+S` in the TUI (or completing a benchmark run) serializes the active conversation state into a `*.roost` snapshot with a **128-byte header** (`magic: b"ROST"`, `author_tag: "roost // created by mjojo"`), storing:
* **Compressed MLA Latent KV-Cache** ($c_t^{KV}$ vectors in FP8/INT8).
* **Warm Cache Profile** (`Vec<ExpertId>`) so subsequent launches pre-warm the exact set of active experts via linear DMA reads.

---

## Quick Start

### Prerequisites
* **Rust Toolchain:** `1.85+` (supports Rust `2024` edition; verified on `rustc 1.97.0`).

### 1. Clone & Run the Test Suite

```bash
git clone https://github.com/mjojo/apus.git
cd apus-moe
cargo test
```

### 2. First-Launch OOBE Wizard & Model Location Resolution (`apus wizard`)

When APUS starts for the first time and `apus.toml` does not exist yet, it automatically triggers the **First-Launch Initialization Wizard (`init.rs`)**:
1. **RAM Probing (`sysinfo`):** Queries total and available RAM, reserves 15% (`4–12 GB`) for the host OS, and assigns the remaining RAM to the 4KB Pinned L2 Slab Arena.
2. **Physical Disk & Volume Probing (`sysinfo::Disks`):** Enumerates all mounted drives (distinguishing `NVMe / SSD` from `HDD / Legacy` and checking free gigabytes for 300+ GB `.apus` containers).
3. **Deterministic Model Path Resolution (`resolve_model_path`):** APUS locates `.apus` model containers in the following priority order:
   * **Priority 1 (CLI argument):** Explicit file path passed to `apus run <path/to/model.apus>`, `apus serve 8080 <path/to/model.apus>`, `apus bench <path>`, or `apus inspect <path>`.
   * **Priority 2 (Environment variables):** `APUS_MODEL_PATH` (exact `.apus` file) or `APUS_MODELS_DIR` (directory on any NVMe drive, e.g. `D:\APUS_Models`).
   * **Priority 3 (`apus.toml` configuration):** `engine.model_path` or automatic discovery of the first `*.apus` container inside `engine.models_dir`.

```bash
# Run OOBE Wizard and optionally point the models directory to a dedicated NVMe drive:
cargo run --release --bin apus -- wizard apus.toml "C:/APUS_Models"

# Launch the interactive Model Hub & Downloader (with hardware compatibility badges & indicatif progress bar):
cargo run --release --bin apus -- pull
```

### 3. Import Upstream Hugging Face `.safetensors` / `.gguf` or Pack Synthetic MoE Weights

```bash
# Import a Hugging Face .safetensors or .gguf shard into a 4KB-aligned *.apus container:
cargo run --release --bin apus -- import model.safetensors glm-5.2-744b-q4.apus

# Or pack a reference INT4 (imatrix) *.apus container directly:
cargo run --release --bin apus -- convert glm-5.2-744b-q4.apus 4 8
```

### 4. Inspect Container Header, LBA Alignment & BLAKE3 Integrity

```bash
cargo run --release --bin apus -- inspect glm-5.2-744b-q4.apus
```

Example output:
```text
╭─── APUS Container Inspection (glm-5.2-744b-q4.apus) ───╮
│ Magic Bytes     : APUS (v1)
│ Author Sig      : apus-moe // authored by mjojo
│ Model ID        : GLM-5.2-744B-Q4_K_M
│ Topology        : 4 layers, 32 routed experts
│ Dense Base Size : 69120 bytes (4KB aligned)
│ LBA Offset Table: offset 0x00011000 (32 entries x 16 bytes)
│ BLAKE3 Root Tag : 0xEAAA7E0C4EC9C8E0
│ 4KB Alignment   : PASSED (O_DIRECT / DirectStorage compatible)
╰──────────────────────────────────────────────────────────╯
```

### 5. Run the End-to-End SSD Streaming Benchmark, Quality Audit & Multi-Thread Stress Suite

```bash
# End-to-end SSD streaming + AVX2 FMA3 + MLA KV + .roost snapshot benchmark:
cargo run --release --bin apus -- bench glm-5.2-744b-q4.apus

# Mathematical SQNR (dB), Cosine Similarity & Perplexity drift audit for INT4/INT2:
cargo run --release --bin apus -- audit

# 8-thread Zipfian ARC + 4KB SlabArena concurrency & invariant stress test (~1.74M ops/sec):
cargo run --release --bin apus -- stress

# Generate & verify a resumable *.apus.idx BLAKE3 chunk manifest:
cargo run --release --bin apus -- manifest glm-5.2-744b-q4.apus

# List supported frontier MoE model profiles:
cargo run --release --bin apus -- profiles
```

---

## Aerospace Telemetry TUI

Launch the 60 FPS `mpsc`-decoupled Truecolor RGB terminal interface (featuring the animated Boot Splash screen, live BLAKE3 verification counter, NVMe throughput sparklines, dynamic 8-dot Unicode Braille expert activation heatmap, on-the-fly Markdown & Rust syntax highlighting, and interactive prompt streaming):

```bash
cargo run --release --bin apus -- run
```

> For automated CI environments without an interactive TTY, run `cargo run --bin apus -- run --headless`.

### TUI Interactive Controls
* **`Space` / `Enter` / `Esc` (on Splash Screen):** Skip boot animation and enter the Telemetry HUD immediately.
* **Type text + `Enter`:** Submit a live prompt to the background `ApusRuntimeSession` worker over `tokio::sync::mpsc` and stream tokens into the chat panel.
* **`Tab`:** Cycle active focus across the 5 HUD panels (`SESSION`, `ARC ENGINE`, `NVMe I/O`, `EXPERT HEATMAP`, `TICKET QUEUE`).
* **`Up` / `Down` / `PageUp` / `PageDown`:** Scroll conversation history up or down.
* **`Ctrl + S`:** Freeze and persist the active MLA KV-cache and warm expert profile to `coding_rust.roost`.
* **`Ctrl + T`:** Cycle the OS Drive QoS NVMe bandwidth ceiling (`80%` / `100%` / `60%`).
* **`Esc` (or `Q` when prompt is empty):** Exit the TUI cleanly.

```text
┏━ APUS // ULTRA-STREAMING MoE ━ GLM-5.2-744B-Q4_K_M.apus ━━━━━━━━━━━━━━━━━━━━━━━⟪ engineered by mjojo ⟫━┓
┃ ◥ ██ ◣  ▄▀█ █▀█ █ █ █▀        ▸ STREAM   :: 6.58 tok/s [MTP 2x Active]▸ I/O PATH :: DirectStorage + O_DIRECT ┃
┃  ◥ █▀ █▀█ █▀▀ █▄█ ▄█ by mjojo ▸ ARC HIT  :: 90.7%  ▰▰▰▰▰▰▰▰▰▱         ▸ NVMe QoS :: 52°C │ VRAM: 14.2/16 GB  ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
╭─ SESSION #04 [coding_rust.roost] │ Context: 48.3K MLA ───╮╭─ MEMORY & CACHE HIERARCHY (ARC ENGINE) ──────────╮
│ pub struct IoRingQueue<T, const N: usize> {              ││ L1 VRAM [GPU0+1] ▰▰▰▰▰▰▰▰▰▰▰▰▰▰▱▱ 14.2/16 GB     │
│     buffer: [UnsafeCell<MaybeUninit<T>>; N],             ││ L2 Slab [Pinned] ▰▰▰▰▰▰▰▰▰▰▰▱▱▱▱▱ 18.4/24 GB     │
│     head: CachePadded<AtomicUsize>,                      ││ L3 NVMe [Stream] ▰▰▱▱▱▱▱▱▱▱▱▱▱▱▱▱ 346.4 GB       │
│     tail: CachePadded<AtomicUsize>,                      ││ L1/L2/Miss: L1 25% │ L2 25% │ Miss 50% (1.4ms)   │
│ }                                                        │╰──────────────────────────────────────────────────╯
│                                                          │╭─ NVMe I/O & S.M.A.R.T. [Samsung 990 PRO] ────────╮
│ [You]                                                    ││ Read BW: ▁▂▃▅▇██▆▄▂▁▁▃▆█▇▄▂▁  6.13 GB/s          │
│ Optimize O_DIRECT 4K SlabArena                           ││ I/O QD:  ▁▁▂▃▄▆▄▂▁▁▁▁▂▄▆▅▃▁▁  QD: 20 / 64        │
│                                                          ││ Thermal: [███████████░░░░░░░] 52°C (Nominal)     │
│ [APUS :: GLM-5.2-744B]                                   │╰──────────────────────────────────────────────────╯
│ [APUS // GLM-5.2-744B-Q4_K_M // engineered by mjojo]     │╭─ EXPERT ACTIVATION HEATMAP (21,504 / LAYER 16) ──╮
│ Streamed 12 tokens (2400.2 tok/s, MTP 2x) via 4K Direct  ││ ⣿⡈⠄⠢⢁⡀⠠⠐⠉⠄⠂⢁⡄⠠ Mode: Code/Rust [Cluster #07]     │
│ I/O + CPU AVX2 + FMA3 | ARC Hit: 50.0% | MLA KV: 12.     ││ ⠐⠈⠄⡂⢁⡀⠠⠐⠈⠄⠂⢃⡀⠠ Active: 15 / 288 per layer        │
│ ▸ Prompt > █                                             ││ ⠐⠈⠄⢂⢁⡀⠠⠐⣿⡈⠄⠢⢁⡀ Look-ahead (L+2): Ready           │
╰──────────────────────────────────────────────────────────╯╰──────────────────────────────────────────────────╯
╭─ TICKET QUEUE (I/O-AWARE CONTINUOUS BATCHING SCHEDULER) ─────────────────────────────────────────────────────╮
│ ▸ [ACTIVE] #01 Generating IoRingQueue (Layer 42/75) ── ETA ~12s                                              │
│ ▹ [QUEUED] #02 Auditing unsafe memory blocks in crate apus-core                                              │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
┗━ [Tab] Panel#1 ━ [Ctrl+S] Save .roost ━ [Ctrl+T] NVMe Cap (100%) ━ [Q/Esc] Quit ━━ APUS // authored by mjojo ┛
```

---

## OpenAI & Anthropic Compatible HTTP/SSE API

Start the embedded `axum` inference server backed by `ApusRuntimeSession` to connect IDE extensions (**VS Code Continue**, **Cursor**, **Claude Code**) or web frontends (**Open WebUI**, **SillyTavern**):

```bash
cargo run --release --bin apus -- serve 8080
```

### Endpoints
* `GET  /health` — Runtime health and `mjojo` author signature check.
* `GET  /v1/models` — Lists mounted `.apus` MoE models.
* `POST /v1/chat/completions` — OpenAI-compatible chat completion endpoint supporting both JSON (`"stream": false`) and Server-Sent Events (`"stream": true` with `chat.completion.chunk` + `data: [DONE]`).
* `POST /v1/messages` — Anthropic Messages API compatible endpoint.

### Example Requests (`curl`)

```bash
# 1. Standard JSON Chat Completion (OpenAI format)
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "GLM-5.2-744B-Q4_K_M.apus",
    "messages": [
      {"role": "user", "content": "Explain 4KB page alignment for O_DIRECT NVMe streaming."}
    ],
    "max_tokens": 64,
    "stream": false
  }'

# 2. Server-Sent Events (SSE) Token Streaming
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "GLM-5.2-744B-Q4_K_M.apus",
    "messages": [
      {"role": "user", "content": "Stream tokens in real time"}
    ],
    "stream": true
  }'

# 3. Anthropic Messages API (/v1/messages)
curl -s http://127.0.0.1:8080/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "GLM-5.2-744B-Q4_K_M.apus",
    "messages": [
      {"role": "user", "content": "Verify Anthropic endpoint compatibility."}
    ],
    "max_tokens": 64
  }'
```

---

## License & Attribution

Licensed under the **Apache License, Version 2.0**.  
**APUS (`apus-moe`)** is architected and engineered by **`mjojo`**.

