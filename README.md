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
[![Engineered By](https://img.shields.io/badge/Engineered_by-mjojo-FFB800?style=for-the-badge)](https://github.com/mjojo/apus-moe)

*Run frontier Mixture-of-Experts models (**GLM-5.2 744B**, **DeepSeek-V3 / R1 671B**, **Qwen-MoE**, **Mixtral 8x22B**) on consumer workstations with **24 GB RAM** and budget GPUs by streaming 4KB page-aligned expert blocks directly from NVMe storage.*

[Architecture](#system-architecture) •
[Key Innovations](#key-engineering-innovations) •
[Binary Format (`*.apus` / `*.roost`)](#binary-specifications-apus--roost) •
[Quick Start](#quick-start) •
[Telemetry HUD](#aerospace-telemetry-tui) •
[OpenAI Server](#openai-compatible-http-api)

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
| **`apus-core`** | [`crates/apus-core`](./crates/apus-core) | 4096B `PageAlignedSlot`, zero-fragmentation `SlabArena`, 3-tier `ArcCache` (L1/L2/L3), 576-dim `MlaKvCache` (PagedAttention + Attention Sinks), `MoERouter`, `LookAheadPrefetcher`, and native `MtpSpeculativeEngine`. |
| **`apus-quant`** | [`crates/apus-quant`](./crates/apus-quant) | Zero-copy binary specifications for `*.apus` (512B header + 16B LBA Offset Table + BLAKE3 Merkle Tree) and `*.roost` (128B session snapshot), `ApusConverter`, and `ImatrixCalibrator` INT4/INT2 quantizers. |
| **`apus-io`** | [`crates/apus-io`](./crates/apus-io) | Pluggable storage engines (`ApusBlockStorage`, `KvCompressedStorage`), `OsDriveQosController` (I/O priority & bandwidth cap), and `SmartThermalController` (PID thermal guard). |
| **`apus-backends`** | [`crates/apus-backends`](./crates/apus-backends) | Runtime CPU SIMD capability dispatcher (`AVX-512 VNNI`, `AVX2+FMA3`, `ARM NEON`) and on-the-fly INT4 register dequantization with SwiGLU GEMV execution. |
| **`apus-net`** | [`crates/apus-net`](./crates/apus-net) | `DualGpuPipeline` (2x 8GB VRAM asymmetric L1 cache + Chunked LM Head logits computation), P2P DMA / Pinned Host staging, `IoAwareBatchScheduler`, and LAN cluster layer balancing. |
| **`apus-tui`** | [`crates/apus-tui`](./crates/apus-tui) | 60 FPS Truecolor RGB Aerospace/Cyberpunk Terminal HUD (`ratatui`): animated Boot Splash screen, Braille 21,504-expert heatmap, live NVMe sparklines, and ticket queue. |
| **`apus-cli`** | [`crates/apus-cli`](./crates/apus-cli) | Unified `apus` command-line interface (`convert`, `inspect`, `bench`, `wizard`, `run`, `serve`) and embedded `axum` OpenAI-compatible HTTP server. |

---

## Key Engineering Innovations

1. **Zero-Double-Caching & 4KB Slab Arena (`SlabArena`)**
   Standard file reads pollute the OS Page Cache, duplicating expert weights in RAM and triggering swap thrashing. APUS aligns every expert block to **4096-byte sector boundaries** (`O_DIRECT` / `FILE_FLAG_NO_BUFFERING`) and streams directly into pre-allocated, memory-locked (`mlock` / `VirtualLock`) `PageAlignedSlot` buffers—guaranteeing zero heap fragmentation over multi-day uptime.
2. **Adaptive Replacement Cache (`ArcCache`)**
   Unlike naive LRU caches that flush hot coding/math experts during a single topic shift, APUS implements a lock-free 3-tier **ARC** engine tracking both *Recency* ($T_1, B_1$) and *Frequency* ($T_2, B_2$) across **L1 VRAM** and **L2 Pinned Slab RAM**.
3. **OS Drive Protection QoS & S.M.A.R.T. PID Thermal Guard**
   Streaming gigabytes per second from a primary OS drive can starve the operating system and overheat NVMe controllers past 70°C. `apus-io` assigns background I/O priority (`IOPRIO_CLASS_IDLE` / `IoPriorityHintVeryLow`), enforces a configurable bandwidth ceiling (e.g., 5.5 GB/s out of 7.0 GB/s), and runs a closed-loop **PID thermal controller** that injects micro-sleeps at 65°C and trips an emergency thermal brake at 70°C.
4. **Multi-Head Latent Attention (`MlaKvCache`) & Native 2x MTP**
   Compresses KV-cache states down to **576 latent values per token** in FP8/INT8 (~584 bytes/token), preserving Attention Sinks while offloading cold pages to system RAM, paired with native **Multi-Token Prediction (MTP)** heads to amortize each NVMe expert read across multiple generated tokens.
5. **Dual-GPU Pipeline Parallelism (`2x 8GB VRAM`) & Chunked LM Head**
   Splits dense layers across two budget GPUs (`Layers 0..37` on GPU 0, `Layers 38..75` on GPU 1) with asymmetric per-GPU L1 expert caches and computes the 150,000+ vocabulary `LM Head` in 16K sub-chunks to prevent VRAM Out-Of-Memory errors.

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
git clone https://github.com/mjojo/apus-moe.git
cd apus-moe
cargo test
```

### 2. Generate a Hardware & QoS Configuration Profile (`apus.toml`)

```bash
cargo run --release --bin apus -- wizard apus.toml
```

### 3. Pack Quantized MoE Weights into a 4KB-Aligned `*.apus` Container

```bash
# Syntax: apus convert <output.apus> <layers> <experts_per_layer>
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

### 5. Run the End-to-End SSD Streaming Benchmark

```bash
cargo run --release --bin apus -- bench glm-5.2-744b-q4.apus
```

---

## Aerospace Telemetry TUI

Launch the 60 FPS Truecolor RGB terminal interface (featuring the animated Boot Splash screen, live BLAKE3 verification counter, NVMe throughput sparklines, and 21,504-expert Braille activation heatmap):

```bash
cargo run --release --bin apus -- run
```

> For automated CI environments without an interactive TTY, run `cargo run --bin apus -- run --headless`.

```text
┏━ APUS // ULTRA-STREAMING MoE ━ GLM-5.2-744B-Q4_K_M.apus ━━━━━━━━━━━━━⟪ engineered by mjojo ⟫━┓
┃  ◥ ██ ◣  ▄▀█ █▀█ █ █ █▀   ▸ STREAM   :: 6.42 tok/s [MTP 2x Active] ▸ I/O PATH :: DirectStorage┃
┃   ◥ █▀ █▀█ █▀▀ █▄█ ▄█     ▸ ARC HIT  :: 91.8%  ▰▰▰▰▰▰▰▰▰▱          ▸ NVMe QoS :: 54°C │ 14/16G┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
╭─ SESSION #04 [coding_rust.roost] │ Context: 48.2K MLA ─╮╭─ MEMORY & CACHE HIERARCHY (ARC) ───╮
│ [You]                                                  ││ L1 VRAM [GPU0+1] ▰▰▰▰▰▰▰▰▰▰▰▰▰▰▱▱  │
│ Write a lock-free SQ/CQ ring buffer for io_uring.      ││ L2 Slab [Pinned] ▰▰▰▰▰▰▰▰▰▰▰▱▱▱▱▱  │
│                                                        ││ L3 NVMe [Stream] ▰▰▱▱▱▱▱▱▱▱▱▱▱▱▱▱  │
│ [APUS :: GLM-5.2-744B]                                 ││ Hit Ratio: 91.8% ── Penalty: 1.4ms │
│ pub struct IoRingQueue<T, const N: usize> {            │╰────────────────────────────────────╯
│     buffer: [UnsafeCell<MaybeUninit<T>>; N],           │╭─ EXPERT HEATMAP (21,504 / L42) ────╮
│     head: CachePadded<AtomicUsize>,                    ││  ⠀⢀⣠⣴⣶⣾⣿⣿⣿⣿⣷⣶⣦ Mode: Code/Rust  │
│     tail: CachePadded<AtomicUsize>,                    ││  ⠀⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀ Active: 8 / 288  │
│ }                                                      ││  ⠀⠙⠿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠋⠀ Look-ahead: OK   │
╰────────────────────────────────────────────────────────╯╰────────────────────────────────────╯
┗━ [Ctrl+S] Save .roost ━ [Ctrl+T] NVMe Cap (80%) ━ [Q/Esc] Quit ━━━ APUS // authored by mjojo ┛
```

---

## OpenAI-Compatible HTTP API

Start the embedded `axum` inference server to connect IDE extensions (**VS Code Continue**, **Cursor**) or web frontends (**Open WebUI**):

```bash
cargo run --release --bin apus -- serve 8080
```

### Endpoints
* `GET  /health` — Runtime health and author signature check.
* `GET  /v1/models` — Lists mounted `.apus` MoE models.
* `POST /v1/chat/completions` — OpenAI-compatible chat completion endpoint with ARC cache & MTP speedup telemetry.

### Example Request (`curl`)

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "GLM-5.2-744B-Q4_K_M.apus",
    "messages": [
      {"role": "user", "content": "Explain 4KB page alignment for O_DIRECT NVMe streaming."}
    ],
    "max_tokens": 128
  }'
```

---

## License & Attribution

Licensed under the **Apache License, Version 2.0**.  
**APUS (`apus-moe`)** is architected and engineered by **`mjojo`**.

