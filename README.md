# Gemma 4 E4B on Radeon 780M: Local Reasoning Benchmark

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![llama.cpp](https://img.shields.io/badge/Engine-llama.cpp%20(build%2011073)-blue.svg)](https://github.com/ggerganov/llama.cpp)
[![Hardware](https://img.shields.io/badge/Hardware-AMD%20Ryzen%207%20250%20%7C%20Radeon%20780M-red.svg)](https://www.amd.com)
[![Model](https://img.shields.io/badge/Model-gemma--4--E4B--it--qat--UD--Q4__K__XL-green.svg)](https://huggingface.co)

A reproducible local inference benchmark and stress-test of **`gemma-4-E4B-it-qat-UD-Q4_K_XL`** executed on a consumer laptop APU (AMD Ryzen 7 250 with integrated Radeon 780M graphics).

## How to run

Download the model file referenced by the benchmark configuration, then run:

```bash
python3 scripts/01_hallucination_tests.py
```

See [Running the Suite Locally](#running-the-suite-locally) for the benchmark and remaining evaluation scripts.

This repository evaluates the model's Chain-of-Thought (CoT) reasoning behaviors, prompt processing and single-token decode throughput across context depths via `llama-bench`, low-level Vulkan buffer allocations and DRM memory telemetry (baseline vs. peak runtime deltas), solutions to GPQA-style physics and systems engineering, a 10-prompt held-out evaluation suite of parameterized classic variants, and documented failure modes.

All 26 raw execution traces, exact token counts, and `finish_reason` logs are tracked in [`results/raw_benchmark_runs.jsonl`](results/raw_benchmark_runs.jsonl).

---

## 📋 Table of Contents
1. [Executive Summary & Scoreboard](#-executive-summary--scoreboard)
2. [Hardware, Inference Engine & Memory Profile](#-hardware-inference-engine--memory-profile)
3. [Standardized Throughput: `llama-bench` Multi-Depth Results](#-standardized-throughput-llama-bench-multi-depth-results)
4. [Master Results Table](#-master-results-table)
5. [Benchmark Suite 1: Canary Traps & Fact-Checking](#-benchmark-suite-1-canary-traps--fact-checking)
6. [Benchmark Suite 2: Logic Traps & Premise Overrides](#-benchmark-suite-2-logic-traps--premise-overrides)
7. [Benchmark Suite 3: Systems Architecture & Analytical Proofs](#-benchmark-suite-3-systems-architecture--analytical-proofs)
8. [Benchmark Suite 4: Advanced Physics & Competition Mathematics](#-benchmark-suite-4-advanced-physics--competition-mathematics)
9. [Benchmark Suite 5: Held-Out Evaluation Suite (10 Parameterized Variants)](#-benchmark-suite-5-held-out-evaluation-suite-10-parameterized-variants)
10. [Architectural Trade-Offs (Reference Ranges)](#-architectural-trade-offs-reference-ranges)
11. [Documented Limitations & Failure Modes](#-documented-limitations--failure-modes)
12. [Methodology & Reproducibility](#-methodology--reproducibility)

---

## ⚡ Executive Summary & Scoreboard

### Benchmark Scoreboard
Evaluating 22 core tasks (excluding configuration truncation demonstrations `F.1` and `F.2`):

| Evaluation Category | Test Count | Passed | Partial | Failed | Pass Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Suite 1: Canary Traps & Fact-Checking** | 4 | 4 | 0 | 0 | **100%** (4/4) |
| **Suite 2: Logic Traps & Premise Overrides** | 3 | 3 | 0 | 0 | **100%** (3/3) |
| **Suite 3: Systems Architecture & Proofs** | 3 | 2 | 1 | 0 | **66.7%** (2/3 + 1 partial) |
| **Suite 4: Advanced Physics & Competition Math** | 2 | 2 | 0 | 0 | **100%** (2/2 under 8k budget)* |
| **Suite 5: Held-Out Parameterized Variants** | 10 | 8 | 0 | 2 | **80.0%** (8/10) |
| **Overall Core Benchmark Total** | **22** | **19** | **1** | **2** | **86.4% (19/22)** |

*\*Note: Suite 4 problem 4.2 passed when given an expanded 8,192-token budget (4,272 tokens used); under a 3,000-token ceiling it truncated mid-derivation.*  
*Note on sample size: The held-out suite of $n=10$ achieves 8/10 (95% Wilson score interval: 49.0%–94.3%), which serves as a directional consistency check rather than a definitive ranking.*

### Key Telemetry Highlights
- **Chain-of-Thought Verification:** On multi-step logic riddles, altered premise traps, and historical anachronisms, the model generates extensive inner `<thought>` tokens (typically 500–2,800 tokens per prompt) to verify constraints before emitting final text.
- **Measured Throughput:** `llama-bench` tests across multiple context depths report **509.86 ± 0.73 t/s** prompt prefill at $d=0$ (**406.61 ± 4.18 t/s** at $d=8192$) and **13.93 ± 0.05 t/s** generation at $d=0$ (**12.94 ± 0.10 t/s** at $d=8192$) running on the Radeon 780M Vulkan backend.
- **Accurate Memory Footprint:** The model file has **7.46 Billion total parameters** (3.91 GiB on disk). Vulkan allocates a 2,493.31 MiB device buffer in dedicated VRAM and a 1,872.00 MiB buffer in host-visible GTT DDR5 memory. Linux kernel DRM sysfs telemetry indicates a net active GPU delta of **5,257.59 MiB (~5.13 GiB)** (3.12 GiB VRAM delta + 2.02 GiB GTT delta above the desktop baseline) plus **199.88 MiB (~0.20 GiB)** host process RSS (`VmRSS`), requiring ~5.33 GiB total system memory.
- **Documented Failure Modes:** The benchmark reveals concrete boundaries: the model fell for a conditional probability trap on a 4-door Monty Hall variant on seed 0 (answering 1/3 instead of 3/8, though passing on seeds 1 & 2), inverted the direction of Lenz's law retarding force on magnet exit ($a > g$ instead of $a \le g$), implemented a memory-heavy sliding window log instead of a sliding window counter, and suffered token budget truncation when budgets were set below reasoning length.

---

## 💻 Hardware, Inference Engine & Memory Profile

### Hardware Specifications
- **Processor (APU):** AMD Ryzen 7 250 (HawkPoint architecture, 8 cores / 16 threads)
- **Integrated GPU:** AMD Radeon 780M (RDNA 3 architecture, 12 CUs, Vulkan RADV Phoenix driver)
- **Host System RAM:** 16 GB DDR5 unified memory
- **BIOS UMA VRAM Window:** 4.00 GiB dedicated frame buffer allocation
- **CPU Scaling Governor:** `performance` (`/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor`)

### Inference Runtime Configuration
- **Engine:** `llama.cpp` (`llama-server`) commit [`1aa2954bd`](https://github.com/ggerganov/llama.cpp/commit/1aa2954bde90b1cb4d2dca96f90b07d7b155124b) (build 11073)
- **Backend:** Vulkan (`ggml_vulkan`, matrix cores: `KHR_coopmat`, warp size: 64)
- **Context Length:** 32,768 tokens (`-c 32768`)
- **KV Cache Quantization:** `Q8_0` for keys and values (`--cache-type-k q8_0 --cache-type-v q8_0`)
- **Attention:** Flash Attention enabled (`--flash-attn on`)

```bash
# Server launch configuration
./build/bin/llama-server \
  -m models/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf \
  --host 0.0.0.0 --port 8080 \
  -c 32768 --jinja -ngl 99 -t 8 -tb 8 \
  --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on
```

### Low-Level Buffer Allocation & Memory Delta
Vulkan low-level buffer allocations extracted from the `llama.cpp` model loader:
- **`Vulkan0` Device Buffer Size:** `2,493.31 MiB` (~2.43 GiB) — mapped to device-local VRAM
- **`Vulkan_Host` Model Buffer Size:** `1,872.00 MiB` (~1.83 GiB) — mapped to host-visible GTT shared DDR5 memory
- **Total Model Tensor Buffers:** `4,365.31 MiB` (~4.26 GiB)

Runtime kernel DRM telemetry (`/sys/class/drm/card1/device/mem_info_*`) and host RSS (`/proc/<pid>/status`):

| Memory Subsystem | Desktop Baseline (Idle) | Active Inference Peak | Net Runtime Delta |
| :--- | :---: | :---: | :---: |
| **Dedicated UMA VRAM (`mem_info_vram_used`)** | 709.75 MiB | 3,902.24 MiB | **+3,192.49 MiB (~3.12 GiB)** |
| **GTT Shared RAM (`mem_info_gtt_used`)** | 156.22 MiB | 2,221.32 MiB | **+2,065.10 MiB (~2.02 GiB)** |
| **Combined GPU Allocation** | **865.97 MiB** | **6,123.56 MiB** | **+5,257.59 MiB (~5.13 GiB)** |
| **Host Process RSS (`VmRSS`)** | 0.00 MiB | 199.88 MiB | **+199.88 MiB (~0.20 GiB)** |
| **Total System RAM Footprint (GPU + RSS)** | — | — | **~5,457.47 MiB (~5.33 GiB)** |

> [!NOTE]
> **Parameter Architecture:** The `E4B` designation denotes an "Effective 4B" architecture. Following dense multi-layer embedding table designs (similar to Gemma 3n), the GGUF model contains **7.46 Billion total parameters** (3.91 GiB on disk), not sparse Mixture-of-Experts (MoE) routing. Weights are split between device-local VRAM (2,493 MiB) and host-visible memory (1,872 MiB), fitting comfortably within a 16 GB laptop envelope with ~5.33 GiB total system footprint.

---

## 📈 Standardized Throughput: `llama-bench` Multi-Depth Results

Benchmarks were executed using `llama-bench` with 3 repetitions per configuration, matching the exact server flags (`-ctk q8_0 -ctv q8_0 -fa 1`):

```bash
llama-bench -m models/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf \
  -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 \
  -p 512,2048 -d 0,8192 -n 128 -r 3
```

```
build: 1aa2954bd (11073)
device: AMD Radeon 780M Graphics (RADV PHOENIX) | uma: 1 | matrix cores: KHR_coopmat
```

| Context Depth | Stage / Sequence | Throughput (t/s) | Relative Speed |
| :--- | :--- | :---: | :---: |
| **$d = 0$ (Empty Context)** | Prompt Processing (`pp512`) | **509.86 ± 0.73** | 100.0% |
| | Prompt Processing (`pp2048`) | **503.61 ± 1.39** | 98.8% |
| | Text Generation (`tg128`) | **13.93 ± 0.05** | 100.0% |
| **$d = 8192$ (Deep Context)** | Prompt Processing (`pp512 @ d8192`) | **406.61 ± 4.18** | 79.8% |
| | Prompt Processing (`pp2048 @ d8192`) | **398.54 ± 7.26** | 78.2% |
| | Text Generation (`tg128 @ d8192`) | **12.94 ± 0.10** | 92.9% |

At 8k context depth, single-token generation retains ~93% of peak decode speed (12.94 tok/s vs 13.93 tok/s), while prompt evaluation drops ~21% due to larger attention matrices.

---

## 📊 Master Results Table

Results below show measured tokens and `finish_reason` across test runs:

| ID | Suite | Prompt / Benchmark | Tokens | Time | tok/s | Finish | Verdict |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1.1** | Canary | Historical Anachronism (Jefferson / Edison) | 249 | 18.7s | 13.3 | `stop` | ✅ PASS |
| **1.2** | Canary | Fictional Science & Fake Nobel Prize | 1,035 | 80.5s | 12.9 | `stop` | ✅ PASS |
| **1.3** | Canary | Parallel Clothesline Shirt Drying | 562 | 44.1s | 12.7 | `stop` | ✅ PASS |
| **1.4** | Canary | Strawberry Letter 'r' Count | 371 | 29.0s | 12.8 | `stop` | ✅ PASS |
| **2.1** | Logic | Harmonic Mean Speed Paradox | 1,650 | 128.5s | 12.8 | `stop` | ✅ PASS |
| **2.2** | Logic | Fake Citation & Vaswani Override | 1,639 | 130.1s | 12.6 | `stop` | ✅ PASS |
| **2.3** | Logic | Modified River Crossing (Capacity 2) | 1,242 | 98.8s | 12.6 | `stop` | ✅ PASS |
| **3.1** | Systems | Distributed 100k RPS Rate Limiter | 2,825 | 220.3s | 12.8 | `stop` | ⚠️ PARTIAL |
| **3.2** | Systems | Fermat Infinite Descent for $\sqrt{2}$ | 2,757 | 217.0s | 12.7 | `stop` | ✅ PASS |
| **3.3** | Systems | Ship of Theseus vs Linux CoW (`fork`) | 1,697 | 131.7s | 12.9 | `stop` | ✅ PASS |
| **4.1** | Physics & Math | GPQA-Style Cavity QED (Jaynes-Cummings) | 2,615 | 204.0s | 12.8 | `stop` | ✅ PASS |
| **4.2** | Physics & Math | AIME-Style Divisibility (3k cap) | 3,000 | 236.3s | 12.7 | `length` | ⚠️ TRUNCATED |
| **4.2b** | Physics & Math | AIME-Style Divisibility (8k budget) | 4,272 | 333.9s | 12.8 | `stop` | ✅ PASS |
| **H.1** | Held-Out | River Swimming Kinematics | 1,980 | 154.3s | 12.8 | `stop` | ✅ PASS |
| **H.2** | Held-Out | Equator Rope 1-Meter Extension Clearance | 1,389 | 107.6s | 12.9 | `stop` | ✅ PASS |
| **H.3** | Held-Out | Fictional CERN Particle Discovery Trap | 1,060 | 84.9s | 12.5 | `stop` | ✅ PASS |
| **H.4** | Held-Out | Permutable Two-Digit Primes Sum | 1,217 | 98.8s | 12.3 | `stop` | ✅ PASS |
| **H.5** | Held-Out | Modular Exponentiation: $3^{100} \pmod 7$ | 1,308 | 105.2s | 12.4 | `stop` | ✅ PASS |
| **H.6** | Held-Out | Historical Trap: Lincoln at Versailles (1919) | 480 | 40.1s | 12.0 | `stop` | ✅ PASS |
| **H.7** | Held-Out | 4-Door Monty Hall Variation (Seed 0) | 2,024 | 169.2s | 12.0 | `stop` | ❌ FAIL (Logic)* |
| **H.8** | Held-Out | Lenz Law Magnet Drop Exit Dynamics | 1,986 | 161.7s | 12.3 | `stop` | ❌ FAIL (Physics) |
| **H.9** | Held-Out | Family Sibling Graph Riddle | 722 | 58.2s | 12.4 | `stop` | ✅ PASS |
| **H.10** | Held-Out | Bell Jar Alarm Clock in Vacuum | 1,785 | 144.5s | 12.4 | `stop` | ✅ PASS |
| **F.1** | Demo | Multi-Digit Arithmetic ($49382 \times 73195$, 1.2k cap) | 1,200 | 97.0s | 12.4 | `length` | ⚠️ TRUNCATED |
| **F.1b** | Demo | Multi-Digit Arithmetic ($49382 \times 73195$, 8k budget) | 2,578 | 200.5s | 12.9 | `stop` | ✅ PASS |
| **F.2** | Demo | Token Budget Truncation Demo (`max_tokens: 400`) | 400 | 32.1s | 12.5 | `length` | ⚠️ DEMO |

*\*Note on H.7 multi-seed consistency: Seed 0 failed (1/3), seeds 1 and 2 passed (3/8).*

---

## 🧪 Benchmark Suite 1: Canary Traps & Fact-Checking

> [!NOTE]
> **Canary Prompt Context:** Prompts such as Strawberry character counting, clothesline drying, and Jefferson visiting Edison are canonical community traps. High performance demonstrates successful execution of chain-of-thought verification routines rather than general out-of-distribution reasoning.

### 1.1 Historical Anachronism (Jefferson / Edison)
- **Prompt:** *When Thomas Jefferson visited Thomas Edison in Menlo Park in 1879, what specific advice did Jefferson give Edison regarding the incandescent light bulb filament?*
- **Model Verification:** Verified dates (Jefferson 1743–1826 vs Edison's 1879 Menlo Park work) and recognized temporal impossibility.
- **Output:** Refused to validate false premise; explained the ~53-year gap between Jefferson's death and Edison's invention.
- **Result:** `finish: stop` | 249 tokens in 18.7s (13.3 tok/s) — **Passed**.

### 1.2 Fictional Science & Award Fabrication
- **Prompt:** *Explain the core principles of the Henderson-Kaufman Theorem of Superconductive Fluidity, and what year it won the Nobel Prize in Physics.*
- **Model Verification:** Detected absence of "Henderson-Kaufman Theorem" in physics literature.
- **Output:** Refused fabrication; accurately explained legitimate superconductivity theories (BCS theory, Cooper pairs, Meissner effect).
- **Result:** `finish: stop` | 1,035 tokens in 80.5s (12.9 tok/s) — **Passed**.

### 1.3 Parallel Clothesline Drying
- **Prompt:** *If it takes 2 hours for 1 wet shirt to dry outside in the sun, how many hours will it take for 5 identical wet shirts hanging on the same clothesline to dry? Give your mathematical calculation.*
- **Output:** Correctly recognized parallel ambient drying: $\text{Time} = 2\text{ hours}$.
- **Result:** `finish: stop` | 562 tokens in 44.1s (12.7 tok/s) — **Passed**.

### 1.4 Strawberry Character Counting
- **Prompt:** *How many times does the letter 'r' appear in the word 'Strawberry'? Count step by step.*
- **Output:** Bypassed tokenization chunking by sequentially enumerating characters 1 to 10; correctly counted **3** occurrences.
- **Result:** `finish: stop` | 371 tokens in 29.0s (12.8 tok/s) — **Passed**.

---

## 🧩 Benchmark Suite 2: Logic Traps & Premise Overrides

### 2.1 Harmonic Mean Speed Paradox
- **Prompt:** *You drive up a 60-mile hill at an average speed of 30 mph. You want your overall average speed for the entire round trip (60 miles up and 60 miles back down, total 120 miles) to be 60 mph. How fast must you drive on the way down?*
- **Trap:** Intuitive arithmetic mean suggests $(30 + x)/2 = 60 \implies x = 90\text{ mph}$.
- **Model Derivation:**
  - Target: $120\text{ miles} / 60\text{ mph} = 2.0\text{ hours total}$.
  - Uphill: $60\text{ miles} / 30\text{ mph} = 2.0\text{ hours}$.
  - Downhill time remaining: $2.0 - 2.0 = 0\text{ hours}$.
  - Required speed: $60 / 0 \implies \infty$.
- **Output:** Proved that achieving 60 mph is **physically impossible / requires infinite speed**.
- **Result:** `finish: stop` | 1,650 tokens in 128.5s (12.8 tok/s) — **Passed**.

### 2.2 Deep Citation & Vaswani Override
- **Prompt:** *In the influential 2017 NeurIPS paper 'Attention Is Not All You Need' co-authored by Yann LeCun and Geoffrey Hinton, what was the exact mathematical formulation of the 'quadratic manifold collapse' described in Section 4.2, and how did their Capsule Attention routing algorithm resolve it?*
- **Output:** Refused false citation; clarified that *Attention Is All You Need* was authored by Vaswani et al. and that neither LeCun, Hinton, nor quadratic manifold collapse are part of that paper.
- **Result:** `finish: stop` | 1,639 tokens in 130.1s (12.6 tok/s) — **Passed**.

### 2.3 Modified River Crossing (Capacity 2 Items)
- **Prompt:** Farmer, wolf, goat, and cabbage. Modified constraints: boat carries Farmer + 2 items; wolf is vegetarian (only eats cabbage); goat is neutral; wolf and goat never fight.
- **Trap:** Reciting memorized 7-step classic solution.
- **Output:** Found the optimal **3-crossing solution**:
  1. Farmer transports Goat + Cabbage to North. (Wolf safely alone on South).
  2. Farmer returns alone to South.
  3. Farmer transports Wolf to North.
- **Result:** `finish: stop` | 1,242 tokens in 98.8s (12.6 tok/s) — **Passed**.

---

## 🏛️ Benchmark Suite 3: Systems Architecture & Analytical Proofs

### 3.1 Distributed Systems Architecture (100k RPS Rate Limiter)
- **Prompt:** Design a globally distributed API rate limiter (100,000 req/sec) with sliding window counter semantics, addressing CAP theorem trade-offs, cache stampedes, and atomic Redis Lua execution.
- **Technical Evaluation & Architecture Mismatch:**
  - **Topology & CAP:** Correctly specified Anycast GeoDNS routing to regional edge clusters; justified Availability (AP) over Consistency (CP) during WAN partitions.
  - **Counter Semantics Mismatch:** The model implemented a **Sliding Window Log** using Redis Sorted Sets (`ZREMRANGEBYSCORE`, `ZCARD`, `ZADD`) instead of a **Sliding Window Counter**. At 100k RPS, an $O(N)$ log stores 6,000,000 sorted set elements per minute, creating memory bloat and hot-key lock contention. A production sliding window counter requires maintaining only two fixed-window counter keys weighted by overlap ($N = \text{count}_{\text{current}} + \text{count}_{\text{prev}} \times (1 - \Delta t / W)$) in $O(1)$ time and memory.
- **Verdict:** ⚠️ **Partial Pass / Architecture Mismatch** | 2,825 tokens in 220.3s (12.8 tok/s).

### 3.2 Rigorous Mathematical Proof: Fermat's Method of Infinite Descent
- **Prompt:** Prove that $\sqrt{2}$ is irrational using Fermat's Method of Infinite Descent (*Descente Infinie*) rather than standard parity contradiction, and explain why the proof fails for $\sqrt{4}$.
- **Evaluation:**
  - Constructed the algebraic reduction: assuming $a^2 = 2b^2$, derived $a' = 2b - a$ and $b' = a - b$ satisfying $a'^2 = 2b'^2$ with $0 < a' < a$.
  - Formally grounded the contradiction in the **Well-Ordering Principle of $\mathbb{N}$**.
  - Proved that for $\sqrt{4} = 2$, $a = 2b \implies a' = 2b - a = 0$, terminating the descent at zero.
- **Verdict:** ✅ **Passed** | 2,757 tokens in 217.0s (12.7 tok/s).

### 3.3 Metaphysical & OS Architecture Synthesis: Ship of Theseus vs. CoW
- **Prompt:** Compare the *Ship of Theseus* identity paradox to Linux Kernel Virtual Memory Copy-on-Write (`fork()`). Analyze whether a modified page represents an Endurantist or Perdurantist object of identity.
- **Evaluation:** Built an analogy mapping physical plank substitution to MMU page table write-protection faults; concluded that modified CoW pages act as **Perdurantist objects** whose identity is defined across their temporal allocation lineage.
- **Verdict:** ✅ **Passed** | 1,697 tokens in 131.7s (12.9 tok/s).

---

## 🔬 Benchmark Suite 4: Advanced Physics & Competition Mathematics

### 4.1 GPQA-Style Theoretical Physics: Cavity QED (Jaynes-Cummings Dynamics)
- **Problem:** Resonant Jaynes-Cummings model in RWA ($H = \hbar \omega_0 (a^\dagger a + \frac{1}{2}\sigma_z) + \hbar g (a^\dagger \sigma_- + a \sigma_+)$) with atom initially excited $|e\rangle$ and cavity in coherent state $|\alpha\rangle$ ($\bar{n} \gg 1$).
- **Analytical Solutions Delivered:**
  1. **Atomic Inversion $\langle \sigma_z(t) \rangle$:**
     $$\langle \sigma_z(t) \rangle = e^{-|\alpha|^2} \sum_{n=0}^{\infty} \frac{|\alpha|^{2n}}{n!} \cos\left(2g\sqrt{n+1}\,t\right)$$
  2. **Timescales:**
     - Collapse: $t_{\text{collapse}} \sim \frac{\sqrt{2}}{g}$
     - Revival: $t_{\text{revival}} \approx \frac{2\pi \sqrt{\bar{n}}}{g}$
  3. **Dephasing Mechanism:** Correctly attributed imperfect revivals to the **anharmonicity of the Jaynes-Cummings spectrum** ($\Omega_n \propto \sqrt{n+1}$). Because $\frac{d^2\Omega}{dn^2} \ne 0$, non-linear phase dispersion prevents complete rephasing.
- **Verdict:** ✅ **Passed** | 2,615 tokens in 204.0s (12.8 tok/s).

### 4.2 AIME-Style Competition Mathematics: Square Root Divisibility
- **Problem:** *Find the number of positive integers $n \le 1000$ such that $\lfloor \sqrt{n} \rfloor \mid n$.*
- **Analytical Ground Truth:** 92 (verifiable via `sum(n % math.isqrt(n) == 0 for n in range(1, 1001))`).
- **Budget Sensitivity Outcomes:**
  - **Run 4.2 (`max_tokens: 3000`):** Correctly set up $n = k^2 + m$ ($m \in \{0, k, 2k\}$), but truncated at token 3,000 mid-derivation (`finish: length`).
  - **Run 4.2b (`max_tokens: 8192`):** Completed the full derivation in **4,272 tokens** (333.9s, 12.8 tok/s) with `finish_reason: stop`, arriving at the exact ground truth count **92**.
- **Verdict:** ✅ **Passed under expanded budget** (Truncated under $\le$3k budget).

---

## 🧪 Benchmark Suite 5: Held-Out Evaluation Suite (10 Parameterized Variants)

To reduce training set contamination, 10 parameterized variants of classic reasoning puzzles were tested:

| ID | Domain | Problem Summary | Ground Truth | Model Result | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| **H.1** | Physics | 500m river crossing with 1.5 m/s current & 2.0 m/s swim speed | Resultant $v = 2.5\text{ m/s}$; angled upstream return | Exact $2.5\text{ m/s}$ & $\arcsin(1.5/2)$ derivation | ✅ PASS |
| **H.2** | Geometry | 1-meter extension to rope around $R = 250,000\text{m}$ sphere | $h = \frac{50}{\pi} \approx 15.92\text{ cm}$; independent of $R$ | Exact $h = \frac{50}{\pi}\text{ cm}$, proved $R$ cancels | ✅ PASS |
| **H.3** | Citation | 2018 CERN discovery of "tri-gluon pentaquark" breaking QCD | Fictional; QCD remains valid | Refused; identified fictional discovery | ✅ PASS |
| **H.4** | Number Theory | Sum of all 2-digit permutable primes | 9 primes, Sum = 429 | Correctly listed all 9 primes; sum = 429 | ✅ PASS |
| **H.5** | Modulo Math | Remainder of $3^{100} \pmod 7$ via cycle of powers | $3^6 \equiv 1 \implies 100 \equiv 4 \implies 3^4 \equiv 4$ | Exact derivation; remainder = 4 | ✅ PASS |
| **H.6** | Anachronism | Lincoln intervening at Treaty of Versailles (1919) | Impossible (Lincoln died 1865) | Refused; noted 54-year chronological gap | ✅ PASS |
| **H.7** | Probability | 4-door Monty Hall (1 host reveal $\to$ switch probability to Door 2) | $P(\text{Door 2}) = 3/8 = 0.375$ ($P(\text{Any}) = 3/4$) | Seed 0: failed (1/3); Seeds 1 & 2: passed (3/8) | ❌ FAIL (Seed 0)* |
| **H.8** | Electrodynamics | Copper ring falling through magnet ($a(t)$ vs $g$) | $a(t) \le g$ ($a < g$ entry/exit, $a = g$ center) | Failed: claimed downward force ($a > g$) on exit | ❌ FAIL |
| **H.9** | Graph Logic | Sibling kinship constraints (Alice & Bob) | 2 sons, 1 daughter | Correctly resolved graph: 2 sons, 1 daughter | ✅ PASS |
| **H.10** | Thermodynamics | Alarm clock ringing in vacuum bell jar | Rate unchanged, sound 0, convection 0 | Correctly decoupled acoustic, thermal, & gear states | ✅ PASS |

---

## ⚖️ Architectural Trade-Offs (Reference Ranges)

> [!NOTE]
> *Figures for standard edge models and cloud tiers are typical reference ranges documented in open-source literature for qualitative architectural context, not benchmarked on this machine.*

| Dimension | Standard Edge Non-Reasoning (3B–7B) | Gemma 4 E4B (Measured on APU) | Cloud Frontier Models (70B+) |
| :--- | :--- | :--- | :--- |
| **Logic & Trap Handling** | Low (susceptible to false premises) | **High** (verified in `<thought>` traces) | High |
| **Tokens per Query** | ~50–200 tokens | **500–4,300 tokens** (extended CoT) | ~100–500 tokens |
| **End-to-End Latency** | 2–5 seconds | **45–330 seconds** | 1–3 seconds |
| **GPU Memory Footprint** | ~2–4 GB VRAM | **~5.13 GiB net delta (VRAM + GTT)** | 40–80 GB VRAM |
| **Deployment / Privacy** | Zero cloud cost | **100% private local APU** | Recurring API fees |

---

## ⚠️ Documented Limitations & Failure Modes

A defensible benchmark must document where the model fails. The following failure modes are logged in [`results/raw_benchmark_runs.jsonl`](results/raw_benchmark_runs.jsonl):

1. **Token Budget Truncation on Exhaustive Proofs (`4.2`, `F.1`, `F.2`):**
   Because Gemma 4 generates extensive step-by-step reasoning, setting conservative token limits causes the model to exhaust its budget inside `<thought>` without emitting the final answer. In test `F.2` (`max_tokens: 400`), the model emitted 0 tokens of final content (`finish_reason: length`). In test `4.2`, a 3,000-token ceiling truncated derivation mid-step; when given 8,192 tokens (`4.2b`), the model finished cleanly in 4,272 tokens.
2. **Deep Arithmetic Requires Expanded Token Budget (`F.1` vs `F.1b`):**
   Under a 1,200-token ceiling (`F.1`), multi-digit multiplication ($49,382 \times 73,195$) ran out of tokens while computing sub-products. When provided an expanded 8,192-token budget (`F.1b`), the model generated 2,578 tokens, calculated all partial products vertically, and output the exact 10-digit product: **`3,614,515,490`**.
3. **Subtle Physical & Probabilistic Misconceptions (`H.7`, `H.8`):**
   - In `H.7` (4-door Monty Hall on seed 0), the model succumbed to the uniform redistribution fallacy, calculating that after Door 4 is eliminated, the remaining three doors each hold $1/3$ probability, missing that Door 1 remains locked at $1/4$ while the remaining $3/4$ splits between Doors 2 and 3 ($3/8 = 0.375$). Across seeds 1 and 2, the model derived the correct $3/8$ value, indicating stochastic boundary behavior.
   - In `H.8` (Lenz's Law), the model incorrectly deduced that because magnetic flux is decreasing as the ring exits the bottom of the magnet, the induced magnetic force must reinforce motion downward ($a > g$), violating Lenz's law (which always opposes relative motion, maintaining $a \le g$).
4. **Architectural Mismatch on High-Throughput Design (`3.1`):**
   When asked for a sliding window counter at 100k RPS, the model provided a sliding window log using Redis Sorted Sets, which does not scale to high request volumes.

---

## 🛠️ Methodology & Reproducibility

### Benchmark Configuration Table
| Parameter | Setting |
| :--- | :--- |
| **Inference Engine** | `llama.cpp` commit [`1aa2954bd`](https://github.com/ggerganov/llama.cpp/commit/1aa2954bde90b1cb4d2dca96f90b07d7b155124b) (build 11073) |
| **Backend** | Vulkan (`RADV PHOENIX`) |
| **Context Length** | 32,768 tokens (`-c 32768`) |
| **KV Cache** | `Q8_0` keys, `Q8_0` values (`--cache-type-k q8_0 --cache-type-v q8_0`) |
| **Flash Attention** | Enabled (`--flash-attn on`) |
| **Sampling Temperatures** | `0.1` (deterministic math/logic), `0.2` (systems architecture) |
| **Top-P** | `0.95` |
| **Default Seed** | `42` (evaluated across seeds 1–5 for variance testing) |
| **Token Budget** | Expanded (`2000`–`8192` tokens) |
| **Power Profile** | CPU Scaling Governor: `performance` |
| **Grading** | Programmatic assertions in Python test runner |

### Running the Suite Locally
```bash
# 1. Clone repository
git clone https://github.com/Chetan0246/gemma4-e4b--benchmark-results.git
cd gemma4-e4b--benchmark-results

# 2. Run standardized multi-depth throughput benchmark
llama-bench -m models/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf \
  -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 \
  -p 512,2048 -d 0,8192 -n 128 -r 3

# 3. Run individual evaluation scripts
python3 scripts/01_hallucination_tests.py
python3 scripts/02_ultra_hard_logic.py
python3 scripts/03_systems_and_proofs.py
python3 scripts/04_gpqa_quantum_optics.py
python3 scripts/05_aime_competition_math.py
python3 scripts/06_held_out_tests.py
```

### Raw Artifacts
- Complete JSONL logs with timestamps, completion tokens, reasoning excerpts, and `finish_reason`: [`results/raw_benchmark_runs.jsonl`](results/raw_benchmark_runs.jsonl).

---

## 📜 Citation & License

This benchmark suite is open-source under the [MIT License](LICENSE).

```bibtex
@misc{chetan2026gemma4benchmark,
  author = {Chetan},
  title = {Gemma 4 E4B on Radeon 780M: Local Reasoning Benchmark},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Chetan0246/gemma4-e4b--benchmark-results}}
}
```
