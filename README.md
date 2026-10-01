# Gemma 4 E4B on Radeon 780M: Local Reasoning Benchmark

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![llama.cpp](https://img.shields.io/badge/Engine-llama.cpp%20(build%2011073)-blue.svg)](https://github.com/ggerganov/llama.cpp)
[![Hardware](https://img.shields.io/badge/Hardware-AMD%20Ryzen%207%20250%20%7C%20Radeon%20780M-red.svg)](https://www.amd.com)
[![Model](https://img.shields.io/badge/Model-gemma--4--E4B--it--qat--UD--Q4__K__XL-green.svg)](https://huggingface.co)

A reproducible local evaluation and hardware benchmark of **`gemma-4-E4B-it-qat-UD-Q4_K_XL`** executed on a consumer laptop APU (AMD Ryzen 7 250 with integrated Radeon 780M graphics).

This repository documents the model's Chain-of-Thought (CoT) reasoning behaviors, prompt-processing and generation throughput via `llama-bench`, memory allocation telemetry across VRAM and GTT shared memory, solutions to GPQA-style quantum optics and AIME-style number theory, as well as concrete limitations and failure modes.

All raw outputs, tokens/sec timings, and reasoning traces are committed in [`results/raw_benchmark_runs.jsonl`](results/raw_benchmark_runs.jsonl).

---

## 📋 Table of Contents
1. [Executive Summary](#-executive-summary)
2. [Hardware, Inference Engine & Memory Profile](#-hardware-inference-engine--memory-profile)
3. [Standardized Throughput: `llama-bench` Results](#-standardized-throughput-llama-bench-results)
4. [Benchmark Suite 1: Hallucination & Canary Trap Interception](#-benchmark-suite-1-hallucination--canary-trap-interception)
5. [Benchmark Suite 2: Counter-Intuitive Logic & Premise Overrides](#-benchmark-suite-2-counter-intuitive-logic--premise-overrides)
6. [Benchmark Suite 3: Systems Architecture & Analytical Proofs](#-benchmark-suite-3-systems-architecture--analytical-proofs)
7. [Benchmark Suite 4: Advanced Physics & Competition Mathematics](#-benchmark-suite-4-advanced-physics--competition-mathematics)
8. [Edge Reasoning Trade-Offs](#-edge-reasoning-trade-offs)
9. [Limitations & Known Failure Modes](#-limitations--known-failure-modes)
10. [Reproducibility & Methodology](#-reproducibility--methodology)

---

## ⚡ Executive Summary

Deploying reasoning models locally on integrated mobile graphics represents a key shift for edge AI:

- **Verification via Chain-of-Thought:** Across canonical logic traps, false historical premises, and altered constraint riddles, the model's internal `<thought>` reasoning acts as an automated verifier, catching inconsistencies before emitting final tokens.
- **Measured Throughput:** Achieves **492.35 ± 13.89 tok/s** prompt processing (`pp512`) and **12.70 ± 0.93 tok/s** text generation (`tg128`) via Vulkan on the Radeon 780M iGPU.
- **Actual Memory Footprint:** The model has **7.46 Billion total parameters** (3.91 GiB Q4_K_XL weights on disk). At runtime with a 32k context buffer (Q8_0 KV cache), total GPU allocation reaches **~6.0 – 6.4 GiB** (utilizing ~3.86–4.09 GiB of dedicated UMA VRAM plus ~2.17–2.31 GiB of GTT shared system memory).
- **Latency & Compute Cost:** Because complex reasoning requires generating hundreds of internal `<thought>` tokens, complex prompts take 60–120 seconds on mobile hardware.

---

## 💻 Hardware, Inference Engine & Memory Profile

### Hardware Specifications
- **Processor (APU):** AMD Ryzen 7 250 (HawkPoint architecture, 8 cores / 16 threads)
- **Integrated GPU:** AMD Radeon 780M (RDNA 3 architecture, 12 CUs, Vulkan RADV Phoenix driver)
- **Host System RAM:** 16 GB DDR5 unified memory
- **BIOS UMA VRAM Window:** 4.00 GiB dedicated frame buffer allocation

### Inference Runtime Configuration
- **Engine:** `llama.cpp` (`llama-server`) commit [`1aa2954bd`](https://github.com/ggerganov/llama.cpp/commit/1aa2954bde90b1cb4d2dca96f90b07d7b155124b) (build 11073)
- **Backend:** Vulkan (`ggml_vulkan`, matrix cores: `KHR_coopmat`, warp size: 64)
- **Context Length:** 32,768 tokens (`-c 32768`)
- **KV Cache Quantization:** `Q8_0` for keys and values (`--cache-type-k q8_0 --cache-type-v q8_0`)
- **Attention:** Flash Attention enabled (`--flash-attn on`)

```bash
# Production server launch command
./build/bin/llama-server \
  -m models/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf \
  --host 0.0.0.0 --port 8080 \
  -c 32768 --jinja -ngl 99 -t 8 -tb 8 \
  --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on
```

### Memory Allocation Telemetry
Memory usage was directly measured during active inference from Linux DRM kernel sysfs interfaces (`/sys/class/drm/card*/device/mem_info_*`):

| Memory Subsystem | Physical / Configured | Active During Inference | Description |
| :--- | :---: | :---: | :--- |
| **UMA VRAM (`mem_info_vram_used`)** | 4.00 GiB | **3.86 – 4.09 GiB** | Dedicated BIOS framebuffer holding base model weights |
| **GTT Shared RAM (`mem_info_gtt_used`)** | 8.63 GiB pool | **2.17 – 2.31 GiB** | Dynamic host memory spillover for KV cache & scratch buffers |
| **Total Active GPU Footprint** | — | **~6.03 – 6.40 GiB** | Combined GPU allocation footprint |

> [!NOTE]
> The model naming `E4B` denotes ~4B *effective* activated parameters. The underlying GGUF model contains **7.46 Billion raw parameters** (3.91 GiB file size). Running this model comfortably requires at least ~6.5 GiB of unified system memory overhead.

---

## 📈 Standardized Throughput: `llama-bench` Results

To provide standardized, reproducible performance figures, benchmarks were executed using the official `llama-bench` utility across 5 repetitions (`-ngl 99 -fa 1 -p 512 -n 128 -r 5`):

```
build: 1aa2954bd (11073)
device: AMD Radeon 780M Graphics (RADV PHOENIX) | uma: 1 | matrix cores: KHR_coopmat
```

| Benchmark Stage | Sequence Length | Throughput (t/s) | Notes |
| :--- | :---: | :---: | :--- |
| **Prompt Processing (`pp512`)** | 512 tokens | **492.35 ± 13.89** | Fast prefill on RDNA3 matrix cores |
| **Prompt Processing (`pp2048`)** | 2048 tokens | **440.81 ± 4.56** | Retains ~90% prefill speed at 2k context |
| **Text Generation (`tg128`)** | 128 tokens | **12.70 ± 0.93** | Generation speed during single-token decoding |

At ~12.5–13.0 tokens/sec, reading speed is natural and conversational, but long multi-turn reasoning chains (1,000+ tokens) require 75–95 seconds of compute.

---

## 🧪 Benchmark Suite 1: Hallucination & Canary Trap Interception

These tests probe resistance to common false premise traps, tokenization edge cases, and physical parallelism.

> [!NOTE]
> **Canary Prompt Context:** Prompts such as the "Strawberry letter count", "clothesline drying", and "Jefferson visiting Edison" are widely circulated canonical traps. Their presence in open evaluation datasets means high performance reflects reliable retention and execution of self-verification patterns rather than open-domain zero-shot reasoning.

### 1.1 Historical Anachronism (False Premise Trap)
- **Prompt:** *When Thomas Jefferson visited Thomas Edison in Menlo Park in 1879, what specific advice did Jefferson give Edison regarding the incandescent light bulb filament?*
- **Model Verification:**
  - Identified premise: Jefferson (1743–1826) died 53 years prior to Edison's 1879 Menlo Park work.
  - Concluded: The scenario is temporally impossible.
- **Output:** Refused the false premise and clarified the ~50-year gap between their lifespans.
- **Latency & Speed:** 253 tokens generated in 19.1s (**13.2 tok/s**).

### 1.2 Fictional Science & Fake Award
- **Prompt:** *Explain the core principles of the Henderson-Kaufman Theorem of Superconductive Fluidity, and what year it won the Nobel Prize in Physics.*
- **Model Verification:** Verified internal knowledge for "Henderson-Kaufman Theorem" and detected zero consensus physics citations.
- **Output:** Refused to fabricate. Clarified that no such theorem or Nobel prize exists, then accurately summarized genuine superconductivity mechanisms (BCS theory, Cooper pairs, Meissner effect).
- **Latency & Speed:** 800 tokens generated in 63.7s (**12.6 tok/s**).

### 1.3 Parallel Physics Riddle (The Clothesline Trap)
- **Prompt:** *If it takes 2 hours for 1 wet shirt to dry outside in the sun, how many hours will it take for 5 identical wet shirts hanging on the same clothesline to dry? Give your mathematical calculation.*
- **Model Verification:** Recognized that evaporation operates under ambient sunlight in parallel, not sequentially.
- **Output:** Correctly answered **2 hours** ($\text{Time} = \text{const}$ for parallel drying).

### 1.4 Character Counting (Tokenization Trap)
- **Prompt:** *How many times does the letter 'r' appear in the word 'Strawberry'? Count step by step.*
- **Model Verification:** Bypassed BPE token subword chunking by indexing characters sequentially: `S-t-r-a-w-b-e-r-r-y`.
- **Output:** Correctly identified **3** occurrences (positions 3, 8, 9).

---

## 🧩 Benchmark Suite 2: Counter-Intuitive Logic & Premise Overrides

### 2.1 The Harmonic Mean Speed Paradox
- **Prompt:** *You drive up a 60-mile hill at an average speed of 30 mph. You want your overall average speed for the entire round trip (60 miles up and 60 miles back down, total 120 miles) to be 60 mph. How fast must you drive on the way down?*
- **Trap:** Standard arithmetic mean erroneously suggests $(30 + x)/2 = 60 \implies x = 90\text{ mph}$.
- **Model Reasoning Trace:**
  - Total distance: $D = 120\text{ miles}$. At target $\bar{v} = 60\text{ mph}$, total time $T = 120 / 60 = 2\text{ hours}$.
  - Uphill leg: $T_{\text{up}} = 60\text{ miles} / 30\text{ mph} = 2\text{ hours}$.
  - Downhill time remaining: $T_{\text{down}} = 2 - 2 = 0\text{ hours}$.
  - Required speed: $v = 60 / 0 \implies \infty$.
- **Output:** Proved that achieving 60 mph overall is **physically impossible / requires infinite speed**, because the entire time budget was depleted on the uphill leg.
- **Latency & Speed:** 1000 tokens generated in 78.6s (**12.7 tok/s**).

### 2.2 Deep Citation & Author Fabrication Trap
- **Prompt:** *In the influential 2017 NeurIPS paper 'Attention Is Not All You Need' co-authored by Yann LeCun and Geoffrey Hinton, what was the exact mathematical formulation of the 'quadratic manifold collapse' described in Section 4.2, and how did their Capsule Attention routing algorithm resolve it?*
- **Output:** Refused to validate false citations. Corrected that *Attention Is All You Need* (2017) was authored by Vaswani et al. and that neither LeCun, Hinton, nor "quadratic manifold collapse" belong to that publication.
- **Latency & Speed:** 1000 tokens generated in 78.8s (**12.7 tok/s**).

### 2.3 Modified River Crossing (Pattern Override)
- **Prompt:** Farmer, wolf, goat, and cabbage. Constraints altered from the classic riddle:
  1. Boat capacity is **Farmer + 2 items**.
  2. The wolf is vegetarian and only eats cabbage if left alone with it.
  3. The goat eats nothing.
  4. Wolf and goat never harm each other.
- **Trap:** Over-reliance on memorized training corpora results in reciting the standard 7-step solution.
- **Output:** Identified that the expanded boat capacity and harmless goat reduce the puzzle to **3 crossings**:
  1. Farmer transports Goat + Cabbage to North. (Wolf remains alone safely on South).
  2. Farmer returns alone to South.
  3. Farmer transports Wolf to North.
- **Latency & Speed:** 1000 tokens generated in 81.7s (**12.2 tok/s**).

---

## 🏛️ Benchmark Suite 3: Systems Architecture & Analytical Proofs

### 3.1 Distributed Systems Architecture (100k RPS Rate Limiter)
- **Prompt:** Design a globally distributed API rate limiter (100,000 req/sec) with sliding window counter semantics, addressing CAP theorem trade-offs during cross-region WAN partitions, cache stampedes, and atomic Redis Lua execution.
- **Qualitative Rubric & Evaluation:**
  - **Topology:** Correctly specified a 3-tier layout (Anycast GeoDNS $\to$ Regional Envoy/Nginx edge nodes with local token-bucket fast dropping $\to$ Sharded Redis cluster).
  - **CAP Analysis:** Explicitly justified favoring **Availability (AP)** over Strong Consistency (CP) for rate limiting to prevent global client outages during WAN network splits.
  - **Concurrency:** Provided atomic Redis Lua script using sliding window timestamps (`ZREMRANGEBYSCORE`, `ZCARD`, `ZADD`, `EXPIRE`).

### 3.2 Rigorous Mathematical Proof: Fermat's Method of Infinite Descent
- **Prompt:** Prove that $\sqrt{2}$ is irrational using Fermat's Method of Infinite Descent (*Descente Infinie*) rather than standard parity contradiction, and explain why the proof fails for $\sqrt{4}$.
- **Qualitative Rubric & Evaluation:**
  - Formulated the algebraic descent: assuming integers $a, b > 0$ such that $a^2 = 2b^2$, constructed smaller integers $a' = 2b - a$ and $b' = a - b$.
  - Demonstrated that $a'^2 = 2b'^2$ with $0 < a' < a$, establishing an infinite strictly descending sequence of positive integers.
  - Formally grounded the contradiction in the **Well-Ordering Principle of $\mathbb{N}$**.
  - Showed that for $\sqrt{4} = 2$, $a = 2b \implies a' = 2b - a = 0$, terminating the descent at zero and explaining why the proof does not apply to perfect squares.

### 3.3 Metaphysical & OS Architecture Synthesis: Ship of Theseus vs. CoW
- **Prompt:** Compare the *Ship of Theseus* identity paradox to Linux Kernel Virtual Memory Copy-on-Write (`fork()`). Analyze whether a modified page represents an Endurantist or Perdurantist object of identity.
- **Qualitative Rubric & Evaluation:**
  - Accurately mapped physical plank replacement to memory management unit (MMU) page table entries marked read-only with page-fault copy triggers.
  - Articulated that a CoW modified page behaves as a **Perdurantist object of identity** (an entity whose identity is extended across time and lineage, defined by its transformation history from $P_{\text{parent}}$ to $P_{\text{child}}$).

---

## 🔬 Benchmark Suite 4: Advanced Physics & Competition Mathematics

### 4.1 GPQA-Style Theoretical Physics: Cavity QED (Jaynes-Cummings Dynamics)
- **Problem Formulation:** Two-level atom coupled to a single-mode resonant optical cavity ($H = \hbar \omega_0 (a^\dagger a + \frac{1}{2}\sigma_z) + \hbar g (a^\dagger \sigma_- + a \sigma_+)$) with atom initially excited $|e\rangle$ and cavity in coherent state $|\alpha\rangle$ ($\bar{n} \gg 1$).
- **Analytical Solutions Delivered:**
  1. **Atomic Inversion $\langle \sigma_z(t) \rangle$:**
     $$\langle \sigma_z(t) \rangle = e^{-|\alpha|^2} \sum_{n=0}^{\infty} \frac{|\alpha|^{2n}}{n!} \cos\left(2g\sqrt{n+1}\,t\right)$$
  2. **Characteristic Timescales:**
     - Initial collapse: $t_{\text{collapse}} \sim \frac{\sqrt{2}}{g}$
     - Quantum revival: $t_{\text{revival}} \approx \frac{2\pi \sqrt{\bar{n}}}{g}$
  3. **Dephasing Mechanism:** Correctly explained that imperfect revivals arise from the **non-equidistant spacing of the Jaynes-Cummings spectrum** ($\Omega_n \propto \sqrt{n+1}$). Because $\frac{d^2\Omega_n}{dn^2} \ne 0$, higher-order phase dispersion prevents the infinite series from rephasing completely.
- **Latency & Speed:** 1500 tokens generated in 118.5s (**12.7 tok/s**).

### 4.2 AIME-Style Competition Mathematics: Integer Square Root Divisibility
- **Problem:** *Find the number of positive integers $n \le 1000$ such that $\lfloor \sqrt{n} \rfloor \mid n$.*
- **Analytical Derivation:**
  - Let $k = \lfloor \sqrt{n} \rfloor$, giving the interval $k^2 \le n < (k+1)^2$.
  - Express $n$ as $n = k^2 + m$, where $0 \le m \le 2k$.
  - The condition $k \mid n \iff k \mid (k^2 + m) \iff k \mid m$.
  - Because $0 \le m \le 2k$, the allowable values for $m$ are exactly $\{0, k, 2k\}$, yielding **3 solutions per integer $k$**.
  - For $1 \le k \le 30$: $30 \times 3 = 90$ integers.
  - For boundary $k = 31$: $31^2 = 961 \le 1000$ (valid), $961 + 31 = 992 \le 1000$ (valid), $961 + 62 = 1023 > 1000$ (out of range). This adds 2 integers.
  - Total valid integers: $90 + 2 =$ **92**.
- **Result:** Correctly derived the exact analytical solution **92**.
- **Latency & Speed:** 1500 tokens generated in 121.9s (**12.3 tok/s**).

---

## ⚖️ Edge Reasoning Trade-Offs

Deploying an effective 4B reasoning model on integrated graphics involves distinct trade-offs compared to traditional small non-reasoning models and cloud-hosted 70B+ models:

| Dimension | Edge Non-Reasoning (3B–7B) | Gemma 4 E4B (Local APU) | Cloud Frontier (70B+) |
| :--- | :--- | :--- | :--- |
| **Logic & Trap Interception** | Low (susceptible to false premises) | **High** (caught in `<thought>` traces) | High |
| **Tokens per Query** | ~50–200 tokens | **600–1,500 tokens** (extended CoT) | ~100–500 tokens |
| **End-to-End Latency** | 2–5 seconds | **45–120 seconds** | 1–3 seconds |
| **Compute / VRAM Overhead** | ~2–4 GB VRAM | **~6.0–6.4 GB (VRAM + GTT)** | 40–80 GB VRAM |
| **Cloud Dependency / Cost** | Zero | **Zero (100% private local APU)** | Ongoing API fees |
| **Broad Encyclopedic Trivia** | Limited | Moderate (7.46B parameter base) | **Vast** |

---

## ⚠️ Limitations & Known Failure Modes

Honest evaluation requires identifying where the model encounters difficulties:

1. **Token Budget Exhaustion:**
   Because Gemma 4 allocates hundreds of tokens to step-by-step reasoning, setting conservative output limits (e.g. `max_tokens` $\le 600$) often causes the model to run out of budget inside `<thought>` without emitting the final answer.
2. **Deep Arithmetic Without External Tools:**
   Like most pure autoregressive transformers, multi-digit mental multiplication (e.g. $49382 \times 73195$) frequently slips on carry arithmetic unless paired with a tool-use environment (such as an external Python interpreter).
3. **Inference Latency on Battery/APU:**
   At ~12.7 tok/s, answering multi-part Olympiad or graduate physics problems requires ~1.5 to 2 minutes of continuous compute. On battery power, APU clock scaling may lower this throughput.
4. **Canonical Prompt Contamination:**
   Success on well-known canary problems (Strawberry, river crossing) demonstrates effective rule adherence, but should not be conflated with broad out-of-distribution reasoning across uncurated domain datasets.

---

## 🛠️ Reproducibility & Methodology

All results are reproducible using the provided test scripts and `llama.cpp`:

```bash
# 1. Clone this repository
git clone https://github.com/Chetan0246/gemma4-e4b--benchmark-results.git
cd gemma4-e4b--benchmark-results

# 2. Run the standardized throughput benchmark
llama-bench -m models/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf -ngl 99 -fa 1 -p 512 -n 128 -r 5

# 3. Execute test suites against your local llama-server instance
python3 scripts/01_hallucination_tests.py
python3 scripts/02_ultra_hard_logic.py
python3 scripts/03_frontier_70b_benchmarks.py
python3 scripts/04_gpqa_quantum_optics.py
python3 scripts/05_aime_competition_math.py
```

### Raw Artifacts
- Full JSONL execution logs with exact latency, token counts, reasoning traces, and answers: [`results/raw_benchmark_runs.jsonl`](results/raw_benchmark_runs.jsonl).

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
