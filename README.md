# llama-gfx906-furnace

A retune of llama.cpp for AMD Vega 20 (gfx906: Instinct MI50/MI60,
Radeon VII / Pro VII). Adds a GCN weight-repacking GPU buffer type and
DPP-based warp reductions, ported from the
[reinstinct](https://github.com/sixvolts/reinstinct) inference engine
where these techniques ship in production, and the kernels, memory
layout and server changes needed to serve a 177B sparse-attention MoE
(Qwen3.8-Flash-Next / Swift 1.5, `qwen4exp`) across four MI50s.
Everything is measured on real hardware with cold-start A/B discipline
and validated against the canonical paths (KL divergence against an
unfused single-token baseline, test-backend-ops cases, a single-op
reference check for the repacked kernels).

Recently updated for Qwen-3.8-flash-next (9/28)

Branches:

- `gfx906-perf` (default, the only branch): everything below. The retune rebased onto
  upstream e6ab7c1a4 (2026-09-22), the multi-GPU and qwen4exp work, and
  the qwen4exp NextN/MTP draft head (port of upstream PR #28243) with two
  fixes: the scheduler is re-reserved when the nextn output is switched
  on (without it every prompt ubatch reallocated and drained all
  devices), and the draft's prompt catch-up is deferred by one batch so
  the target pipeline is not drained. A merge commit joins the earlier
  b9587-based history; each commit message carries its measured A/B
  numbers and validation. Updated 2026-09-30 with the pipeline,
  correctness and adaptive-MTP round, and 2026-10-04 with the kernel round
  (rank top-k, MoE down+reduce, Q8_0 multi-column and K-mapped matvecs,
  GDN/norm/conv-step fusions, split-K, Q4_0 / IQ4 / IQ3_S repack, Q3_K->Q6_K
  relabel, MMQ LDS layout, thread-packed expert matvecs) that the furnace
  production server runs. The former development branches
  (`gfx906-perf-upstream`, `qwen4exp-mtp`, `gfx906-multislot-multicard`) were
  merged here or superseded and removed.

## Single GPU

Measured on an MI50 32 GB (1825 MHz sclk / 1125 MHz mclk / 300 W),
`-fa 1`, vs the stock master this branch is based on (throughput in
tok/s, llama-bench pp512 / tg128):

| Model | stock pp / tg | retune pp / tg |
|---|---:|---:|
| Gemma 4 31B UD-Q4_K_XL (dense) | 186.7 / 22.4 | **220.3 / 27.9** (+18% / +24%) |
| Qwen 3.6 35B-A3B UD-Q4_K_XL (MoE) | 864.4 / 89.9 | **967.9 / 89.9** (+12% / parity) |
| Qwen 3.5 0.8B UD-Q4_K_XL | 4772 / 219.9 | **5324 / 220.3** (+12% / par) |

What it does:

- **Three-plane K-quant weight repack** (Q4_K/Q5_K/Q6_K, 2D and MoE
  expert stacks): weights are transformed at load so the decode matvec
  and the int8 MMQ prefill GEMM stream them fully coalesced on wave64
  (the on-disk superblock layout caps a GCN matvec at ~58% of HBM
  bandwidth; repacked sustains ~89%). dp4a (`v_dot4_i32_i8`)
  throughout. On by default on GCN devices; `GGML_CUDA_REPACK=0`
  disables it (required for `llama-quantize`/save: repacked weights
  cannot be read back).
- **DPP/ds_swizzle warp reductions** under the GCN define - single
  VALU lane exchanges instead of `ds_bpermute` LDS roundtrips in every
  warp reduction backend-wide.
- **MoE (MUL_MAT_ID) support** with direct in-kernel expert routing at
  decode and a grouped 16-token-tile GEMM for prefill.
- `GGML_CUDA_REPACK_Q8_0=1` and `GGML_CUDA_REPACK_Q5_1=1` additionally
  repack Q8_0 and Q5_1 (opt-in; both are on in the multi-GPU config).

## Four MI50s: Qwen3.8-Flash-Next / Swift 1.5 (177B MoE, UD-Q4_K_XL)

Layer split across 4x MI50 32 GB (ROCm 7.1). gfx906 PCIe peer copies
crash inside the HIP runtime, so the build uses
`-DGGML_CUDA_NO_PEER_COPY=ON`; the fork replaces the synchronous
host-staged copy that implies with asynchronous staged copies so the
layer-split prompt pipeline still overlaps. On top of the single-GPU
work this branch adds:

- qwen4exp kernels: LDS-resident gated delta net prefill and decode,
  the QSA indexer key pooling and block scoring as fused kernels, a
  block-layout cache kept between decode calls (one sequence, and a
  unified cache holding several), sparse decode attention that reads
  only the selected keys (a gather path and a one-query kernel that now
  takes up to 16 query rows), a parallel top-k radix select, and the
  hyper-connection and shared-expert tail fusions.
- Small-batch paths for speculative verify and for several decoding
  sequences: multi-column repacked Q8_0 matvecs, MoE through the
  per-slot decode kernels, and the single-token fusions (quantize on
  the way out, grouped matvecs, hc mix tail) extended to a few columns.
- Server: context checkpoints copied on the device streams into pinned
  memory (`--ctx-checkpoints-async`), so a checkpoint no longer drains
  the devices and splits the prompt pipeline; `--ctx-checkpoints-tail-only`
  drops the checkpoint one ubatch before the prompt end.
- Prompt pipeline: on this HIP runtime a host-synchronous copy into a
  device waits for any cross-device wait already queued on it (the staged
  hop from the previous layer-split device), so the scheduler's graph-input
  copies ran the devices one after another. Graph inputs of prompt-sized
  ubatches now go asynchronously from pinned per-copy-slot staging
  (llama-bench pp8192 877 -> 1978 t/s). A scheduler reallocation during
  multi-sequence decode no longer leaves the allocator planned for that
  small graph (every later prompt ubatch reallocated and drained all
  devices): the worst-case plan is restored before prompt ubatches.
- QSA prompt attention reads only each row's selected keys once the
  context is 3x the selection (flat cost in depth: pp4096 at 48K depth
  622 -> 1150 t/s together with the larger staging).
- Correctness: the gated-delta-net state read elided through the cache
  raced with the fused cache write-back when several new sequences shared
  a prompt ubatch; multi-sequence batches are reproducible again. Async
  checkpoint reads no longer outlive the context at shutdown.
- Adaptive MTP: `--spec-draft-max-slots N` (or `LLAMA_SPEC_MAX_GEN=N`)
  drafts only while at most N slots are generating. A verify batch costs
  every slot its draft tokens' matvec and expert work, so with several
  users the aggregate is higher without drafts.

Production configuration used for the numbers below: 6 slots over a
unified 256K cache (65536 per slot), `-b 16384 -ub 1024 -fa on`,
`--tensor-split 0.20,0.267,0.267,0.266`, 4 async checkpoints, MTP draft
head on the first GPU with `--spec-draft-n-max 2 --spec-draft-max-slots 2`,
`GGML_CUDA_REPACK_Q8_0=1 GGML_CUDA_REPACK_Q5_1=1`. Server prefill is one
novel-text request (the second of two for 8K); decode is 256 tokens at
temperature 0. For prefill and the mixed load, the previous column is
the same server config on the previous default branch (c575827e9); for
the decode rows it is drafting on every step (the previous behaviour)
against `--spec-draft-max-slots 2`.

| Measurement | Previous | This branch |
|---|---:|---:|
| Prefill 8K, t/s | 887 | **1251** |
| Prefill 32K, t/s | 867 | **1561** |
| Prefill 32K with ~210K of the pool held by 5 slots, t/s | - | **1547** |
| Prefill 40K after multi-user decode, t/s | ~470 | **1300-1420** |
| Prefill 32K while 5 slots decode, t/s | - | **1342** |
| Decode, 1 user (aggregate t/s) | 62 | 62 |
| Decode, 2 users | 77 | **78** |
| Decode, 6 users | 57 | **98** |
| 6-stream mixed load, 20 min, same seeds (requests completed) | 124 | **191** |

llama-bench / batched-bench on the same split, no speculation: pp8192
1977 t/s, tg64 49.2 t/s (43.7 at 32K depth), 6 decoding sequences 104.6
t/s aggregate.

Where it started: the same model on the same machine did 796 t/s
prefill and 51 t/s decode with one slot before this work. MTP's draft
acceptance depends on the text (60-65% on raw wikitext continuation,
80-89% on instruction-style output), so its single-user decode gain
ranges from +17% to +40%; with drafting limited to two generating slots
it no longer costs aggregate throughput under load.

Environment switches for A/B (all default on): `GGML_CUDA_NO_Q8_NC`,
`GGML_CUDA_NO_MOE_SMALL`, `GGML_CUDA_NO_F32_NC`, `GGML_CUDA_NO_FA_DEC`,
`GGML_CUDA_NO_FA_GATHER`, `GGML_CUDA_NO_QSA_POOL`,
`GGML_CUDA_NO_QSA_SCORE`, `GGML_CUDA_NO_Q8_MULTI`,
`GGML_CUDA_NO_HC_UP_PRE`, `GGML_CUDA_NO_Q8_EMIT`,
`GGML_CUDA_NO_REPACK_FLATTEN`, `LLAMA_QSA_NO_MS_CACHE`,
`GGML_CUDA_NO_FA_SPARSE_PREFILL`, `GGML_SCHED_SYNC_INPUTS`,
`LLAMA_NO_REALLOC_RESTORE`, `LLAMA_RS_COPY_ALL_EXTRA`. Tuning:
`GGML_SCHED_INPUT_STAGE_MB` (pinned staging per device and copy slot,
256), `LLAMA_ASYNC_INPUT_MIN_TOKENS` (64), `GGML_CUDA_FA_SPARSE_PREFILL_MIN`
(3), `GGML_CUDA_GDN_CPW` (2 or 4), `LLAMA_KV_SEQ_HEADROOM` (4096: cells
kept free after a sequence in a unified qwen4exp cache, and the spacing of
sequences that start together, at least 256; 256 helps short requests
started together, but long ones then interleave and decode slower).
Diagnostics:
`LLAMA_QSA_LAYOUT_CHECK=1` (rebuild and compare the cached block layout
every call), `GGML_SCHED_TIME=1` (per-split waits, synchronous input
copies and graph reallocation reasons), `LLAMA_DECODE_TIME=1`
(per-ubatch phase times), `LLAMA_SPEC_TIME=1` and `LLAMA_SPEC_NO_DEFER=1`.

Build (ROCm, gfx906; add `-DGGML_CUDA_NO_PEER_COPY=ON` for more than
one gfx906 card):

    cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906 \
          -DGGML_HIP_GRAPHS=ON -DGGML_CUDA_NO_PEER_COPY=ON \
          -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j

This repo is not affiliated with the upstream llama.cpp project -
upstream README follows below.

---

# llama.cpp

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Quick start

A few options to get `llama.cpp` installed on your machine:

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
