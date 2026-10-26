# Experimental Kirin HiAI backend

Base: THU-MIG/edge-dit.cpp `97cc585d9dbd9ac5e30b78ef2e2c4a0d8bfbe9f2`.
This branch excludes Hscccc DreamLite commits. The ggml gitlink stays at upstream
`f270346d688c7bec7c56bae2ae73343aba6817ad`; the backend builds alongside ggml
from `src/core/backend/ggml/hiai/`, without changing the submodule.

## Execution contract

`HIAI0` is an opt-in synchronous Android arm64 accelerator using the documented
`/vendor/lib64/libai_single_op.so` C interface. No vendor binaries are distributed.
Missing library/symbols yields no device; explicit selection fails visibly.

Non-batched contiguous GGML MUL_MAT nodes are mapped to 1x1 convolution:
weights [K,N] -> filters [N,K,1,1], activations [K,M] -> [1,K,1,M],
and output -> [N,M]. Weights with a GGML to_float decoder (including Q4_0)
are decoded on CPU. The single-op data path uses FP16 weights and activations;
this is not native INT4 arithmetic. Explicit GGML_PREC_F32 requests stay on CPU.
Limits per convolution are K=4096, N=512 and M=256; partial K outputs accumulate
in FP32. Host tensors use ordinary CPU buffers and are copied into ION buffers.

Compiled executors use a weight-content fingerprint and an LRU cache with a
default estimated 128 MiB budget (`ED_HIAI_CACHE_MB`). Vendor-internal allocations
are not fully observable and are outside this estimate. Pointer reuse and LoRA
weight updates must not reuse stale executors.

The edge runner supplies an already allocated graph. The backend explicitly
executes CPU graph views between HiAI matmuls, preserving tensor bindings and
allocation. supports_op describes structural candidates; device PreCheck and
execution can still reject them. Failed candidates are logged and recomputed on
CPU; `ED_HIAI_REQUIRE_NPU=1` makes candidate execution failures fatal. It does not
forbid intentionally unsupported CPU nodes. Counters report completed HiAI
matmuls, API dispatches, CPU nodes and failed matmuls separately.

On the test phone, FP16 calls affected caller SIMD saved values. A non-inlined
execution boundary preserves d8-d15 and FPCR. Integration tests retain floating
point thresholds across that boundary.

## Build and validate

```bash
export ANDROID_NDK_HOME=/path/to/android-ndk
bash scripts/build_android_hiai.sh
adb -s SERIAL push build-android-hiai/bin/ed-test-hiai /data/local/tmp/ed-test-hiai
adb -s SERIAL shell 'chmod 755 /data/local/tmp/ed-test-hiai; ED_HIAI_REQUIRE_NPU=1 /data/local/tmp/ed-test-hiai'
```

Tests compare mixed graphs with CPU references: F32/F16/Q4_0 weights, CPU nodes
before/after matmul, input/output/token tiling, cache reuse and weight changes,
and CPU-only explicit-F32 precision. Synthetic absolute error budget: 0.01.
This does not define final image quality acceptance.

## Model experiment

```bash
python3 tools/mobile/download_flux2.py --serial SERIAL
# Optional alternative transport, retaining source SHA256 verification:
python3 tools/mobile/download_flux2.py --serial SERIAL --transport-origin https://hf-mirror.com --direct
adb -s SERIAL shell mkdir -p /data/local/tmp/edge-dit-npu
adb -s SERIAL push build-android-hiai/bin/ed-cli /data/local/tmp/edge-dit-npu/ed-cli
adb -s SERIAL shell chmod 755 /data/local/tmp/edge-dit-npu/ed-cli
python3 tools/mobile/run_flux2.py --serial SERIAL --width 256 --height 256 --steps 4
```

Assets are pinned: Local Dream Q4_0 DiT, Q4_0 Qwen3 text encoder, FLUX.2 VAE,
and official FP8 DiT as an additional reference. The pinned upstream ggml lacks
the fork's F8_E4M3 type; the initial inference experiment therefore uses Q4_0.
The downloader records revision, size, SHA256 and on-device verification in a
manifest. Weights remain outside Git, beside the source checkout.

## Coverage

Eligible Linear/MatMul nodes use HiAI. Attention, RoPE, normalization, modulation,
VAE convolution and other unsupported nodes use CPU. Batched/permuted matmuls
stay on CPU. Selecting hiai does not mean the whole model runs on NPU. This
initial path makes no speedup claim over an optimized CPU baseline.

Device: PLR-AL30 / Kirin 9020, Android API 31 compatibility environment, ~16 GB RAM.
NPU devfreq statistics are permission-denied to shell. Counters prove our API
dispatch path, not physical NPU utilization; hardware tracing remains unavailable.

API: https://developer.huawei.com/consumer/cn/doc/hiai-guides/single-operator-development-0000001890402068
