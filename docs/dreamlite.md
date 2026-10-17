# DreamLite-mobile

[Back to supported models](models.md)

DreamLite-mobile supports text-to-image generation and single-image editing
on CPU and CUDA. Inference uses the native C++ runtime and does not require
Python. This implementation targets DreamLite-mobile; DreamLite-base uses a
different architecture.

## Model files

Use either the original Diffusers directory or a complete GGUF export.
The Diffusers directory must contain `model_index.json`, the UNet weights in
`unet/`, Qwen3-VL weights and configuration in `text_encoder/`, and Tiny VAE
weights in `vae/`.

The model's vision tower encodes the input image for editing. It is part of
Qwen3-VL and is loaded with the language encoder.

## Build

Initialize the submodules before building:

```bash
git submodule update --init --recursive
```

CPU:

```bash
cmake -S . -B build-dreamlite-cpu \
  -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF
cmake --build build-dreamlite-cpu --target ed-cli ed-convert -j 4
```

CUDA requires an NVIDIA GPU and a CUDA toolkit:

```bash
cmake -S . -B build-dreamlite-cuda \
  -DCMAKE_BUILD_TYPE=Release -DED_GGML_CUDA=ON \
  -DED_ENABLE_CUDNN_SDPA=OFF
cmake --build build-dreamlite-cuda --target ed-cli ed-convert -j 4
```

Set `CMAKE_CUDA_ARCHITECTURES` when targeting a specific GPU. DreamLite's CUDA
path does not require cuDNN. See [Build and installation](build.md) for compiler
and backend configuration.

## Generate an image

```bash
./build-dreamlite-cpu/bin/ed-cli \
  --backend cpu --model /path/to/DreamLite-mobile \
  --prompt "A small red robot on a wooden table, studio photograph" \
  --width 1024 --height 1024 --steps 4 --seed 42 --threads 4 \
  --output dreamlite.png
```

For CUDA, use `./build-dreamlite-cuda/bin/ed-cli --backend cuda` with the same
model and generation arguments. To use a GGUF, replace the model directory
with the GGUF file path.

The reference resolution is 1024×1024. Text-to-image dimensions must be
positive multiples of 32. Set `--steps 4` explicitly to use the model's native
four-step schedule.

## Edit an image

```bash
./build-dreamlite-cpu/bin/ed-cli \
  --backend cpu \
  --model /path/to/dreamlite-mobile-deploy-language-q4_1-f16.gguf \
  --image input.png --prompt "Change the robot to blue" \
  --width 1024 --height 1024 --steps 4 --seed 42 --threads 4 \
  --vae-tiling on --output dreamlite-edit.png
```

Editing requires a 1024×1024 canvas; the input image is resized automatically.
Through the C API, provide `init_image` or one `ref_images` entry.

Use `--vae-tiling on` to reduce VAE decoding memory. `--vae-tiling off` uses
untiled decoding and requires more temporary memory. Total runtime memory
exceeds the GGUF file size because it also includes intermediate tensors and
compute buffers.

## Convert to GGUF

Build `ed-convert` as shown above, then install the Python export dependencies:

```bash
python -m pip install torch safetensors
python scripts/convert_dreamlite_deploy.py \
  --model /path/to/DreamLite-mobile \
  --converter build-dreamlite-cpu/bin/ed-convert \
  --output /path/to/dreamlite-mobile-deploy-language-q4_1-f16.gguf
```

The default export stores:

| Component | Weight format |
|---|---|
| UNet | FP16 |
| Tiny VAE | FP16 |
| Qwen3-VL vision tower | FP16 |
| Compatible language matrices, including token embeddings | Q4_1 |
| Other language tensors, including biases and norms | FP16 |

Q4_1 stores 32 affine four-bit weights per block with FP16 scale and offset,
using 5 bits per weight including metadata. This follows the component
precision policy of DreamLite's deploy workflow, while using ggml's GGUF
quantizer and runtime. It differs from MLX's group-64 quantization.

Export temporarily stages FP16 weights next to the destination and writes a
JSON sidecar describing the conversion. Allow space for the staging weights
and final output. The converter preserves the source checkpoint and refuses
to overwrite an existing GGUF or sidecar.

For an export without low-bit quantization, add `--text-type f16` and choose a
different output filename. `--quantize-vision` additionally quantizes compatible
vision matrices; the default keeps the entire vision tower in FP16.

Only conversion requires Python; the resulting GGUF loads directly in
`ed-cli`.

## Computation precision

DreamLite uses BF16 operation boundaries to follow the original Python
pipeline. Floating weights may be converted when loaded, while many
reductions, attention operations, and elementwise operations use FP32
intermediate buffers. FP16 in the GGUF table describes stored weights;
inference uses mixed precision.

Q4_1 language weights remain compressed in memory. Embedding lookup
dequantizes the selected rows, and quantized matrix multiplication reconstructs
floating results inside its kernels. On CPU, these matrix multiplications can
use temporary Q8_1 activation blocks. Q4_1 storage therefore does not make all
computation or activations four-bit.

Quantized results can differ from the original BF16 checkpoint. CPU, CUDA,
and Python outputs can also differ because of kernels, random-number streams,
and image resizing.

## Supported options and limitations

- CPU and CUDA are supported. Other backends are rejected by this pipeline.
- One source image is supported for editing.
- The model uses its native dynamically shifted Euler schedule.
- CFG scales are unused; negative prompts are unsupported.
- Masks, ControlNet, multiple reference images, LoRA, transformer caches,
  other samplers, custom flow shifts, and video generation are unsupported.

See [Command-line usage](cli.md) for shared CLI arguments.
