"""Export a deploy-inspired DreamLite GGUF (not bit-identical to MLX).

Upstream deploy: FP16 UNet/VAE and FP16 -> MLX affine 4-bit/group-64 Qwen3-VL.
Here: FP16 UNet/VAE/vision and FP16 -> GGUF Q4_1/group-32 language matrices.
Q4_1 is supported by both ggml matmul and embedding lookup. Norms, biases,
vision position embeddings and non-block-aligned tensors remain FP16.
Requires torch and safetensors; original checkpoint files are never modified.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import torch
from safetensors.torch import load_file, save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--converter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-type", choices=("q4_1", "f16"), default="q4_1",
                        help="Use f16 for an unquantized control export")
    parser.add_argument("--quantize-vision", action="store_true",
                        help="Experimental: also quantize vision; current MLX-VLM defaults skip it")
    args = parser.parse_args()
    model = args.model.resolve()
    output = args.output.resolve()
    if output.exists() or output.with_suffix(".json").exists():
        parser.error("output GGUF and sidecar must not already exist")
    with (model / "model_index.json").open() as file:
        if json.load(file).get("_class_name") != "DreamLiteMobilePipeline":
            parser.error("expected a DreamLiteMobilePipeline checkpoint")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    weights = list(model.rglob("*.safetensors"))
    if not weights:
        parser.error("no safetensors weights found")
    rules = rf"^text_encoders\.llm\.={args.text_type}"
    if not args.quantize_vision:
        rules = r"^text_encoders\.llm\.visual\.=f16," + rules
    with tempfile.TemporaryDirectory(prefix="dreamlite-fp16-", dir=output.parent) as tmp:
        stage = Path(tmp)
        # Preserve configs, shard indices and tokenizer metadata, not other artifacts.
        for source in model.rglob("*.json"):
            target = stage / source.relative_to(model)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        for source in weights:
            print(f"FP16 staging: {source.relative_to(model)}", flush=True)
            state = load_file(str(source))
            for name, tensor in state.items():
                if tensor.is_floating_point():
                    converted = tensor.to(torch.float16)
                    if not torch.isfinite(converted).all():
                        raise ValueError(f"non-finite FP16 weights: {source}: {name}")
                    state[name] = converted
            target = stage / source.relative_to(model)
            target.parent.mkdir(parents=True, exist_ok=True)
            save_file(state, str(target))
            del state
        command = [str(args.converter.resolve()), "--model", str(stage), "--type", "f16",
                   "--tensor-type-rules", rules, "--output", str(output)]
        subprocess.run(command, check=True)
    report = {
        "source": str(model), "output": str(output),
        "reference": "DreamLite/deploy/export_{text_encoder,unet,vae_encoder,vae_decoder}.py",
        "unet": "FP16", "vae": "FP16",
        "language": args.text_type,
        "vision": args.text_type if args.quantize_vision else "f16",
        "mlx_vlm_scope_reference": "https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/convert.py (base_quant_predicate skips multimodal modules; checked 2026-09-20)",
        "mlx_equivalent": False,
        "difference": "Official MLX uses 4-bit/group-64; GGUF Q4_1 uses 4-bit/group-32 (5 bits/weight including metadata). f16 is an unquantized control. Runtime remains mixed precision.",
        "tensor_type_rules": rules,
        "source_weight_bytes": sum(path.stat().st_size for path in weights),
        "gguf_bytes": output.stat().st_size,
    }
    with output.with_suffix(".json").open("w") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
