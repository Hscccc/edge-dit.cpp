#include "dit_models/pipelines/dreamlite_pipeline.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <set>
#include <stdexcept>

#include "dit_models/components/autoencoders/vae.hpp"
#include "dit_models/components/autoencoders/tae.hpp"
#include "dit_models/components/text_encoders/llm.hpp"
#include "dit_models/models/dreamlite.hpp"
#include "dit_models/pipelines/dit_pipeline_utils.hpp"

namespace edgedit {
namespace {

sd::Tensor<float> resize_rgb(const ed_image_t& image, int width, int height) {
    sd::Tensor<float> input({image.width, image.height, 3, 1});
    const int64_t pixels = static_cast<int64_t>(image.width) * image.height;
    for (int64_t i = 0; i < pixels; ++i) {
        for (int c = 0; c < 3; ++c) {
            input[i + pixels * c] = image.data[i * image.channels + (image.channels < 3 ? 0 : c)] / 255.f;
        }
    }
    auto result = sd::ops::interpolate(input, {width, height, 3, 1}, sd::ops::InterpolateMode::Lanczos, false, true);
    // PIL resizes RGB bytes before preprocessing in the reference pipeline.
    for (auto& value : result.values()) value = std::round(std::clamp(value, 0.f, 1.f) * 255.f) / 255.f;
    return result;
}

void round_bf16(sd::Tensor<float>& tensor) {
    for (float& value : tensor.values()) value = ggml_bf16_to_fp32(ggml_fp32_to_bf16(value));
}

void require_tensor(const sd::Tensor<float>& tensor, const char* stage) {
    if (tensor.empty()) throw std::runtime_error(std::string("DreamLite ") + stage + " returned an empty tensor");
    for (float value : tensor.values()) {
        if (!std::isfinite(value)) throw std::runtime_error(std::string("DreamLite ") + stage + " produced non-finite values");
    }
}

} // namespace

DreamLitePipeline::DreamLitePipeline() = default;
DreamLitePipeline::~DreamLitePipeline() = default;

bool DreamLitePipeline::prepare(const ed_context_params_t&, ModelRuntime& runtime, const ModelLoader& loader,
                                PipelineTensorRegistry& registry, std::string* error) {
    ready_ = false;
    runtime_ = &runtime;
    try {
        auto supported = [](ggml_backend_t backend) {
            return sd_backend_is(backend, "CPU") || sd_backend_is(backend, "CUDA");
        };
        if (!supported(runtime.backend()) || !supported(runtime.clip_backend()) || !supported(runtime.vae_backend())) {
            throw std::runtime_error("DreamLite BF16 precision requires CPU or CUDA backends");
        }
        const auto& storage = loader.get_tensor_storage_map();
        runtime.reset_auto_allocate_state();
        size_t remaining = runtime.effective_budget_bytes();
        const bool unet_offload = runtime.dit_offload_params_to_cpu() ||
            runtime.plan_component_offload(loader, "model.diffusion_model", remaining);
        const bool text_offload = runtime.clip_offload_params_to_cpu() ||
            runtime.plan_component_offload(loader, "text_encoders", remaining);
        const bool vae_offload = runtime.vae_offload_params_to_cpu() ||
            runtime.plan_component_offload(loader, "first_stage_model", remaining);
        runtime.finalize_auto_segment_budget(runtime.effective_budget_bytes());
        unet_ = std::make_unique<DreamLite::Runner>(runtime.backend(), unet_offload, storage);
        text_ = std::make_unique<LLM::LLMEmbedder>(LLM::LLMArch::QWEN3_VL, runtime.clip_backend(),
            text_offload, storage, "text_encoders.llm", true, true);
        vae_ = std::make_unique<TinyImageAutoEncoder>(runtime.vae_backend(), vae_offload, storage,
            "first_stage_model", false, VERSION_DREAMLITE_MOBILE);
        LOG_INFO("DreamLite Python BF16 reference mode: BF16 operator boundaries, FP32 reduction/elementwise scratch; quantized weights remain approximate");

        registry.clear();
        unet_->get_param_tensors(registry.tensors());
        text_->get_param_tensors(registry.tensors(), "text_encoders.llm");
        vae_->get_param_tensors(registry.tensors(), "first_stage_model");
        // Fail before allocation for incompatible variants or incomplete models.
        for (const auto& entry : registry.tensors()) {
            const auto it = storage.find(entry.first);
            if (it == storage.end()) throw std::runtime_error("DreamLite missing weight: " + entry.first);
            for (int d = 0; d < 4; ++d) {
                if (it->second.ne[d] != entry.second->ne[d]) {
                    throw std::runtime_error("Unsupported DreamLite-mobile weight shape: " + entry.first);
                }
            }
        }
        for (auto* runner : {static_cast<GGMLRunner*>(unet_.get()), static_cast<GGMLRunner*>(&text_->model),
                            static_cast<GGMLRunner*>(vae_.get())}) {
            runner->set_flash_attention_enabled(runtime.flash_attention());
            runner->set_max_graph_vram_bytes(runtime.max_graph_vram_bytes());
            if (!runner->alloc_params_buffer()) throw std::runtime_error("DreamLite parameter allocation failed");
        }
        text_->model.set_max_graph_vram_bytes(runtime.text_encoder_segment_budget(text_->model.get_params_buffer_size()));
        registry.ignore_prefix("text_encoders.llm.lm_head.");
        registry.ignore_prefix("text_encoders.llm.output.weight");
        return true;
    } catch (const std::exception& e) {
        if (error) *error = e.what();
        return false;
    }
}

sd::Tensor<float> DreamLitePipeline::encode_prompt(const std::string& prompt, const ed_image_t* image) {
    std::vector<std::pair<int, sd::Tensor<float>>> image_embeds;
    std::vector<LLM::LLMImageEmbedInfo> image_infos;
    std::vector<std::vector<std::pair<int, sd::Tensor<float>>>> deepstack(3);
    std::string text;
    int drop = 34;
    if (!image) {
        text = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, "
               "quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
               "<|im_start|>user\n[Generate]: " + prompt + "<|im_end|>\n<|im_start|>assistant\n";
    } else {
        drop = 64;
        text = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, "
               "texture, objects, background), then explain how the user's text instruction should alter "
               "or modify the image. Generate a new image that meets the user's requirements while maintaining "
               "consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n"
               "<|vision_start|>";
        const int embed_index = static_cast<int>(text_->tokenizer->encode(text, nullptr).size());
        auto pixels = resize_rgb(*image, 256, 256) * 2.f - 1.f;
        auto outputs = text_->model.encode_image_outputs(runtime_->n_threads(), pixels);
        if (outputs.size() != 4) throw std::runtime_error("DreamLite Qwen3-VL vision encoding failed");
        require_tensor(outputs[0], "vision encoder");
        const auto count = outputs[0].shape()[1];
        image_embeds.emplace_back(embed_index, std::move(outputs[0]));
        for (int i = 0; i < 3; ++i) deepstack[i].emplace_back(embed_index, std::move(outputs[i + 1]));
        image_infos.push_back({embed_index, count, 1, 16, 16});
        for (int64_t i = 0; i < count; ++i) text += "<|image_pad|>";
        text += "<|vision_end|>[Edit]: A diptych with two side-by-side images of the same scene. "
                "Compared to the right side, the left one has " + prompt + "<|im_end|>\n<|im_start|>assistant\n";
    }
    const auto tokens = text_->tokenizer->encode(text, nullptr);
    // DreamLite's Python pipeline conditions diffusion on
    // outputs.hidden_states[-1], which is the final decoder block output
    // before Qwen3-VL's terminal RMSNorm. Request only that layer here;
    // the ordinary LLM output (and other users of this runner) stays normalized.
    const std::set<int> diffusion_prompt_layer{static_cast<int>(text_->model.params.num_layers)};
    auto context = text_->model.compute(runtime_->n_threads(), sd::Tensor<int32_t>::from_vector(tokens),
                                        {}, image_embeds, diffusion_prompt_layer, image_infos, "", deepstack);
    text_->model.free_compute_buffer();
    require_tensor(context, "text encoder");
    if (context.shape()[1] <= drop) throw std::runtime_error("DreamLite prompt is shorter than its template");
    context = sd::ops::slice(context, 1, drop, context.shape()[1]);
    round_bf16(context);
    return context;
}

ed_status_t DreamLitePipeline::generate_image(const ed_image_generation_params_t* p, ed_image_batch_t* out,
                                              std::string* error) {
    if (out) { out->images = nullptr; out->count = 0; }
    auto fail = [&](ed_status_t status, const std::string& message) {
        if (error) *error = message;
        return status;
    };
    if (!ready_) return fail(ED_STATUS_MODEL_LOAD_FAILED, "DreamLite pipeline is not initialized");
    if (!p || !out || !p->prompt || p->width <= 0 || p->height <= 0 || p->batch_count <= 0 ||
        p->width % 32 || p->height % 32) {
        return fail(ED_STATUS_INVALID_ARGUMENT, "DreamLite requires a prompt, positive batch count, and dimensions divisible by 32");
    }
    if (p->ref_image_count < 0 || (p->ref_image_count && !p->ref_images)) {
        return fail(ED_STATUS_INVALID_ARGUMENT, "Invalid DreamLite reference image list");
    }
    if (p->mask_image || p->control_image || p->ref_image_count > 1 || p->lora_count ||
        (p->init_image && p->ref_image_count == 1 && p->init_image->data != p->ref_images[0].data)) {
        return fail(ED_STATUS_UNSUPPORTED, "DreamLite supports text generation and one --image; masks, ControlNet, extra references and LoRA are not supported");
    }
    // CLI supplies the same image through both fields; the C API can use either.
    const ed_image_t* edit_image = p->init_image ? p->init_image : (p->ref_image_count == 1 ? p->ref_images : nullptr);
    if (edit_image && (!edit_image->data || !edit_image->width || !edit_image->height ||
        edit_image->channels < 1 || edit_image->channels > 4)) {
        return fail(ED_STATUS_INVALID_ARGUMENT, "Invalid DreamLite edit image");
    }
    if (edit_image && (p->width != 1024 || p->height != 1024)) {
        return fail(ED_STATUS_INVALID_ARGUMENT, "DreamLite-mobile editing requires --width 1024 --height 1024, matching the reference pipeline");
    }
    if ((p->sample.sampler != ED_SAMPLER_AUTO && p->sample.sampler != ED_SAMPLER_EULER) ||
        (p->sample.scheduler != ED_SCHEDULER_AUTO && p->sample.scheduler != ED_SCHEDULER_DISCRETE)) {
        return fail(ED_STATUS_UNSUPPORTED, "DreamLite-mobile uses the flow-matching Euler sampler and discrete schedule");
    }
    if (p->negative_prompt && p->negative_prompt[0]) {
        return fail(ED_STATUS_UNSUPPORTED, "DreamLite-mobile is guidance-distilled and does not use a negative prompt");
    }
    if (p->sample.cache_mode != ED_CACHE_DISABLED || p->sample.flow_shift > 0.f) {
        return fail(ED_STATUS_UNSUPPORTED, "DreamLite-mobile requires its native dynamic shift schedule and does not support transformer caches");
    }
    if (p->sample.steps == 0 || p->sample.steps < -1) {
        return fail(ED_STATUS_INVALID_ARGUMENT, "DreamLite steps must be positive or -1 for the default 4 steps");
    }
    const int steps = p->sample.steps > 0 ? p->sample.steps : 4;
    if (p->batch_count > std::numeric_limits<int>::max() / steps) {
        return fail(ED_STATUS_INVALID_ARGUMENT, "DreamLite batch count times steps exceeds the progress counter limit");
    }
    if (p->sample.cfg_scale != 1.f || p->sample.image_cfg_scale != 1.f) {
        LOG_INFO("DreamLite-mobile is guidance-distilled; CFG scales are unused, matching the reference pipeline");
    }
    auto control = runtime_->generation_control();
    if (control) control->start(steps * p->batch_count);
    auto images = static_cast<ed_image_t*>(std::calloc(p->batch_count, sizeof(ed_image_t)));
    if (!images) return fail(ED_STATUS_OUT_OF_MEMORY, "DreamLite image batch allocation failed");
    try {
        auto check_cancel = [&]() {
            if (control && control->should_cancel()) {
                control->mark_cancelled();
                throw std::runtime_error("generation cancelled");
            }
        };
        emit_phase_marker("encode", "begin");
        check_cancel();
        const auto context = encode_prompt(p->prompt, edit_image);
        check_cancel();
        const int w = p->width / 8, h = p->height / 8;
        auto condition = sd::Tensor<float>::zeros({w, h, 4, 1});
        if (edit_image) {
            condition = vae_->encode(runtime_->n_threads(), resize_rgb(*edit_image, p->width, p->height), runtime_->vae_tiling());
            require_tensor(condition, "VAE encoder");
            round_bf16(condition);
        }
        emit_phase_marker("encode", "end");
        // Dynamic exponential shift uses the single-image area, not the diptych.
        const float mu = calculate_shift(static_cast<int64_t>(w) * h / 4);
        std::vector<float> sigmas;
        for (int i = 0; i < steps; ++i) {
            const float t = 1.f - static_cast<float>(i) / steps;
            sigmas.push_back(std::exp(mu) / (std::exp(mu) + 1.f / t - 1.f));
        }
        sigmas.push_back(0.f);
        const sd::Tensor<float> size({2}, std::vector<float>{static_cast<float>(p->width), static_cast<float>(p->height)});
        for (int batch = 0; batch < p->batch_count; ++batch) {
            auto rng = runtime_->rng_ptr();
            rng->manual_seed(static_cast<uint64_t>(p->seed < 0 ? 42 : p->seed) + batch);
            auto x = sd::Tensor<float>::randn({w, h, 4, 1}, rng);
            round_bf16(x);
            emit_phase_marker("denoise", "begin");
            for (int i = 0; i < steps; ++i) {
                check_cancel();
                sd::Tensor<float> time({1}, std::vector<float>{sigmas[i] * 1000.f});
                round_bf16(time);
                auto prediction = unet_->compute(runtime_->n_threads(), sd::ops::concat(x, condition, 0), time, context, size);
                require_tensor(prediction, "UNet");
                prediction = sd::ops::slice(prediction, 0, 0, w);
                round_bf16(prediction);
                // PyTorch promotes sample to F32, but a scalar F32 dt times
                // a BF16 prediction still produces a BF16 tensor.
                auto delta = prediction * (sigmas[i + 1] - sigmas[i]);
                round_bf16(delta);
                x = x + delta;
                round_bf16(x);
                if (control) control->step_done();
                LOG_INFO("dreamlite-mobile: image %d/%d step %d/%d", batch + 1, p->batch_count, i + 1, steps);
            }
            unet_->free_compute_buffer();
            emit_phase_marker("denoise", "end");
            check_cancel();
            emit_phase_marker("decode", "begin");
            auto decoded = vae_->decode(runtime_->n_threads(), x, runtime_->vae_tiling());
            require_tensor(decoded, "VAE decoder");
            const size_t pixels = static_cast<size_t>(p->width) * p->height;
            images[batch] = {static_cast<uint32_t>(p->width), static_cast<uint32_t>(p->height), 3,
                             static_cast<uint8_t*>(std::malloc(pixels * 3))};
            if (!images[batch].data) throw std::bad_alloc();
            for (size_t i = 0; i < pixels; ++i) {
                for (size_t c = 0; c < 3; ++c) {
                    images[batch].data[3 * i + c] = static_cast<uint8_t>(std::round(std::clamp(decoded[i + pixels * c], 0.f, 1.f) * 255.f));
                }
            }
            emit_phase_marker("decode", "end");
        }
        out->images = images;
        out->count = p->batch_count;
        return ED_STATUS_OK;
    } catch (const std::exception& e) {
        for (int i = 0; i < p->batch_count; ++i) std::free(images[i].data);
        std::free(images);
        unet_->free_compute_buffer();
        text_->model.free_compute_buffer();
        vae_->free_compute_buffer();
        return fail(ED_STATUS_GENERATION_FAILED, e.what());
    }
}

ed_status_t DreamLitePipeline::generate_video(const ed_video_generation_params_t*, ed_video_t*, std::string* error) {
    if (error) *error = "DreamLite-mobile does not support video generation";
    return ED_STATUS_UNSUPPORTED;
}

} // namespace edgedit
