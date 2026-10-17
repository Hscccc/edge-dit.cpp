#pragma once

#include "dit_models/pipelines/dit_pipeline.hpp"

namespace DreamLite { struct Runner; }
namespace LLM { struct LLMEmbedder; }
struct TinyImageAutoEncoder;

namespace edgedit {

class DreamLitePipeline final : public DiTPipeline {
public:
    DreamLitePipeline();
    ~DreamLitePipeline() override;
    const char* name() const override { return "dreamlite-mobile"; }
    SDVersion version() const override { return VERSION_DREAMLITE_MOBILE; }
    bool ready() const override { return ready_; }
    bool supports_image_generation() const override { return ready_; }
    bool supports_video_generation() const override { return false; }
    ed_sampler_t default_sample_method() const override { return ED_SAMPLER_EULER; }
    ed_scheduler_t default_scheduler(ed_sampler_t) const override { return ED_SCHEDULER_DISCRETE; }
    bool prepare(const ed_context_params_t&, ModelRuntime&, const ModelLoader&,
                 PipelineTensorRegistry&, std::string*) override;
    void mark_ready() override { ready_ = true; }
    ed_status_t generate_image(const ed_image_generation_params_t*, ed_image_batch_t*, std::string*) override;
    ed_status_t generate_video(const ed_video_generation_params_t*, ed_video_t*, std::string*) override;
private:
    bool ready_ = false;
    ModelRuntime* runtime_ = nullptr;
    std::unique_ptr<DreamLite::Runner> unet_;
    std::unique_ptr<LLM::LLMEmbedder> text_;
    std::unique_ptr<TinyImageAutoEncoder> vae_;
    sd::Tensor<float> encode_prompt(const std::string&, const ed_image_t*);
};

} // namespace edgedit
