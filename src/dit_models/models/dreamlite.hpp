#pragma once

#include "backend/ggml/ggml_extend.hpp"

// DreamLite-mobile's Diffusers names are intentionally retained. This UNet is
// not an SD1/SDXL UNet: separable resnets expand by two, attention uses a single
// KV head, and the first two resolutions omit self attention.
namespace DreamLite {

inline ggml_tensor* add(GGMLRunnerContext* c, ggml_tensor* a, ggml_tensor* b) {
    return bf16_boundary(c, ggml_add(c->ggml_ctx, a, b));
}
inline ggml_tensor* silu(GGMLRunnerContext* c, ggml_tensor* x) {
    return bf16_boundary(c, ggml_silu(c->ggml_ctx, x));
}

class DepthwiseConv : public UnaryBlock {
    int64_t channels_;
    void init_params(ggml_context* ctx, const String2TensorStorage&, const std::string) override {
        params["weight"] = ggml_new_tensor_4d(ctx, GGML_TYPE_BF16, 3, 3, 1, channels_);
    }
public:
    explicit DepthwiseConv(int64_t channels) : channels_(channels) {}
    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
        // Depthwise CPU fallback: exact BF16 values expanded to F32, accumulate
        // without the old intermediate FP16 im2col conversion, then round once.
        auto w = ggml_cast(ctx->ggml_ctx, params.at("weight"), GGML_TYPE_F32);
        return bf16_boundary(ctx, ggml_conv_2d_dw_direct(ctx->ggml_ctx, w,
            bf16_boundary(ctx, x), 1, 1, 1, 1, 1, 1));
    }
};

class UNet : public GGMLBlock {
    void linear(const std::string& p, int in, int out, bool bias = true) {
        blocks[p] = std::make_shared<Linear>(in, out, bias);
    }
    void conv(const std::string& p, int in, int out, int kernel = 3, int stride = 1, bool bias = true) {
        blocks[p] = std::make_shared<Conv2d>(in, out, std::make_pair(kernel, kernel),
            std::make_pair(stride, stride), std::make_pair(kernel / 2, kernel / 2),
            std::make_pair(1, 1), bias);
    }
    void norm(const std::string& p, int channels) {
        // Parameters only; apply group/layer normalization with the appropriate
        // epsilon explicitly below (the shared GroupNorm uses a fixed 1e-6).
        blocks[p] = std::make_shared<LayerNorm>(channels);
    }
    void resnet(const std::string& p, int in, int out) {
        norm(p + ".norm1", in);
        norm(p + ".norm2", out * 2);
        blocks[p + ".conv1.depthwise"] = std::make_shared<DepthwiseConv>(in);
        conv(p + ".conv1.pointwise", in, out * 2, 1, 1, false);
        blocks[p + ".conv2.depthwise"] = std::make_shared<DepthwiseConv>(out * 2);
        conv(p + ".conv2.pointwise", out * 2, out, 1, 1, false);
        linear(p + ".time_emb_proj", 1024, out * 2);
        if (in != out) conv(p + ".conv_shortcut", in, out, 1);
    }
    void attention(const std::string& p, int channels, int context) {
        linear(p + ".to_q", channels, channels, false);
        linear(p + ".to_k", context, 64, false);
        linear(p + ".to_v", context, 64, false);
        linear(p + ".to_out.0", channels, channels);
        blocks[p + ".norm_q"] = std::make_shared<RMSNorm>(64, 1e-5f);
        blocks[p + ".norm_k"] = std::make_shared<RMSNorm>(64, 1e-5f);
    }
    void spatial(const std::string& p, int channels, int depth, bool self) {
        norm(p + ".norm", channels);
        linear(p + ".proj_in", channels, channels);
        linear(p + ".proj_out", channels, channels);
        for (int i = 0; i < depth; ++i) {
            const auto b = p + ".transformer_blocks." + std::to_string(i);
            for (int j = 1; j <= 3; ++j) norm(b + ".norm" + std::to_string(j), channels);
            if (self) attention(b + ".attn1", channels, channels);
            attention(b + ".attn2", channels, 2304);
            linear(b + ".ff.net.0.proj", channels, channels * 6);
            linear(b + ".ff.net.2", channels * 3, channels);
        }
    }
    ggml_tensor* run(GGMLRunnerContext* c, const std::string& p, ggml_tensor* x) {
        return bf16_boundary(c, std::dynamic_pointer_cast<UnaryBlock>(blocks.at(p))->forward(c, x));
    }
    ggml_tensor* normalize(GGMLRunnerContext* c, const std::string& p, ggml_tensor* x,
                           bool group = false, float eps = 1e-5f) {
        std::map<std::string, ggml_tensor*> weights;
        blocks.at(p)->get_param_tensors(weights);
        auto w = weights.at("weight");
        auto b = weights.at("bias");
        if (group) {
            x = ggml_group_norm(c->ggml_ctx, x, 32, eps);
            w = ggml_reshape_4d(c->ggml_ctx, w, 1, 1, w->ne[0], 1);
            b = ggml_reshape_4d(c->ggml_ctx, b, 1, 1, b->ne[0], 1);
        } else {
            x = ggml_norm(c->ggml_ctx, x, eps);
        }
        return bf16_boundary(c, ggml_add(c->ggml_ctx, ggml_mul(c->ggml_ctx, x, w), b));
    }
    ggml_tensor* resnet_forward(GGMLRunnerContext* c, const std::string& p, ggml_tensor* x, ggml_tensor* t) {
        auto ctx = c->ggml_ctx;
        auto h = silu(c, normalize(c, p + ".norm1", x, true));
        h = run(c, p + ".conv1.pointwise", run(c, p + ".conv1.depthwise", h));
        auto emb = run(c, p + ".time_emb_proj", silu(c, t));
        h = add(c, h, ggml_reshape_4d(ctx, emb, 1, 1, emb->ne[0], emb->ne[1]));
        h = silu(c, normalize(c, p + ".norm2", h, true));
        h = run(c, p + ".conv2.pointwise", run(c, p + ".conv2.depthwise", h));
        if (blocks.count(p + ".conv_shortcut")) x = run(c, p + ".conv_shortcut", x);
        return add(c, x, h);
    }
    ggml_tensor* attention_forward(GGMLRunnerContext* c, const std::string& p,
                                   ggml_tensor* x, ggml_tensor* context) {
        auto ctx = c->ggml_ctx;
        auto q = run(c, p + ".to_q", x);
        auto k = run(c, p + ".to_k", context);
        auto v = run(c, p + ".to_v", context);
        const auto heads = q->ne[0] / 64;
        q = ggml_reshape_4d(ctx, q, 64, heads, q->ne[1], q->ne[2]);
        k = ggml_reshape_4d(ctx, k, 64, 1, k->ne[1], k->ne[2]);
        q = std::dynamic_pointer_cast<RMSNorm>(blocks.at(p + ".norm_q"))->forward(c, q);
        k = std::dynamic_pointer_cast<RMSNorm>(blocks.at(p + ".norm_k"))->forward(c, k);
        q = ggml_reshape_3d(ctx, q, heads * 64, x->ne[1], x->ne[2]);
        k = ggml_reshape_3d(ctx, k, 64, context->ne[1], context->ne[2]);
        auto h = ggml_ext_attention_ext(ctx, c->backend, q, k, v, heads, nullptr, false, c->flash_attn_enabled);
        return run(c, p + ".to_out.0", bf16_boundary(c, h));
    }
    ggml_tensor* spatial_forward(GGMLRunnerContext* c, const std::string& p,
                                 ggml_tensor* x, ggml_tensor* context, int depth, bool self) {
        auto ctx = c->ggml_ctx;
        const auto w = x->ne[0], h = x->ne[1], channels = x->ne[2], batch = x->ne[3];
        auto y = normalize(c, p + ".norm", x, true, 1e-6f);
        y = ggml_cont(ctx, ggml_permute(ctx, y, 1, 2, 0, 3));
        y = run(c, p + ".proj_in", ggml_reshape_3d(ctx, y, channels, w * h, batch));
        for (int i = 0; i < depth; ++i) {
            const auto b = p + ".transformer_blocks." + std::to_string(i);
            auto n = normalize(c, b + ".norm1", y);
            // The reference replaces the residual with norm1 when self attention
            // is removed; skipping the entire sublayer gives incorrect images.
            y = self ? add(c, y, attention_forward(c, b + ".attn1", n, n)) : n;
            n = normalize(c, b + ".norm2", y);
            y = add(c, y, attention_forward(c, b + ".attn2", n, context));
            n = run(c, b + ".ff.net.0.proj", normalize(c, b + ".norm3", y));
            const auto inner = n->ne[0] / 2;
            auto value = ggml_view_3d(ctx, n, inner, n->ne[1], n->ne[2], n->nb[1], n->nb[2], 0);
            auto gate = ggml_view_3d(ctx, n, inner, n->ne[1], n->ne[2], n->nb[1], n->nb[2], inner * n->nb[0]);
            n = bf16_boundary(c, ggml_mul(ctx, ggml_cont(ctx, value),
                bf16_boundary(c, ggml_gelu_erf(ctx, ggml_cont(ctx, gate)))));
            y = add(c, y, run(c, b + ".ff.net.2", n));
        }
        y = run(c, p + ".proj_out", y);
        y = ggml_cont(ctx, ggml_permute(ctx, y, 1, 0, 2, 3));
        return add(c, x, ggml_reshape_4d(ctx, y, w, h, channels, batch));
    }
public:
    UNet() {
        conv("conv_in", 4, 256);
        conv("conv_out", 256, 4);
        norm("conv_norm_out", 256);
        linear("time_embedding.linear_1", 256, 1024);
        linear("time_embedding.linear_2", 1024, 1024);
        linear("add_embedding.linear_1", 512, 1024);
        linear("add_embedding.linear_2", 1024, 1024);
        linear("encoder_hid_proj.0", 2048, 2304);
        blocks["encoder_hid_proj.1"] = std::make_shared<RMSNorm>(2304, 1e-5f);
        const int channels[] = {256, 512, 896};
        const int depths[] = {1, 2, 4};
        int in = 256;
        std::vector<int> skips{in};
        for (int i = 0; i < 3; ++i) {
            const auto p = "down_blocks." + std::to_string(i);
            for (int j = 0; j < 2; ++j) {
                resnet(p + ".resnets." + std::to_string(j), in, channels[i]);
                spatial(p + ".attentions." + std::to_string(j), channels[i], depths[i], i == 2);
                in = channels[i];
                skips.push_back(in);
            }
            if (i < 2) {
                conv(p + ".downsamplers.0.conv", in, in, 3, 2);
                skips.push_back(in);
            }
        }
        resnet("mid_block.resnets.0", 896, 896);
        spatial("mid_block.attentions.0", 896, 4, true);
        resnet("mid_block.resnets.1", 896, 896);
        for (int i = 0; i < 3; ++i) {
            const auto p = "up_blocks." + std::to_string(i);
            const int out = channels[2 - i];
            for (int j = 0; j < 3; ++j) {
                resnet(p + ".resnets." + std::to_string(j), in + skips.back(), out);
                skips.pop_back();
                if (i < 2) spatial(p + ".attentions." + std::to_string(j), out, depths[2 - i], i == 0);
                in = out;
            }
            if (i < 2) conv(p + ".upsamplers.0.conv", in, in);
        }
    }
    ggml_tensor* forward(GGMLRunnerContext* c, ggml_tensor* x, ggml_tensor* t,
                         ggml_tensor* context, ggml_tensor* size) {
        auto ctx = c->ggml_ctx;
        x = bf16_boundary(c, x);
        t = bf16_boundary(c, t);
        context = bf16_boundary(c, context);
        auto emb = run(c, "time_embedding.linear_1", bf16_boundary(c, ggml_timestep_embedding(ctx, t, 256, 10000)));
        emb = run(c, "time_embedding.linear_2", silu(c, emb));
        auto aug = ggml_reshape_2d(ctx, ggml_timestep_embedding(ctx, size, 256, 10000), 512, 1);
        aug = run(c, "add_embedding.linear_2", silu(c, run(c, "add_embedding.linear_1", bf16_boundary(c, aug))));
        emb = add(c, emb, aug);
        context = run(c, "encoder_hid_proj.0", context);
        context = std::dynamic_pointer_cast<RMSNorm>(blocks.at("encoder_hid_proj.1"))->forward(c, context);
        x = run(c, "conv_in", x);
        std::vector<ggml_tensor*> skips{x};
        const int depths[] = {1, 2, 4};
        for (int i = 0; i < 3; ++i) {
            const auto p = "down_blocks." + std::to_string(i);
            for (int j = 0; j < 2; ++j) {
                x = resnet_forward(c, p + ".resnets." + std::to_string(j), x, emb);
                x = spatial_forward(c, p + ".attentions." + std::to_string(j), x, context, depths[i], i == 2);
                skips.push_back(x);
            }
            if (i < 2) {
                x = run(c, p + ".downsamplers.0.conv", x);
                skips.push_back(x);
            }
        }
        x = resnet_forward(c, "mid_block.resnets.0", x, emb);
        x = spatial_forward(c, "mid_block.attentions.0", x, context, 4, true);
        x = resnet_forward(c, "mid_block.resnets.1", x, emb);
        for (int i = 0; i < 3; ++i) {
            const auto p = "up_blocks." + std::to_string(i);
            for (int j = 0; j < 3; ++j) {
                x = ggml_concat(ctx, x, skips.back(), 2);
                skips.pop_back();
                x = resnet_forward(c, p + ".resnets." + std::to_string(j), x, emb);
                if (i < 2) x = spatial_forward(c, p + ".attentions." + std::to_string(j), x, context, depths[2 - i], i == 0);
            }
            if (i < 2) x = run(c, p + ".upsamplers.0.conv", ggml_upscale(ctx, x, 2, GGML_SCALE_MODE_NEAREST));
        }
        return run(c, "conv_out", silu(c, normalize(c, "conv_norm_out", x, true)));
    }
};

struct Runner : GGMLRunner {
    UNet unet;
    Runner(ggml_backend_t backend, bool offload, const String2TensorStorage& storage)
        : GGMLRunner(backend, offload) {
        set_bf16_compute(true);
        unet.set_bf16_weights(true);
        unet.init(params_ctx, storage, "model.diffusion_model");
    }
    std::string get_desc() override { return "dreamlite-mobile"; }
    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) {
        unet.get_param_tensors(tensors, "model.diffusion_model");
    }
    sd::Tensor<float> compute(int threads, const sd::Tensor<float>& x, const sd::Tensor<float>& t,
                              const sd::Tensor<float>& text, const sd::Tensor<float>& size) {
        auto graph = [&]() {
            auto gf = new_graph_custom(32768);
            auto ctx = get_context();
            auto out = unet.forward(&ctx, make_input(x), make_input(t), make_input(text), make_input(size));
            ggml_build_forward_expand(gf, out);
            return gf;
        };
        return restore_trailing_singleton_dims(GGMLRunner::compute<float>(graph, threads, false), 4);
    }
};
} // namespace DreamLite
