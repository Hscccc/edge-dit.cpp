#include "ggml-hiai.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include <cmath>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cstdlib>

static bool run(ggml_backend_t backend,ggml_type type,int K,int N,int M,bool force_f32=false){
    auto *ctx=ggml_init({size_t(96)<<20,nullptr,false});
    if(!ctx)return false;
    auto *w=ggml_new_tensor_2d(ctx,type,K,N);
    auto *x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,K,M);
    auto *bias=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,N,M);
    std::vector<float>wf(size_t(K)*N),xf(size_t(K)*M);
    for(size_t i=0;i<wf.size();i++)wf[i]=std::sin(float(i)*.17f)*.1f;
    for(size_t i=0;i<xf.size();i++)xf[i]=std::cos(float(i)*.13f)*.1f;
    if(type==GGML_TYPE_F32)std::memcpy(w->data,wf.data(),wf.size()*4);
    else ggml_quantize_chunk(type,wf.data(),w->data,0,N,K,nullptr);
    // Compare against the actual quantized/dequantized weights.
    if(type==GGML_TYPE_Q4_0){
        // CPU reference uses the same GGML quantized tensor, below.
    }
    std::memcpy(x->data,xf.data(),xf.size()*4);
    auto *scaled=ggml_scale(ctx,x,.5f);
    auto *ref=ggml_mul_mat(ctx,w,scaled);ggml_set_name(ref,"test_linear");
    if(force_f32)ggml_mul_mat_set_prec(ref,GGML_PREC_F32);
    auto *out=ggml_add(ctx,ref,bias);
    auto *g=ggml_new_graph_custom(ctx,32,false);ggml_build_forward_expand(g,out);
    ggml_backend_t cpu=ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU,nullptr);
    if(!cpu){ggml_free(ctx);return false;}
    bool ok=true;
    for(int pass=0;pass<3;pass++){
        // Reuse the same address with changed weights: cached executors must
        // be invalidated by content changes, even without a pointer change.
        if(pass==2){
            for(float &v:wf)v*=.5f;
            if(type==GGML_TYPE_F32)std::memcpy(w->data,wf.data(),wf.size()*4);
            else ggml_quantize_chunk(type,wf.data(),w->data,0,N,K,nullptr);
        }
        std::fill_n(static_cast<float*>(bias->data),size_t(N)*M,.125f);
        if(ggml_backend_graph_compute(cpu,g)!=GGML_STATUS_SUCCESS){ok=false;break;}
        std::vector<float>reference(static_cast<float*>(out->data),static_cast<float*>(out->data)+size_t(N)*M);
        auto before=ggml_backend_hiai_stats(backend);
        if(ggml_backend_graph_compute(backend,g)!=GGML_STATUS_SUCCESS){ok=false;break;}
        auto after=ggml_backend_hiai_stats(backend);
        double sq=0,maxe=0;
        for(size_t i=0;i<reference.size();i++){
            double e=static_cast<float*>(out->data)[i]-reference[i];sq+=e*e;maxe=std::max(maxe,std::abs(e));
        }
        double rmse=std::sqrt(sq/reference.size());
        bool pass_ok=std::isfinite(rmse)&&maxe<.01&&(force_f32 ? after.npu_matmuls==before.npu_matmuls : after.npu_matmuls>before.npu_matmuls)&&after.cpu_nodes>before.cpu_nodes;
        std::printf("{\"type\":\"%s\",\"k\":%d,\"n\":%d,\"m\":%d,\"pass\":%d,\"rmse\":%.8g,\"max_error\":%.8g,\"dispatches\":%llu,\"ok\":%s}\n",ggml_type_name(type),K,N,M,pass,rmse,maxe,(unsigned long long)(after.npu_dispatches-before.npu_dispatches),pass_ok?"true":"false");
        ok=ok&&pass_ok;
    }
    ggml_backend_free(cpu);ggml_free(ctx);return ok;
}
int main(){
    ggml_backend_load_all();ggml_backend_register(ggml_backend_hiai_reg());
    auto *backend=ggml_backend_hiai_init();if(!backend)return 3;
    bool ok=run(backend,GGML_TYPE_F32,32,32,16);
    ok=run(backend,GGML_TYPE_F16,256,64,32)&&ok;
    ok=run(backend,GGML_TYPE_Q4_0,256,64,32)&&ok;
    // Exercise both output-channel and token tiling.
    ok=run(backend,GGML_TYPE_Q4_0,256,544,272)&&ok;
    // Exercise accumulation across input-channel tiles.
    ok=run(backend,GGML_TYPE_F32,4128,32,16)&&ok;
    ok=run(backend,GGML_TYPE_F32,32,32,16,true)&&ok;
    ggml_backend_free(backend);return ok?0:1;
}
