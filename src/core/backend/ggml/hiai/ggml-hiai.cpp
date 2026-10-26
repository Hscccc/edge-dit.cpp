#include "ggml-hiai.h"
#include "hiai_runtime.hpp"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <list>
#include <memory>
#include <tuple>
#include <vector>

namespace {
using Key=std::tuple<uintptr_t,uint64_t,int64_t,int64_t,int,int,int,int,int>;
struct Entry { Key key; std::unique_ptr<ed_hiai::Conv> conv; size_t bytes; };
struct Context {
    ggml_backend_t cpu=ggml_backend_cpu_init();
    std::list<Entry> cache;
    size_t cache_bytes=0;
    size_t cache_limit=128ULL<<20;
    ed_hiai_stats stats{};
    bool strict=false;
    Context() {
        if(const char *s=std::getenv("ED_HIAI_CACHE_MB")) {
            long n=std::strtol(s,nullptr,10);
            if(n>=0&&n<=2048)cache_limit=size_t(n)<<20;
        }
        const char *require=std::getenv("ED_HIAI_REQUIRE_NPU");
        strict=require && std::strcmp(require,"0")!=0;
        if(cpu)ggml_backend_cpu_set_n_threads(cpu,4);
    }
    ~Context(){ggml_backend_free(cpu);}
};
bool candidate(const ggml_tensor *o){
    if(!o||o->op!=GGML_OP_MUL_MAT||o->type!=GGML_TYPE_F32)return false;
    if(ggml_get_op_params_i32(o,0)==GGML_PREC_F32)return false;
    const auto *w=o->src[0],*x=o->src[1];
    if(!w||!x||w->ne[2]!=1||w->ne[3]!=1||x->ne[2]!=1||x->ne[3]!=1)return false;
    if(!ggml_is_contiguous(w)||!ggml_is_contiguous(x)||!ggml_is_contiguous(o))return false;
    if(x->type!=GGML_TYPE_F32&&x->type!=GGML_TYPE_F16&&x->type!=GGML_TYPE_BF16)return false;
    if(w->type!=GGML_TYPE_F32&&!ggml_get_type_traits(w->type)->to_float)return false;
    return w->ne[0]>=32&&w->ne[0]<=32768&&w->ne[1]>=32&&w->ne[1]<=32768&&x->ne[1]>=1&&x->ne[1]<=16384;
}
uint64_t fingerprint(const ggml_tensor *w){
    // Content, not just pointer identity: LoRA updates and reused allocations
    // must never execute an old compiled convolution.
    const auto *p=static_cast<const uint8_t*>(w->data);
    uint64_t h=1469598103934665603ULL;
    for(size_t i=0;i<ggml_nbytes(w);i++){h^=p[i];h*=1099511628211ULL;}
    return h;
}
void row_f32(const ggml_tensor *t,int64_t row,float *dst){
    const char *p=static_cast<const char*>(t->data)+row*t->nb[1];
    if(t->type==GGML_TYPE_F32)std::memcpy(dst,p,size_t(t->ne[0])*4);
    else ggml_get_type_traits(t->type)->to_float(p,dst,t->ne[0]);
}
bool matmul(Context &ctx,ggml_tensor *o){
    const auto *w=o->src[0],*x=o->src[1];
    const int K=int(w->ne[0]),N=int(w->ne[1]),M=int(x->ne[1]);
    const uint64_t hash=fingerprint(w);
    std::vector<float> row(K),weights,activation,result;
    float *dst=static_cast<float*>(o->data);
    for(int m0=0;m0<M;m0+=256){
        int m=std::min(256,M-m0);
        for(int n0=0;n0<N;n0+=512){
            int n=std::min(512,N-n0);
            for(int k0=0;k0<K;k0+=4096){
                int k=std::min(4096,K-k0);
                Key key{uintptr_t(w->data),hash,K,N,k0,n0,k,n,m};
                auto it=std::find_if(ctx.cache.begin(),ctx.cache.end(),[&](const Entry &e){return e.key==key;});
                if(it==ctx.cache.end()){
                    weights.resize(size_t(k)*n);
                    for(int i=0;i<n;i++){
                        row_f32(w,n0+i,row.data());
                        std::copy_n(row.data()+k0,k,weights.data()+size_t(i)*k);
                    }
                    auto conv=std::make_unique<ed_hiai::Conv>(k,n,m,weights.data());
                    if(!conv->valid()){
                        std::fprintf(stderr,"[HIAI] rejected %s k=%d n=%d m=%d: %s\n",w->name,k,n,m,conv->error.c_str());
                        return false;
                    }
                    size_t bytes=2*size_t(k)*n*4+size_t(k+n)*m*4+conv->workspace_bytes;
                    while(!ctx.cache.empty()&&ctx.cache_bytes+bytes>ctx.cache_limit){ctx.cache_bytes-=ctx.cache.back().bytes;ctx.cache.pop_back();}
                    ctx.cache.push_front({key,std::move(conv),bytes});ctx.cache_bytes+=bytes;it=ctx.cache.begin();
                }else ctx.cache.splice(ctx.cache.begin(),ctx.cache,it);
                activation.resize(size_t(k)*m);result.resize(size_t(n)*m);
                for(int j=0;j<m;j++){
                    row_f32(x,m0+j,row.data());
                    std::copy_n(row.data()+k0,k,activation.data()+size_t(j)*k);
                }
                if(!it->conv->run(activation.data(),result.data()))return false;
                ctx.stats.npu_dispatches++;
                for(int j=0;j<m;j++)for(int i=0;i<n;i++){
                    float &d=dst[size_t(m0+j)*N+n0+i];
                    if(k0==0)d=result[size_t(j)*n+i];else d+=result[size_t(j)*n+i];
                }
                if(ctx.cache_limit==0||ctx.cache_bytes>ctx.cache_limit){ctx.cache_bytes-=ctx.cache.front().bytes;ctx.cache.pop_front();}
            }
        }
    }
    return true;
}
const char *backend_name(ggml_backend_t){return "HIAI0";}
void backend_free(ggml_backend_t b){
    auto *c=static_cast<Context*>(b->context);
    std::fprintf(stderr,"[HIAI] totals npu_matmuls=%llu npu_dispatches=%llu cpu_nodes=%llu failed_matmuls=%llu\n",
        (unsigned long long)c->stats.npu_matmuls,(unsigned long long)c->stats.npu_dispatches,
        (unsigned long long)c->stats.cpu_nodes,(unsigned long long)c->stats.failed_matmuls);
    delete c;delete b;
}
ggml_status compute(ggml_backend_t b,ggml_cgraph *g){
    auto &c=*static_cast<Context*>(b->context);
    // edge's runner allocates the complete graph in host buffers first. Execute
    // CPU ranges over views of that graph, preserving its allocation and sources.
    // This is explicit synchronous partitioning, not silent NPU emulation.
    auto cpu_range=[&](int first,int last){
        if(first==last)return GGML_STATUS_SUCCESS;
        auto view=ggml_graph_view(g,first,last);
        for(int i=first;i<last;i++)if(g->nodes[i]->op!=GGML_OP_NONE && g->nodes[i]->op!=GGML_OP_VIEW && g->nodes[i]->op!=GGML_OP_RESHAPE && g->nodes[i]->op!=GGML_OP_PERMUTE && g->nodes[i]->op!=GGML_OP_TRANSPOSE)c.stats.cpu_nodes++;
        return ggml_backend_graph_compute(c.cpu,&view);
    };
    try {
        int first=0;
        for(int i=0;i<g->n_nodes;i++){
            auto *o=g->nodes[i];if(!candidate(o))continue;
            auto status=cpu_range(first,i);if(status!=GGML_STATUS_SUCCESS)return status;
            if(matmul(c,o)){c.stats.npu_matmuls++;}
            else {
                c.stats.failed_matmuls++;
                if(c.strict)return GGML_STATUS_FAILED;
                status=cpu_range(i,i+1);if(status!=GGML_STATUS_SUCCESS)return status;
            }
            first=i+1;
        }
        return cpu_range(first,g->n_nodes);
    }catch(const std::exception &e){std::fprintf(stderr,"[HIAI] graph failed: %s\n",e.what());return GGML_STATUS_FAILED;}
}
const char *dev_name(ggml_backend_dev_t){return "HIAI0";}
const char *dev_desc(ggml_backend_dev_t){return "Kirin HiAI single-op MatMul via 1x1 Conv; explicit CPU fallback";}
void dev_mem(ggml_backend_dev_t,size_t *f,size_t *t){*f=0;*t=0;}
enum ggml_backend_dev_type dev_type(ggml_backend_dev_t){return GGML_BACKEND_DEVICE_TYPE_ACCEL;}
void dev_props(ggml_backend_dev_t d,ggml_backend_dev_props *p){
    *p={};p->name=dev_name(d);p->description=dev_desc(d);p->type=dev_type(d);p->caps.buffer_from_host_ptr=true;
}
ggml_backend_t dev_init(ggml_backend_dev_t,const char*){return ggml_backend_hiai_init();}
ggml_backend_buffer_type_t dev_buft(ggml_backend_dev_t){return ggml_backend_cpu_buffer_type();}
ggml_backend_buffer_t from_ptr(ggml_backend_dev_t,void *p,size_t s,size_t){return ggml_backend_cpu_buffer_from_ptr(p,s);}
bool dev_support(ggml_backend_dev_t,const ggml_tensor *o){
    if(o->op==GGML_OP_NONE||o->op==GGML_OP_VIEW||o->op==GGML_OP_RESHAPE||o->op==GGML_OP_TRANSPOSE||o->op==GGML_OP_PERMUTE)return true;
    return candidate(o);
}
bool dev_support_buft(ggml_backend_dev_t,ggml_backend_buffer_type_t t){return ggml_backend_buft_is_host(t);}
const char *reg_name(ggml_backend_reg_t){return "HIAI";}
size_t reg_count(ggml_backend_reg_t){return ed_hiai::Api::get().available()?1:0;}
ggml_backend_dev_t reg_device(ggml_backend_reg_t r,size_t index){
    GGML_ASSERT(index==0);
    static ggml_backend_device_i iface=[](){
        ggml_backend_device_i v{};v.get_name=dev_name;v.get_description=dev_desc;v.get_memory=dev_mem;
        v.get_type=dev_type;v.get_props=dev_props;v.init_backend=dev_init;v.get_buffer_type=dev_buft;
        v.buffer_from_host_ptr=from_ptr;v.supports_op=dev_support;v.supports_buft=dev_support_buft;return v;
    }();
    static ggml_backend_device d{iface,r,nullptr};return &d;
}
void *reg_proc(ggml_backend_reg_t,const char *name){
    if(std::strcmp(name,"ggml_backend_set_n_threads")==0)return reinterpret_cast<void*>(ggml_backend_hiai_set_n_threads);
    return nullptr;
}
}
ggml_backend_reg_t ggml_backend_hiai_reg(){
    static ggml_backend_reg r{GGML_BACKEND_API_VERSION,{reg_name,reg_count,reg_device,reg_proc},nullptr};return &r;
}
ggml_backend_t ggml_backend_hiai_init(){
    if(!ed_hiai::Api::get().available()){
        std::fprintf(stderr,"[HIAI] unavailable: %s\n",ed_hiai::Api::get().error.c_str());return nullptr;
    }
    auto c=std::make_unique<Context>();if(!c->cpu)return nullptr;
    static ggml_guid guid={0x65,0x64,0x68,0x69,0x61,0x69,0x90,0x20,0x88,0x71,0x23,0x19,0x54,0x18,0x06,0x26};
    ggml_backend_i iface{};iface.get_name=backend_name;iface.free=backend_free;iface.graph_compute=compute;
    return new ggml_backend{&guid,iface,reg_device(ggml_backend_hiai_reg(),0),c.release()};
}
void ggml_backend_hiai_set_n_threads(ggml_backend_t b,int threads){ggml_backend_cpu_set_n_threads(static_cast<Context*>(b->context)->cpu,threads);}
ed_hiai_stats ggml_backend_hiai_stats(ggml_backend_t b){return static_cast<Context*>(b->context)->stats;}
