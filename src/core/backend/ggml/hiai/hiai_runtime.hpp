#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <stdexcept>
#include <string>
#include <vector>

// ABI declarations for Huawei's documented Android single-operator interface.
// The device supplies the implementation; no proprietary binary is distributed.
namespace ed_hiai {
struct FpcrGuard {
    uint64_t value;
    FpcrGuard() { asm volatile("mrs %0, fpcr" : "=r"(value)); }
    ~FpcrGuard() { asm volatile("msr fpcr, %0" :: "r"(value) : "memory"); }
};

struct Desc; struct Tensor; struct Buffer; struct Op; struct Executor; struct Options;
class Api {
public:
    void *handle = nullptr;
    std::string error;
    Desc *(*desc_create)(const int64_t *,size_t,int,int,bool) = nullptr;
    void (*desc_destroy)(Desc **) = nullptr;
    Tensor *(*tensor_create)(const Desc *) = nullptr;
    Tensor *(*tensor_const)(const Desc *,void *,size_t) = nullptr;
    uint32_t (*tensor_destroy)(Tensor **) = nullptr;
    Buffer *(*tensor_buffer)(const Tensor *) = nullptr;
    void *(*buffer_data)(const Buffer *) = nullptr;
    Buffer *(*buffer_create)(size_t) = nullptr;
    uint32_t (*buffer_destroy)(Buffer **) = nullptr;
    Op *(*conv_create)(int,const int64_t *,const int64_t *,const int64_t *,int64_t,int) = nullptr;
    void (*op_destroy)(Op **) = nullptr;
    Options *(*options_create)() = nullptr;
    void (*options_destroy)(Options **) = nullptr;
    int (*precheck)(const Options *,const Op *,const Desc *,const Desc *,const Tensor *,const Tensor *) = nullptr;
    Executor *(*executor_create)(const Options *,const Op *,const Desc *,const Desc *,const Tensor *,const Tensor *) = nullptr;
    uint32_t (*executor_destroy)(Executor **) = nullptr;
    size_t (*workspace_size)(const Executor *) = nullptr;
    uint32_t (*executor_init)(Executor *,void *,size_t) = nullptr;
    uint32_t (*execute)(Executor *,Tensor **,int32_t,Tensor **,int32_t) = nullptr;

    Api() {
        FpcrGuard fpcr;
        handle=dlopen("/vendor/lib64/libai_single_op.so",RTLD_NOW|RTLD_LOCAL);
        if (!handle) { const char *e=dlerror(); error=e?e:"dlopen failed"; return; }
#define LOAD(member,symbol) member=reinterpret_cast<decltype(member)>(dlsym(handle,symbol)); if(!member){error=std::string("missing ")+symbol;return;}
        LOAD(desc_create,"HiAI_SingleOpTensorDesc_Create")
        LOAD(desc_destroy,"HiAI_SingleOpTensorDesc_Destroy")
        LOAD(tensor_create,"HiAI_SingleOpTensor_CreateFromTensorDesc")
        LOAD(tensor_const,"HiAI_SingleOpTensor_CreateFromConst")
        LOAD(tensor_destroy,"HiAI_SingleOpTensor_Destroy")
        LOAD(tensor_buffer,"HiAI_SingleOpTensor_GetBuffer")
        LOAD(buffer_data,"HiAI_SingleOpBuffer_GetData")
        LOAD(buffer_create,"HiAI_SingleOpBuffer_Create")
        LOAD(buffer_destroy,"HiAI_SingleOpBuffer_Destroy")
        LOAD(conv_create,"HiAI_SingleOpDescriptor_CreateConvolution")
        LOAD(op_destroy,"HiAI_SingleOpDescriptor_Destroy")
        LOAD(options_create,"HiAI_SingleOpOptions_Create")
        LOAD(options_destroy,"HiAI_SingleOpOptions_Destroy")
        LOAD(precheck,"HiAI_SingleOpExecutor_PreCheckConvolution")
        LOAD(executor_create,"HiAI_SingleOpExecutor_CreateConvolution")
        LOAD(executor_destroy,"HiAI_SingleOpExecutor_Destroy")
        LOAD(workspace_size,"HiAI_SingleOpExecutor_GetWorkspaceSize")
        LOAD(executor_init,"HiAI_SingleOpExecutor_Init")
        LOAD(execute,"HiAI_SingleOpExecutor_Execute")
#undef LOAD
    }
    bool available() const { return handle && error.empty(); }
    static Api &get() { static Api api; return api; }
};

class Conv {
    Api &a=Api::get();
    Desc *id=nullptr,*od=nullptr,*wd=nullptr,*bd=nullptr;
    Tensor *in=nullptr,*out=nullptr,*weight=nullptr,*bias=nullptr;
    Op *op=nullptr; Options *opt=nullptr; Executor *ex=nullptr;
    Buffer *workspace=nullptr;
    std::vector<_Float16> weights,biases;
public:
    int k,n,m;
    size_t workspace_bytes=0;
    int support=2;
    std::string error;
    Conv(int k_,int n_,int m_,const float *w):weights(w,w+size_t(k_)*n_),biases(n_,0),k(k_),n(n_),m(m_) {
        FpcrGuard fpcr;
        if(!a.available()){error=a.error;return;}
        int64_t is[]={1,k,1,m},os[]={1,n,1,m},ws[]={n,k,1,1},bs[]={1,n,1,1};
        id=a.desc_create(is,4,1,0,false); od=a.desc_create(os,4,1,0,false);
        wd=a.desc_create(ws,4,1,0,false); bd=a.desc_create(bs,4,1,0,false);
        if(!id||!od||!wd||!bd){error="tensor descriptor creation failed";return;}
        weight=a.tensor_const(wd,weights.data(),weights.size()*sizeof(_Float16));
        bias=a.tensor_const(bd,biases.data(),biases.size()*sizeof(_Float16));
        opt=a.options_create();
        int64_t strides[]={1,1},dilations[]={1,1},pads[]={0,0,0,0};
        op=a.conv_create(0,strides,dilations,pads,1,0);
        if(!weight||!bias||!opt||!op){error="convolution setup failed";return;}
        support=a.precheck(opt,op,id,od,weight,bias);
        if(support==2){error="convolution unsupported by device";return;}
        ex=a.executor_create(opt,op,id,od,weight,bias);
        if(!ex){error="executor creation failed";return;}
        workspace_bytes=a.workspace_size(ex);
        if(workspace_bytes){workspace=a.buffer_create(workspace_bytes);if(!workspace){error="workspace allocation failed";return;}}
        if(a.executor_init(ex,workspace?a.buffer_data(workspace):nullptr,workspace_bytes)!=0){error="executor init failed";return;}
        in=a.tensor_create(id);out=a.tensor_create(od);
        if(!in||!out){error="IO allocation failed";return;}
        if(!a.buffer_data(a.tensor_buffer(in))||!a.buffer_data(a.tensor_buffer(out)))error="IO mapping failed";
    }
    ~Conv(){
        FpcrGuard fpcr;
        if(ex)a.executor_destroy(&ex);
        if(in)a.tensor_destroy(&in);if(out)a.tensor_destroy(&out);
        if(weight)a.tensor_destroy(&weight);if(bias)a.tensor_destroy(&bias);
        if(workspace)a.buffer_destroy(&workspace);
        if(op)a.op_destroy(&op);if(opt)a.options_destroy(&opt);
        if(id)a.desc_destroy(&id);if(od)a.desc_destroy(&od);if(wd)a.desc_destroy(&wd);if(bd)a.desc_destroy(&bd);
    }
    Conv(const Conv&)=delete;Conv&operator=(const Conv&)=delete;
    bool valid() const{return error.empty();}
    __attribute__((noinline)) bool run(const float *x,float *y){
        // Some device FP16 conversion routines clobber SIMD callee-saved
        // registers. Force this boundary to preserve the caller's d8-d15.
        asm volatile("" ::: "v8","v9","v10","v11","v12","v13","v14","v15");
        FpcrGuard fpcr;
        if(!valid())return false;
        auto *input=static_cast<_Float16*>(a.buffer_data(a.tensor_buffer(in)));
        // ggml X[K,M] (K contiguous) -> NCHW [1,K,1,M].
        for(int i=0;i<k;i++)for(int j=0;j<m;j++)input[size_t(i)*m+j]=x[size_t(j)*k+i];
        Tensor *ins[]={in},*outs[]={out};
        uint32_t status=a.execute(ex,ins,1,outs,1);
        asm volatile("" ::: "v8","v9","v10","v11","v12","v13","v14","v15");
        if(status){error="execute returned "+std::to_string(status);return false;}
        auto *output=static_cast<const _Float16*>(a.buffer_data(a.tensor_buffer(out)));
        for(int i=0;i<n;i++)for(int j=0;j<m;j++)y[size_t(j)*n+i]=output[size_t(i)*m+j];
        return true;
    }
};
}
