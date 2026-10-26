#pragma once
#include "ggml-backend.h"
#ifdef __cplusplus
extern "C" {
#endif
ggml_backend_reg_t ggml_backend_hiai_reg(void);
ggml_backend_t ggml_backend_hiai_init(void);
void ggml_backend_hiai_set_n_threads(ggml_backend_t backend, int threads);
struct ed_hiai_stats {
    uint64_t npu_matmuls, npu_dispatches, cpu_nodes, failed_matmuls;
};
struct ed_hiai_stats ggml_backend_hiai_stats(ggml_backend_t backend);
#ifdef __cplusplus
}
#endif
