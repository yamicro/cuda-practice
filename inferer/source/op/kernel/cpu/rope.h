#ifndef ROPE_H
#define ROPE_H
#include "tensor/tensor.h"
namespace kernel {
    void sin_cos_cache_calc_cpu(int head_size, int max_seq_len, float* sin_cache, float* cos_cache);

    void rope_kernel_cpu(int32_t dim, int32_t kv_dim, int32_t head_size, const tensor::Tensor& input_q,
                         const tensor::Tensor& input_k, const tensor::Tensor& input_pos,
                         const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache,
                         void* stream);
}
#endif //ROPE_H
