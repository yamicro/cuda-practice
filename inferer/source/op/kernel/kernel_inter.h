//
// Created by yami on 25-3-16.
//

#ifndef KERNEL_INTER_H
#define KERNEL_INTER_H
#include "tensor/tensor.h"
#include <base/cuda_config.h>

namespace kernel {
    typedef void (*AddKernel)(const tensor::Tensor& input1, const tensor::Tensor& input2,
                              const tensor::Tensor& output, void* stream);
    typedef void (*RMSNormKernel)(const tensor::Tensor& input1, const tensor::Tensor& weight,
                                const tensor::Tensor& output, void* stream);
    typedef void (*MatmulKernel)(const tensor::Tensor& input, const tensor::Tensor& weight,
                             const tensor::Tensor& output, float scale, const CudaConfig* config);
    typedef void (*EmbeddingKernel)(const tensor::Tensor& input, const tensor::Tensor& weight,
                                const tensor::Tensor& output, int32_t vocab_size, void* stream);

    AddKernel get_add_kernel(base::DeviceType device_type);
    RMSNormKernel get_rmsnorm_kernel(base::DeviceType device_type);
    MatmulKernel get_matmul_kernel(base::DeviceType device_type);
    EmbeddingKernel get_emb_kernel(base::DeviceType device_type);



}





#endif //KERNEL_INTER_H
