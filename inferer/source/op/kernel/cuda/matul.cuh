#ifndef MATUL_CUH
#define MATUL_CUH
#include <tensor/tensor.h>
#include "base/cuda_config.h"
namespace kernel {
    void matmul_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                          const tensor::Tensor& output, float scale = 1.f,
                          const CudaConfig* config = nullptr);
}


#endif //MATUL_CUH
