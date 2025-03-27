#ifndef SCALER_SUM_H
#define SCALER_SUM_H
#include <tensor/tensor.h>
namespace kernel {
    void scale_sum_kernel_cpu(const tensor::Tensor& value, const tensor::Tensor& scale,
                              const tensor::Tensor& output, int t, int d, int stride,
                              void* stream = nullptr);
}
#endif //SCALER_SUM_H
