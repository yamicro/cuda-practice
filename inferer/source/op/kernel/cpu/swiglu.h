#ifndef SWIGLU_H
#define SWIGLU_H
#include "tensor/tensor.h"
namespace kernel {
    void swiglu_kernel_cpu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                           const tensor::Tensor& output, void* stream);
}
#endif //SWIGLU_H
