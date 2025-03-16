//
// Created by yami on 25-3-16.
//

#ifndef ADD_KERNEL_H
#define ADD_KERNEL_H
#include "tensor/tensor.h"

namespace kernel {
    void add_kernel_cpu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                 const tensor::Tensor& output, void* stream);
}



#endif //ADD_KERNEL_H
