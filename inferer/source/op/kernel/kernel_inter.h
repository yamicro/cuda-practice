//
// Created by yami on 25-3-16.
//

#ifndef KERNEL_INTER_H
#define KERNEL_INTER_H
#include "tensor/tensor.h"

namespace kernel {
    typedef void (*AddKernel)(const tensor::Tensor& input1, const tensor::Tensor& input2,
                              const tensor::Tensor& output, void* stream);

    AddKernel get_add_kernel(base::DeviceType device_type);
}





#endif //KERNEL_INTER_H
