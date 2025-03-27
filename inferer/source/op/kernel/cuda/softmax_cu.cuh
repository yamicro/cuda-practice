//
// Created by yami on 25-3-26.
//

#ifndef SOFTMAX_CU_CUH
#define SOFTMAX_CU_CUH
#include "tensor/tensor.h"
#include "float.h"
namespace kernel {
    void softmax_inplace_cu(const tensor::Tensor& input, void* stream);
}
#endif //SOFTMAX_CU_CUH
