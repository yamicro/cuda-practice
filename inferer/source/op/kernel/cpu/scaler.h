//
// Created by yami on 25-3-27.
//

#ifndef SCALER_H
#define SCALER_H
#include <tensor/tensor.h>
namespace kernel {
    void scale_inplace_cpu(float scale, const tensor::Tensor& tensor, void* stream = nullptr);
}
#endif //SCALER_H
