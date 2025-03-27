#include "softmax.h"
namespace kernel {
    void softmax_inplace_cpu(const tensor::Tensor& input, void* stream) {
        int32_t size = static_cast<int32_t>(input.size());
        const float* input_ptr = input.ptr<float>();

        float max_value = *std::max_element(input_ptr, input_ptr + size);

        arma::fvec input_mat(const_cast<float*>(input_ptr), size, false, true);
        input_mat = arma::exp(input_mat - max_value);

        float sum_value = arma::sum(input_mat);
        input_mat = input_mat / sum_value;
    }
}