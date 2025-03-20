#include "rmsnorm.h"

namespace kernel {
    void rmsnorm_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                            const tensor::Tensor& output, void* stream) {
        UNUSED(stream);
        CHECK(!input.is_empty());
        CHECK(!weight.is_empty());
        CHECK(!output.is_empty());

        CHECK(input.device_type() == base::DeviceType::kDeviceCPU &&
              weight.device_type() == base::DeviceType::kDeviceCPU &&
              output.device_type() == base::DeviceType::kDeviceCPU);
        // std::cout << "Address from index cpu: " << &input.index<float>(0) << std::endl;
        // std::cout << "Address from ptr cpu: " << reinterpret_cast<const void*>(input.ptr<float>()) << std::endl;
        // int n = input.index<float>(3);
        //
        // std::cout << "input_cpu:" << std::endl;
        // for (int i = 0; i < 32 * 15; ++i) {
        //     std::cout << input.index<float>(i) << " ";
        // }
        // std::cout << std::endl;
        //
        // std::cout << "weight_cpu:" << std::endl;
        // for (int i = 0; i < 32 * 15; ++i) {
        //     std::cout << weight.index<float>(i) << " ";
        // }
        // std::cout << std::endl;


        const float* in_ptr = input.ptr<float>();
        const float* wei_ptr = weight.ptr<float>();
        const float* out_ptr = output.ptr<float>();
        const int32_t dim = static_cast<int32_t>(input.size());

        arma::fvec in_tensor(const_cast<float*>(in_ptr), dim, false, true);
        arma::fvec out_tensor(const_cast<float*>(out_ptr), dim, false, true);
        arma::fvec wei_tensor(const_cast<float*>(wei_ptr), dim, false, true);
        const float eps = 1e-5f;
        const float mean = arma::as_scalar(arma::mean(arma::pow(in_tensor, 2))) + eps;
        const float rsqrt = 1.f / std::sqrt(mean);
        out_tensor = wei_tensor % (rsqrt * in_tensor);
        std::cout << "output_cpu:" << std::endl;
        out_tensor.brief_print();
        for (int i = 0; i < 32 * 15; ++i) {
            std::cout << output.index<float>(i) << " ";
        }
        std::cout << std::endl;
    }
}  // namespace kernel