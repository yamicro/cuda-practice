#include "matul.cuh"

#include <base/cuda_config.h>

namespace kernel
{
    template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
    __global__ void matmul_kernel_cu_fp32(const float* input, const float* weight, float* C, int M,
                                      int K) {
        __shared__ float sdata[THREAD_PER_BLOCK];
        unsigned int tid = threadIdx.x;

        int start_row = blockIdx.x * ROW_PER_BLOCK;
        int end_row = start_row + ROW_PER_BLOCK;
        if (start_row >= K) {
            return;
        }
        for (int p = start_row; p < end_row; ++p) {
            sdata[tid] = 0;
            for (int i = tid; i < M; i += THREAD_PER_BLOCK) {
                sdata[tid] += input[i] * weight[p * M + i];
            }
            __syncthreads();
            for (unsigned int s = 1; s < THREAD_PER_BLOCK; s *= 2) {
                if ((tid & (2 * s - 1)) == 0) {
                    sdata[tid] += sdata[tid + s];
                }
                __syncthreads();
            }

            if (tid == 0) {
                C[p] = sdata[0];
            }
            __syncthreads();
        }
    }

    void matmul_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                          const tensor::Tensor& output, float scale,
                          const CudaConfig* config)
    {
        CHECK(config != nullptr);
        if (config->stream) {
        }
        CHECK(input.is_empty() == false && input.dims_size() <= 2);
        CHECK(input.device_type() == base::DeviceType::kDeviceCUDA);

        CHECK(weight.is_empty() == false && weight.dims_size() == 2);
        CHECK(weight.device_type() == base::DeviceType::kDeviceCUDA);
        const int32_t K = weight.get_dim(0);  // row
        const int32_t M = weight.get_dim(1);  // col
        CHECK_EQ(M, input.get_dim(0));
        matmul_kernel_cu_fp32<256, 1><<<K, 256>>>(input.ptr<float>(), weight.ptr<float>(),
                                                  const_cast<float*>(output.ptr<float>()), M, K);
    }
}
