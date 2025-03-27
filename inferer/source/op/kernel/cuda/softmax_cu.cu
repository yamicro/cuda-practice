//
// Created by yami on 25-3-26.
//

#include "softmax_cu.cuh"

#define BLOCK_SIZE 1024

namespace kernel
{
    __global__ void softmax_kernel_cu_fp32(const int64_t total_elements, const int64_t cols, float* input) {
        int row = blockIdx.x;
        int idx = threadIdx.x;
        int offset = row * cols;

        float* row_ptr = input + offset;

        float max_val = -FLT_MAX;
        for (int i = idx; i < cols; i += blockDim.x) {
            max_val = fmaxf(max_val, row_ptr[i]);
        }

        __shared__ float shared_max;
        max_val = blockReduceMax(max_val);
        if (idx == 0) {
            shared_max = max_val;
        }
        __syncthreads();

        float sum_val = 0.0f;
        for (int i = idx; i < cols; i += blockDim.x) {
            sum_val += expf(row_ptr[i] - shared_max);
        }

        __shared__ float shared_sum;
        sum_val = blockReduceSum(sum_val);
        if (idx == 0) {
            shared_sum = sum_val;
        }
        __syncthreads();

        for (int i = idx; i < cols; i += blockDim.x) {
            row_ptr[i] = expf(row_ptr[i] - shared_max) / shared_sum;
        }
    }

    __inline__ __device__ float blockReduceMax(float val) {
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
        }

        __shared__ float shared[32];
        int lane = threadIdx.x % warpSize;
        int warp_id = threadIdx.x / warpSize;

        if (lane == 0) {
            shared[warp_id] = val;
        }
        __syncthreads();

        val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : -FLT_MAX;
        if (warp_id == 0) {
            for (int offset = warpSize / 2; offset > 0; offset /= 2) {
                val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
            }
        }
        return val;
    }

    __inline__ __device__ float blockReduceSum(float val) {
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }

        __shared__ float shared[32];
        int lane = threadIdx.x % warpSize;
        int warp_id = threadIdx.x / warpSize;

        if (lane == 0) {
            shared[warp_id] = val;
        }
        __syncthreads();

        val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;
        if (warp_id == 0) {
            for (int offset = warpSize / 2; offset > 0; offset /= 2) {
                val += __shfl_down_sync(0xFFFFFFFF, val, offset);
            }
        }
        return val;
    }


    void softmax_inplace_cu(const tensor::Tensor& input, void* stream) {
        const int block_size = BLOCK_SIZE;
        int dims = input.dims_size();

        if (dims < 1) {
            printf("Error: Invalid tensor dimension.\n");
            return;
        }

        int64_t rows = 1;
        int64_t cols = input.get_dim(dims - 1);

        for (int i = 0; i < dims - 1; ++i) {
            rows *= input.get_dim(i);
        }

        float* x_ptr = const_cast<float*>(input.ptr<float>());

        dim3 grid(rows);
        dim3 block(block_size);

        cudaStream_t stream_ = stream ? static_cast<cudaStream_t>(stream) : 0;

        // 启动 kernel 处理 softmax
        softmax_kernel_cu_fp32<<<grid, block, 0, stream_>>>(rows * cols, cols, x_ptr);

        // 错误检查
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
        }
    }
}

