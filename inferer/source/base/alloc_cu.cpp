#include "base/alloc.h"
#include <cuda_runtime_api.h>


namespace base {

    CUDADeviceAllocator::CUDADeviceAllocator() : DeviceAllocator(DeviceType::kDeviceCUDA){}

    void* CUDADeviceAllocator::allocate(size_t bytes, size_t alignment) const
    {
        return nullptr;
    }


    void* CUDADeviceAllocator::allocate(size_t byte_size) const {
        if (!byte_size) {
            return nullptr;
        }
        void* ptr = nullptr;
        cudaError_t err = cudaMallocManaged(&ptr, byte_size);
        CHECK_EQ(err, cudaSuccess);
        return ptr;
    }

    void CUDADeviceAllocator::release(void* ptr) const {
        if (ptr) {
            cudaFree(ptr);
        }
    }
    std::shared_ptr<CUDADeviceAllocator> GPUDeviceAllocatorFactory::instance = nullptr;
}