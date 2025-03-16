#include "base/alloc.h"
#include <glog/logging.h>
#include <cstdlib>
#include "base/alloc.h"

#if (defined(_POSIX_ADVISORY_INFO) && (_POSIX_ADVISORY_INFO >= 200112L))
#define INFEROR_HAVE_POSIX_MEMALIGN
#endif

namespace base
{
    DeviceAllocatorCPU::DeviceAllocatorCPU() : DeviceAllocator(DeviceType::kDeviceCPU) {};

    void* DeviceAllocatorCPU::allocate(size_t bytes) const
    {
        if (!bytes)
        {
            return nullptr;
        }

#ifdef INFEROR_HAVE_POSIX_MEMALIGN
        void* data = nullptr;
        const size_t alignment = (bytes > 1024) ? size_t(32) : size_t(16);
        int status = posix_memalign(&data, (alignment >= sizeof(void*))? alignment : sizeof(void*), bytes);
        if (status != 0)
        {
            return nullptr;
        }
        return data;
#else
        void* data = malloc(bytes);
        return data;
#endif
    }

    void* DeviceAllocatorCPU::allocate(size_t bytes, size_t alignment) const {
        // 如果请求的字节数为0，直接返回nullptr
        if (bytes == 0) {
            return nullptr;
        }

        // 确保对齐要求至少为 sizeof(void*)，并且符合 posix_memalign 要求（必须为2的幂）
        if (alignment < sizeof(void*)) {
            alignment = sizeof(void*);
        }

        void* ptr = nullptr;
        // posix_memalign 的第二个参数必须是符合要求的对齐值，返回值为0表示成功
        int result = posix_memalign(&ptr, alignment, bytes);
        if (result != 0) {
            // 如果分配失败，返回nullptr
            return nullptr;
        }
        return ptr;
    }

    void DeviceAllocatorCPU::release(void* p) const
    {
        if (p)
        {
            free(p);
        }
    }

    std::shared_ptr<DeviceAllocatorCPU> CPUDeviceAllocatorFactory::instance = nullptr;
}

