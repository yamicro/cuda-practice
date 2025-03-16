//
// Created by yami on 25-3-9.
//

#ifndef ALLOC_H
#define ALLOC_H
#include <cstdlib>
#include <base/base.h>

#define UNUSED(expr) \
do {               \
(void)(expr);    \
} while (0)

namespace base {
enum class MemcpyKind {
    kMemcpyCPU2CPU = 0,
    kMemcpyCPU2CUDA = 1,
    kMemcpyCUDA2CPU = 2,
    kMemcpyCUDA2CUDA = 3,
  };

class DeviceAllocator {
    public:
    explicit DeviceAllocator(DeviceType device_type) : device_type_(device_type) {}

    virtual DeviceType device_type() const { return device_type_; }

    virtual void* allocate(size_t bytes, size_t alignment) const = 0;

    virtual void* allocate(size_t bytes) const = 0;

    virtual void release(void* p) const = 0;

    virtual void memcpy(const void* src_ptr, void* dest_ptr, size_t byte_size,
                      MemcpyKind memcpy_kind = MemcpyKind::kMemcpyCPU2CPU,
                      void* stream = nullptr, bool need_sync = false) const;

    virtual void memset_zero(void* ptr, size_t byte_size, void* stream,
                             bool need_sync = false);
    private:
        DeviceType device_type_ = DeviceType::kDeviceUnknown;
};

class DeviceAllocatorCPU : public DeviceAllocator {
  public:
    explicit DeviceAllocatorCPU();

    void* allocate(size_t bytes, size_t alignment) const override;

    void* allocate(size_t bytes) const override;

    void release(void* p) const override;
};

class CUDADeviceAllocator : public DeviceAllocator {
public:
    explicit CUDADeviceAllocator();

    void* allocate(size_t bytes, size_t alignment) const override;

    void* allocate(size_t byte_size) const override;

    void release(void* ptr) const override;
};

class CPUDeviceAllocatorFactory {
  public:
    static std::shared_ptr<DeviceAllocatorCPU> get_instance() {
        if (instance == nullptr) {
          instance = std::make_shared<DeviceAllocatorCPU>();
        }

        return instance;
    };
  private:
    static std::shared_ptr<DeviceAllocatorCPU> instance;
};

class GPUDeviceAllocatorFactory
{
    public:
        static std::shared_ptr<CUDADeviceAllocator> get_instance()
        {
            if (instance == nullptr) {
                instance = std::make_shared<CUDADeviceAllocator>();
            }

            return instance;
        }
    private:
        static std::shared_ptr<CUDADeviceAllocator> instance;
};
}
#endif //ALLOC_H
