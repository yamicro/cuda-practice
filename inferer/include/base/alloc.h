//
// Created by yami on 25-3-9.
//

#ifndef ALLOC_H
#define ALLOC_H
#include <cstdlib>
#include <base/base.h>


namespace base {

class DeviceAllocator {
    public:
    virtual ~DeviceAllocator() = default;
    explicit DeviceAllocator(DeviceType device_type) : device_type_(device_type) {};

        virtual DeviceType device_type() const { return device_type_; }

        virtual void* allocate(size_t bytes, size_t alignment) = 0;

        virtual void* allocate(size_t bytes) = 0;

        virtual void release(void* p, size_t bytes) = 0;

        virtual void memcpy(const void* src_ptr, void* dest_ptr, size_t size) const = 0;
    private:
        DeviceType device_type_ = DeviceType::kDeviceUnknown;
};

class DeviceAllocatorCPU : public DeviceAllocator {
  public:
    explicit DeviceAllocatorCPU(DeviceType device_type) : DeviceAllocator(device_type) {};

    void* allocate(size_t bytes, size_t alignment) override;

    void* allocate(size_t bytes) override;

    void release(void* p, size_t bytes) override;

    void memcpy(const void* src_ptr, void* dest_ptr, size_t size) const override;
};

class CPUDeviceAllocatorFactory {
  public:
    static std::shared_ptr<DeviceAllocator> get_instance(DeviceType device_type) {
        if (instance_ == nullptr) {
          instance_ = std::make_shared<DeviceAllocatorCPU>(device_type);
        }

        return instance_;
    };
  private:
    static std::shared_ptr<DeviceAllocator> instance_;
};
}
#endif //ALLOC_H
