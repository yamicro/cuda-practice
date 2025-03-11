//
// Created by yami on 25-3-11.
//

#ifndef BUFFER_H
#define BUFFER_H
#include <memory>
#include <base/alloc.h>
namespace base
{
    class Buffer : public NoCopyable, std::enable_shared_from_this<Buffer>
    {
    public:
        explicit Buffer() = default;
        explicit Buffer(size_t byte_size, std::shared_ptr<DeviceAllocator> allocator = nullptr,
                  void* ptr = nullptr, bool use_external = false);

        virtual ~Buffer();

        bool allocate();

        void copy_from(const Buffer& buffer) const;

        void copy_from(const Buffer* buffer) const;

        void* ptr();

        const void* ptr() const;

        size_t byte_size() const;

        std::shared_ptr<DeviceAllocator> allocator() const;

        DeviceType device_type() const;

        void set_device_type(DeviceType device_type);

        std::shared_ptr<Buffer> get_shared_from_this();

        bool is_external() const;

    private:
        size_t byte_size_ = 0;
        void* ptr_ = nullptr;
        bool use_external_ = false;
        DeviceType device_type_ = DeviceType::kDeviceUnknown;
        std::shared_ptr<DeviceAllocator> allocator_;
    };

}
#endif //BUFFER_H
