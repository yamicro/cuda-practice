#include "tensor/tensor.h"
#include <glog/logging.h>
#include <numeric>

namespace tensor {
    template <typename T, typename Tp>
    static inline size_t ReduceDimension(T begin, T end, Tp init) {
        if (begin >= end) {
            return 0;
        }
        size_t size = std::accumulate(begin, end, init, std::multiplies<>());
        return size;
    }

    Tensor::Tensor(base::DataType data_type, int32_t dim0, bool need_alloc,
                   std::shared_ptr<base::DeviceAllocator> alloc)
        : data_type_(data_type) {
        dims_.push_back(dim0);
        size_ = dim0;
        if (need_alloc && alloc) {
            allocate(alloc);
        }
    }

    Tensor::Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, bool need_alloc,
                   std::shared_ptr<base::DeviceAllocator> alloc)
        : data_type_(data_type) {
        dims_.push_back(dim0);
        dims_.push_back(dim1);
        size_ = dim0 * dim1;
        if (need_alloc && alloc) {
            allocate(alloc);
        }
    }

    Tensor::Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2,
                   bool need_alloc, std::shared_ptr<base::DeviceAllocator> alloc)
        : data_type_(data_type) {
        dims_.push_back(dim0);
        dims_.push_back(dim1);
        dims_.push_back(dim2);
        size_ = dim0 * dim1 * dim2;
        if (need_alloc && alloc) {
            allocate(alloc);
        }
    }

    Tensor::Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2,
                   int32_t dim3, bool need_alloc,
                   std::shared_ptr<base::DeviceAllocator> alloc)
        : data_type_(data_type) {
        dims_.push_back(dim0);
        dims_.push_back(dim1);
        dims_.push_back(dim2);
        dims_.push_back(dim3);
        size_ = dim0 * dim1 * dim2 * dim3;
        if (need_alloc && alloc) {
            allocate(alloc);
        }
    }

    Tensor::Tensor(base::DataType data_type, std::vector<int32_t> dims)
        : dims_(std::move(dims)), data_type_(data_type) {
        size_ = ReduceDimension(dims_.begin(), dims_.end(), 1);
    }

    bool Tensor::is_empty() const {
        return size_ == 0 || buffer_ == nullptr || buffer_->ptr() == nullptr;
    }

    size_t Tensor::byte_size() const
    {
        return this->size() * DataTypeSize(data_type_);
    }

    size_t Tensor::size() const {
        return this->size_;
    }

    int32_t Tensor::dims_size() const {
      return static_cast<int32_t>(this->dims_.size());
    }

    base::DataType Tensor::data_type() const {
      return this->data_type_;
    }

    int32_t Tensor::get_dim(int32_t idx) const {
      CHECK_LT(idx, dims_.size());
      CHECK_GE(idx, 0);
      return static_cast<int32_t>(this->dims_.at(idx));
    }

    const std::vector<int32_t>& Tensor::dims() const {
      return this->dims_;
    }

    std::vector<size_t> Tensor::strides() const {
        std::vector<size_t> strides;
        if (!dims_.empty()) {
            for (int32_t i = 0; i < dims_.size() - 1; ++i) {
                size_t stride = ReduceDimension(dims_.begin() + i + 1, dims_.end(), 1);
                strides.push_back(stride);
            }
            strides.push_back(1);
        }
        return strides;
    }

    bool Tensor::assign(std::shared_ptr<base::Buffer> buffer) {
        if (!buffer) {
            LOG(ERROR) << "The buffer parameter in the assign function is null pointer!";
            return false;
        }

        size_t byte_size = this->byte_size();
        if (byte_size > buffer->byte_size()) {
            LOG(ERROR) << "The size of buffer is too small for the tensor!";
            return false;
        }
        buffer_ = buffer;
        return true;
    }

    void Tensor::reset(base::DataType data_type, const std::vector<int32_t>& dims) {
        this->data_type_ = data_type;
        this->dims_ = dims;
        this->size_ = ReduceDimension(dims.begin(), dims.end(), 1);
        this->buffer_ = nullptr;
    }

    void Tensor::set_device_type(base::DeviceType device_type) {
        if (buffer_) {
            buffer_->set_device_type(device_type);
        }
    }

    base::DeviceType Tensor::device_type() const {
        if (!buffer_) {
            return base::DeviceType::kDeviceUnknown;
        }
        return buffer_->device_type();
    }


    bool Tensor::allocate(std::shared_ptr<base::DeviceAllocator> allocator,
                      bool need_realloc) {
        if (!allocator) {
            LOG(ERROR) << "The allocator parameter in the allocate function is null "
                          "pointer!";
            return false;
        }

        size_t byte_size = this->byte_size();
        if (!byte_size) {
            LOG(ERROR) << "The byte_size parameter in the allocate function is equal to zero!";
            return false;
        }

        if (buffer_ && byte_size <= buffer_->byte_size()) {
            if (!need_realloc) {
                return true;
            }
        }

        buffer_ = std::make_shared<base::Buffer>(byte_size, allocator, nullptr);
        if (!buffer_->ptr()) {
            LOG(ERROR) << "The memory allocated is a null pointer!";
            return false;
        }
        return true;
    }

    Tensor Tensor::clone() const {
        Tensor new_tensor = *this;
        size_t byte_size = this->byte_size();

        auto allocator = buffer_->allocator();
        new_tensor.buffer_ = std::make_shared<base::Buffer>(byte_size, allocator);
        new_tensor.buffer_->copy_from(buffer_.get());
        return new_tensor;
    }
    //TODO: cuda 拷贝错误
    void Tensor::to_cuda(cudaStream_t stream, int u) {
        CHECK_NE(buffer_, nullptr);
        const base::DeviceType device_type = this->device_type();
        if (device_type == base::DeviceType::kDeviceUnknown) {
            LOG(ERROR) << "The device type of the tensor is unknown.";
        } else if (device_type == base::DeviceType::kDeviceCPU) {
            size_t byte_size = this->byte_size();

            auto cu_alloc = base::GPUDeviceAllocatorFactory::get_instance();
            auto cu_buffer = std::make_shared<base::Buffer>(byte_size, cu_alloc);
            cu_alloc->memcpy(buffer_->ptr(), cu_buffer->ptr(), byte_size, base::MemcpyKind::kMemcpyCPU2CUDA,
                             stream);
            this->buffer_ = cu_buffer;
        } else {
            LOG(INFO) << "The device type of the tensor is already cpu.";
        }
    }

    void Tensor::to_cpu() {
        CHECK_NE(buffer_, nullptr);
        const base::DeviceType device_type = this->device_type();

        if (device_type == base::DeviceType::kDeviceUnknown) {
            LOG(ERROR) << "The device type of the tensor is unknown.";
        } else if (device_type == base::DeviceType::kDeviceCUDA) {
            size_t byte_size = this->byte_size();
            auto cpu_alloc = base::CPUDeviceAllocatorFactory::get_instance();
            auto cpu_buffer = std::make_shared<base::Buffer>(byte_size, cpu_alloc);
            cpu_alloc->memcpy(buffer_->ptr(), cpu_buffer->ptr(), byte_size,
                              base::MemcpyKind::kMemcpyCUDA2CPU);
            this->buffer_ = cpu_buffer;
        } else {
            LOG(INFO) << "The device type of the tensor is already cuda.";
        }
    }
}