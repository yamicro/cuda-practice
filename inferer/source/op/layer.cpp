#include "op/layer.h"
#include <glog/logging.h>
#include <cstdarg>
#include <numeric>
#include <utility>

namespace op {
    BaseLayer::BaseLayer(base::DeviceType device_type, LayerType layer_type,
                         base::DataType data_type, std::string layer_name)
        : device_type_(device_type),
          layer_type_(layer_type),
          data_type_(data_type),
          layer_name_(std::move(layer_name)) {
    }

    base::DataType BaseLayer::data_type() const {
        return data_type_;
    }

    LayerType BaseLayer::layer_type() const {
        return layer_type_;
    }

    const std::string& BaseLayer::get_layer_name() const {
        return layer_name_;
    }

    void BaseLayer::set_layer_name(const std::string& layer_name) {
        layer_name_ = layer_name;
    }
    base::DeviceType BaseLayer::device_type() const {
        return device_type_;
    }

    void BaseLayer::set_device_type(base::DeviceType device_type) {
        device_type_ = device_type;
    }

    Layer::Layer(base::DeviceType device_type, LayerType layer_type, std::string layer_name)
        : BaseLayer(device_type, layer_type, base::DataType::kDataTypeFp32,
                    std::move(layer_name)) {
    }

    base::Status Layer::init() {
        return base::error::Success();
    }

    base::Status Layer::forward() {
        return base::error::FunctionNotImplement("");
    }

    base::Status Layer::check_tensor(const tensor::Tensor& tensor,
                                     base::DeviceType device_type,
                                     base::DataType data_type) const {
        if (tensor.is_empty()) {
            return base::error::InvalidArgument("The tensor parameter is empty.");
        }
        if (tensor.device_type() != device_type) {
            return base::error::InvalidArgument("The tensor has a wrong device type.");
        }
        if (tensor.data_type() != data_type) {
            return base::error::InvalidArgument("The tensor has a wrong data type.");
        }
        return base::error::Success();
    }

    base::Status Layer::check_tensor_with_dim(const tensor::Tensor& tensor,
                                          base::DeviceType device_type,
                                          base::DataType data_type, ...) const {
        std::va_list args;
        if (tensor.is_empty()) {
            return base::error::InvalidArgument("The tensor parameter is empty.");
        }
        if (tensor.device_type() != device_type) {
            return base::error::InvalidArgument("The tensor has a wrong device type.");
        }
        if (tensor.data_type() != data_type) {
            return base::error::InvalidArgument("The tensor has a wrong data type.");
        }

        va_start(args, data_type);
        int32_t dims = tensor.dims_size();
        for (int32_t i = 0; i < dims; ++i) {
            int32_t dim = va_arg(args, int32_t);
            if (dim != tensor.get_dim(i)) {
                return base::error::InvalidArgument("The tensor has a wrong dim in dim" +
                                                  std::to_string(i));
            }
        }
        va_end(args);
        return base::error::Success();
    }

    base::Status Layer::check() const{
        return base::error::FunctionNotImplement("The check function is not implement yet");
    };

    void Layer::set_input(int32_t idx, const tensor::Tensor& input) {
        CHECK_GE(idx, 0);
        CHECK_LT(idx, inputs_.size());
        this->inputs_.at(idx) = input;
    }

    void Layer::set_output(int32_t idx, const tensor::Tensor& output) {
        CHECK_GE(idx, 0);
        CHECK_LT(idx, outputs_.size());
        this->outputs_.at(idx) = output;
    }

    const tensor::Tensor& Layer::get_input(int32_t idx) const {
        CHECK_GE(idx, 0);
        CHECK_LT(idx, inputs_.size());
        return inputs_.at(idx);
    }

    tensor::Tensor& Layer::get_input(int32_t idx) {
        CHECK_GE(idx, 0);
        CHECK_LT(idx, inputs_.size());
        return inputs_.at(idx);
    }

    tensor::Tensor& Layer::get_output(int32_t idx) {
        CHECK_GE(idx, 0);
        CHECK_LT(idx, outputs_.size());
        return outputs_.at(idx);
    }

    const tensor::Tensor& Layer::get_output(int32_t idx) const {
        CHECK_GE(idx, 0);
        CHECK_LT(idx, outputs_.size());
        return outputs_.at(idx);
    }

    void Layer::reset_input_size(size_t size) {
        inputs_.resize(size);
    }

    void Layer::reset_output_size(size_t size) {
        outputs_.resize(size);
    }

    size_t Layer::input_size() const {
        return inputs_.size();
    }

    size_t Layer::output_size() const {
        return outputs_.size();
    }

    LayerFp32Param::LayerFp32Param(base::DeviceType device_type, LayerType layer_type,
                               std::string layer_name)
    : Layer(device_type, layer_type, std::move(layer_name)) {
    }

    void LayerFp32Param::set_weight(int32_t idx, const tensor::Tensor& weight) {
        CHECK_GE(idx, 0);
        CHECK_LT(idx, weights_.size());
        CHECK(weight.data_type() == base::DataType::kDataTypeFp32);
        weights_.at(idx) = weight;
    }

    const tensor::Tensor& LayerFp32Param::get_weight(int32_t idx) const {
        CHECK_GE(idx, 0);
        CHECK_LT(idx, weights_.size());
        return weights_.at(idx);
    }

    void LayerFp32Param::set_weight(int32_t idx, const std::vector<int32_t>& dims,
                                    const float* weight_ptr, base::DeviceType device_type) {
        CHECK_GE(idx, 0);
        CHECK_LT(idx, weights_.size());

        size_t size =
            std::accumulate(dims.begin(), dims.end(), sizeof(float), std::multiplies<>());
        std::shared_ptr<base::Buffer> buffer =
            std::make_shared<base::Buffer>(size, nullptr, (void*)(weight_ptr), true);
        if (device_type != base::DeviceType::kDeviceUnknown) {
            buffer->set_device_type(device_type);
        }

        tensor::Tensor weight(base::DataType::kDataTypeFp32, dims);
        CHECK(weight.assign(buffer));
        weights_.at(idx) = weight;
    }

    void LayerFp32Param::reset_weight_size(size_t size) {
        weights_.resize(size);
    }

    size_t LayerFp32Param::weight_size() const {
        return weights_.size();
    }

    tensor::Tensor& LayerFp32Param::get_weight(int32_t idx) {
        CHECK_GE(idx, 0);
        CHECK_LT(idx, weights_.size());
        return weights_.at(idx);
    }


    base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& output1) {
        this->set_input(0, input1);
        this->set_output(0, output1);
        return this->forward();
    }

    base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                                const tensor::Tensor& output1) {
        this->set_input(0, input1);
        this->set_input(1, input2);

        this->set_output(0, output1);
        return this->forward();
    }

    base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                                const tensor::Tensor& input3, const tensor::Tensor& output1) {
        this->set_input(0, input1);
        this->set_input(1, input2);
        this->set_input(2, input3);

        this->set_output(0, output1);
        return this->forward();
    }

    base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                                const tensor::Tensor& input3, const tensor::Tensor& input4,
                                const tensor::Tensor& output1) {
        this->set_input(0, input1);
        this->set_input(1, input2);
        this->set_input(2, input3);
        this->set_input(3, input4);

        this->set_output(0, output1);
        return this->forward();
    }

    base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                                const tensor::Tensor& input3, const tensor::Tensor& input4,
                                const tensor::Tensor& input5, const tensor::Tensor& output1) {
        this->set_input(0, input1);
        this->set_input(1, input2);
        this->set_input(2, input3);
        this->set_input(3, input4);
        this->set_input(4, input5);

        this->set_output(0, output1);
        return this->forward();
    }

}