//
// Created by yami on 25-3-13.
//

#ifndef LAYER_H
#define LAYER_H
#include <vector>
#include <string>
#include "tensor/tensor.h"
#include "base/base.h"

namespace op {

enum class  LayerType: uint8_t {
  kLayerUnknown = 0,
  kLayerLinear = 1,
  kLayerEncode = 2,
  kLayerEmbedding = 3,
  kLayerRMSNorm = 4,
  kLayerMatmul = 5,
  kLayerRoPe = 6,
  kLayerMHA = 7,
  kLayerSoftmax = 8,
  kLayerAdd = 9,
  kLayerSwiGLU = 10,
};

class BaseLayer {
  public:
  explicit BaseLayer(base::DeviceType device_type, LayerType layer_type,
                   base::DataType data_type, std::string layer_name = "");

    base::DataType data_type() const;

    LayerType layer_type() const;

    virtual base::Status init() = 0;

    virtual base::Status forward() = 0;

    virtual base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& output1) = 0;

    virtual base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                                 const tensor::Tensor& output1) = 0;

    virtual base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                                 const tensor::Tensor& input3, const tensor::Tensor& output1) = 0;

    virtual base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                                 const tensor::Tensor& input3, const tensor::Tensor& input4,
                                 const tensor::Tensor& output1) = 0;

    virtual base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                                 const tensor::Tensor& input3, const tensor::Tensor& input4,
                                 const tensor::Tensor& input5, const tensor::Tensor& output1) = 0;

    virtual void set_input(int32_t idx, const tensor::Tensor& input) = 0;

    virtual void set_output(int32_t idx, const tensor::Tensor& output) = 0;

    virtual size_t input_size() const = 0;

    virtual size_t output_size() const = 0;

    virtual base::Status check() const = 0;

    virtual tensor::Tensor& get_input(int32_t idx) = 0;

    virtual tensor::Tensor& get_output(int32_t idx) = 0;

    virtual const tensor::Tensor& get_input(int32_t idx) const = 0;

    virtual const tensor::Tensor& get_output(int32_t idx) const = 0;

    const std::string& get_layer_name() const;

    void set_layer_name(const std::string& layer_name);

    base::DeviceType device_type() const;

    void set_device_type(base::DeviceType device_type);

protected:
    std::string layer_name_;
    LayerType layer_type_ = LayerType::kLayerUnknown;
    base::DataType data_type_ = base::DataType::kDataTypeUnknown;
    base::DeviceType device_type_ = base::DeviceType::kDeviceUnknown;
};

class Layer : public BaseLayer {
 public:
  explicit Layer(base::DeviceType device_type, LayerType layer_type,
                 std::string layer_name = "");

  base::Status init() override;

  base::Status check_tensor(const tensor::Tensor& tensor, base::DeviceType device_type,
                            base::DataType data_type) const;

  base::Status check_tensor_with_dim(const tensor::Tensor& tensor,
                                     base::DeviceType device_type,
                                     base::DataType data_type, ...) const;

  base::Status check() const override;

  virtual base::Status forward() = 0;

  virtual base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& output1) = 0;

  virtual base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                             const tensor::Tensor& output1) = 0;

  virtual base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                             const tensor::Tensor& input3, const tensor::Tensor& output1) = 0;

  virtual base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                             const tensor::Tensor& input3, const tensor::Tensor& input4,
                             const tensor::Tensor& output1) = 0;

  virtual base::Status forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                             const tensor::Tensor& input3, const tensor::Tensor& input4,
                             const tensor::Tensor& input5, const tensor::Tensor& output1) = 0;

  void set_input(int32_t idx, const tensor::Tensor& input) override;

  void set_output(int32_t idx, const tensor::Tensor& output) override;

  const tensor::Tensor& get_input(int32_t idx) const override;

  const tensor::Tensor& get_output(int32_t idx) const override;

  tensor::Tensor& get_input(int32_t idx) override;

  tensor::Tensor& get_output(int32_t idx) override;

  size_t input_size() const override;

  size_t output_size() const override;

  void reset_input_size(size_t size);

  void reset_output_size(size_t size);

 private:
  std::vector<tensor::Tensor> inputs_;
  std::vector<tensor::Tensor> outputs_;
};

class LayerFp32Param : public Layer {
public:
    explicit LayerFp32Param(base::DeviceType device_type, LayerType layer_type,
                            std::string layer_name = "");

    size_t weight_size() const;

    void reset_weight_size(size_t size);

    tensor::Tensor& get_weight(int32_t idx);

    const tensor::Tensor& get_weight(int32_t idx) const;

    void set_weight(int32_t idx, const tensor::Tensor& weight);

    void set_weight(int32_t idx, const std::vector<int32_t>& dims, const float* weight_ptr,
                    base::DeviceType device_type = base::DeviceType::kDeviceUnknown);

private:
    std::vector<tensor::Tensor> weights_;
    std::vector<tensor::Tensor> inputs_;
    std::vector<tensor::Tensor> outputs_;
};
}

#endif //LAYER_H
