// Updated on March 15, 2026
#include "op/layer.h"
#include <base/cuda_backend_runtime.h>
#include <base/cuda_config.h>
#include <glog/logging.h>
#include <cstdarg>
#include <numeric>
#include <limits>
#include <utility>

namespace op {
namespace {

base::Status RebindExternalTensor(tensor::Tensor* tensor, const void* host_base,
                                  void* device_base, uint64_t allocation_bytes,
                                  base::DataType dtype, uint64_t* views,
                                  uint64_t* logical_bytes) {
  if (!tensor || tensor->is_empty() || !host_base || !device_base ||
      allocation_bytes == 0 || tensor->data_type() != dtype) {
    return base::error::InvalidArgument("invalid shared weight tensor binding");
  }
  const uintptr_t host = reinterpret_cast<uintptr_t>(host_base);
  const uintptr_t pointer = reinterpret_cast<uintptr_t>(tensor->get_buffer()->ptr());
  if (pointer < host) return base::error::InvalidArgument("shared weight offset underflow");
  const uint64_t offset = pointer - host;
  const uint64_t bytes = tensor->byte_size();
  if (offset > allocation_bytes || bytes > allocation_bytes - offset) {
    return base::error::InvalidArgument("shared weight tensor exceeds allocation");
  }
  auto* mapped = static_cast<uint8_t*>(device_base) + offset;
  tensor::Tensor replacement(dtype, tensor->dims(), false, nullptr, mapped);
  replacement.set_device_type(base::DeviceType::kDeviceCUDA);
  *tensor = std::move(replacement);
  if (views) ++*views;
  if (logical_bytes) *logical_bytes += bytes;
  return base::error::Success();
}

}  // namespace
BaseLayer::BaseLayer(base::DeviceType device_type, LayerType layer_type, base::DataType data_type,
                     std::string layer_name)
    : device_type_(device_type),
      layer_type_(layer_type),
      data_type_(data_type),
      layer_name_(std::move(layer_name)) {}

base::DataType BaseLayer::data_type() const { return data_type_; }

LayerType BaseLayer::layer_type() const { return layer_type_; }

base::Status BaseLayer::set_weight(int32_t idx, const tensor::Tensor& weight) {
  return base::error::FunctionNotImplement();
}

base::Status BaseLayer::set_weight(int32_t idx, const std::vector<int32_t>& dims,
                                   const void* weight_ptr, base::DeviceType device_type,
                                   base::DataType data_type) {
  return base::error::FunctionNotImplement();
}

const std::string& BaseLayer::get_layer_name() const { return layer_name_; }

void BaseLayer::set_layer_name(const std::string& layer_name) { layer_name_ = layer_name; }
base::DeviceType BaseLayer::device_type() const { return device_type_; }

void BaseLayer::set_device_type(base::DeviceType device_type) { device_type_ = device_type; }

void BaseLayer::set_data_type(base::DataType data_type) { data_type_ = data_type; }

Layer::Layer(base::DeviceType device_type, LayerType layer_type, std::string layer_name)
    : BaseLayer(device_type, layer_type, base::DataType::kDataTypeFp32, std::move(layer_name)) {}

base::Status Layer::init() { return base::error::Success(); }

base::Status Layer::forward() { return base::error::FunctionNotImplement(""); }

base::Status Layer::check_tensor(const tensor::Tensor& tensor, base::DeviceType device_type,
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
                                          base::DeviceType device_type, base::DataType data_type,
                                          ...) const {
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
      return base::error::InvalidArgument("The tensor has a wrong dim in dim" + std::to_string(i));
    }
  }
  va_end(args);
  return base::error::Success();
}

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

base::Status Layer::check() const {
  return base::error::FunctionNotImplement("The check function is not implement yet");
}

const tensor::Tensor& Layer::get_output(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, outputs_.size());
  return outputs_.at(idx);
}

void Layer::reset_input_size(size_t size) { inputs_.resize(size); }

void Layer::reset_output_size(size_t size) { outputs_.resize(size); }

void* Layer::compute_queue() const {
  if (device_context_ != nullptr && device_context_->compute_queue != nullptr) {
    return device_context_->compute_queue;
  }
  return cuda_config_ ? cuda_config_->stream : nullptr;
}

kernel::CudaConfig* Layer::cuda_config_or_null() const {
  if (cuda_config_ != nullptr) {
    return cuda_config_.get();
  }
  if (device_context_ != nullptr && device_context_->backend == base::BackendType::kCUDA) {
    auto config = base::cuda_config_from_device_context(device_context_);
    return config.get();
  }
  return nullptr;
}

void Layer::materialize() {
  const void* queue = compute_queue();
  for (auto& input : inputs_) {
    if (!input.is_empty()) {
      input.to_device(device_type_, const_cast<void*>(queue));
    }
  }
  for (auto& output : outputs_) {
    if (!output.is_empty()) {
      output.to_device(device_type_, const_cast<void*>(queue));
    }
  }
}

void Layer::to_cuda() {
  materialize();
}

void Layer::set_cuda_config(std::shared_ptr<kernel::CudaConfig> config) {
  if (!config) {
    return;
  }
  this->cuda_config_ = config;
  if (device_context_ == nullptr) {
    device_context_ = std::make_shared<base::DeviceContext>();
  }
  device_context_->backend = base::BackendType::kCUDA;
  device_context_->compute_queue = config->stream;
  device_context_->transfer_queue = config->stream;
  device_context_->blas_handle = config->cublas_handle;
  device_context_->blaslt_handle = config->cublas_lt_handle;
}

std::shared_ptr<kernel::CudaConfig> Layer::cuda_config() const { return cuda_config_; }

void Layer::set_device_context(std::shared_ptr<base::DeviceContext> context) {
  if (!context) {
    return;
  }
  device_context_ = std::move(context);
  if (device_context_->backend == base::BackendType::kCUDA) {
    cuda_config_ = base::cuda_config_from_device_context(device_context_);
  }
}

std::shared_ptr<base::DeviceContext> Layer::device_context() const {
  return device_context_;
}

size_t Layer::input_size() const { return inputs_.size(); }

size_t Layer::output_size() const { return outputs_.size(); }

LayerParam::LayerParam(base::DeviceType device_type, LayerType layer_type, bool is_quant_layer,
                       std::string layer_name)
    : Layer(device_type, layer_type, std::move(layer_name)), is_quant_layer_(is_quant_layer) {}

base::Status LayerParam::set_weight(int32_t idx, const tensor::Tensor& weight) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, weights_.size());
  CHECK(weight.data_type() == base::DataType::kDataTypeFp32 ||
        weight.data_type() == base::DataType::kDataTypeBf16);
  if (!weight.is_empty()) {
    CHECK(weight.device_type() == device_type_);
  }
  weights_.at(idx) = weight;
  return base::error::Success();
}

const tensor::Tensor& LayerParam::get_weight(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, weights_.size());
  return weights_.at(idx);
}

void LayerParam::materialize() {
  Layer::materialize();
  const void* queue = compute_queue();
  for (auto& weight : weights_) {
    if (!is_quant_layer_) {
      weight.to_device(device_type_, const_cast<void*>(queue), data_type_);
    } else {
      weight.to_device(device_type_, const_cast<void*>(queue));
    }
  }
  if (!scales_.is_empty()) {
    scales_.to_device(device_type_, const_cast<void*>(queue));
  }
}

void LayerParam::to_cuda() {
  materialize();
}

base::Status LayerParam::set_weight(int32_t idx, const std::vector<int32_t>& dims,
                                    const void* weight_ptr, base::DeviceType device_type,
                                    base::DataType data_type) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, weights_.size());
  CHECK_NE(weight_ptr, nullptr);

  size_t size =
      std::accumulate(dims.begin(), dims.end(), base::DataTypeSize(data_type), std::multiplies<>());
  std::shared_ptr<base::Buffer> buffer =
      std::make_shared<base::Buffer>(size, nullptr, const_cast<void*>(weight_ptr), true);
  if (device_type != base::DeviceType::kDeviceUnknown) {
    buffer->set_device_type(device_type);
  }

  if (!is_quant_layer_) {
    tensor::Tensor weight(data_type, dims);
    weight.set_device_type(device_type);
    CHECK(weight.assign(buffer));
    weights_.at(idx) = weight;
  } else {
    // is quant layer
    tensor::Tensor weight(base::DataType::kDataTypeInt8, dims);
    weight.set_device_type(device_type);
    CHECK(weight.assign(buffer));
    weights_.at(idx) = weight;

    const int32_t weight_size = static_cast<int32_t>(weight.size());
    CHECK(weight_size % group_size_ == 0);

    int32_t scale_nums = weight_size / group_size_;
    scales_ = tensor::Tensor{base::DataType::kDataTypeFp32, scale_nums, false, nullptr,
                             reinterpret_cast<float*>((int8_t*)weight_ptr + weight_size)};
    scales_.set_device_type(device_type);
  }

  return base::error::Success();
}

void LayerParam::set_scales(const tensor::Tensor& scales) {
  CHECK(!scales.is_empty());
  this->scales_ = scales;
}

void LayerParam::set_group_size(int32_t group_size) { this->group_size_ = group_size; }

int32_t LayerParam::get_scale_num() const {
  CHECK(!scales_.is_empty());
  return static_cast<int32_t>(scales_.size());
}

base::Status LayerParam::bind_external_weights(const void* host_base,
                                                void* device_base,
                                                uint64_t allocation_bytes,
                                                base::DataType dtype,
                                                uint64_t* views,
                                                uint64_t* logical_bytes) {
  if (is_quant_layer_) {
    return base::error::InvalidArgument("quantized shared weights are outside E4 scope");
  }
  for (auto& weight : weights_) {
    auto status = RebindExternalTensor(&weight, host_base, device_base,
                                       allocation_bytes, dtype, views,
                                       logical_bytes);
    if (!status) return status;
  }
  return base::error::Success();
}

void LayerParam::reset_weight_size(size_t size) { weights_.resize(size); }

size_t LayerParam::weight_size() const { return weights_.size(); }

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

tensor::Tensor& LayerParam::get_weight(int32_t idx) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, weights_.size());
  return weights_.at(idx);
}

}  // namespace op
