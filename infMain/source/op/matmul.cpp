// Updated on March 15, 2026
#include "op/matmul.h"
#include "kernels/cpu/matmul_kernel.h"
#include "kernels/kernels_interface.h"
namespace op {
MatmulLayer::MatmulLayer(base::DeviceType device_type, int32_t dim0, int32_t dim1,
                         bool is_quant_layer, bool has_bias)
    : LayerParam(device_type, LayerType::kLayerMatmul, is_quant_layer, "Matmul"),
      dim0_(dim0),
      dim1_(dim1),
      has_bias_(has_bias) {
  reset_input_size(1);
  reset_output_size(1);
  reset_weight_size(1);
  if (has_bias_) {
    bias_.resize(1);
  }
}

base::Status MatmulLayer::check() const {
  auto status = check_tensor_with_dim(get_input(0), device_type_, data_type_, dim1_);
  if (!status) {
    LOG(ERROR) << "The input tensor error in the matmul layer.";
    return status;
  }

  if (!is_quant_layer_) {
    status = check_tensor_with_dim(get_weight(0), device_type_, data_type_, dim0_, dim1_);
    if (!status) {
      LOG(ERROR) << "The weight tensor error in the matmul layer.";
      return status;
    }
  } else {
    status = check_tensor_with_dim(get_weight(0), device_type_, base::DataType::kDataTypeInt8,
                                   dim0_, dim1_);
    if (!status) {
      LOG(ERROR) << "The weight tensor error in the matmul layer.";
      return status;
    }
  }

  if (is_quant_layer_) {
    status = check_tensor_with_dim(scales_, device_type_, base::DataType::kDataTypeFp32, scales_.size());
    if (!status) {
      LOG(ERROR) << "The scale tensor error in the matmul layer.";
      return status;
    }
  }

  status = check_tensor_with_dim(get_output(0), device_type_, data_type_, dim0_);
  if (!status) {
    LOG(ERROR) << "The output tensor error in the matmul layer.";
    return status;
  }
  return base::error::Success();
}

base::Status MatmulLayer::forward() {
  auto status = check();
  if (!status) {
    return status;
  }
  const base::DeviceContext* context = device_context_.get();
  void* queue = compute_queue();
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    CHECK(cuda_config_or_null() != nullptr);
  }
  if (is_quant_layer_) {
    kernel::get_matmul_kernel_quant8(device_type_)(get_input(0), get_weight(0), get_output(0),
                                                   group_size_, scales_, context);
  } else {
    kernel::get_matmul_kernel(device_type_)(get_input(0), get_weight(0), get_output(0), 1.f,
                                            context);
  }

  if (has_bias_) {
    kernel::get_add_kernel(device_type_)(get_output(0), get_bias(0), get_output(0), queue);
  }

  return base::error::Success();
}

base::Status MatmulLayer::set_bias(int32_t idx, int32_t& dim, const void* bias_ptr,
                                   base::DeviceType device_type, base::DataType data_type) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, bias_.size());
  CHECK_NE(bias_ptr, nullptr);

  size_t size = dim * base::DataTypeSize(data_type);
  std::shared_ptr<base::Buffer> buffer =
      std::make_shared<base::Buffer>(size, nullptr, const_cast<void*>(bias_ptr), true);
  if (device_type != base::DeviceType::kDeviceUnknown) {
    buffer->set_device_type(device_type);
  }

  if (!is_quant_layer_) {
    tensor::Tensor bias(data_type, dim);
    bias.set_device_type(device_type);
    CHECK(bias.assign(buffer));
    // LOG(INFO) << "bias:" << bias.index<float>(0);
    bias_.at(idx) = bias;
  } else {
    // is quant layer
    tensor::Tensor bias(base::DataType::kDataTypeInt8, dim);
    bias.set_device_type(device_type);
    CHECK(bias.assign(buffer));
    bias_.at(idx) = bias;

    const int32_t bias_size = static_cast<int32_t>(bias.size());
    CHECK(bias_size % group_size_ == 0);

    int32_t scale_nums = bias_size / group_size_;
    scales_ = tensor::Tensor{base::DataType::kDataTypeFp32, scale_nums, false, nullptr,
                             reinterpret_cast<float*>((int8_t*)bias_ptr + bias_size)};
    scales_.set_device_type(device_type);
  }

  return base::error::Success();
}

tensor::Tensor& MatmulLayer::get_bias(int32_t idx) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, bias_.size());
  return bias_.at(idx);
}

const tensor::Tensor& MatmulLayer::get_bias(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, bias_.size());
  return bias_.at(idx);
}

void MatmulLayer::materialize() {
  LayerParam::materialize();
  const void* queue = compute_queue();
  if (has_bias_) {
    for (auto& bias : bias_) {
      if (!is_quant_layer_) {
        bias.to_device(device_type_, const_cast<void*>(queue), data_type_);
      } else {
        bias.to_device(device_type_, const_cast<void*>(queue));
      }
    }
  }
}

void MatmulLayer::to_cuda() {
  materialize();
}

base::Status MatmulLayer::bind_external_weights(const void* host_base,
                                                void* device_base,
                                                uint64_t allocation_bytes,
                                                base::DataType dtype,
                                                uint64_t* views,
                                                uint64_t* logical_bytes) {
  auto status = LayerParam::bind_external_weights(host_base, device_base,
                                                  allocation_bytes, dtype,
                                                  views, logical_bytes);
  if (!status) return status;
  const uintptr_t host = reinterpret_cast<uintptr_t>(host_base);
  for (auto& bias : bias_) {
    if (bias.is_empty() || bias.data_type() != dtype)
      return base::error::InvalidArgument("invalid shared weight bias binding");
    const uintptr_t pointer = reinterpret_cast<uintptr_t>(bias.get_buffer()->ptr());
    if (pointer < host) return base::error::InvalidArgument("shared bias offset underflow");
    const uint64_t offset = pointer - host;
    const uint64_t bytes = bias.byte_size();
    if (offset > allocation_bytes || bytes > allocation_bytes - offset)
      return base::error::InvalidArgument("shared bias exceeds allocation");
    tensor::Tensor replacement(dtype, bias.dims(), false, nullptr,
                               static_cast<uint8_t*>(device_base) + offset);
    replacement.set_device_type(base::DeviceType::kDeviceCUDA);
    bias = std::move(replacement);
    if (views) ++*views;
    if (logical_bytes) *logical_bytes += bytes;
  }
  return base::error::Success();
}

}  // namespace op
