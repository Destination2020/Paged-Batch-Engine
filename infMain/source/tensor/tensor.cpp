// Updated on March 15, 2026
#include "tensor/tensor.h"
#include <base/bf16.h>
#include <cuda_device_runtime_api.h>
#include <glog/logging.h>
#include <numeric>
#include <vector>

namespace tensor {
template <typename T, typename Tp>
static size_t reduce_dimension(T begin, T end, Tp init) {
  if (begin >= end) {
    return 0;
  }
  size_t size = std::accumulate(begin, end, init, std::multiplies<>());
  return size;
}

static size_t data_type_size(base::DataType data_type) {
  switch (data_type) {
    case base::DataType::kDataTypeFp32: {
      return 4;
    }
    case base::DataType::kDataTypeInt8: {
      return 1;
    }
    case base::DataType::kDataTypeInt32: {
      return 4;
    }
    case base::DataType::kDataTypeBf16: {
      return 2;
    }
    default: {
      LOG(FATAL) << "Unknown data type size for " << int(data_type);
      return 0;
    }
  }
}

static std::vector<uint16_t> fp32_to_bf16_vector(const float* src, size_t size) {
  std::vector<uint16_t> dst(size);
  for (size_t i = 0; i < size; ++i) {
    dst[i] = base::float_to_bf16_bits(src[i]);
  }
  return dst;
}

static std::vector<float> bf16_to_fp32_vector(const uint16_t* src, size_t size) {
  std::vector<float> dst(size);
  for (size_t i = 0; i < size; ++i) {
    dst[i] = base::bf16_bits_to_float(src[i]);
  }
  return dst;
}

Tensor::Tensor(base::DataType data_type, int32_t dim0, bool need_alloc,
               std::shared_ptr<base::DeviceAllocator> alloc, void* ptr)
    : data_type_(data_type) {
  dims_.push_back(dim0);
  size_ = dim0;
  if (need_alloc && alloc) {
    allocate(alloc);
  } else {
    if (ptr != nullptr) {
      CHECK(need_alloc == false)
          << "The need_alloc is is true when ptr parameter is not a null pointer.";
      init_buffer(alloc, data_type_, need_alloc, ptr);
    }
  }
}

Tensor::Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, bool need_alloc,
               std::shared_ptr<base::DeviceAllocator> alloc, void* ptr)
    : data_type_(data_type) {
  dims_.push_back(dim0);
  dims_.push_back(dim1);
  size_ = dim0 * dim1;
  if (need_alloc && alloc) {
    allocate(alloc);
  } else {
    init_buffer(alloc, data_type_, need_alloc, ptr);
  }
}

Tensor::Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2, bool need_alloc,
               std::shared_ptr<base::DeviceAllocator> alloc, void* ptr)
    : data_type_(data_type) {
  dims_.push_back(dim0);
  dims_.push_back(dim1);
  dims_.push_back(dim2);
  size_ = dim0 * dim1 * dim2;
  if (need_alloc && alloc) {
    allocate(alloc);
  } else {
    init_buffer(alloc, data_type_, need_alloc, ptr);
  }
}

Tensor::Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2, int32_t dim3,
               bool need_alloc, std::shared_ptr<base::DeviceAllocator> alloc, void* ptr)
    : data_type_(data_type) {
  dims_.push_back(dim0);
  dims_.push_back(dim1);
  dims_.push_back(dim2);
  dims_.push_back(dim3);
  size_ = dim0 * dim1 * dim2 * dim3;
  if (need_alloc && alloc) {
    allocate(alloc);
  } else {
    init_buffer(alloc, data_type_, need_alloc, ptr);
  }
}

Tensor::Tensor(base::DataType data_type, std::vector<int32_t> dims, bool need_alloc,
               std::shared_ptr<base::DeviceAllocator> alloc, void* ptr)
    : dims_(std::move(dims)), data_type_(data_type) {
  size_ = reduce_dimension(dims_.begin(), dims_.end(), 1);
  if (need_alloc && alloc) {
    allocate(alloc);
  } else {
    init_buffer(alloc, data_type_, need_alloc, ptr);
  }
}

void Tensor::to_device(base::DeviceType target_device_type,
                       void* queue,
                       base::DataType target_data_type) {
  CHECK_NE(buffer_, nullptr);
  if (target_data_type == base::DataType::kDataTypeUnknown) {
    target_data_type = data_type_;
  }

  const base::DeviceType current_device_type = this->device_type();
  if (current_device_type == base::DeviceType::kDeviceUnknown) {
    LOG(ERROR) << "The device type of the tensor is unknown.";
    return;
  }

  if (target_device_type == base::DeviceType::kDeviceCPU) {
    this->to_host();
    CHECK_EQ(this->data_type_, target_data_type)
        << "Tensor::to_device does not support host-side dtype conversion.";
    return;
  }

  CHECK_EQ(target_device_type, base::DeviceType::kDeviceCUDA)
      << "Tensor::to_device currently only supports CPU/CUDA backends.";

  if (current_device_type == base::DeviceType::kDeviceCPU) {
    auto cu_alloc = base::CUDADeviceAllocatorFactory::get_instance();
    auto cu_buffer = std::make_shared<base::Buffer>(size_ * base::DataTypeSize(target_data_type), cu_alloc);

    if (data_type_ == target_data_type) {
      size_t byte_size = this->byte_size();
      cu_alloc->memcpy(buffer_->ptr(), cu_buffer->ptr(), byte_size,
                       base::MemcpyKind::kMemcpyCPU2CUDA, queue);
    } else if (data_type_ == base::DataType::kDataTypeFp32 &&
               target_data_type == base::DataType::kDataTypeBf16) {
      auto host_bf16 = fp32_to_bf16_vector(reinterpret_cast<const float*>(buffer_->ptr()), size_);
      cu_alloc->memcpy(host_bf16.data(), cu_buffer->ptr(), host_bf16.size() * sizeof(uint16_t),
                       base::MemcpyKind::kMemcpyCPU2CUDA, queue, true);
    } else if (data_type_ == base::DataType::kDataTypeBf16 &&
               target_data_type == base::DataType::kDataTypeFp32) {
      auto host_fp32 = bf16_to_fp32_vector(reinterpret_cast<const uint16_t*>(buffer_->ptr()), size_);
      cu_alloc->memcpy(host_fp32.data(), cu_buffer->ptr(), host_fp32.size() * sizeof(float),
                       base::MemcpyKind::kMemcpyCPU2CUDA, queue, true);
    } else {
      LOG(FATAL) << "Unsupported tensor dtype conversion from " << data_type_ << " to "
                 << target_data_type << " in Tensor::to_device";
    }
    this->buffer_ = cu_buffer;
    this->data_type_ = target_data_type;
  } else {
    CHECK_EQ(data_type_, target_data_type)
        << "Tensor::to_device does not support in-place CUDA dtype conversion.";
  }
}

void Tensor::to_host() {
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
    LOG(INFO) << "The device type of the tensor is already cpu.";
  }
}

void Tensor::to_cuda(void* queue, base::DataType target_data_type) {
  to_device(base::DeviceType::kDeviceCUDA, queue, target_data_type);
}

void Tensor::to_cpu() {
  to_host();
}

size_t Tensor::size() const { return this->size_; }

int32_t Tensor::get_dim(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, this->dims_.size());
  return this->dims_.at(idx);
}

base::DeviceType Tensor::device_type() const {
  if (!buffer_) {
    return base::DeviceType::kDeviceUnknown;
  }
  return buffer_->device_type();
}

bool Tensor::assign(std::shared_ptr<base::Buffer> buffer) {
  if (!buffer) {
    LOG(ERROR) << "The buffer parameter in the assign function is null pointer!";
    return false;
  }
  if (buffer_) {
    if (buffer_->device_type() != buffer->device_type()) {
      LOG(ERROR) << "The device type of the new buffer is different from the original one.";
    }
  }

  size_t byte_size = this->byte_size();
  if (byte_size > buffer->byte_size()) {
    LOG(ERROR) << "The size of buffer is too small for the tensor!";
    return false;
  }
  buffer_ = buffer;
  return true;
}

bool Tensor::allocate(std::shared_ptr<base::DeviceAllocator> allocator, bool need_realloc) {
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

const std::vector<int32_t>& Tensor::dims() const { return this->dims_; }

void Tensor::set_device_type(base::DeviceType device_type) const {
  if (buffer_) {
    buffer_->set_device_type(device_type);
  }
}

void Tensor::reset(base::DataType data_type, const std::vector<int32_t>& dims) {
  this->data_type_ = data_type;
  this->dims_ = dims;
  this->size_ = reduce_dimension(dims.begin(), dims.end(), 1);
  this->buffer_ = nullptr;
}

int32_t Tensor::dims_size() const { return static_cast<int32_t>(dims_.size()); }

base::DataType Tensor::data_type() const { return data_type_; }

void Tensor::reshape(const std::vector<int32_t>& dims) {
  size_t size = reduce_dimension(dims.begin(), dims.end(), 1);
  if (!buffer_) {
    this->dims_ = dims;
    this->size_ = size;
    return;
  }

  const size_t requested_bytes = size * base::DataTypeSize(this->data_type_);
  const size_t buffer_bytes = buffer_ ? buffer_->byte_size() : 0;
  if (requested_bytes > buffer_bytes) {
    auto new_buffer = std::make_shared<base::Buffer>(size * base::DataTypeSize(this->data_type_),
                                                     buffer_->allocator());
    CHECK(new_buffer->allocate());
    new_buffer->copy_from(buffer_.get());
    this->buffer_ = new_buffer;
  }
  this->dims_ = dims;
  this->size_ = size;
}

void Tensor::reshape_no_realloc(const std::vector<int32_t>& dims) {
  size_t size = reduce_dimension(dims.begin(), dims.end(), 1);
  CHECK(buffer_ != nullptr && buffer_->ptr() != nullptr)
      << "Tensor::reshape_no_realloc requires existing allocated storage.";
  CHECK_NE(this->data_type_, base::DataType::kDataTypeUnknown)
      << "Tensor::reshape_no_realloc requires a known data type.";

  const size_t requested_bytes = size * base::DataTypeSize(this->data_type_);
  const size_t buffer_bytes = buffer_->byte_size();
  CHECK_LE(requested_bytes, buffer_bytes)
      << "Tensor::reshape_no_realloc would exceed preallocated storage: requested_bytes="
      << requested_bytes << ", buffer_bytes=" << buffer_bytes;

  this->dims_ = dims;
  this->size_ = size;
}

std::shared_ptr<base::Buffer> Tensor::get_buffer() const { return buffer_; }

Tensor Tensor::clone() const {
  Tensor new_tensor = *this;
  size_t byte_size = this->byte_size();

  auto allocator = buffer_->allocator();
  new_tensor.buffer_ = std::make_shared<base::Buffer>(byte_size, allocator);
  new_tensor.buffer_->copy_from(buffer_.get());
  return new_tensor;
}

size_t Tensor::byte_size() const { return this->size() * DataTypeSize(data_type_); }

std::vector<size_t> Tensor::strides() const {
  std::vector<size_t> strides;
  if (!dims_.empty()) {
    for (int32_t i = 0; i < dims_.size() - 1; ++i) {
      size_t stride = reduce_dimension(dims_.begin() + i + 1, dims_.end(), 1);
      strides.push_back(stride);
    }
    strides.push_back(1);
  }
  return strides;
}

bool Tensor::is_empty() const {
  return size_ == 0 || buffer_ == nullptr || buffer_->ptr() == nullptr;
}

void Tensor::init_buffer(std::shared_ptr<base::DeviceAllocator> alloc, base::DataType data_type,
                         bool need_alloc, void* ptr) {
  if (!alloc && !need_alloc) {
    std::shared_ptr<base::Buffer> buffer =
        std::make_shared<base::Buffer>(data_type_size(data_type) * size_, nullptr, ptr, true);
    this->buffer_ = buffer;
  } else {
    allocate(alloc, true);
  }
}
}  // namespace tensor
