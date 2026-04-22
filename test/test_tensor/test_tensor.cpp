// Updated on March 15, 2026
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <tensor/tensor.h>
#include "../utils.cuh"
#include "base/buffer.h"

TEST(test_tensor, to_cpu) {
  using namespace base;
  auto alloc_cu = CUDADeviceAllocatorFactory::get_instance();
  tensor::Tensor t1_cu(DataType::kDataTypeFp32, 32, 32, true, alloc_cu);
  ASSERT_EQ(t1_cu.is_empty(), false);
  set_value_cu(t1_cu.ptr<float>(), 32 * 32);

  t1_cu.to_cpu();
  ASSERT_EQ(t1_cu.device_type(), base::DeviceType::kDeviceCPU);
  float* cpu_ptr = t1_cu.ptr<float>();
  for (int i = 0; i < 32 * 32; ++i) {
    ASSERT_EQ(*(cpu_ptr + i), 1.f);
  }
}

TEST(test_tensor, clone_cuda) {
  using namespace base;
  auto alloc_cu = CUDADeviceAllocatorFactory::get_instance();
  tensor::Tensor t1_cu(DataType::kDataTypeFp32, 32, 32, true, alloc_cu);
  ASSERT_EQ(t1_cu.is_empty(), false);
  set_value_cu(t1_cu.ptr<float>(), 32 * 32, 1.f);

  tensor::Tensor t2_cu = t1_cu.clone();
  float* p2 = new float[32 * 32];
  cudaMemcpy(p2, t2_cu.ptr<float>(), sizeof(float) * 32 * 32, cudaMemcpyDeviceToHost);
  for (int i = 0; i < 32 * 32; ++i) {
    ASSERT_EQ(p2[i], 1.f);
  }

  cudaMemcpy(p2, t1_cu.ptr<float>(), sizeof(float) * 32 * 32, cudaMemcpyDeviceToHost);
  for (int i = 0; i < 32 * 32; ++i) {
    ASSERT_EQ(p2[i], 1.f);
  }

  ASSERT_EQ(t2_cu.data_type(), base::DataType::kDataTypeFp32);
  ASSERT_EQ(t2_cu.size(), 32 * 32);

  t2_cu.to_cpu();
  std::memcpy(p2, t2_cu.ptr<float>(), sizeof(float) * 32 * 32);
  for (int i = 0; i < 32 * 32; ++i) {
    ASSERT_EQ(p2[i], 1.f);
  }
  delete[] p2;
}

TEST(test_tensor, clone_cpu) {
  using namespace base;
  auto alloc_cpu = CPUDeviceAllocatorFactory::get_instance();
  tensor::Tensor t1_cpu(DataType::kDataTypeFp32, 32, 32, true, alloc_cpu);
  ASSERT_EQ(t1_cpu.is_empty(), false);
  for (int i = 0; i < 32 * 32; ++i) {
    t1_cpu.index<float>(i) = 1.f;
  }

  tensor::Tensor t2_cpu = t1_cpu.clone();
  float* p2 = new float[32 * 32];
  std::memcpy(p2, t2_cpu.ptr<float>(), sizeof(float) * 32 * 32);
  for (int i = 0; i < 32 * 32; ++i) {
    ASSERT_EQ(p2[i], 1.f);
  }

  std::memcpy(p2, t1_cpu.ptr<float>(), sizeof(float) * 32 * 32);
  for (int i = 0; i < 32 * 32; ++i) {
    ASSERT_EQ(p2[i], 1.f);
  }
  delete[] p2;
}

TEST(test_tensor, to_cu) {
  using namespace base;
  auto alloc_cpu = CPUDeviceAllocatorFactory::get_instance();
  tensor::Tensor t1_cpu(DataType::kDataTypeFp32, 32, 32, true, alloc_cpu);
  ASSERT_EQ(t1_cpu.is_empty(), false);
  float* p1 = t1_cpu.ptr<float>();
  for (int i = 0; i < 32 * 32; ++i) {
    *(p1 + i) = 1.f;
  }

  t1_cpu.to_cuda();
  float* p2 = new float[32 * 32];
  cudaMemcpy(p2, t1_cpu.ptr<float>(), sizeof(float) * 32 * 32, cudaMemcpyDeviceToHost);
  for (int i = 0; i < 32 * 32; ++i) {
    ASSERT_EQ(*(p2 + i), 1.f);
  }
  delete[] p2;
}

TEST(test_tensor, init1) {
  using namespace base;
  auto alloc_cu = base::CPUDeviceAllocatorFactory::get_instance();

  int32_t size = 32 * 151;

  tensor::Tensor t1(base::DataType::kDataTypeFp32, size, true, alloc_cu);
  ASSERT_EQ(t1.is_empty(), false);
}

TEST(test_tensor, init3) {
  using namespace base;
  float* ptr = new float[32];
  ptr[0] = 31;
  tensor::Tensor t1(base::DataType::kDataTypeFp32, 32, false, nullptr, ptr);
  ASSERT_EQ(t1.is_empty(), false);
  ASSERT_EQ(t1.ptr<float>(), ptr);
  ASSERT_EQ(*t1.ptr<float>(), 31);
}

TEST(test_tensor, init2) {
  using namespace base;
  auto alloc_cu = base::CPUDeviceAllocatorFactory::get_instance();

  int32_t size = 32 * 151;

  tensor::Tensor t1(base::DataType::kDataTypeFp32, size, false, alloc_cu);
  ASSERT_EQ(t1.is_empty(), true);
}

TEST(test_tensor, reshape_no_realloc_within_capacity) {
  using namespace base;
  auto alloc_cpu = CPUDeviceAllocatorFactory::get_instance();
  tensor::Tensor workspace(DataType::kDataTypeFp32, 64, true, alloc_cpu);
  ASSERT_EQ(workspace.is_empty(), false);

  void* original_ptr = workspace.ptr<float>();
  workspace.reshape_no_realloc({8, 8});

  ASSERT_EQ(workspace.dims_size(), 2);
  ASSERT_EQ(workspace.get_dim(0), 8);
  ASSERT_EQ(workspace.get_dim(1), 8);
  ASSERT_EQ(workspace.size(), 64);
  ASSERT_EQ(workspace.ptr<float>(), original_ptr);
}

TEST(test_tensor, reshape_no_realloc_exceeds_capacity) {
  using namespace base;
  auto alloc_cpu = CPUDeviceAllocatorFactory::get_instance();
  tensor::Tensor workspace(DataType::kDataTypeFp32, 64, true, alloc_cpu);
  ASSERT_EQ(workspace.is_empty(), false);

  EXPECT_DEATH(workspace.reshape_no_realloc({65}),
               "Tensor::reshape_no_realloc would exceed preallocated storage");
}

TEST(test_tensor, assign1) {
  using namespace base;
  auto alloc_cpu = CPUDeviceAllocatorFactory::get_instance();
  tensor::Tensor t1_cpu(DataType::kDataTypeFp32, 32, 32, true, alloc_cpu);
  ASSERT_EQ(t1_cpu.is_empty(), false);

  int32_t size = 32 * 32;
  float* ptr = new float[size];
  for (int i = 0; i < size; ++i) {
    ptr[i] = float(i);
  }
  std::shared_ptr<Buffer> buffer =
      std::make_shared<Buffer>(size * sizeof(float), nullptr, ptr, true);
  buffer->set_device_type(DeviceType::kDeviceCPU);

  ASSERT_EQ(t1_cpu.assign(buffer), true);
  ASSERT_EQ(t1_cpu.is_empty(), false);
  ASSERT_NE(t1_cpu.ptr<float>(), nullptr);
  delete[] ptr;
}

// 测试 ptr(int64_t index) 函数 - CPU版本
TEST(test_tensor, ptr_with_index_cpu) {
  using namespace base;
  auto alloc_cpu = CPUDeviceAllocatorFactory::get_instance();
  int32_t size = 100;
  
  tensor::Tensor t1(DataType::kDataTypeFp32, size, true, alloc_cpu);
  ASSERT_EQ(t1.is_empty(), false);
  
  // 测试非const版本的 ptr(int64_t index) - 用于写入
  for (int64_t i = 0; i < size; ++i) {
    float* ptr_at_i = t1.ptr<float>(i);
    ASSERT_NE(ptr_at_i, nullptr);
    *ptr_at_i = float(i * 2.5f);
  }
  
  // 验证数据正确写入 - 使用 const 版本的 ptr(int64_t index)
  for (int64_t i = 0; i < size; ++i) {
    const float* ptr_at_i = t1.ptr<float>(i);
    ASSERT_NE(ptr_at_i, nullptr);
    ASSERT_FLOAT_EQ(*ptr_at_i, float(i * 2.5f));
  }
  
  // 验证 ptr(int64_t index) 与 ptr() + index 等价
  float* base_ptr = t1.ptr<float>();
  for (int64_t i = 0; i < size; ++i) {
    float* ptr_at_i = t1.ptr<float>(i);
    ASSERT_EQ(ptr_at_i, base_ptr + i);
    ASSERT_FLOAT_EQ(*ptr_at_i, *(base_ptr + i));
  }
  
  // 测试边界情况：index = 0 应该等于 ptr()
  ASSERT_EQ(t1.ptr<float>(0), t1.ptr<float>());
  
  // 测试最后一个元素
  int64_t last_idx = size - 1;
  float* last_ptr = t1.ptr<float>(last_idx);
  ASSERT_FLOAT_EQ(*last_ptr, float(last_idx * 2.5f));
}

// 测试 ptr(int64_t index) 函数 - const版本的正确性
TEST(test_tensor, ptr_with_index_const) {
  using namespace base;
  auto alloc_cpu = CPUDeviceAllocatorFactory::get_instance();
  int32_t size = 50;
  
  tensor::Tensor t1(DataType::kDataTypeFp32, size, true, alloc_cpu);
  ASSERT_EQ(t1.is_empty(), false);
  
  // 先填充数据
  float* base_ptr = t1.ptr<float>();
  for (int i = 0; i < size; ++i) {
    base_ptr[i] = float(i * 3.0f);
  }
  
  // 使用 const 引用测试 const 版本的 ptr(int64_t index)
  const tensor::Tensor& t1_const = t1;
  for (int64_t i = 0; i < size; ++i) {
    const float* ptr_at_i = t1_const.ptr<float>(i);
    ASSERT_NE(ptr_at_i, nullptr);
    ASSERT_FLOAT_EQ(*ptr_at_i, float(i * 3.0f));
    
    // 验证 const 指针确实指向正确的位置
    ASSERT_EQ(ptr_at_i, base_ptr + i);
  }
  
  // 验证 const 版本的 ptr(int64_t index) 与 const ptr() + index 等价
  const float* const_base_ptr = t1_const.ptr<float>();
  for (int64_t i = 0; i < size; ++i) {
    const float* ptr_at_i = t1_const.ptr<float>(i);
    ASSERT_EQ(ptr_at_i, const_base_ptr + i);
  }
}
