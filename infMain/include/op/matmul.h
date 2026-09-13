// Updated on March 15, 2026
//
// Created by hello on 2024/5/2.
//

#ifndef KUIPER_INCLUDE_OP_MATMUL_H_
#define KUIPER_INCLUDE_OP_MATMUL_H_
#include "layer.h"
namespace op {
class MatmulLayer : public LayerParam {
 public:
  explicit MatmulLayer(base::DeviceType device_type, int32_t dim0, int32_t dim1,
                       bool is_quant_layer = false, bool has_bias = false);

  base::Status check() const override;

  base::Status forward() override;

  base::Status set_bias(int32_t idx, int32_t& dims, const void* bias_ptr,
                        base::DeviceType device_type,
                        base::DataType data_type = base::DataType::kDataTypeFp32);

  tensor::Tensor& get_bias(int32_t idx);

  const tensor::Tensor& get_bias(int32_t idx) const;

  void materialize() override;

  void to_cuda() override;

  base::Status bind_external_weights(const void* host_base, void* device_base,
                                     uint64_t allocation_bytes,
                                     base::DataType dtype, uint64_t* views,
                                     uint64_t* logical_bytes) override;

 private:
  int32_t dim0_ = 0;
  int32_t dim1_ = 0;
  bool has_bias_ = false;
  std::vector<tensor::Tensor> bias_;
};
}  // namespace op
#endif  // KUIPER_INCLUDE_OP_MATMUL_H_
