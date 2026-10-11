#include "serving/serving_runtime.h"

namespace serving {
KVPoolDescriptor ServingRuntime::pd_prefill_kv_pool() const {
  const ServingCapacityInfo info = pd_prefill_serving_capacity_info();
  KVPoolDescriptor pool;
  pool.device_id = bench_config().pd.prefill_device_id;
  pool.layer_num = info.layer_num;
  pool.block_size = info.block_size;
  pool.kv_head_num = info.kv_head_num;
  pool.head_size = info.head_size;
  pool.dtype = info.runtime_data_type;
  pool.storage_mode = base::BlockStorageMode::kPlain;
  return pool;
}

KVPoolDescriptor ServingRuntime::pd_decode_kv_pool() const {
  const ServingCapacityInfo info = pd_decode_serving_capacity_info();
  KVPoolDescriptor pool;
  pool.device_id = bench_config().pd.decode_device_id;
  pool.layer_num = info.layer_num;
  pool.block_size = info.block_size;
  pool.kv_head_num = info.kv_head_num;
  pool.head_size = info.head_size;
  pool.dtype = info.runtime_data_type;
  pool.storage_mode = base::BlockStorageMode::kPlain;
  return pool;
}

} // namespace serving
