// Updated on March 15, 2026
#include "model/model.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <utility>

namespace model {

static bool ReadWeightDataTypeHeaderIfExists(FILE* file, base::DataType* data_type,
                                             int32_t* extra_bytes) {
  long pos_before = ftell(file);
  if (pos_before < 0) {
    return false;
  }
  WeightDataTypeHeader header{};
  size_t read_count = fread(&header, sizeof(WeightDataTypeHeader), 1, file);
  if (read_count != 1) {
    fseek(file, pos_before, SEEK_SET);
    return false;
  }
  if (header.magic != kWeightDataTypeMagic) {
    fseek(file, pos_before, SEEK_SET);
    return false;
  }
  if (data_type) {
    *data_type = static_cast<base::DataType>(header.data_type);
  }
  if (extra_bytes) {
    *extra_bytes += static_cast<int32_t>(sizeof(WeightDataTypeHeader));
  }
  return true;
}

static bool ReadMoeHeaderIfExists(FILE* file, TransformerConfig* cfg, int32_t* extra_bytes) {
  long pos_before = ftell(file);
  if (pos_before < 0) {
    return false;
  }
  MoeHeader header{};
  size_t read_count = fread(&header, sizeof(MoeHeader), 1, file);
  if (read_count != 1) {
    fseek(file, pos_before, SEEK_SET);
    return false;
  }
  if (header.magic != kMoeMagic) {
    fseek(file, pos_before, SEEK_SET);
    return false;
  }
  if (cfg) {
    cfg->moe_expert_num_ = header.moe_expert_num;
    cfg->moe_topk_ = header.moe_topk;
    cfg->moe_shared_expert_num_ = header.moe_shared_expert_num;
    cfg->moe_hidden_dim_ = header.moe_hidden_dim;
    cfg->moe_shared_hidden_dim_ = header.moe_shared_hidden_dim;
    cfg->moe_sparse_step_ = header.moe_sparse_step;
    cfg->moe_norm_topk_prob_ = header.moe_norm_topk_prob;
  }
  
  if (extra_bytes) {
    *extra_bytes += static_cast<int32_t>(sizeof(MoeHeader));
  }
  return true;
}


Model::Model(base::TokenizerType tokenizer_type, base::ModelType model_type, std::string token_path,
             std::string model_path, bool is_quant_model)
    : tokenizer_type_(tokenizer_type),
      model_type_(model_type),
      token_path_(std::move(token_path)),
      model_path_(std::move(model_path)),
      is_quant_model_(is_quant_model) {}

base::ModelType Model::model_type() const { return model_type_; }

const std::string& Model::token_path() const { return token_path_; }

const std::string& Model::model_path() const { return model_path_; }

void Model::set_runtime_data_type(base::DataType data_type) { runtime_data_type_ = data_type; }

base::DataType Model::runtime_data_type() const { return runtime_data_type_; }

void Model::set_kv_cache_storage_mode(base::BlockStorageMode storage_mode) {
  kv_cache_storage_mode_ = storage_mode;
}

base::BlockStorageMode Model::kv_cache_storage_mode() const {
  return kv_cache_storage_mode_;
}

base::KVCacheStorageSpec Model::kv_cache_storage_spec() const {
  return base::MakeKVCacheStorageSpec(runtime_data_type_, kv_cache_storage_mode_);
}

base::Status Model::validate_kv_cache_runtime(base::DeviceType device_type) const {
  return base::ValidateKVCacheStorageSpec(kv_cache_storage_spec(), device_type, runtime_data_type_);
}

ServingWorkspaceProfile Model::serving_workspace_profile(
    base::DataType /*runtime_data_type*/) const {
  return {};
}

size_t Model::serving_workspace_bytes_per_token(base::DataType runtime_data_type) const {
  return serving_workspace_profile(runtime_data_type).total_bytes_per_token();
}

void Model::set_device_context(std::shared_ptr<base::DeviceContext> context) {
  device_context_ = std::move(context);
}

std::shared_ptr<base::DeviceContext> Model::device_context() const {
  return device_context_;
}

base::Status Model::insert_buffer(ModelBufferType buffer_idx, const tensor::Tensor& tensor) {
  if (buffers_.count(buffer_idx) > 0) {
    return base::error::KeyHasExits(std::to_string(int(buffer_idx)) + " has exits in the buffers");
  }
  if (tensor.is_empty()) {
    return base::error::InvalidArgument("The tensor is empty for inserting buffer.");
  }
  buffers_.insert({buffer_idx, tensor});
  return base::error::Success();
}

tensor::Tensor& Model::get_buffer(ModelBufferType buffer_idx) {
  CHECK_GT(buffers_.count(buffer_idx), 0) << int(buffer_idx);
  return buffers_.at(buffer_idx);
}

const tensor::Tensor& Model::get_buffer(ModelBufferType buffer_idx) const {
  CHECK_GT(buffers_.count(buffer_idx), 0);
  return buffers_.at(buffer_idx);
}

base::Status Model::read_model_file() {
  using namespace base;
  if (model_path_.empty()) {
    return error::PathNotValid("Failed to open the weight file, the model path is empty!");
  }
  int32_t fd = open(model_path_.data(), O_RDONLY);
  if (fd == -1) {
    return error::PathNotValid("Failed to open the weight file " + model_path_ +
                               " may be the path does not exist!");
  }

  FILE* file = fopen(model_path_.data(), "rb");
  if (!file) {
    return error::PathNotValid("Failed to open the file. The path may be invalid.");
  }

  auto config = ModelConfig{};
  if (fread(&config, sizeof(ModelConfig), 1, file) != 1) {
    return error::ModelParseError(
        "Failed to retrieve the configuration information from the model "
        "file.");
  }
  if (is_quant_model_) {
    if (fread(&group_size_, sizeof(int32_t), 1, file) != 1) {
      return error::ModelParseError(
          "Failed to retrieve the group size information from the model "
          "file.");
    }
  }

  int32_t header_extra_bytes = 0;
  base::DataType weight_data_type = is_quant_model_ ? base::DataType::kDataTypeInt8
                                                    : base::DataType::kDataTypeFp32;
  if (!is_quant_model_) {
    ReadWeightDataTypeHeaderIfExists(file, &weight_data_type, &header_extra_bytes);
  }
  ReadMoeHeaderIfExists(file, config_.get(), &header_extra_bytes);

  if (weight_data_type == base::DataType::kDataTypeFp32) {
    raw_model_data_ = std::make_shared<RawModelDataFp32>();
  } else if (weight_data_type == base::DataType::kDataTypeInt8) {
    raw_model_data_ = std::make_shared<RawModelDataInt8>();
  } else if (weight_data_type == base::DataType::kDataTypeBf16) {
    raw_model_data_ = std::make_shared<RawModelDataBf16>();
  } else {
    return error::ModelParseError("Unsupported weight data type in model file.");
  }
  raw_model_data_->data_type = weight_data_type;
  raw_model_data_->header_extra_bytes = header_extra_bytes;

  auto gen_status = generate_model_infos(config);
  if (!gen_status) {
    return gen_status;
  }

  struct stat sb;
  if (fstat(fd, &sb) == -1) {
    close(fd);
    return error::ModelParseError(
        "Failed to retrieve the file size information from the model "
        "file.");
  }
  raw_model_data_->file_size = sb.st_size;

  raw_model_data_->fd = fd;
  raw_model_data_->data =
      mmap(nullptr, raw_model_data_->file_size, PROT_READ, MAP_PRIVATE, raw_model_data_->fd, 0);

  if (raw_model_data_->data == MAP_FAILED || raw_model_data_->data == nullptr) {
    return error::ModelParseError("Failed to map the weight file " + model_path_ + " into memory.");
  }
  if (!is_quant_model_) {
    raw_model_data_->weight_data =
        static_cast<int8_t*>(raw_model_data_->data) + sizeof(ModelConfig) +
        raw_model_data_->header_extra_bytes;
  } else {
    raw_model_data_->weight_data =
        static_cast<int8_t*>(raw_model_data_->data) + sizeof(ModelConfig) + sizeof(group_size_) +
        raw_model_data_->header_extra_bytes;
  }
  if (raw_model_data_ == nullptr) {
    LOG(ERROR);
    return error::ModelParseError("Failed to map the weight file " + model_path_ +
                                  " into memory, the pointer to weight start address is null");
  }
  return error::Success();
}

base::Status Model::generate_model_infos(const ModelConfig& config) const {
  config_->dim_ = config.dim;
  config_->hidden_dim_ = config.hidden_dim;
  config_->layer_num_ = config.layer_num;
  config_->head_num_ = config.head_num;
  config_->kv_head_num_ = config.kv_head_num;
  config_->seq_len_ = config.seq_len;

  config_->kv_dim_ = (config.dim * config.kv_head_num) / config.head_num;
  config_->kv_mul_ = config.head_num / config.kv_head_num;
  config_->head_size_ = config.dim / config.head_num;
#if defined(QWEN3_SUPPORT)
  config_->immediate_dim_ = config.immediate_dim_;
#endif
  if (config.vocab_size > 0) {
    config_->is_shared_weight_ = true;
  } else {
    config_->is_shared_weight_ = false;
  }

  // Qwen tokenizer size and embedding size is mismatched
  // refer: https://github.com/QwenLM/Qwen2.5/issues/29
  // if (std::abs(config.vocab_size) != config_->vocab_size_) {
  //   return base::error::ModelParseError(
  //       "Vocabulary size mismatch between the model file and the token list.");
  // }
  config_->vocab_size_ = std::abs(config.vocab_size);
  return base::error::Success();
}

base::Status Model::create_encode_layer() {
  using namespace base;

  // create token encode decode layer
  if (tokenizer_type_ == TokenizerType::kEncodeSpe) {
    encode_layer_ = std::make_unique<op::SpeEncodeLayer>(this->token_path_, true, false);
  } else {
#ifdef LLAMA3_SUPPORT
    encode_layer_ = std::make_unique<op::BpeEncodeLayer>(this->token_path_, true, false);
#endif

#if defined(QWEN2_SUPPORT) || defined(QWEN3_SUPPORT) || defined(QWEN_MOE_SUPPORT)
    encode_layer_ = std::make_unique<op::QwenEncodeLayer>(this->token_path_, false, false);
#endif
  }
  if (!encode_layer_) {
    return error::InternalError("Create the encode layer failed.");
  }

  config_->vocab_size_ = encode_layer_->vocab_size();
  if (config_->vocab_size_ <= 0) {
    return error::InternalError("The vocab size param read error from the model file!");
  }
  return error::Success();
}

base::Status Model::gen_model_from_file() {
  using namespace base;
  config_ = std::make_unique<TransformerConfig>();

  // init sentence piece processor
  // google sentence piece
  auto create_encode_status = create_encode_layer();
  if (!create_encode_status) {
    LOG(ERROR) << "Create the encode layer failed!";
    return create_encode_status;
  }
  // mmap
  auto mmap_status = read_model_file();
  if (!mmap_status) {
    LOG(ERROR) << "Handle model file " << model_path_ << " failed!";
    return mmap_status;
  }
  auto layer_create_status = create_layers();
  if (!layer_create_status) {
    LOG(ERROR) << "Create layers for the model file " << model_path_ << " failed!";
    return layer_create_status;
  }

  return error::Success();
}

std::vector<int32_t> Model::encode(const std::string& sentence) const {
  CHECK(encode_layer_ != nullptr);
  return encode_layer_->encode(sentence);
}

bool Model::is_sentence_ending(int32_t token_idx) const {
  CHECK(this->encode_layer_ != nullptr);
  return this->encode_layer_->is_sentence_ending(token_idx);
}

std::string Model::decode(int32_t token_idx) const {
  CHECK(this->encode_layer_ != nullptr);
  return this->encode_layer_->decode(token_idx);
}

std::string Model::decode(const std::vector<int32_t>& token_idxs) const {
  CHECK(this->encode_layer_ != nullptr);
  return this->encode_layer_->decode(token_idxs);
}

std::pair<tensor::Tensor, tensor::Tensor> Model::slice_kv_cache(int32_t layer_idx,
                                                                int32_t token_pos) const {
  int32_t layer_offset = layer_idx * config_->seq_len_ * config_->kv_dim_;
  int32_t cache_offset = layer_offset + token_pos * config_->kv_dim_;
  const auto& key_cache = get_buffer(ModelBufferType::kKeyCache);
  const auto& value_cache = get_buffer(ModelBufferType::kValueCache);

  void* key_cache_ptr = nullptr;
  void* val_cache_ptr = nullptr;
  if (key_cache.data_type() == base::DataType::kDataTypeFp32) {
    key_cache_ptr = const_cast<float*>(key_cache.ptr<float>(cache_offset));
    val_cache_ptr = const_cast<float*>(value_cache.ptr<float>(cache_offset));
  } else if (key_cache.data_type() == base::DataType::kDataTypeBf16) {
    key_cache_ptr = const_cast<uint16_t*>(key_cache.ptr<uint16_t>(cache_offset));
    val_cache_ptr = const_cast<uint16_t*>(value_cache.ptr<uint16_t>(cache_offset));
  } else {
    LOG(FATAL) << "Unsupported kv cache dtype: " << key_cache.data_type();
  }

  tensor::Tensor key(key_cache.data_type(), config_->kv_dim_, false, nullptr, key_cache_ptr);
  tensor::Tensor val(value_cache.data_type(), config_->kv_dim_, false, nullptr, val_cache_ptr);
  key.set_device_type(device_type_);
  val.set_device_type(device_type_);
  return {key, val};
}

tensor::Tensor Model::fill_input(const tensor::Tensor& pos_tensor,
                                 const op::EmbeddingOutput& embedding_output,
                                 bool is_prompt) const {
  const int32_t pos = pos_tensor.index<int32_t>(0);
  auto [input_tokens, input_embeddings, input_token_num] = embedding_output;

  int32_t index = 0;
  if (is_prompt) {
    index = pos;
  }
#if defined(QWEN3_SUPPORT)
  std::shared_ptr<base::Buffer> input_emb_buffer = std::make_shared<base::Buffer>(
      config_->hidden_dim_ * base::DataTypeSize(input_embeddings.data_type()), nullptr,
      const_cast<void*>(reinterpret_cast<const void*>(input_embeddings.ptr<uint8_t>(
          index * config_->hidden_dim_ * base::DataTypeSize(input_embeddings.data_type())))),
      true);
  tensor::Tensor input(input_embeddings.data_type(), config_->hidden_dim_);

#else
  std::shared_ptr<base::Buffer> input_emb_buffer =
      std::make_shared<base::Buffer>(
          config_->dim_ * base::DataTypeSize(input_embeddings.data_type()), nullptr,
          const_cast<void*>(reinterpret_cast<const void*>(input_embeddings.ptr<uint8_t>(
              index * config_->dim_ * base::DataTypeSize(input_embeddings.data_type())))),
          true);
  tensor::Tensor input(input_embeddings.data_type(), config_->dim_);
#endif
  input.assign(input_emb_buffer);
  input.set_device_type(device_type_);
  return input;
}

}  // namespace model
