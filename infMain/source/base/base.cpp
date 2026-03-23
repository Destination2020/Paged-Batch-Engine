// Updated on March 15, 2026
#include "base/base.h"

#include <string>
namespace base {
Status::Status(int code, std::string err_message)
    : code_(code), message_(std::move(err_message)) {}

Status& Status::operator=(int code) {
  code_ = code;
  return *this;
};

bool Status::operator==(int code) const {
  if (code_ == code) {
    return true;
  } else {
    return false;
  }
};

bool Status::operator!=(int code) const {
  if (code_ != code) {
    return true;
  } else {
    return false;
  }
};

Status::operator int() const { return code_; }

Status::operator bool() const {
  return code_ == kSuccess;
}

int32_t Status::get_err_code() const {
  return code_;
}

const std::string& Status::get_err_msg() const { return message_; }

void Status::set_err_msg(const std::string& err_msg) { message_ = err_msg; }

namespace error {
Status Success(const std::string& err_msg) { return Status{kSuccess, err_msg}; }

Status FunctionNotImplement(const std::string& err_msg) {
  return Status{kFunctionUnImplement, err_msg};
}

Status PathNotValid(const std::string& err_msg) {
  return Status{kPathNotValid, err_msg};
}

Status ModelParseError(const std::string& err_msg) {
  return Status{kModelParseError, err_msg};
}

Status InternalError(const std::string& err_msg) {
  return Status{kInternalError, err_msg};
}

Status InvalidArgument(const std::string& err_msg) {
  return Status{kInvalidArgument, err_msg};
}

Status KeyHasExits(const std::string& err_msg) {
  return Status{kKeyValueHasExist, err_msg};
}
}  // namespace error

std::ostream& operator<<(std::ostream& os, const Status& x) {
  os << x.get_err_msg();
  return os;
}

std::ostream& operator<<(std::ostream& os, DeviceType x) {
  switch (x) {
    case DeviceType::kDeviceUnknown:
      os << "DeviceUnknown";
      break;
    case DeviceType::kDeviceCPU:
      os << "DeviceCPU";
      break;
    case DeviceType::kDeviceCUDA:
      os << "DeviceCUDA";
      break;
    default:
      os << "Device(" << static_cast<int>(x) << ")";
      break;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, DataType x) {
  switch (x) {
    case DataType::kDataTypeUnknown:
      os << "Unknown";
      break;
    case DataType::kDataTypeFp32:
      os << "Fp32";
      break;
    case DataType::kDataTypeInt8:
      os << "Int8";
      break;
    case DataType::kDataTypeInt32:
      os << "Int32";
      break;
    case DataType::kDataTypeBf16:
      os << "Bf16";
      break;
    default:
      os << "DataType(" << static_cast<int>(x) << ")";
      break;
  }
  return os;
}

}  // namespace base
