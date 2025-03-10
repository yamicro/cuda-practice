//
// Created by yami on 25-3-10.
//

#ifndef BASE_H
#define BASE_H
#include <glog/logging.h>
#include <cstdint>
#include <string>


namespace base {

enum class DeviceType: uint8_t {
    kDeviceUnknown = 0,
    kDeviceCPU = 1,
};

enum class ModelType : uint8_t {
    kModelTypeUnknown = 0,
    kModelTypeLLama2 = 1,
};

enum class DataType: uint8_t {
    kDataTypeUnknown = 0,
    kDataTypeFp32 = 1,
    kDataTypeInt8 = 2,
    kDataTypeInt32 = 3,
};

inline size_t DataTypeSize(DataType dataType) {
  if (dataType == DataType::kDataTypeFp32) {
    return sizeof(float);
  }
  if (dataType == DataType::kDataTypeInt8) {
    return sizeof(int8_t);
  }
  if (dataType == DataType::kDataTypeInt32) {
    return sizeof(int32_t);
  }
  return 0;
};

class NoCopyable {
protected:
    NoCopyable() = default;

    ~NoCopyable() = default;

    NoCopyable(const NoCopyable&) = delete;

    NoCopyable& operator=(const NoCopyable&) = delete;
};

enum StatusCode : uint8_t {
    kSuccess = 0,
    kFunctionUnImplement = 1,
    kPathNotValid = 2,
    kModelParseError = 3,
    kInternalError = 5,
    kKeyValueHasExist = 6,
    kInvalidArgument = 7,
};

class Status {
  public:
    Status(int code = StatusCode::kSuccess, std::string err_message = "");
    Status(const Status& other) = default;
    Status& operator=(const Status& other) = default;
    Status& operator=(int code);
    bool operator==(int code) const;
    bool operator!=(int code) const;
    operator int() const;
    operator bool() const;
    const std::string& get_err_message() const;
    void set_err_message(const std::string& err_message);
    private:
      int code_ = StatusCode::kSuccess;
      std::string err_message_;
};

namespace error {
#define STATUS_CHECK(call)        \
  do {                        \
    const base::Status call_status = call;  \
    if(!status) {        \
        const size_t buf_size = 512;        \
        std::unique_ptr<char[]> buffer(new char[buf_size]);  \
        snprintf(buffer, buf_size - 1, \
            "Infer error\n File:%s Line:%d\n Error code:%d\n Error msg:%s\n", __FILE__, \
        _LINE__, int(status), status.get_err_msg().c_str());                       \
        LOG(FATAL) << buf;                                                                   \
    }                                \
  } while (0)

Status Success(const std::string& err_msg = "");

Status FunctionNotImplement(const std::string& err_msg = "");

Status PathNotValid(const std::string& err_msg = "");

Status ModelParseError(const std::string& err_msg = "");

Status InternalError(const std::string& err_msg = "");

Status KeyHasExits(const std::string& err_msg = "");

Status InvalidArgument(const std::string& err_msg = "");

}  // namespace error

std::ostream& operator<<(std::ostream& os, const Status& x);
}

#endif //BASE_H
