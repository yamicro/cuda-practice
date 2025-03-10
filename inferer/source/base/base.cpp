//
// Created by yami on 25-3-10.
//

#include "base/base.h"
#include <string>


namespace base {
    Status::Status(int code, std::string err_message): code_(code), err_message_(std::move(err_message)) {}

    Status& Status::operator=(int code) {
      code_ = code;
      return *this;
    }

    bool Status::operator==(int code) const {
        if (code_ == code) {
            return true;
        } else {
            return false;
        }
    }

    bool Status::operator!=(int code) const {
      if (code_ == code) {
        return false;
      }
      return true;
    }

    Status::operator int() const { return code_; }

    Status::operator bool() const { return code_ == kSuccess; }

    const std::string& Status::get_err_message() const { return err_message_; }

    void Status::set_err_message(const std::string& err_message) { err_message_ = err_message; }

namespace error {

    Status Success(const std::string& err_msg) { return Status{kSuccess, err_msg}; };

    Status FunctionNotImplement(const std::string& err_msg) { return Status{kFunctionUnImplement, err_msg}; };

    Status PathNotValid(const std::string& err_msg) { return Status{kPathNotValid, err_msg}; };

    Status ModelParseError(const std::string& err_msg) { return Status{kModelParseError, err_msg}; };

    Status InvalidArgument(const std::string& err_msg) { return Status{kInvalidArgument, err_msg}; };

    Status KeyHasExits(const std::string& err_msg) { return Status{kKeyValueHasExist, err_msg}; };

    Status InternalError(const std::string& err_msg) { return Status{kInternalError, err_msg}; };
}

std::ostream& operator<<(std::ostream& os, const Status& x) {
    os << x.get_err_message();
    return os;
}
} // base