#include <base/alloc.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <base/buffer.h>

// 测试用例：测试 1 是否等于 1
TEST(test_buffer, allocate) {
    using namespace base;
    auto alloc = base::CPUDeviceAllocatorFactory().get_instance();
    Buffer buffer(32, alloc);
    printf("The Buffer length is %lu: ", buffer.byte_size());
    ASSERT_NE(buffer.ptr(), nullptr);
}
