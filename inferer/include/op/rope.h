#ifndef ROPE_H
#define ROPE_H
#include "layer.h"
namespace op {
    class RoPELayer : public Layer {
    public:
        explicit RoPELayer(base::DeviceType device_type, int32_t dim, int32_t kv_dim, int32_t head_size);

        base::Status check() const override;

        base::Status forward() override;

    private:
        int32_t dim_ = 0;
        int32_t head_size_ = 0;
    };
}
#endif //ROPE_H
