#ifndef RMSNORM_H
#define RMSNORM_H
#include <op/layer.h>
#include <base/cuda_config.h>

namespace op {
    class RmsNormLayer : public LayerFp32Param {
    public:
        explicit RmsNormLayer(base::DeviceType device_type, int32_t dim);

        base::Status check() const override;

        base::Status forward() override;

    private:
        int32_t dim_ = 0;
    };
}

#endif //RMSNORM_H
